import asyncio
import numpy as np
import torch
from arsenal import colors
from arsenal.maths import log1mexp
from genlm.control.util import logsumexp
import warnings

from genlm.control.util import select
from genlm.control.sampler.set import SetSampler
from genlm.control.sampler.util import _validate_proposal_vocab, _drive_or_hop
from genlm.control.sampler.controller import BurstDraw


# `_drive_sync` / `_drive_or_hop` / `_CoroutineSuspended` live in `sampler.util`
# so the Controller (which `token` imports from) can share the same inline-or-hop
# primitive for its factor eval without a circular import.


class TokenSampler:
    """Base class for sampling a token from a potential's vocabulary.

    `TokenSampler`s generate properly weighted samples with respect to a `target` potential.

    Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
    a `TokenSampler` samples a token $x_n \\in \\textsf{target.vocab_eos}$ and weight $w$.

    The sampled token and weight are properly weighted with respect to
    $$
    \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
    $$

    A `TokenSampler` collapses to a single per-step transition that the SMC controller
    calls: :meth:`transition` maps a particle context to
    ``(to_append, logw, logp)``. The controller owns the population, ESS test,
    resampling and log marginal likelihood; samplers never reimplement the loop.

    Args:
        target (Potential): The potential that samples are properly weighted with respect to.
    """

    def __init__(self, target):
        self.target = target
        self.token_type = self.target.token_type

    def supports_burst(self) -> bool:
        """Whether this sampler can be driven inside the engine burst -- i.e. it
        implements :meth:`burst_draw_batch`. Default ``False`` (stays on ``StepLoop``).

        The control loop (``Controller``/``BurstLoop``) is uniform; each
        burst-capable sampler plugs its own batched draw in via ``burst_draw_batch``,
        the same way every sampler plugs its slow per-step draw in via
        ``transition`` -- no sampler-type dispatch in the loop.
        """
        return False

    def burst_free_running(self) -> bool:
        """Whether this sampler's burst is FREE-RUNNING (token grain).

        Free-running means the sampler completes one SMC step every engine decode
        step, so all rows advance one token per step and stay synchronized for free:
        the controller tests ESS after every step and pops every row at the
        crossing. A SYNCHRONIZED grain (the unit sampler) returns ``False`` -- it
        completes one SMC step (a whole unit) only at a per-row boundary, so the
        controller runs exactly one unit round per burst, pops each row at its
        boundary, and tests ESS once at the synced boundary (matching the slow
        per-unit loop's timing). Default ``True`` (token grain)."""
        return True

    def burst_max_steps(self, live) -> int:
        """Engine decode-step budget for ONE burst over the ``live`` particles.

        Token grain: long enough to reach the longest particle's remaining token
        budget (+1 so a budget-exhausted row is dropped by the control's abort, not
        length-capped by the engine first). The unit sampler overrides this with one
        unit's worth of subunit steps."""
        return max(p.max_tokens_left for p in live) + 1

    def burst_draw_batch(self, lm_batch, contexts, externals, ctx):
        """Engine-burst draw for the WHOLE batch of live rows (the burst analog of
        ``sample``, vectorized).

        ``lm_batch`` is the ``[L, V+1]`` on-device control-vocab log-weights for
        the L live rows (``PromptedLLM._process_logw_next_batch`` -- the expensive
        50k-vocab processing done ONCE for all rows, never per-row). ``contexts``
        are the L particles' token contexts (the factor is scored from them via
        ``ctx.product_logws``; its ``_consume`` cache keeps that incremental), and
        ``externals`` their stable particle indices (so a unit sampler can key a
        per-burst accumulator across the shrinking live set); ``ctx`` is the
        sampler-facing :class:`~genlm.control.sampler.controller.BurstContext`.

        Returns a list of L :class:`~genlm.control.sampler.controller.BurstDraw`,
        one per row -- the per-row analog of ``sample`` plus the engine-continuation
        token and (for the unit grain) a completed-step / pop signal. The Controller
        banks the step / advances / terminates per particle; the sampler only
        computes the draw. Samplers that ride the burst (``supports_burst()``)
        override; the LM half is already batched, so a sampler vectorizes its control
        where it can (Direct) or loops over the cheap ``[V+1]`` proposals where the
        control is inherently per-particle (AWRS rejection, Set trie).
        """
        raise NotImplementedError

    async def start_weight(self):
        """Compute the weight of the empty sequence under the target potential."""
        return await self.target.prefix([])

    async def transition(self, context):
        """The controller-facing per-step transition.

        Draws a single token, returning the list of items to append to the
        particle context together with the importance-weight increment and the
        log-probability of the random choices.

        Args:
            context (list): The particle's current token context.

        Returns:
            (to_append, logw, logp): ``to_append`` is ``[token]`` (subclasses
                that produce multi-token units may return more than one item),
                ``logw`` is the importance-weight increment, ``logp`` the
                log-probability of the sampler's random choices.
        """
        token, logw, logp = await self.sample(context)
        return [token], logw, logp

    async def sample(self, context, *args, **kwargs):
        """Sample a token and weight from the `target`potential's vocabulary.

        Args:
            context (list[int]): A sequence of tokens in the `target` potential's vocabulary.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the sampled token.
        """
        raise NotImplementedError(
            "Subclasses must implement sample method"
        )  # pragma: no cover

    async def cleanup(self):
        pass  # pragma: no cover

    async def smc(
        self,
        n_particles,
        ess_threshold,
        max_tokens,
        critic=None,
        *,
        accelerate="auto",
        **kwargs,
    ):
        """Generate sequences using sequential Monte Carlo (SMC) inference with this token sampler and an optional critic.

        This method is a convenience wrapper around [`SMC`][genlm.control.sampler.sequence.SMC].
        See [`SMC`][genlm.control.sampler.sequence.SMC] for more details on the generation process.

        Args:
            n_particles (int): The number of particles to use in the SMC algorithm.
            ess_threshold (float): The threshold for the effective sample size (ESS).
            max_tokens (int): The maximum number of tokens to generate.
            critic (Potential, optional): A potential function that guides the generation process
                by scoring candidate sequences. Must have the same token type as the token sampler.
            accelerate (str | bool, optional): Engine-acceleration knob, forwarded to
                `SMC.__call__`. One of ``"auto"`` (default; engine path when
                burst-capable, else the exact per-token path), ``"off"`` (force the
                exact per-token path), or ``"require"`` (engine path or raise
                `NotAcceleratable`). ``True``/``False`` alias ``"auto"``/``"off"``.
                See `SMC.__call__` for the full contract.
            **kwargs (dict): Additional keyword arguments to pass to `SMC`'s `__call__` method.
        """
        from genlm.control.sampler.sequence import SMC

        return await SMC(self, critic)(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            accelerate=accelerate,
            **kwargs,
        )


class DirectTokenSampler(TokenSampler):
    """Samples individual tokens directly from the log-normalized `logw_next` function
    of a potential.

    Args:
        potential (Potential): The potential function to sample from.
        proposal (Potential, optional): If supplied, tokens are drawn from
            `proposal.logw_next` and reweighted by `target/proposal` so the result
            stays properly weighted with respect to `potential.logw_next`. Must
            share `potential.vocab_eos` (cross-tokenizer not yet supported). When
            `None` (the default), the target acts as its own proposal. The proposal
            must place positive mass on every token the target weights positively.

    Warning:
        Only use this sampler if the potential's `logw_next` method is efficient. This is the case
        for potentials like `PromptedLLM`, but for custom potentials with a large vocabulary size,
        the default implementation of `logw_next` generally will not be efficient, and thus this
        sampler will be slow.
    """

    def __init__(self, potential, proposal=None):
        super().__init__(target=potential)
        self.potential = potential
        if proposal is not None:
            _validate_proposal_vocab(potential, proposal)
        self.proposal = proposal

    def supports_burst(self) -> bool:
        # An unbiased direct sampler (no separate proposal) draws exactly from
        # the target's normalized ``logw_next``, which the engine reproduces. With
        # a proposal the per-step importance weight is non-trivial, so it stays
        # on the slow loop.
        return self.proposal is None

    def burst_draw_batch(self, lm_batch, contexts, externals, ctx):
        """Batched burst draw (the burst analog of ``sample``), one BurstDraw per row.

        ``lm_batch`` is the ``[L, V+1]`` on-device LM log-weights (already
        processed ONCE for the whole batch). Two cases:

        * **unconstrained** (no factor): normalize + Gumbel-max across all rows in
          one on-device op; only the L drawn ids come back. This is a batched
          torch Gumbel-max -- a different RNG stream than the slow numpy
          numpy ``select``, so the burst samples the same distribution
          without tracking the slow path token-for-token (parity stays the no-bias
          check the warm-KV residual already requires).
        * **factored** (additive ``ctx.factor``): the LM half is already batched;
          per particle, form ``Product(llm, factor).logw_next`` from the row's
          context (``ctx.product_logws`` scores ``factor.logw_next(context)``, kept
          incremental by the factor's ``_consume`` cache) and draw with the slow
          path's exact numpy ``select`` (same RNG -> tight
          warm-KV-only parity). The per-particle product is the one bit still
          looped -- vectorizing it is the eventual ideal.

        Token grain: every row completes one SMC step every decode step, so each
        ``BurstDraw`` carries ``step=([token], logw, logp)`` and never pops mid-burst
        (the ESS crossing pops the whole population, controller-side). ``externals``
        is unused (no per-row burst state to key).
        """
        if ctx.factor is None:
            logZ = torch.logsumexp(lm_batch, dim=1)  # [L] proposal log-mass (~0)
            logps = lm_batch - logZ[:, None]  # [L, V+1] normalized
            u = torch.rand_like(logps).clamp_(min=torch.finfo(logps.dtype).tiny)
            idx = (logps - torch.log(-torch.log(u))).argmax(dim=1)  # Gumbel-max, on-device
            rows = torch.arange(idx.shape[0], device=idx.device)
            logp_drawn = logps[rows, idx].tolist()
            logZ = logZ.tolist()
            vocab_eos = self.target.vocab_eos
            tokens = [vocab_eos[j] for j in idx.tolist()]
            return [
                BurstDraw(token=t, step=([t], zt, lpt))
                for t, zt, lpt in zip(tokens, logZ, logp_drawn)
            ]

        # Factored: one host transfer of the batched LM weights, then per-particle
        # product + numpy draw over the cheap [V+1] proposals (no 50k re-processing).
        lm_np = lm_batch.cpu().numpy()
        out = []
        for k, context in enumerate(contexts):
            logw = ctx.product_logws(lm_np[k], context)
            logps = logw.normalize()
            token = select(logps)
            out.append(BurstDraw(token=token, step=([token], logw.sum(), logps[token])))
        return out

    async def sample(self, context, draw=None):
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
        this method samples a token $x_n \\in \\textsf{target.vocab_eos}$ and weight $w$.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        Without a proposal, the returned weight is the log normalizing constant
        of $\\textsf{target.logw_next}$. With a proposal $q$, the returned weight
        is the importance weight
        $\\textsf{target.logw_next}[x_n] - \\log q_{\\text{norm}}(x_n)$.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the sampled token.
        """
        if self.proposal is None:
            logws = await self.potential.logw_next(context)
            logps = logws.normalize()
            token = select(logps) if draw is None else draw(logps.exp().materialize())
            return token, logws.sum(), logps[token]

        proposal_logws, target_logws = await asyncio.gather(
            self.proposal.logw_next(context),
            self.potential.logw_next(context),
        )
        proposal_logps = proposal_logws.normalize()
        if draw is None:
            token = select(proposal_logps)
        else:
            token = draw(proposal_logps.exp().materialize())
        logw = target_logws[token] - proposal_logws[token] + proposal_logws.sum()
        return token, logw, proposal_logps[token]

    async def cleanup(self):
        pass  # pragma: no cover


class SetTokenSampler(TokenSampler):
    """Samples individual tokens by sampling a weighted set of tokens and then selecting one
    proportional to its weight.

    This class wraps a `SetSampler`.

    Args:
        set_sampler (SetSampler): The set sampler to sample from
    """

    def __init__(self, set_sampler):
        assert isinstance(set_sampler, SetSampler)
        super().__init__(set_sampler.target)
        self.set_sampler = set_sampler

    def supports_burst(self) -> bool:
        # Decision 2 (UX): reported as NOT engine-accelerated for now. The burst
        # machinery for the set draw exists and is correct (see ``burst_draw_batch``
        # below and the gate-2 set parity test), but in-engine it is marginal/at-best
        # and can regress versus the per-token path until the trie is vectorized, so
        # the capability gate routes Set to ``StepLoop`` rather than silently
        # delivering a possible slowdown. ``burst_capability`` reports the reason
        # "SetTokenSampler is not engine-accelerated".
        return False

    def burst_draw_batch(self, lm_batch, contexts, externals, ctx):
        """Set construction over the engine's warm-KV iterable weights, one BurstDraw
        per row. The LM half (``iter_potential.logw_next``) is batched ONCE; the
        per-particle set draw (the exact slow ``sample`` with only the iterable
        weights substituted) is then run for ALL rows in a SINGLE hop via
        ``asyncio.gather`` -- so the backend trie / ``item_potential`` calls batch
        across the population exactly as they do under the slow path's
        ``asyncio.gather``. (The old per-particle ``run_sync`` serialized them, one
        hop each, losing that batching.) Token grain: each row completes one SMC
        step per decode step. ``externals`` is unused (the set sampler carries no
        per-row burst state)."""
        iter_potential = self.set_sampler.iter_potential
        lm_np = lm_batch.cpu().numpy()  # one host transfer of the batched LM weights
        iter_logws = [
            iter_potential.make_lazy_weights(lm_np[k]) for k in range(len(contexts))
        ]

        async def _draw_one(context, il):
            # Mirrors the slow ``sample`` per particle, with iter weights substituted.
            logws, logp = await self.set_sampler.sample_set(context, iter_logws=il)
            logps = logws.normalize()
            token = select(logps)
            return token, logws.sum(), logp + logps[token]

        async def _gather():
            return await asyncio.gather(
                *(_draw_one(c, il) for c, il in zip(contexts, iter_logws))
            )

        # ONE event-loop hop for the whole population; the concurrent sample_sets
        # let the backend trie batch across particles.
        results = ctx.run_sync(_gather())
        return [
            BurstDraw(token=tok, step=([tok], logw, logp))
            for (tok, logw, logp) in results
        ]

    async def sample(self, context, draw=None):
        """Sample a token and weight by sampling a weighted set of tokens from the `set_sampler`
        and then selecting one proportional to its weight.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the vocabulary of the set sampler's target potential,
        this method samples a token $x_n \\in \\textsf{set_sampler.target.vocab_eos}$ and a weight.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{set_sampler.target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        The returned weight corresponds to the sum of the weights of the sampled set.

        Args:
            context (list[int]): A sequence of tokens in the vocabulary of the set sampler's target potential.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the random
                choices made in sampling that token.

        Note:
            For properly weighted sampling, the `set_sampler` must assign correct weights to each token. See
            `SetSampler` for more details.
        """
        logws, logp = await self.set_sampler.sample_set(context, draw=draw)
        logps = logws.normalize()
        token = select(logps) if draw is None else draw(logps.exp().materialize())
        return token, logws.sum(), logp + logps[token]

    async def cleanup(self):
        """Clean up the sampler.

        This method should be called when the sampler is no longer needed.
        """
        await self.set_sampler.cleanup()


class AWRS(TokenSampler):
    """Samples individual tokens through an adaptive weighted rejection sampling algorithm.

    This sampler is based on the algorithm described in [Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling](https://arxiv.org/abs/2504.05410)

    It draws properly weighted samples from the product of a non-boolean potential and a boolean condition.

    Args:
        potential (Potential): The non-boolean potential.
        condition (Potential): The boolean condition. This potential must only output boolean values (0 or -inf in log-space).
        seed (int or None): The seed for the random number generator.
        prune_logws (bool): Whether to prune the logws to only include the tokens in the intersection of the potential and condition vocabularies
        proper_weights (bool): Whether to return properly weighted samples.
            If False, the sampler will only run one round of adaptive rejection sampling.
        max_accepts (int): The maximum number of tokens to accept - higher values will decrease the variance of the weight estimate.
        max_rejects (int or float('inf')): The maximum number of tokens to reject - lower values will run faster, but at the cost of returning a weight of zero for some samples where there are tokens that would be accepted if tested.
        n_monte_carlo_samples (int): The number of Monte Carlo samples to use to estimate the weight. Higher values will decrease the variance of the weight estimate, but will run slower.
        proposal (Potential, optional): If supplied, the rejection loop proposes
            from `proposal.logw_next` instead of `potential.logw_next`, and the
            returned weight is corrected by `target/proposal` so the sample stays
            properly weighted with respect to `(potential * condition).logw_next`.
            Must share `potential.vocab_eos` (cross-tokenizer not supported). With
            `proper_weights=False`, the proposal still steers sampling but no
            correction is applied (matching the `proper_weights=False` contract).
            The proposal must place positive mass on every token the target
            weights positively.
    """

    def __init__(
        self,
        potential,
        condition,
        seed=None,
        prune_logws=True,
        proper_weights=True,
        max_accepts=2,
        max_rejects=float("inf"),
        n_monte_carlo_samples=None,
        proposal=None,
    ):
        super().__init__(target=potential * condition)
        self.potential = potential
        self.condition = condition
        if proposal is not None:
            _validate_proposal_vocab(potential, proposal)
        self.proposal = proposal

        self.prune_logws = prune_logws
        self.proper_weights = proper_weights

        if max_accepts < 2 and proper_weights:
            raise ValueError("`max_accepts` must be at least 2")

        if max_rejects < 2 and proper_weights:
            raise ValueError("`max_rejects` must be at least 2")

        if n_monte_carlo_samples is not None:
            warnings.warn(
                "n_monte_carlo_samples no longer does anything.",
                DeprecationWarning,
            )

        self.max_accepts = max_accepts
        self.max_rejects = max_rejects or float("inf")

        self.valid_idxs = np.array(
            [self.potential.lookup[t] for t in self.target.vocab_eos]
        )

        self.vocab_eos_set = set(self.target.vocab_eos)
        self.V = len(self.potential.vocab_eos)
        self.rng = np.random.default_rng(seed=seed)

    def supports_burst(self) -> bool:
        # No separate proposal: the rejection runs over the engine LM logits with
        # the boolean condition checked per probed token via `self._accept`
        # (`condition.prefix`/`complete`), exactly as the slow `sample` path.
        return self.proposal is None

    def burst_draw_batch(self, lm_batch, contexts, externals, ctx):
        """AWRS rejection over the engine LM weights, one BurstDraw per row. The LM
        half is batched ONCE; per particle, reconstruct the cheap ``[V+1]`` proposal
        and run the shared ``_run_rejection`` with the SAME ``self._accept(context,
        tok)`` the slow ``sample`` path uses -- ``condition.prefix(context + [tok]) ==
        0`` (``complete(context)`` for EOS). Only the few tokens the rejection walk
        PROBES are checked, NEVER the whole vocab; the condition's ``_consume`` cache
        keeps each probe incremental (only ``f([tok])``'s bytes are newly consumed).
        ``_accept`` is pure-CPU async for an FSA condition, so the rejection coroutine
        never suspends and ``_drive_sync`` runs it inline (no per-particle event-loop
        hop). Same rejection algorithm + returned weight as the slow path (``logp`` is
        ``nan`` as in ``sample``). Token grain: each row completes one SMC step per
        decode step; ``externals`` is unused.

        The whole O(V) PREP -- prune, normalize (``logps``), the rejection
        normalizer ``logZ``, and the Gumbel ``keys`` -- is built ON-DEVICE for all
        rows here (one batched pass), then handed to ``_run_rejection``. That moves
        the per-particle 50k-vocab numpy work (especially the ``-log(-log(u))``
        key-gen, the dominant CPU cost) off the CPU and vectorizes it across the
        whole population; only the cheap top-K sort + the sequential accept walk
        stay per-particle. The keys use a torch (on-device) RNG stream rather than
        the slow path's numpy one, so the burst is no-bias (like the Direct
        unconstrained burst), not token-for-token tied to the slow path."""
        # Prune (mask non-valid columns to -inf) so they normalize/sort out and are
        # never drawn -- the on-device form of `_prune_logws`.
        if self.prune_logws:
            vidx = torch.as_tensor(
                self.valid_idxs, device=lm_batch.device, dtype=torch.long
            )
            masked = torch.full_like(lm_batch, float("-inf"))
            masked[:, vidx] = lm_batch[:, vidx]
        else:
            masked = lm_batch
        logZ_t = torch.logsumexp(masked, dim=1, keepdim=True)
        logps_t = masked - logZ_t
        # Gumbel keys; clamp u off 0 so a finite logps never yields a -inf key
        # (key == -inf then iff logps == -inf, i.e. exactly the pruned tokens).
        u = torch.rand_like(logps_t).clamp_(min=torch.finfo(logps_t.dtype).tiny)
        keys_t = logps_t - torch.log(-torch.log(u))
        # Sort on-device too -> transfer the descending order (ints), not the keys.
        # Pruned (-inf) tokens sort last; the CPU walk stops at the first one.
        order_t = torch.argsort(keys_t, dim=1, descending=True)
        logps_np = logps_t.cpu().numpy()
        order_np = order_t.cpu().numpy()
        logZ_batch = logZ_t.squeeze(1).tolist()

        # Per particle, run the SAME ``self._accept`` the slow ``sample`` path uses,
        # checking ONLY the tokens the rejection walk probes (never the whole vocab).
        # ``_accept`` is pure-CPU async for an FSA condition, so ``_drive_sync`` runs
        # the rejection inline -- no per-particle event-loop hop.
        out = []
        for k, context in enumerate(contexts):
            lm_logws = self.potential.make_lazy_weights(logps_np[k])

            async def accept(tok, c=context):
                return await self._accept(c, tok)

            # Inline for a non-suspending (FSA) condition; hop to the loop if the
            # condition actually awaits (autobatched / LM-backed / IPC) -- correct
            # either way, no per-particle hop in the common FSA case.
            def _rej(lw=lm_logws, ac=accept, kk=k):
                return self._run_rejection(
                    lw, ac, logZ=logZ_batch[kk], logps=logps_np[kk], order=order_np[kk]
                )

            token, logw, _ = _drive_or_hop(_rej, ctx.run_sync)
            out.append(BurstDraw(token=token, step=([token], logw, float("nan"))))
        return out

    def _prune_logws(self, logws):
        # Prune the logws to only include the tokens in the
        # target vocabulary. (This zeros-out tokens which we know a priori
        # will be rejected.) Note: We need an additional correction term
        # to account for the fact that we're throwing away some probability mass.
        # This should be handled in `sample`.
        pruned = self.potential.alloc_logws()
        pruned[self.valid_idxs] = logws.weights[self.valid_idxs]
        logws.weights = pruned
        return logws

    async def _accept(self, context, token, verbosity=0):
        if self.prune_logws or token in self.vocab_eos_set:
            if token is self.target.eos:
                logscore = await self.condition.complete(context)
            else:
                logscore = await self.condition.prefix(context + [token])
            assert logscore in {-np.inf, 0}, "`condition` must be Boolean"
        else:
            logscore = -np.inf

        do_accept = logscore == 0

        if verbosity > 0:
            if do_accept:
                print(colors.green % f". {repr(token)}")
            else:
                print(colors.red % ".", end="")

        return do_accept

    async def sample(self, context, verbosity=0):
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method via adaptive weighted rejection sampling.

        With no proposal, the returned weight is the log normalizing constant of
        $\\textsf{target.logw_next}$. With a proposal $q$, the inner loop proposes
        from $q$ and the returned weight adds an importance correction
        $\\textsf{potential.logw_next}[x_n] - q.\\textsf{logw_next}[x_n]$.

        Returns:
            (token, weight, np.nan): A tuple containing the sampled token, weight, and a dummy value for the log-probability of the sampled token.
        """
        if self.proposal is None:
            logws = await self.potential.logw_next(context)
            target_logws = None
        else:
            target_logws, logws = await asyncio.gather(
                self.potential.logw_next(context),
                self.proposal.logw_next(context),
            )

        async def accept(tok):
            return await self._accept(context, tok, verbosity)

        return await self._run_rejection(logws, accept, target_logws)

    async def _run_rejection(
        self, logws, accept, target_logws=None, logZ=None, logps=None, order=None
    ):
        """Shared AWRS rejection over next-token ``logws`` with a boolean
        ``accept(tok)`` coroutine (and an optional ``target_logws`` proposal
        correction).

        ``sample`` (slow path: context-based ``accept``, ``logws`` from
        ``potential.logw_next``) and the engine burst (stateful-condition
        ``accept``, ``logws`` from the engine LM logits) both call this, so the
        rejection algorithm and the returned weight stay identical -- only the
        source of ``logws`` and the form of ``accept`` differ.

        ``logZ``/``logps``/``order`` may be supplied pre-computed: the burst builds
        all three ON-DEVICE for the whole batch (``logps`` = pruned+normalized LM
        log-weights, ``order`` = descending Gumbel-key order, ``logZ`` = the
        normalizer), so the per-particle O(V) numpy work (prune, logsumexp,
        key-gen, AND the sort) moves off the CPU -- only the sequential accept walk
        remains. When ``None`` (slow path) they are computed here in numpy as before.
        """
        if logps is None:
            if self.prune_logws:
                logws = self._prune_logws(logws)
            if logZ is None:
                logZ = logsumexp(logws.weights)
            logps = logws.weights - logZ
        toks = logws.decode

        # Cache successful calls (geometric_awrs may revisit a token).
        cache = {}

        async def cached_accept(tok):
            try:
                return cache[tok]
            except KeyError:
                pass
            result = await accept(tok)
            if result:
                cache[tok] = result
            return result

        if not self.proper_weights:
            return await improper_sample(
                logps=logps,
                toks=toks,
                accept=cached_accept,
                rng=self.rng,
                max_rejects=self.max_rejects,
                order=order,
            )
        # geometric_awrs when max_accepts>2 (recursive_awrs ignores it) or when
        # the distribution is peaked (then geometric is more efficient).
        elif self.max_accepts > 2 or logps.max() >= GEOMETRIC_THRESHOLD:
            tok, w, _ = await geometric_awrs(
                logps=logps,
                toks=toks,
                accept=cached_accept,
                rng=self.rng,
                max_rejects=self.max_rejects,
                max_accepts=self.max_accepts,
                order=order,
            )
        else:
            tok, w, _ = await recursive_awrs(
                logps=logps,
                toks=toks,
                accept=cached_accept,
                rng=self.rng,
                max_rejects=self.max_rejects,
                order=order,
            )

        if target_logws is None:
            return tok, w + logZ, np.nan

        # `tok` was drawn with finite sampling weight, so `_prune_logws` left
        # `logws.weights[tok_idx]` untouched even when pruning is on. When
        # `w == -inf` (rejection failure) the result stays -inf.
        tok_idx = self.potential.lookup[tok]
        log_ratio = target_logws.weights[tok_idx] - logws.weights[tok_idx]
        return tok, w + logZ + log_ratio, np.nan


# If the top log probability exceeds this value, then it will be
# more efficient to use geometric_awrs. This is because the
# expected number of distinct calls is bounded above by 1 +
# a negative binomial distribution with parameters 2, p (
# the number of calls that would be made by sampling with
# replacement before seeing two of the top probability), so
# has expected value 1 + 2(1 - p) / p, and so is < 2 whenever
# p > 2/ 3. As recursive_awrs always makes at least two calls,
# geometric_awrs dominates here.
GEOMETRIC_THRESHOLD = np.log(2 / 3)


class _TopK:
    """Descending-key order, computed lazily as a top-K ``argpartition`` instead
    of a full ``argsort``.

    The AWRS rejection walks tokens in descending Gumbel-key order and stops at
    the first accepted one -- almost always within the first few -- so fully
    sorting the ~50k-token vocabulary every step is wasted work (sort 50k, walk
    1-2). This materializes only the top ``k`` (one O(V) ``argpartition`` + an
    O(k log k) sort of those k); if a walk ever consumes past ``k`` (rare: the
    top-k tokens all rejected), it falls back **once** to a full sort. Indexable
    and ``len``-able like the ``np.argsort(-keys)`` array it replaces, and -- for
    distinct keys (continuous Gumbel keys are distinct a.s.) -- produces the
    identical index sequence, so parity is unchanged.
    """

    __slots__ = ("_keys", "_n", "_order", "_full")

    def __init__(self, keys, k=256):
        # Keep only a reference to ``keys`` (no full-array negate/copy); negate
        # just the k-element top slice. ``argpartition(keys, n-k)[n-k:]`` are the
        # k largest keys, unordered; sort that small slice descending.
        self._keys = keys
        self._n = len(keys)
        k = min(k, self._n)
        top = np.argpartition(keys, self._n - k)[self._n - k :]
        self._order = top[np.argsort(-keys[top])]
        self._full = self._n <= k

    def __len__(self):
        return self._n

    def _ensure_full(self):
        if not self._full:  # rare deep walk: one full sort
            self._order = np.argsort(-self._keys)
            self._full = True

    def __getitem__(self, i):
        if i < 0 or i >= self._n:
            raise IndexError(i)
        if i >= len(self._order):  # walked past the top-k -> one full-sort fallback
            self._ensure_full()
        return self._order[i]


async def improper_sample(*, logps, toks, accept, rng, max_rejects, order=None):
    """Implements a single rejection sampling loop which returns
    the first value found with no attempt to make a properly
    weighted sample. ``order`` (descending Gumbel-key order) may be precomputed --
    the burst draws the keys AND sorts them on-device for the whole batch; when
    ``None`` the keys are drawn and sorted here (slow path). The walk stops at the
    first ``logps == -inf`` (a pruned / zero-probability token, which sorts last)."""
    if order is None:
        keys = logps - np.log(-np.log(rng.random((len(logps),))))
        order = _TopK(keys)
    else:
        # Burst: the keys live on the GPU and aren't transferred; logps marks the
        # pruned tokens with -inf identically (the on-device keys are clamped so a
        # finite logps never yields a -inf key), so logps is the -inf sentinel.
        keys = logps
    for count, item in enumerate(order):
        if count >= max_rejects or keys[item] == -np.inf:
            break
        tok = toks[item]
        if await accept(tok):
            return tok, 0.0, np.nan
    return tok, -float("inf"), np.nan


async def recursive_awrs(*, logps, toks, accept, rng, max_rejects, order=None):
    """Implements Recursive AWRS.

    This uses the observation that

    E(f(X)) = P(X = x) f(x) + (1 - P(X = x)) E(f(X)|X != x)

    To construct a recursive estimator of the weight from a single
    sampling-with-rejection run. The first time accept(x) passes,
    we use a simple coin flip estimator for the tail. ``order`` (descending
    Gumbel-key order) may be precomputed (burst, on-device); when ``None`` the
    keys are drawn and sorted here (slow path).
    """
    n_accepts = 0
    n_rejects = 0

    rejected_mass = 0.0
    log_multiplier = 0.0

    # We treat any number smaller than this as "effetively" zero.
    # This causes us to terminate early in some cases, but those
    # cases are all ones where the remaining weight is very bad.
    error_tolerance = 10e-6

    if order is None:
        keys = logps - np.log(-np.log(rng.random((len(logps),))))
        order = _TopK(keys)
    else:
        # Burst: keys are on the GPU (not transferred); logps is the -inf sentinel
        # (clamped on-device keys give key == -inf iff logps == -inf).
        keys = logps
    for index_into_list, item in enumerate(order):
        assert n_accepts == 0
        tok = toks[item]
        last = (
            index_into_list + 1 == len(order)
            or keys[order[index_into_list + 1]] == -np.inf
        )

        log_q = logps[item] - np.log1p(-rejected_mass)

        # The last check is because in the case where there is a single
        # accepted token with very low log probability, numerical stability
        # issues make it very hard to get this calculation right.
        assert not last or log_q >= -error_tolerance or logps[item] < -32
        assert log_q <= error_tolerance
        assert log_multiplier <= error_tolerance
        assert rejected_mass <= 1

        # Fix some minor numerical stability errors that can come up.
        if last:
            log_q = 0
        log_q = min(log_q, 0)
        log_multiplier = min(log_multiplier, 0)

        if await accept(toks[item]):
            n_accepts += 1
            if n_rejects == max_rejects - 1:
                return tok, log_multiplier, np.nan
            elif last:
                final_estimator = 0.0
            else:
                next_token = toks[order[index_into_list + 1]]
                if await accept(next_token):
                    final_estimator = 0
                else:
                    final_estimator = log_q
            logp = log_multiplier + final_estimator
            return tok, logp, np.nan
        elif last or n_rejects == max_rejects - 1:
            # No token was accepted, return a rejected token and kill the particle.
            return tok, float("-inf"), np.nan
        else:
            n_rejects += 1
            rejected_mass += np.exp(logps[item])
            if rejected_mass >= 1 - error_tolerance:
                # We've explored all the probability mass and still found no
                # accepted token.
                return tok, float("-inf"), np.nan
            m = log1mexp(log_q)
            assert not np.isnan(m)
            log_multiplier += m
        assert not last

    raise AssertionError("Unreachable")


async def geometric_awrs(*, logps, toks, accept, rng, max_rejects, max_accepts, order=None):
    """Implements Geometric AWRS.

    This simulates a single run of sampling with replacement from a sampling
    without replacement run, reconstructing the counts of "phantom" elements
    discarded from the without-replacement run as a series of draws from
    geometric distributions. We can then use an appropriate estimator
    for the with-replacement run at the end. ``order`` may be precomputed for the
    FIRST pass (burst, on-device); subsequent passes redraw and re-sort (the
    algorithm is adaptive -- it marks rejected tokens ``-inf`` and resamples).
    """
    n_accepts = 0
    n_rejects = 0

    rejected_mass = 0.0
    result = None
    rejected_token = None

    for pass_i in range(max_accepts):
        if n_rejects >= max_rejects:
            break
        if pass_i == 0 and order is not None:
            cur_order = order
            cur_keys = logps  # burst pass 0: logps is the -inf sentinel (see above)
        else:
            cur_keys = logps - np.log(-np.log(rng.random((len(logps),))))
            cur_order = _TopK(cur_keys)
        for item in cur_order:
            if cur_keys[item] == -np.inf:
                break

            tok = toks[item]

            if rejected_mass >= 1:
                # If rejected mass is >= 1 but we have a non-zero probability
                # we've really had numerical precision issues that rounded us to 1.
                # However, this means that the correct estimator is ridiculously
                # small, and we'd exceed any reasonable `max_rejects`, so we just
                # immediately terminate in this case.
                #
                # This can technically happen after we've seen an accepted token
                # but this only happens if the distribution / constraint has gone
                # very wrong.
                assert rejected_token is not None
                return rejected_token, -float("inf"), np.nan
            elif rejected_mass > 0:
                # Add a geometric distribution with parameter 1 - rejected_mass
                # to the number of rejects, account for the phantom tokens
                # "hidden" by sampling without replacement.
                phantom_tokens = rng.geometric(1 - rejected_mass) - 1
                assert phantom_tokens >= 0
                n_rejects += phantom_tokens

            if n_rejects >= max_rejects:
                break

            if await accept(tok):
                n_accepts += 1
                if result is None:
                    result = tok
                break
            else:
                if rejected_token is None:
                    rejected_token = tok
                n_rejects += 1
                rejected_mass += np.exp(logps[item])
                logps[item] = -float("inf")

    if n_accepts == 0:
        assert rejected_token is not None
        return rejected_token, -np.inf, np.nan

    # If we stopped in the middle of a sequence of phantom tokens,
    # n_rejects may have gone over max_rejects.
    n_rejects = min(n_rejects, max_rejects)

    # The correctness of this estimator can be verified by applying
    # the Rao-Blackwell theorem to the estimator that just returns
    # 1 if the first sample was accepted and 0 if it was rejected
    # to the sufficient statistic (n_accepts, n_rejects). Some
    # straightforward sequence counting gives you this estimator.
    estimator = min(max_accepts - 1, n_accepts) / (n_accepts + n_rejects - 1)

    assert estimator > 0 or result is None

    return result, np.log(estimator), np.nan
