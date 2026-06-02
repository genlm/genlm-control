import asyncio
import numpy as np
import torch
from arsenal import colors
from arsenal.maths import log1mexp
from genlm.control.util import logsumexp
import warnings

from genlm.control.util import select, picker_indices, draw_key, draw_ordinal, to_numpy
from genlm.control.sampler.set import SetSampler
from genlm.control.sampler.util import _validate_proposal_vocab
from genlm.control.sampler.burst import BurstDraw
from genlm.control.potential.base import burst_logw_next


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

    # Set by samplers that draw from a separate proposal (Direct/AWRS/Set); ``None`` for
    # the rest, so callers read ``s.proposal`` instead of probing for the attribute.
    proposal = None

    def __init__(self, target):
        self.target = target
        self.token_type = self.target.token_type

    def supports_burst(self) -> bool:
        """Whether this sampler can be driven inside the engine burst (implements
        :meth:`burst_draw_batch`). Default ``False`` (stays on ``StepLoop``)."""
        return False

    def burst_draw_sampler(self):
        """The sampler whose per-step ``sample`` the burst actually injects logits for
        -- i.e. whose ``target``/``proposal`` are the injected views. ``self`` for a
        token-grain sampler; a unit sampler delegates to its subunit."""
        return self

    def burst_free_running(self) -> bool:
        """Whether the burst is free-running (token grain): one SMC step per decode
        step, ESS tested every step. ``False`` = synchronized (unit grain): one SMC
        step per unit, ESS once per round."""
        return True

    def burst_max_steps(self, live) -> int:
        """Engine decode-step budget for one burst. Token grain: the longest
        particle's remaining budget (+1). The unit sampler overrides it."""
        return max(p.max_tokens_left for p in live) + 1

    @staticmethod
    def _row_injection(warm_batch, i):
        """Slice the batched warm (``{view: [N, V+1] LazyWeights}``) into particle ``i``'s
        per-row injection (``{view: [V+1] LazyWeights}``) -- the inverse of the burst's stack,
        for the sequential per-particle draw (AWRS/Set/unit)."""
        return {view: view.make_lazy_weights(W.weights[i]) for view, W in warm_batch.items()}

    async def burst_draw_batch(self, warm_batch, contexts, handles, burst):
        """Engine-burst draw, one BurstDraw per particle (token grain). ``warm_batch`` maps
        each view-LM to its batched ``[N, V+1]`` warm logits; slice row ``i`` into a per-row
        injection and run the REAL per-step ``transition`` -- the target composes itself over
        the K views (proposal/prior/...), no proposal reconstruction. One SMC step per decode
        step. This is the sequential path (AWRS/Set, rejection/trie); `DirectTokenSampler`
        overrides with a vectorized ``[N, V]`` draw, the unit sampler for subunit accumulation."""

        async def one(i, context, handle):
            injection = self._row_injection(warm_batch, i)
            with burst_logw_next(injection), draw_key(handle, draw_ordinal(context)):
                to_append, logw, logp = await self.transition(context)
            return BurstDraw(token=to_append[-1], step=(to_append, logw, logp))

        return await asyncio.gather(
            *(one(i, c, h) for i, (c, h) in enumerate(zip(contexts, handles)))
        )

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

    async def sample(self, context):
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
        # The engine reproduces the target's logw_next, and (with a proposal) the
        # proposal's, as injected views; burst_blocker checks every leaf is a view.
        return True

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
            logZ = logws.sum()  # normalizer == the returned weight; compute logsumexp once
            logps = logws.spawn(logws.weights - logZ)  # == logws.normalize()
            token = select(logps) if draw is None else draw(logps.exp().materialize())
            return token, logZ, logps[token]

        proposal_logws, target_logws = await asyncio.gather(
            self.proposal.logw_next(context), self.potential.logw_next(context)
        )
        proposal_logZ = proposal_logws.sum()  # one logsumexp, reused below
        proposal_logps = proposal_logws.spawn(proposal_logws.weights - proposal_logZ)
        if draw is None:
            token = select(proposal_logps)
        else:
            token = draw(proposal_logps.exp().materialize())
        logw = target_logws[token] - proposal_logws[token] + proposal_logZ
        return token, logw, proposal_logps[token]

    async def burst_draw_batch(self, warm_batch, contexts, handles, burst):
        """Vectorized engine-burst draw over the whole population: compose the ``[N, V+1]``
        proposal (injected warm batch + batched factors), one logsumexp + one keyed Gumbel
        draw + one gather, instead of N per-particle ``sample`` calls. Byte-identical to the
        per-particle path -- threefry keyed by ``(row, draw_ordinal)``, so batched ==
        per-particle. Mirrors ``sample``: without a proposal the weight is the normalizer
        ``logZ``; with one it's the importance weight ``target[x] - proposal[x] + logZ``."""
        rows = torch.tensor(handles, dtype=torch.int64)
        ordinals = torch.tensor([draw_ordinal(c) for c in contexts], dtype=torch.int64)
        with burst_logw_next(warm_batch):
            # Draw from the proposal (the target itself when there is no separate proposal).
            if self.proposal is None:
                proposal, target = await self.potential.batch_logw_next(contexts), None
            else:
                proposal, target = await asyncio.gather(
                    self.proposal.batch_logw_next(contexts),
                    self.potential.batch_logw_next(contexts),
                )
            pw = proposal.weights
            pZ = torch.logsumexp(pw, dim=-1)  # [N], == per-row .sum()
            plogps = pw - pZ[:, None]  # normalized proposal [N, V+1]
            with draw_key(rows, ordinals):
                idx = picker_indices(plogps)  # [N]
            ar = torch.arange(len(handles), device=pw.device)
            logp, decode = plogps[ar, idx], proposal.decode
            if target is None:
                weight = pZ  # the normalizer is the weight
            else:
                tw = target.weights.to(pw.device)
                weight = tw[ar, idx] - pw[ar, idx] + pZ  # importance weight
        idx_l, w_l, lp_l = idx.tolist(), weight.tolist(), logp.tolist()
        return [
            BurstDraw(token=decode[idx_l[i]], step=([decode[idx_l[i]]], w_l[i], lp_l[i]))
            for i in range(len(handles))
        ]

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
        # The async-trie set draw runs on the main loop via the per-step hop.
        return True

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
        logZ = logws.sum()  # one logsumexp, reused as the weight
        logps = logws.spawn(logws.weights - logZ)  # == logws.normalize()
        token = select(logps) if draw is None else draw(logps.exp().materialize())
        return token, logZ, logp + logps[token]

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
        # Rejection runs over the engine LM logits (injected), condition checked per
        # probed token (CPU). With a proposal, both LM reads are injected views.
        return True

    def _prune_logws(self, w):
        # Prune the numpy log-weights ``w`` to only the tokens in the target vocabulary
        # (zeroing-out tokens we know a priori will be rejected). The mass thrown away is
        # corrected via logZ in `_run_rejection`.
        pruned = self.potential.alloc_logws()
        pruned[self.valid_idxs] = w[self.valid_idxs]
        return pruned

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
                self.potential.logw_next(context), self.proposal.logw_next(context)
            )

        async def accept(tok):
            return await self._accept(context, tok, verbosity)

        return await self._run_rejection(logws, accept, target_logws)

    async def _run_rejection(self, logws, accept, target_logws=None):
        """Shared AWRS rejection over next-token ``logws`` with a boolean ``accept``
        coroutine (and an optional ``target_logws`` proposal correction). The slow
        ``sample`` and the engine burst both call this, so the algorithm and weight
        stay identical -- only the source of ``logws``/``accept`` differs."""
        # AWRS rejection is CPU/numpy (seeded rng, argsort, data-dependent walk); pull the
        # log-weights to numpy at this edge (no-op if the producer is already numpy) and
        # operate in numpy throughout.
        lw = to_numpy(logws.weights)
        if self.prune_logws:
            lw = self._prune_logws(lw)
        logZ = logsumexp(lw)
        logps = lw - logZ
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
            )
        else:
            tok, w, _ = await recursive_awrs(
                logps=logps,
                toks=toks,
                accept=cached_accept,
                rng=self.rng,
                max_rejects=self.max_rejects,
            )

        if target_logws is None:
            return tok, w + logZ, np.nan

        # `tok` was drawn with finite sampling weight, so `_prune_logws` left
        # `lw[tok_idx]` untouched even when pruning is on. When `w == -inf`
        # (rejection failure) the result stays -inf.
        tok_idx = self.potential.lookup[tok]
        log_ratio = target_logws.weights[tok_idx].item() - lw[tok_idx]
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
    """Descending-key order as a lazy top-K ``argpartition`` instead of a full
    ``argsort``. The AWRS walk almost always stops within the first few, so sorting
    the whole ~50k vocab is wasted; this materializes only the top ``k`` and falls
    back once to a full sort if a walk goes past it. Indexable/``len``-able like
    ``np.argsort(-keys)``; for distinct keys it yields the identical sequence."""

    __slots__ = ("_keys", "_n", "_order", "_full")

    def __init__(self, keys, k=256):
        # Negate only the k-element top slice; sort that small slice descending.
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


async def improper_sample(*, logps, toks, accept, rng, max_rejects):
    """Single rejection loop returning the first accepted value, no proper weight.
    The walk stops at the first ``logps == -inf`` (pruned token)."""
    keys = logps - np.log(-np.log(rng.random((len(logps),))))
    order = _TopK(keys)
    for count, item in enumerate(order):
        if count >= max_rejects or keys[item] == -np.inf:
            break
        tok = toks[item]
        if await accept(tok):
            return tok, 0.0, np.nan
    return tok, -float("inf"), np.nan


async def recursive_awrs(*, logps, toks, accept, rng, max_rejects):
    """Implements Recursive AWRS.

    This uses the observation that

    E(f(X)) = P(X = x) f(x) + (1 - P(X = x)) E(f(X)|X != x)

    To construct a recursive estimator of the weight from a single
    sampling-with-rejection run. The first time accept(x) passes,
    we use a simple coin flip estimator for the tail.
    """
    n_accepts = 0
    n_rejects = 0

    rejected_mass = 0.0
    log_multiplier = 0.0

    # We treat any number smaller than this as "effetively" zero.
    # This causes us to terminate early in some cases, but those
    # cases are all ones where the remaining weight is very bad.
    error_tolerance = 10e-6

    keys = logps - np.log(-np.log(rng.random((len(logps),))))
    order = _TopK(keys)
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


async def geometric_awrs(*, logps, toks, accept, rng, max_rejects, max_accepts):
    """Implements Geometric AWRS.

    This simulates a single run of sampling with replacement from a sampling
    without replacement run, reconstructing the counts of "phantom" elements
    discarded from the without-replacement run as a series of draws from
    geometric distributions. We can then use an appropriate estimator
    for the with-replacement run at the end.
    """
    n_accepts = 0
    n_rejects = 0

    rejected_mass = 0.0
    result = None
    rejected_token = None

    for _ in range(max_accepts):
        if n_rejects >= max_rejects:
            break
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
