import asyncio
import numpy as np
import torch
from arsenal import colors
from arsenal.maths import log1mexp
import warnings

from genlm.control.util import (
    select,
    picker_indices,
    draw_key,
    draw_ordinal,
    awrs_gumbel_keys,
    get_draw_seed,
)
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

    Collapses to a single per-step :meth:`transition` the controller calls,
    mapping a particle context to ``(to_append, logw, logp)``.

    Args:
        target (Potential): The potential that samples are properly weighted with respect to.
    """

    # ``None`` unless the sampler draws from a separate proposal (Direct/AWRS/Set).
    proposal = None

    def __init__(self, target):
        self.target = target
        self.token_type = self.target.token_type

    def supports_burst(self) -> bool:
        """Whether this sampler can run inside the engine burst (implements
        :meth:`burst_draw_batch`). Default ``False`` (stays on ``StepLoop``)."""
        return False

    def burst_draw_sampler(self):
        """The sampler whose ``target``/``proposal`` are the injected views: ``self``
        at token grain; a unit sampler delegates to its subunit."""
        return self

    def burst_free_running(self) -> bool:
        """``True`` = free-running (token grain): one SMC step per decode step, ESS
        every step. ``False`` = synchronized (unit grain): one step per unit."""
        return True

    def burst_max_steps(self, live) -> int:
        """Engine decode-step budget for one burst (token grain). The unit sampler overrides."""
        return max(p.max_tokens_left for p in live) + 1

    @staticmethod
    def _row_injection(warm_batch, i):
        """Slice batched warm ``{view: [N, V+1]}`` into particle ``i``'s per-row
        injection ``{view: [V+1]}`` for the sequential draw."""
        return {view: view.make_lazy_weights(W.weights[i]) for view, W in warm_batch.items()}

    async def burst_draw_batch(self, warm_batch, contexts, handles, burst):
        """Sequential engine-burst draw, one BurstDraw per particle (token grain): slice
        each row's warm logits and run the real per-step ``transition``. `DirectTokenSampler`
        overrides with a vectorized draw, the unit sampler for subunit accumulation."""

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

    async def logw_eos(self, context):
        """EOS log-weight at the ``max_tokens`` boundary, used to force termination."""
        return await self.target.logw_eos(context)

    async def transition(self, context):
        """Controller-facing per-step transition.

        Args:
            context (list): The particle's current token context.

        Returns:
            (to_append, logw, logp): items to append (``[token]``, or more for a
                multi-token unit), the weight increment, and the choice log-prob.
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
            accelerate (str | bool, optional): Engine-acceleration knob forwarded to
                `SMC.__call__`: ``"auto"`` (default), ``"off"`` (force per-token), or
                ``"require"`` (engine or raise `NotAcceleratable`). ``True``/``False``
                alias ``"auto"``/``"off"``.
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
        # Target (and proposal) logw_next are reproduced as injected views.
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
            logZ = logws.sum()  # normalizer == the weight
            logps = logws.spawn(logws.weights - logZ)  # logws.normalize()
            token = select(logps) if draw is None else draw(logps.exp().materialize())
            return token, logZ, logps[token]

        proposal_logws, target_logws = await asyncio.gather(
            self.proposal.logw_next(context), self.potential.logw_next(context)
        )
        proposal_logZ = proposal_logws.sum()
        proposal_logps = proposal_logws.spawn(proposal_logws.weights - proposal_logZ)
        if draw is None:
            token = select(proposal_logps)
        else:
            token = draw(proposal_logps.exp().materialize())
        logw = target_logws[token] - proposal_logws[token] + proposal_logZ
        return token, logw, proposal_logps[token]

    async def burst_draw_batch(self, warm_batch, contexts, handles, burst):
        """Vectorized engine-burst draw over the population: one logsumexp + one keyed
        Gumbel draw + one gather over the ``[N, V+1]`` proposal. Threefry keyed by
        ``(row, draw_ordinal)``, so byte-identical to the per-particle path."""
        rows = torch.tensor(handles, dtype=torch.int64)
        ordinals = torch.tensor([draw_ordinal(c) for c in contexts], dtype=torch.int64)
        with burst_logw_next(warm_batch):
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
        self.rng = np.random.default_rng(seed=seed)  # phantom-geometric (CPU scalar)
        # Gumbel keys: per-instance threefry stream (driver-independent, on-device).
        self._draw_seed = (seed if seed is not None else get_draw_seed()) & 0xFFFFFFFF
        self._draw_ctr = 0
        self._valid_idxs_cache = None

    def supports_burst(self) -> bool:
        # Rejection runs over the engine LM logits (injected), condition checked per
        # probed token (CPU). With a proposal, both LM reads are injected views.
        return True

    def _prune_logws(self, w):
        # Keep only target-vocab tokens (-inf elsewhere; mass corrected via logZ). On-device.
        pruned = torch.full_like(w, float("-inf"))
        idx = self._valid_idxs_t(w.device)
        pruned[idx] = w[idx]
        return pruned

    def _valid_idxs_t(self, device):
        c = self._valid_idxs_cache
        if c is None or c.device != device:
            c = self._valid_idxs_cache = torch.as_tensor(
                self.valid_idxs, dtype=torch.int64, device=device
            )
        return c

    def _make_keys(self, logps):
        """Fresh Gumbel keys for one round; advance the counter so each round is independent."""
        keys = awrs_gumbel_keys(logps, self._draw_seed, self._draw_ctr)
        self._draw_ctr += 1
        return keys

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
        coroutine (optional ``target_logws`` proposal correction)."""
        # On-device: prune/normalize/top-k stay on the native device, only the walked
        # slice crosses to the CPU condition checks.
        lw = torch.as_tensor(logws.weights)  # no-op when already a device tensor
        if self.prune_logws:
            lw = self._prune_logws(lw)
        logZ = float(torch.logsumexp(lw, 0))
        logps = lw - logZ  # device [V]
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
                make_keys=self._make_keys,
                max_rejects=self.max_rejects,
            )
        # geometric_awrs when max_accepts>2 (recursive_awrs ignores it) or when
        # the distribution is peaked (then geometric is more efficient).
        elif self.max_accepts > 2 or float(logps.max()) >= GEOMETRIC_THRESHOLD:
            tok, w, _ = await geometric_awrs(
                logps=logps,
                toks=toks,
                accept=cached_accept,
                make_keys=self._make_keys,
                rng=self.rng,
                max_rejects=self.max_rejects,
                max_accepts=self.max_accepts,
            )
        else:
            tok, w, _ = await recursive_awrs(
                logps=logps,
                toks=toks,
                accept=cached_accept,
                make_keys=self._make_keys,
                max_rejects=self.max_rejects,
            )

        if target_logws is None:
            return tok, w + logZ, np.nan

        # `tok` was drawn with finite sampling weight, so `_prune_logws` left
        # `lw[tok_idx]` untouched even when pruning is on. When `w == -inf`
        # (rejection failure) the result stays -inf.
        tok_idx = self.potential.lookup[tok]
        tw = torch.as_tensor(target_logws.weights)
        log_ratio = float(tw[tok_idx]) - float(lw[tok_idx])
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


class _AwrsOrder:
    """Descending Gumbel-perturbed order over device ``logps``, materialized lazily via
    ``torch.topk`` (only the walked top-k slice crosses to the CPU checks; a deep walk
    triggers one full sort). ``order[i]`` -> ``(vocab_id, logp)`` of the i-th best
    (``logp == -inf`` for a pruned token). ``reject(vid)`` scatters -inf for the next round."""

    __slots__ = ("_logps", "_keys", "_n", "_ids", "_lvals", "_k")

    def __init__(self, logps, make_keys, k=256):
        self._logps = logps  # device [V]; geometric mutates it via reject()
        self._keys = make_keys(logps)  # device [V]; advances AWRS's per-instance counter
        self._n = logps.shape[-1]
        self._materialize(min(k, self._n))

    def _materialize(self, k):
        _, idx = torch.topk(self._keys, k)  # descending order; the key VALUES aren't needed
        self._ids = idx.cpu().numpy()
        self._lvals = self._logps[idx].cpu().numpy()  # logps at materialization
        self._k = k

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if i < 0 or i >= self._n:
            raise IndexError(i)
        if i >= self._k:  # walked past the materialized top-k -> one full sort
            self._materialize(self._n)
        return int(self._ids[i]), float(self._lvals[i])

    def reject(self, vid):
        self._logps[vid] = float("-inf")  # device scatter; next round re-reads it


async def improper_sample(*, logps, toks, accept, make_keys, max_rejects):
    """Single rejection loop returning the first accepted value, no proper weight.
    The walk stops at the first ``key == -inf`` (pruned token)."""
    order = _AwrsOrder(logps, make_keys)
    tok = None
    for count in range(len(order)):
        vid, lp = order[count]
        if count >= max_rejects or lp == -np.inf:
            break
        tok = toks[vid]
        if await accept(tok):
            return tok, 0.0, np.nan
    return tok, -float("inf"), np.nan


async def recursive_awrs(*, logps, toks, accept, make_keys, max_rejects):
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

    order = _AwrsOrder(logps, make_keys)
    n = len(order)
    for index_into_list in range(n):
        vid, lp = order[index_into_list]
        assert n_accepts == 0
        tok = toks[vid]
        nxt = order[index_into_list + 1] if index_into_list + 1 < n else None
        last = nxt is None or nxt[1] == -np.inf  # nxt is (vid, logp)

        log_q = lp - np.log1p(-rejected_mass)

        # The last check is because in the case where there is a single
        # accepted token with very low log probability, numerical stability
        # issues make it very hard to get this calculation right.
        assert not last or log_q >= -error_tolerance or lp < -32
        assert log_q <= error_tolerance
        assert log_multiplier <= error_tolerance
        assert rejected_mass <= 1

        # Fix some minor numerical stability errors that can come up.
        if last:
            log_q = 0
        log_q = min(log_q, 0)
        log_multiplier = min(log_multiplier, 0)

        if await accept(tok):
            n_accepts += 1
            if n_rejects == max_rejects - 1:
                return tok, log_multiplier, np.nan
            elif last:
                final_estimator = 0.0
            else:
                next_token = toks[nxt[0]]
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
            rejected_mass += np.exp(lp)
            if rejected_mass >= 1 - error_tolerance:
                # We've explored all the probability mass and still found no
                # accepted token.
                return tok, float("-inf"), np.nan
            m = log1mexp(log_q)
            assert not np.isnan(m)
            log_multiplier += m
        assert not last

    raise AssertionError("Unreachable")


async def geometric_awrs(*, logps, toks, accept, make_keys, rng, max_rejects, max_accepts):
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
        # Re-perturb the (mutated) device logps; prior-round rejects are -inf and fall out.
        cur_order = _AwrsOrder(logps, make_keys)
        for pos in range(len(cur_order)):
            vid, lp = cur_order[pos]
            if lp == -np.inf:
                break

            tok = toks[vid]

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
                rejected_mass += np.exp(lp)
                cur_order.reject(vid)

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
