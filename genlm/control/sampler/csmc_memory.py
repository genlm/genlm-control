"""Retained-particle support for Conditional Sequential Monte Carlo.

C-SMC (Andrieu, Doucet & Holenstein 2010, §4.3) is a variant of SMC in
which slot 0 of the particle population is a *retained particle* --
extended deterministically along a fixed pre-specified token sequence
and exempt from resampling. The SMC loop itself lives in
:func:`llamppl.csmc_standard`; this module supplies the unit-sampler
wrapper that implements "force the next token to be the next retained
entry" at the genlm-control sampler layer.

Retainedness is a *slot* property, not a particle property: after
resampling, only slot 0 has ``is_retained = True``, even if a free
slot happened to inherit slot 0's trajectory via the multinomial draw.

* :class:`RetainedTokenSampler` -- a transparent wrapper around any
  :class:`TokenSampler`. When its host particle has
  ``is_retained == True`` and the retained sequence is not exhausted,
  it forces the next token to be the next entry of the retained
  sequence; the underlying base sampler is still invoked to compute
  the importance weight at that forced token, so the C-SMC weight
  semantics match standard SMC exactly. Otherwise it is a no-op
  delegate to the base sampler -- which makes it safe to always wrap.
"""

from genlm.control.sampler.token import TokenSampler


class RetainedTokenSampler(TokenSampler):
    """Transparent retained-particle wrapper around a :class:`TokenSampler`.

    When the host particle has ``is_retained == True`` and the retained
    sequence is not yet exhausted, the next token is forced to be the
    next entry of ``retained_sequence``. The underlying base sampler is
    still invoked to compute the importance weight at that forced
    token, so the C-SMC weight semantics match standard SMC exactly.
    Otherwise (free particle, or retained sequence exhausted), this
    wrapper is a no-op delegate to the base sampler -- so it is safe
    to wrap *every* particle's sampler and let the wrapper itself
    decide whether to force on a per-call basis.

    ``retained_idx`` is held per-wrapper-instance, so each particle's
    deepcopy of the wrapper tracks its own position into the retained
    sequence. Only retained particles ever advance the index; free
    particles never advance it, regardless of which slot they were
    resampled from.

    Args:
        base (TokenSampler): The underlying token sampler used in
            free-particle (or post-exhaustion) mode.
        retained_sequence (list): The pre-specified retained-particle
            tokens, in order. The wrapper forces
            ``retained_sequence[t]`` at step ``t`` for any host particle
            with ``is_retained == True``.
    """

    def __init__(self, base, retained_sequence):
        super().__init__(target=base.target)
        self.base = base
        self.retained_sequence = list(retained_sequence)
        self.retained_idx = 0

    async def forward(self):
        parent = self.parent
        draw = None
        if (
            getattr(parent, "is_retained", False)
            and self.retained_idx < len(self.retained_sequence)
        ):
            forced = self.retained_sequence[self.retained_idx]
            draw = lambda _probs, _forced=forced: _forced  # noqa: E731
            self.retained_idx += 1
        token, logw, logp = await self.base.sample(parent.token_ctx, draw=draw)
        parent.score(logw)
        parent.logp += logp
        return token

    async def sample(self, context, draw=None):
        return await self.base.sample(context, draw=draw)

    async def start_weight(self):
        return await self.base.start_weight()

    async def cleanup(self):
        await self.base.cleanup()
