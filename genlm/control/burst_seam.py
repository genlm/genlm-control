"""The burst's serving seam: the ContextVars a burst writes and a potential reads.

Three variables, one protocol. A driver parks a row on ``_burst_row`` and answers its
logits reads from the engine's warm batch; ``_burst_lane_sums`` hands a potential its
banked per-token sums instead of re-scoring. This module imports nothing from
``potential`` or ``sampler`` so both sides can depend on it.
"""

import contextlib
import contextvars

# Per-burst override: {potential: LazyWeights} a potential's ``logw_next`` returns for
# itself instead of computing. Written by ``burst_serve``, read by ``PromptedLLM``.
_burst_logw_next_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_logw_next", default=None
)

# Boundary override: ``(prefix, complete)``, each a {potential: values} map served as
# that potential's ``batch_prefix``/``batch_complete`` instead of scoring. One pair and
# not two variables: the split is live rows vs terminated ones, and a caller that knows
# either always knows both.
_burst_lane_sums: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_lane_sums", default=None
)

# A parked row's lane to the burst, bound for the whole of one row's step.
# Set only by the burst's parked-row lane; ``None`` everywhere else.
_burst_row: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_row", default=None
)


@contextlib.contextmanager
def burst_lane_sums(prefix, complete):
    """Inject each leaf's banked sums: served as ``batch_prefix`` for the live rows and
    ``batch_complete`` for the terminated ones, in the caller's context order."""
    token = _burst_lane_sums.set((prefix, complete))
    try:
        yield
    finally:
        _burst_lane_sums.reset(token)


@contextlib.contextmanager
def burst_row(lane):
    """Bind ``lane`` for one row's step, over a fresh warm-override scope.

    The scope is per STEP even though the row's coroutine spans the whole burst:
    ``burst_serve`` sets the override without a reset token, so without this the previous
    step's warm would still be readable by a ``batch_logw_next`` at the top of the next."""
    tok_lane = _burst_row.set(lane)
    tok_warm = _burst_logw_next_overrides.set(None)
    try:
        yield
    finally:
        _burst_logw_next_overrides.reset(tok_warm)
        _burst_row.reset(tok_lane)


async def burst_serve(context):
    """Park until the burst delivers this row's warm for ``context``, then inject it.

    A no-op outside the parked-row lane. Every read at one context length is one decode
    step (target and proposal share the step's warm); a read past a draw parks, with a
    context ending in the token just drawn."""
    lane = _burst_row.get()
    if lane is not None:
        _burst_logw_next_overrides.set(await lane.next_warm(context))
