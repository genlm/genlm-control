"""The burst's serving seam: the ContextVars a burst writes and a potential reads.

Four variables, one protocol. A driver parks a row on ``_burst_row`` and answers its
logits reads from the engine's warm batch; the three override maps let it hand a
potential a precomputed ``logw_next``/``batch_prefix``/``batch_complete`` instead of a
forward. This module imports nothing from ``potential`` or ``sampler`` so both sides
can depend on it.
"""

import contextlib
import contextvars

# Per-burst override: {potential: LazyWeights} a potential's ``logw_next`` returns for
# itself instead of computing. Read by ``PromptedLLM``.
_burst_logw_next_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_logw_next", default=None
)

# Boundary override: {potential: values} a potential's ``batch_prefix`` returns for
# itself (banked from the burst's warm rows) instead of re-scoring.
_burst_prefix_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_prefix", default=None
)

# Same seam for ``batch_complete``: {potential: values} served instead of scoring.
_burst_complete_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_complete", default=None
)

# A parked row's channel to the burst, bound for the whole of one row's step.
# Set only by the burst's parked-row lane; ``None`` everywhere else.
_burst_row: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_row", default=None
)


@contextlib.contextmanager
def burst_logw_next(overrides):
    """Inject ``{potential: LazyWeights}`` for one burst step (set per particle task)."""
    token = _burst_logw_next_overrides.set(overrides)
    try:
        yield
    finally:
        _burst_logw_next_overrides.reset(token)


@contextlib.contextmanager
def burst_prefix(overrides):
    """Inject ``{potential: values}`` served as that potential's ``batch_prefix`` result."""
    token = _burst_prefix_overrides.set(overrides)
    try:
        yield
    finally:
        _burst_prefix_overrides.reset(token)


@contextlib.contextmanager
def burst_complete(overrides):
    """Inject ``{potential: values}`` served as that potential's ``batch_complete`` result."""
    token = _burst_complete_overrides.set(overrides)
    try:
        yield
    finally:
        _burst_complete_overrides.reset(token)


@contextlib.contextmanager
def burst_row(channel):
    """Bind ``channel`` for one row's step task (the burst's parked-row lane)."""
    token = _burst_row.set(channel)
    try:
        yield
    finally:
        _burst_row.reset(token)


async def burst_serve(context):
    """Park until the burst delivers this row's warm for ``context``, then inject it.

    A no-op outside the parked-row lane. Every read at one context length is one decode
    step (target and proposal share the step's warm); a read past a draw parks, with a
    context ending in the token just drawn."""
    channel = _burst_row.get()
    if channel is not None:
        _burst_logw_next_overrides.set(await channel.next_warm(context))
