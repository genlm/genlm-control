"""Stateful-potential parity (CPU, no engine).

The contract for the stateful interface: advancing `state0()` through a `context`
and calling `logw_next_from_state(state)` must equal `logw_next(context)`
bit-for-bit. This is what lets a consumer carry the potential's state (like the
LLM carries its KV cache) instead of replaying the whole context every step.
"""

import asyncio

import numpy as np

from genlm.control.potential.built_in.wfsa import WFSA, BoolFSA


def _check(pot, n_steps=6):
    async def run():
        state = pot.state0()
        context = []
        for _ in range(n_steps):
            lw_ctx = await pot.logw_next(context)
            lw_state = await pot.logw_next_from_state(state)
            a = np.asarray(lw_state.weights, dtype=float)
            b = np.asarray(lw_ctx.weights, dtype=float)
            assert np.array_equal(a, b), (
                f"diverged at context={context!r}: "
                f"max|d|={np.nanmax(np.abs(a - b))}"
            )
            # extend by a finite, non-eos continuation
            finite = [t for t in pot.vocab if np.isfinite(lw_ctx[t])]
            if not finite:
                break
            tok = finite[0]
            context.append(tok)
            state = pot.advance(state, tok)

    asyncio.run(run())


def test_wfsa_stateful_matches_logw_next():
    _check(WFSA.from_regex(r"[a-z]+"))


def test_boolfsa_stateful_matches_logw_next():
    _check(BoolFSA.from_regex(r"[a-z]+"))


def test_wfsa_stateful_matches_logw_next_branching():
    # a regex with branching / multiple live states
    _check(WFSA.from_regex(r"(ab|ac|b)+"))
    _check(BoolFSA.from_regex(r"(ab|ac|b)+"))
