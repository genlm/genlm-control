"""Internal helpers shared by token samplers."""
from genlm.control.potential.base import Potential


def _validate_proposal_vocab(target_potential, proposal):
    """Require `proposal.vocab_eos` to match `target_potential.vocab_eos`
    token-for-token. Cross-tokenizer proposals are not yet supported."""
    if not isinstance(proposal, Potential):
        raise TypeError(
            f"`proposal` must be a Potential; got {type(proposal).__name__}."
        )
    if proposal.vocab_eos != target_potential.vocab_eos:
        raise ValueError(
            "Proposals with different tokenizers are not yet supported. "
            f"Target has {len(target_potential.vocab_eos)} tokens; proposal has "
            f"{len(proposal.vocab_eos)}."
        )


class _CoroutineSuspended(Exception):
    """Raised by :func:`_drive_sync` when the coroutine it drives hits a real
    ``await`` (suspends) instead of finishing inline. A dedicated type -- not a bare
    ``RuntimeError`` -- so :func:`_drive_or_hop`'s fallback catches *only* suspension
    and never swallows an unrelated error raised from inside the coroutine."""


def _drive_sync(coro):
    """Run a coroutine that never actually suspends to completion in the current
    thread, returning its result.

    The burst path runs a sampler's draw from the engine's worker thread. When the
    control-side potentials it awaits are pure-CPU async (an FSA ``prefix`` /
    ``complete``, no real await), the coroutine finishes on the first ``send(None)``
    and we avoid the per-particle ``run_coroutine_threadsafe`` hop to the main event
    loop (the worker-blocked ``lock.acquire`` that dominated AWRS bursts). If the
    coroutine *does* suspend (an autobatched / IPC / LM-backed potential), it raises
    :class:`_CoroutineSuspended` so the caller can hop instead (:func:`_drive_or_hop`).
    """
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    coro.close()
    raise _CoroutineSuspended(
        "coroutine suspended on a real await; it cannot be driven synchronously"
    )


def _drive_or_hop(make_coro, run_async):
    """Drive ``make_coro()`` inline if it never suspends; otherwise re-run a fresh
    coroutine on the event loop via ``run_async`` (one hop).

    ``make_coro`` is a zero-arg factory because the inline attempt's coroutine is
    closed when it suspends, so the hop needs a fresh one. The inline attempt wastes
    only the work up to the first real ``await`` -- the coroutine suspends there --
    so the re-run is cheap. This keeps a burst draw hop-free for non-suspending (FSA)
    potentials while staying correct (not crashing) for suspending ones -- an
    autobatched / LM-backed / IPC condition. The adaptive inline-or-hop replaces both
    a hard-coded always-inline (fast but crashes on suspension) and a hard-coded
    always-hop (safe but pays the hop on every step)."""
    try:
        return _drive_sync(make_coro())
    except _CoroutineSuspended:
        return run_async(make_coro())
