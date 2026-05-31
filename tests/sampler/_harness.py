"""Shared harness for the SMC parity gates (gate-1 byte-exact + gate-2 no-bias) and
their reference generators. One definition of the bits that MUST agree across the
gate, its generator, and the other gate -- drift here is a silent false-green.

- ``seed_all`` : the single seed call (was copied in 5 files).
- ``ctx_repr`` / ``ctx_ids`` / ``num`` : context + number serialization. TWO ctx
  flavors kept distinct on purpose -- ``ctx_repr`` is the byte/string view gate-1's
  Mock vocab needs; ``ctx_ids`` is the engine token-id view gate-2 needs. Both flatten
  nested unit lists and map EOS to a sentinel.
- ``load_snapshot`` : load a reference snapshot, optionally enforcing a ``__config__``.
- ``assert_unbiased`` : the ``|mean| <= max(floor, k*sem)`` no-bias check gate-2 repeats.
"""

import json

import numpy as np

from genlm.control.constant import EndOfSequence


def seed_all(s):
    """Seed numpy + torch (in that order) -- the per-case seed every gate uses."""
    np.random.seed(s)
    import torch

    torch.manual_seed(s)


def ctx_repr(ctx):
    """Byte/string-level, comparison-stable repr of a context (gate-1: Mock vocab)."""

    def one(t):
        if hasattr(t, "type_"):  # EOS / any EndOfSequence (EOS = EndOfSequence("EOS"))
            return f"<EOS:{getattr(t, 'type_', 'EOS')}>"
        if isinstance(t, list):
            return [one(x) for x in t]
        if isinstance(t, bytes):
            return "b:" + t.hex()
        return repr(t)

    return [one(t) for t in ctx]


def ctx_ids(ctx):
    """Token-id view of a context (gate-2: engine tokens). EOS -> "EOS"; nested
    unit lists flattened so a unit run and a token run with the same tokens compare equal."""
    out = []

    def emit(t):
        if isinstance(t, EndOfSequence):
            out.append("EOS")
        elif isinstance(t, list):
            for s in t:
                emit(s)
        else:
            out.append(t.token_id)

    for t in ctx:
        emit(t)
    return out


def num(x):
    """nan/inf-safe scalar for JSON: special floats -> tagged strings, else float()."""
    if np.isnan(x):
        return "nan"
    if np.isneginf(x):
        return "-inf"
    if np.isposinf(x):
        return "inf"
    return float(x)


def load_snapshot(path, expected_config=None):
    """Load a reference snapshot JSON. If ``expected_config`` is given, pop the stored
    ``__config__`` and raise on mismatch (so a comparison against a snapshot generated
    under a different model/prompt/eos fails loudly, never silently). Returns the dict
    (without ``__config__``). Raises FileNotFoundError if missing -- the caller decides
    whether that is a skip (gate-1) or an empty-snapshot hard-error-per-key (gate-2)."""
    with open(path) as f:
        snap = json.load(f)
    stored = snap.pop("__config__", None)
    if expected_config is not None and stored is not None and stored != expected_config:
        raise RuntimeError(
            f"snapshot at {path} was generated under {stored} but the tests now run "
            f"under {expected_config}; regenerate it."
        )
    return snap


def assert_unbiased(diffs, *, floor, k, label):
    """No-bias check: |mean(diffs)| <= max(floor, k*sem). Returns (mean, sem) for logging.
    Used by every multi-seed gate-2 no-bias assertion (log_ml and length)."""
    diffs = np.asarray(diffs, dtype=float)
    sem = float(diffs.std() / np.sqrt(len(diffs)))
    mean = float(diffs.mean())
    assert abs(mean) <= max(floor, k * sem), (
        f"{label}: biased -- mean {mean:+.4f} exceeds max({floor}, {k}*sem={k * sem:.4f})"
    )
    return mean, sem
