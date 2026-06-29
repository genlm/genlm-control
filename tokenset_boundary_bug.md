# Bug: `TokenSetBoundary` silently never fires on real `Token` subunits

**Status:** pre-existing on `main` (genlm-control + genlm-backend), confirmed empirically.
Fixed on `shepard/engine-native-smc`; `main` still carries it.

## Summary

`TokenSetBoundary` decides a unit boundary by hash-set membership
(`subunit_buffer[-1] in self.boundary_tokens`). With a real LLM the subunits are
`genlm.backend.tokenization.Token` objects, which subclass `bytes` but **hash by
`token_id`**, not by byte content. So a token whose bytes equal a boundary byte is
*not* found in a set of `bytes`:

```python
Token(220, b" ") == b" "      # True   (content equality works)
Token(220, b" ") in {b" "}    # False  (hash(token_id) != hash(b" "))
```

The boundary therefore never fires. A `MultiTokenUnitSampler` configured with
`TokenSetBoundary` for word/sentence segmentation runs to `max_subunits_per_unit`
every unit and (per `unit.py`) returns the buffer with weight `-inf` — i.e. it
silently fails to segment, instead of closing on its boundary token.

## Root cause (two pieces, both on `main`)

| Piece | Location (on `main`) |
|---|---|
| `Token(bytes)` hashes by `token_id`, equals `bytes` by content | genlm-**backend** `genlm/backend/tokenization/token.py` |
| `TokenSetBoundary.__call__` uses hash-set membership | genlm-**control** `genlm/control/sampler/unit.py` (`subunit_buffer[-1] in self.boundary_tokens`) |
| vocab wrapped in `Token`; samplers return `Token`s | genlm-**control** `genlm/control/util.py` (`Token(token_id=i, …)`) |

Neither piece is from the engine-native SMC work. The mismatch is the interaction of
a content-vs-id hash asymmetry (backend) with a hash-membership lookup (control).

## Why it went unnoticed

- Every `TokenSetBoundary` test in `tests/sampler/test_unit_sampler.py` uses a **mock**
  whose vocab is raw `bytes` (`[b"hello", b" ", …]`). On the bytes path `b" " in {b" "}`
  is `True`, so the tests pass.
- No test constructed a real `Token`, and the one real-LLM `MultiTokenUnitSampler`
  gate config uses a custom length boundary (`_ByteLengthBoundary`), not
  `TokenSetBoundary`. So the real-LLM combination was never exercised.

## Confirmation (empirical, on the box)

Run under `main`'s code via the PYTHONPATH shadow, against the real decoded gpt2 vocab
(tokenizer only, no vLLM engine):

```
TokenSetBoundary loaded from: /root/genlm/genlm-control-main/genlm/control/sampler/unit.py
vocab element type: genlm.backend.tokenization.token.Token
chosen token: Token(token_id=220, byte_string=b' ')  bytes=b' '
  token == boundary byte (content): True
  token in {boundary byte} (hash) : False
main TokenSetBoundary fires on the real Token: False
=> BUG PRESENT ON MAIN
```

## Fix (applied on `shepard/engine-native-smc`)

Compare by **byte content** instead of relying on `Token`/`bytes` hash agreement.
Precompute a plain-`bytes` set + an EOS flag in `__init__`; match `bytes(last)` in
`__call__` (EOS by identity). Localized to `TokenSetBoundary` — the only
token-membership site in the predicate library — so it corrects both the
`MultiTokenUnitSampler` use and the engine-native `slow_cadence` reuse.

```python
def __init__(self, boundary_tokens):
    self.boundary_tokens = set(boundary_tokens)
    self._eos_boundary = any(isinstance(t, EndOfSequence) for t in self.boundary_tokens)
    self._byte_boundaries = {
        bytes(t) for t in self.boundary_tokens if not isinstance(t, EndOfSequence)
    }

def __call__(self, unit_context, subunit_buffer):
    if not subunit_buffer:
        return False
    last = subunit_buffer[-1]
    if isinstance(last, EndOfSequence):
        return self._eos_boundary
    return bytes(last) in self._byte_boundaries
```

Not fixed by changing `Token.__hash__` to hash by content: that is deliberately
id-based and changing it risks the `PYTHONHASHSEED`/vocab-ordering nondeterminism the
project already warns about. The one-comparison-site fix keeps the blast radius small.

## Regression guard

`tests/sampler/test_unit_sampler.py::test_token_set_boundary_real_token_grain`
constructs real `Token`s and asserts the boundary fires by content, including the
telling line `assert space not in {b" "} and space == b" "` — the exact membership
that silently failed — so the regression cannot reappear invisibly. The bytes (mock)
path is asserted unchanged in the same test.

## `main` follow-up

`main` is unaffected by this branch. To fix it there, port the same `TokenSetBoundary`
change + the `Token`-based test. (Not done here — out of scope for the SMC branch.)
