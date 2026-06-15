# SMC parity gates

Two gates guard the SMC algorithm. Their reference snapshots are committed here, so a
fresh checkout runs the gates directly — no regeneration needed.

| Snapshot | Used by | Cases |
|---|---|---|
| `parity_snapshot.json` | gate-1 | all |
| `gate2_steploop_snapshot.json` | gate-2 | 10 of 14 (the default, `steploop_cached`) |
| `gate2_snapshot.json` | gate-2 | 2 (`ref`, original-genlm anchor) |

## Gate-1 — local, no GPU (~0.5s)

Byte-exact StepLoop parity. Mock potentials; loads `parity_snapshot.json` and checks the
current StepLoop against it. No vLLM, no llamppl needed at runtime.

```
python -m pytest tests/sampler/test_per_token_parity.py
```

## Gate-2 — GPU box with vLLM 0.21 + the engine-native backend

Unbiased burst-vs-per-token. Loads the two gate-2 snapshots and checks the engine burst
against them; auto-skips when there's no CUDA.

```
VLLM_USE_FLASHINFER_SAMPLER=0 python -m pytest tests/sampler/test_engine_native.py
```

(Maintainers with the rsync box loop: `BOX=root@<ip> scripts/box.sh gate2`.)

## When a gate fails

- **Gate-1 mismatch** is a real bias — investigate; the StepLoop must stay byte-exact.
- **Gate-2 mismatch** is usually one of two things: a genuine bias (a systematic
  length/`log_ml` gap — a bug), or a numerics mismatch because your engine differs from the
  one the snapshot was generated on (vLLM build / GPU), exceeding the warm-KV tolerance.
  The snapshots are pinned to vLLM 0.21 on the generating box; on a materially different
  stack, regenerate against your own engine with the committed generators:

```
# gate-2 tight ref (box):                 python tests/sampler/gen_steploop_reference.py
# gate-2 'ref' cases (box, needs main):   python tests/sampler/gen_original_reference.py
# gate-1 ref (local, needs llamppl):      python tests/sampler/_gen_parity_snapshot.py
```
