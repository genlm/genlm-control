"""Gate-2 parity: the engine-accelerated WindowDriver vs the ground-truth SlowDriver.

Both drivers run the identical hub SMC algorithm; the ONLY intended difference is
the source of next-token logits:

* SlowDriver re-prefills the full context every step (``PromptedLLM.logw_next`` ->
  ``AsyncVirtualLM.next_token_logprobs``);
* WindowDriver reads warm-KV decode logits inside the vLLM engine.

So the two should agree up to the warm-KV-vs-reprefill numerical residual
(~1e-2/logit), which may occasionally flip a single Gumbel-max draw but is small
and unbiased -- NOT a systematic divergence. A consistent ``log_ml`` bias or
sequences that systematically run longer in the window is a BUG, not a tolerance.

Requires CUDA + vLLM; uses gpt2. Run on the box with
``VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1``.
"""

import asyncio
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("engine-native parity needs CUDA + vLLM", allow_module_level=True)

from genlm.control.constant import EndOfSequence  # noqa: E402
from genlm.control.potential.built_in.llm import PromptedLLM  # noqa: E402
from genlm.control.sampler.token import DirectTokenSampler  # noqa: E402
from genlm.control.sampler.hub import (  # noqa: E402
    Hub,
    SlowDriver,
    WindowDriver,
    window_eligible,
)

MODEL = "gpt2"
SEED = 1234


def _seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def _ctx_ids(ctx):
    """Token-id view of a particle context (EOS -> the sentinel string)."""
    out = []
    for t in ctx:
        if isinstance(t, EndOfSequence):
            out.append("EOS")
        else:
            out.append(t.token_id)
    return out


def _build_hub(llm, n_particles, ess_threshold, max_tokens):
    sampler = DirectTokenSampler(llm)
    return Hub(
        unit_sampler=sampler,
        critic=None,
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        twist_with_critic=False,
    )


@pytest.fixture(scope="module")
def llm():
    from genlm.backend.llm import AsyncVirtualLM

    model = AsyncVirtualLM.from_name(
        MODEL,
        engine_opts={
            "gpu_memory_utilization": 0.3,
            "max_model_len": 256,
            "enable_prefix_caching": True,
        },
    )
    return PromptedLLM(model, eos_byte_strings=[b"\n"])


@pytest.mark.parametrize("ess_threshold", [0.0, 0.5])
def test_unconstrained_window_vs_slow(llm, ess_threshold):
    import genlm.backend.llm.vllm as backend

    llm.set_prompt_from_str("The")
    n_particles = 8
    max_tokens = 12

    assert window_eligible(_build_hub(llm, n_particles, ess_threshold, max_tokens))

    # --- slow (ground truth) ---
    _seed()
    slow_hub = _build_hub(llm, n_particles, ess_threshold, max_tokens)
    slow = asyncio.run(SlowDriver(slow_hub).run())

    # --- window (engine) ---
    _seed()
    win_hub = _build_hub(llm, n_particles, ess_threshold, max_tokens)
    win = asyncio.run(WindowDriver(win_hub, backend).run())

    slow_ctx = [_ctx_ids(p.context) for p in slow]
    win_ctx = [_ctx_ids(p.context) for p in win]
    slow_w = np.array([p.logw for p in slow])
    win_w = np.array([p.logw for p in win])

    from genlm.control.sampler.sequence import Sequences, _unpack_particles

    slow_seq = Sequences(*_unpack_particles(slow))
    win_seq = Sequences(*_unpack_particles(win))

    # ---- report ----
    n_ctx_match = sum(a == b for a, b in zip(slow_ctx, win_ctx))
    slow_lens = [len(c) for c in slow_ctx]
    win_lens = [len(c) for c in win_ctx]
    print(f"\n=== ess_threshold={ess_threshold} ===")
    print(f"contexts match: {n_ctx_match}/{n_particles}")
    print(f"slow lens: {slow_lens}")
    print(f"win  lens: {win_lens}")
    print(f"mean len slow={np.mean(slow_lens):.3f} win={np.mean(win_lens):.3f}")
    print(f"slow logw: {np.array2string(slow_w, precision=4)}")
    print(f"win  logw: {np.array2string(win_w, precision=4)}")
    print(f"slow log_ml={slow_seq.log_ml:.6f}  win log_ml={win_seq.log_ml:.6f}  "
          f"diff={win_seq.log_ml - slow_seq.log_ml:+.6f}")
    for i, (a, b) in enumerate(zip(slow_ctx, win_ctx)):
        if a != b:
            print(f"  particle {i} DIFF: slow={a} win={b}")

    # Systematic-divergence guards (NOT bit-exact): window must not run
    # systematically longer, and log_ml must not be biased.
    assert abs(np.mean(win_lens) - np.mean(slow_lens)) <= 1.5, (
        "window sequences run systematically longer/shorter than slow"
    )
    assert abs(win_seq.log_ml - slow_seq.log_ml) <= 0.5, "log_ml diverged"
