"""Gate-2 LoRA: K=2 multi-view burst with q=LoRA / p0=base on ONE engine.

The proposal q and prior p0 are two ``PromptedLLM``s on the same vLLM engine differing
only by adapter (q carries a LoRA, p0 is base). The burst submits two substreams per
particle -- q's with the adapter, p0's without -- via per-request LoRA, injects both
warm logit views, and reweights by ``p0/q``.

Requires CUDA + vLLM + a downloadable SmolLM LoRA adapter. (The backend forces the V1
engine in-process at import; vLLM's V1 multiprocessing engine-core deadlocks on LoRA.)
"""

import asyncio
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("LoRA burst needs CUDA + vLLM", allow_module_level=True)

from huggingface_hub import snapshot_download  # noqa: E402
from genlm.backend.llm.vllm import AsyncVirtualLM  # noqa: E402
from genlm.control.potential.built_in.llm import PromptedLLM  # noqa: E402
from genlm.control.sampler.token import DirectTokenSampler  # noqa: E402
from genlm.control.sampler.smc import Controller, StepLoop  # noqa: E402
from genlm.control.sampler.burst import BurstLoop, burst_blocker  # noqa: E402
from genlm.control.sampler.sequence import Sequences, _unpack_particles  # noqa: E402

ADAPTER = "farpluto/SmolLM-135M-Instruct-Finetune-LoRA"
EOS = [b"\n"]


@pytest.fixture(scope="module")
def lora_model():
    path = snapshot_download(ADAPTER)
    base = json.load(open(os.path.join(path, "adapter_config.json")))[
        "base_model_name_or_path"
    ]
    m = AsyncVirtualLM.from_name(
        base,
        engine_opts={
            "enable_lora": True,
            "max_lora_rank": 16,
            "max_loras": 2,
            "gpu_memory_utilization": 0.3,
            "max_model_len": 256,
            "enforce_eager": True,
        },
    )
    m.add_new_lora(path, "vk")
    return m


def _run(model, q_lora_name, seed, driver_cls):
    np.random.seed(seed)
    torch.manual_seed(seed)
    prompt_ids = model.tokenizer.encode("The capital of France is")
    p0 = PromptedLLM(model, prompt_ids=prompt_ids, eos_byte_strings=EOS)
    q = PromptedLLM(model, prompt_ids=prompt_ids, eos_byte_strings=EOS, lora_name=q_lora_name)
    controller = Controller(
        unit_sampler=DirectTokenSampler(potential=p0, proposal=q),
        critic=None,
        n_particles=8,
        ess_threshold=0.0,
        max_tokens=12,
        twist_with_critic=False,
    )
    if driver_cls is BurstLoop:
        assert burst_blocker(controller) is None, burst_blocker(controller)
    driver = driver_cls(controller)
    parts = asyncio.run(driver.run())
    return getattr(driver, "n_bursts", 0), float(Sequences(*_unpack_particles(parts)).log_ml)


def test_lora_proposal_burst_applies_adapter(lora_model):
    """q=LoRA / p0=base on one engine -> K=2 multi-view burst. The config bursts, and
    the adapter materially changes the draw (q=LoRA vs q=base diverge), proving the
    per-request LoRA flows through the burst per view. (q=base is the degenerate q==p0
    case -> log_ml 0; q=LoRA picks up a non-trivial p0/q correction.)"""
    nb_lora, ml_lora = _run(lora_model, "vk", 1234, BurstLoop)
    nb_base, ml_base = _run(lora_model, None, 1234, BurstLoop)
    assert nb_lora > 0 and nb_base > 0, f"did not burst: lora={nb_lora} base={nb_base}"
    assert abs(ml_lora - ml_base) > 1e-6, (
        f"adapter had no effect on the burst draw: log_ml lora={ml_lora} base={ml_base}"
    )


def test_lora_proposal_burst_vs_steploop(lora_model):
    """The real target: q=LoRA / p0=base K=2 multi-view burst vs StepLoop (which now
    applies q's adapter via slow-path per-view LoRA). Unbiased log_ml across seeds
    (warm-KV residual only) -- validates the LoRA burst's correctness, not just that
    the adapter is wired."""
    seeds = (1234, 7, 99, 2024, 555, 31)
    diffs, n_bursts = [], 0
    for seed in seeds:
        nb, ml_burst = _run(lora_model, "vk", seed, BurstLoop)
        _, ml_slow = _run(lora_model, "vk", seed, StepLoop)
        diffs.append(ml_burst - ml_slow)
        n_bursts = max(n_bursts, nb)
    diffs = np.array(diffs)
    sem = diffs.std() / np.sqrt(len(diffs))
    print(
        f"\nlora burst vs steploop over {len(seeds)} seeds: "
        f"log_ml diff mean={diffs.mean():+.4f} sem={sem:.4f}"
    )
    assert n_bursts > 0, "did not burst"
    assert abs(diffs.mean()) <= max(0.3, 2.5 * sem), (
        f"lora burst log_ml biased vs StepLoop: mean {diffs.mean():+.4f} (sem {sem:.4f})"
    )
