"""Gate-2 LoRA: K=2 multi-view with q=LoRA / p0=base on ONE vLLM engine.

The proposal q and prior p0 are two ``PromptedLLM``s on the same engine differing only by
adapter (q carries a LoRA, p0 is base), so per-request LoRA has to reach each view
independently; draws come from q and are reweighted by ``p0/q``. The reference is the same
run over a HuggingFace ``AsyncTransformer`` carrying the same adapter.

Requires CUDA + vLLM + a downloadable SmolLM LoRA adapter. (The backend forces the V1
engine in-process at import; vLLM's V1 multiprocessing engine-core deadlocks on LoRA.)
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("LoRA multi-view needs CUDA + vLLM", allow_module_level=True)

from huggingface_hub import snapshot_download  # noqa: E402
from genlm.backend.llm.vllm import AsyncVirtualLM  # noqa: E402
from genlm.backend.llm.hf import AsyncTransformer  # noqa: E402
from genlm.control.potential.built_in.llm import PromptedLLM  # noqa: E402
from genlm.control.sampler.token import DirectTokenSampler  # noqa: E402
from _harness import run_case  # noqa: E402

ADAPTER = "farpluto/SmolLM-135M-Instruct-Finetune-LoRA"
EOS = [b"\n"]


@pytest.fixture(scope="module")
def adapter():
    """(local adapter path, base model id) for ``ADAPTER``."""
    path = snapshot_download(ADAPTER)
    base = json.load(open(os.path.join(path, "adapter_config.json")))[
        "base_model_name_or_path"
    ]
    return path, base


@pytest.fixture(scope="module")
def lora_model(adapter):
    path, base = adapter
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


@pytest.fixture(scope="module")
def ref_model(adapter):
    """The reference backend: the same base + adapter behind plain per-context forwards."""
    path, base = adapter
    m = AsyncTransformer.from_name(base)
    m.add_new_lora(path, "vk")
    return m


def _run(model, q_lora_name, seed):
    """One K=2 multi-view run on ``model``: q under ``q_lora_name``, p0 base."""
    prompt_ids = model.tokenizer.encode("The capital of France is")

    def make():
        p0 = PromptedLLM(model, prompt_ids=prompt_ids, eos_byte_strings=EOS)
        q = PromptedLLM(
            model, prompt_ids=prompt_ids, eos_byte_strings=EOS, lora_name=q_lora_name
        )
        return DirectTokenSampler(potential=p0, proposal=q)

    return run_case(make, 8, 0.0, 12, seed)["log_ml"]


def test_lora_proposal_applies_adapter(lora_model):
    """The adapter materially changes the draw: q=LoRA vs q=base diverge, so per-request
    LoRA reaches the proposal view. (q=base is the degenerate q==p0 case -> log_ml 0;
    q=LoRA picks up a non-trivial p0/q correction.)"""
    ml_lora = _run(lora_model, "vk", 1234)
    ml_base = _run(lora_model, None, 1234)
    assert abs(ml_lora - ml_base) > 1e-6, (
        f"adapter had no effect on the draw: log_ml lora={ml_lora} base={ml_base}"
    )


def test_lora_proposal_unbiased(lora_model, ref_model):
    """q=LoRA / p0=base K=2 multi-view on the engine vs the same run over plain forwards.
    Unbiased log_ml across seeds -- validates the LoRA path's correctness, not just that
    the adapter is wired."""
    seeds = (1234, 7, 99, 2024, 555, 31)
    diffs = []
    for seed in seeds:
        diffs.append(_run(lora_model, "vk", seed) - _run(ref_model, "vk", seed))
    diffs = np.array(diffs)
    sem = diffs.std() / np.sqrt(len(diffs))
    print(
        f"\nlora engine vs forwards over {len(seeds)} seeds: "
        f"log_ml diff mean={diffs.mean():+.4f} sem={sem:.4f}"
    )
    assert abs(diffs.mean()) <= max(0.3, 2.5 * sem), (
        f"lora log_ml biased vs plain forwards: mean {diffs.mean():+.4f} (sem {sem:.4f})"
    )
