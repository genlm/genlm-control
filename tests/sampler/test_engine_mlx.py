"""Gate-2 on MLX: SMC served by the MLX engine is UNBIASED vs the same cases served by
plain per-context forwards.

There is one control path, so the reference is a second BACKEND, not a second loop: the
same case, same seeds, run over a HuggingFace ``AsyncTransformer`` of the same model.
What the gate measures is therefore exactly what engine serving adds -- resident KV reuse
and the engine's numerics -- against forwards that recompute every context from scratch.
Draws are RNG-matched through the counter-based picker, so the paired diff stays tight.

Runs LOCALLY on Apple silicon -- no box, no CUDA. This is the only arm where gate-2 does.
"""

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from genlm.control.potential.built_in.llm import PromptedLLM  # noqa: E402

from _harness import (  # noqa: E402
    assert_case_unbiased,
    live_ref_factory,
    threefry_draw,  # noqa: F401 -- autouse fixture, armed by importing it
)
from gate2_cases import CASES, MODEL, PROMPT, EOS_BYTES  # noqa: E402


@pytest.fixture(scope="module")
def llm():
    from genlm.backend.llm import AsyncMlxLM

    return PromptedLLM(AsyncMlxLM.from_name(MODEL), eos_byte_strings=EOS_BYTES)


@pytest.fixture(scope="module")
def ref_llm():
    from genlm.backend.llm import AsyncTransformer

    return PromptedLLM(AsyncTransformer.from_name(MODEL), eos_byte_strings=EOS_BYTES)


@pytest.mark.parametrize("label", list(CASES))
def test_mlx_vs_forwards(label, llm, ref_llm):
    """Every shared case, MLX-served vs plain-forward reference.

    ``need_resample`` is read off the case rather than listed per test: an ``ess > 0``
    case that never crosses would pass vacuously without exercising the fork path, which
    on MLX is the whole KV mechanism."""
    c = CASES[label]
    ref_llm.set_prompt_from_str(PROMPT)
    assert_case_unbiased(
        c,
        llm,
        PROMPT,
        live_ref_factory(ref_llm),
        len_bound=1.5,
        need_resample=c.ess > 0,
    )
