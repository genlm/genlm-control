"""Gate-2 on MLX: the engine-accelerated BurstLoop is UNBIASED vs the exact per-token path.

Same cases, same harness, same no-bias assertion as ``test_engine_native.py`` -- only the
engine behind the ``PromptedLLM`` differs, and with it the reference.

The reference is a LIVE StepLoop at the same seed. The cached snapshots are gpt2 logits
under vLLM, which MLX does not reproduce bit-for-bit, so comparing against them would
measure the two engines rather than the burst. A live StepLoop on the SAME MLX model is
RNG-matched through the counter-based picker, so the paired diff is tight and the only
admissible residual is warm-KV-vs-reprefill.

Runs LOCALLY on Apple silicon -- no box, no CUDA. This is the only arm where gate-2 does.
"""

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from genlm.control.potential.built_in.llm import PromptedLLM  # noqa: E402

from _harness import assert_case_unbiased, threefry_draw  # noqa: E402,F401
from gate2_cases import CASES, MODEL, PROMPT, EOS_BYTES  # noqa: E402


@pytest.fixture(scope="module")
def llm():
    from genlm.backend.llm import AsyncMlxLM

    return PromptedLLM(AsyncMlxLM.from_name(MODEL), eos_byte_strings=EOS_BYTES)


@pytest.mark.parametrize("label", list(CASES))
def test_burst_vs_steploop(label, llm):
    """Every shared case, burst vs live StepLoop on the same MLX model.

    ``need_resample`` is read off the case rather than listed per test: an ``ess > 0``
    case that never crosses would pass vacuously without exercising the fork-by-gather
    path, which on MLX is the whole KV mechanism."""
    c = CASES[label]
    assert_case_unbiased(c, llm, PROMPT, len_bound=1.5, need_resample=c.ess > 0)
