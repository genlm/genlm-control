import importlib.util

import pytest

from genlm.control.potential.built_in.llm import load_model_by_name

# MLX is Apple-silicon only; it cannot be installed on the Linux CI/GPU box, so
# this backend test only runs where the `mlx` package is importable.
_HAS_MLX = importlib.util.find_spec("mlx") is not None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed (Apple silicon only)")
@pytest.mark.asyncio
async def test_mlx_backend():
    """Test that load_model_by_name correctly loads MLX backend."""
    model = load_model_by_name(
        "mlx-community/Llama-3.2-1B-Instruct-4bit", backend="mlx"
    )

    from genlm.backend.llm import AsyncMlxLM

    assert isinstance(model, AsyncMlxLM)
