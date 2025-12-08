import pytest
from genlm.control.potential.built_in.llm import load_model_by_name


@pytest.mark.asyncio
async def test_mlx_backend():
    """Test that load_model_by_name correctly loads MLX backend."""
    model = load_model_by_name(
        "mlx-community/Llama-3.2-1B-Instruct-4bit", backend="mlx"
    )

    from genlm.backend.llm import AsyncMlxLM

    assert isinstance(model, AsyncMlxLM)
