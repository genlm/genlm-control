import pytest
import numpy as np

from genlm.control.sampler import (
    DirectTokenSampler,
    MultiTokenUnitSampler,
    boundary_token_set,
    boundary_fixed_length,
    TokenSetBoundary,
    FixedLengthBoundary,
)
from genlm.control.sampler.unit import flatten_units
from genlm.control.sampler.sequence import SMC
from genlm.control.constant import EOS
from conftest import MockPotential, WeightedSet


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_basic():
    """Test basic multi-token unit sampling with boundary tokens."""
    # Create a simple mock potential
    vocab = [b"hello", b" ", b"world", b"!"]
    logws = np.log([0.4, 0.1, 0.3, 0.1, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" ", b"!", EOS})
    # Unit sampler samples words according to boundary
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    # Sample a unit
    unit, weight, _ = await unit_sampler.sample([], draw=None)
    # Unit should be a list of tokens
    assert isinstance(unit, list)
    assert len(unit) > 0
    # Unit must end with a boundary token
    assert unit[-1] in {b" ", b"!", EOS}
    # Weight should be finite
    assert weight != float("-inf")
    assert not np.isnan(weight)


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_fixed_length():
    """Test fixed-length unit sampling."""
    vocab = [b"a", b"b", b"c"]
    logws = np.log([0.3, 0.3, 0.3, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    # 3 tokens per unit
    boundary = boundary_fixed_length(3)
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    # Sample unit
    unit, _, _ = await unit_sampler.sample([], draw=None)
    # Unit should have exactly 3 tokens (unless EOS was sampled early)
    if EOS not in unit:
        assert len(unit) == 3
    else:
        # EOS can terminate early
        assert len(unit) <= 3


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_with_context():
    """Test that unit sampler correctly flattens multi-token unit context."""
    vocab = [b"hello", b" ", b"world"]
    logws = np.log([0.4, 0.2, 0.3, 0.1])

    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" ", EOS})

    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    # Context with multi-token units
    unit_context = [
        [b"hello", b" "],
        [b"world", b" "],
    ]
    # Sample next unit
    unit, weight, logp = await unit_sampler.sample(unit_context, draw=None)
    assert isinstance(unit, list)
    assert len(unit) > 0


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_exclude_boundary():
    """Test excluding boundary token from unit."""
    vocab = [b"hello", b" ", b"world"]
    logws = np.log([0.4, 0.2, 0.3, 0.1])

    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" "})
    # Exclude boundary token from unit
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
        include_boundary_in_unit=False,
    )
    # Sample a unit
    unit, _, _ = await unit_sampler.sample([], draw=None)
    # Unit should NOT end with boundary token (if boundary was reached)
    if len(unit) > 0:
        assert unit[-1] not in {b" "}


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_timeout():
    """Test that timeout prevents infinite loops."""
    # Create potential where boundary is never reached
    vocab = [b"a", b"b", b"c"]
    logws = np.log([0.4, 0.3, 0.2, 0.1])

    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)

    # Boundary that's never satisfied
    boundary = boundary_token_set({b"NEVER"})

    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=5,  # Small timeout
    )
    unit, _, _ = await unit_sampler.sample([], draw=None)
    # Unit should be terminated either by max_subunits_per_unit or EOS
    # The unit length should not exceed max_subunits_per_unit
    assert len(unit) <= 5
    # If EOS is in the unit, it should be the last element
    if EOS in unit:
        assert unit[-1] is EOS


@pytest.mark.asyncio
async def test_boundary_predicate_classes():
    """Test using BoundaryPredicate classes directly."""
    vocab = [b"hello", b" ", b"world", b"!"]
    logws = np.log([0.3, 0.2, 0.2, 0.2, 0.1])

    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)

    # Test TokenSetBoundary
    boundary = TokenSetBoundary({b" ", b"!", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )

    unit, _, _ = await unit_sampler.sample([], draw=None)
    assert isinstance(unit, list)
    assert len(unit) > 0
    assert unit[-1] in {b" ", b"!", EOS}
    # Test FixedLengthBoundary
    boundary2 = FixedLengthBoundary(3)
    unit_sampler2 = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary2,
        max_subunits_per_unit=10,
    )
    unit2, _, _ = await unit_sampler2.sample([], draw=None)
    if EOS not in unit2:
        assert len(unit2) == 3
    assert "TokenSetBoundary" in repr(boundary)
    assert "FixedLengthBoundary" in repr(boundary2)


@pytest.mark.asyncio
async def test_boundary_predicate_validation():
    """Test FixedLengthBoundary validation."""
    with pytest.raises(ValueError, match="Length must be positive"):
        FixedLengthBoundary(0)

    with pytest.raises(ValueError, match="Length must be positive"):
        FixedLengthBoundary(-5)


@pytest.mark.asyncio
async def test_sequence_model_with_multi_token_units():
    """Test SequenceModel with MultiTokenUnitSampler."""
    vocab = [b"hello", b" ", b"world", b"!"]
    logws = np.log([0.3, 0.2, 0.2, 0.2, 0.1])

    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" ", b"!", EOS})

    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    sequences = await SMC(unit_sampler, critic=None)(
        n_particles=5,
        ess_threshold=0.5,
        max_tokens=10,
    )
    # Each sequence should be a list of units
    for seq, _ in sequences:
        assert isinstance(seq, list)
        # Each unit should be a list of tokens
        for unit in seq:
            assert isinstance(unit, list) or unit is EOS


@pytest.mark.asyncio
async def test_flatten_units_flat_list():
    """Test flatten_units with already flat list."""
    flat_context = [b"hello", b" ", b"world"]
    result = flatten_units(flat_context)
    assert result == [b"hello", b" ", b"world"]


@pytest.mark.asyncio
async def test_flatten_units_nested_list():
    """Test flatten_units with nested list (multi-token units)."""
    nested_context = [[b"hello", b" "], [b"world", b"!"], b"\n"]
    result = flatten_units(nested_context)
    assert result == [b"hello", b" ", b"world", b"!", b"\n"]


@pytest.mark.asyncio
async def test_flatten_units_empty():
    """Test flatten_units with empty list."""
    result = flatten_units([])
    assert result == []


@pytest.mark.asyncio
async def test_flatten_units_mixed():
    """Test flatten_units with mixed flat and nested items."""
    mixed_context = [b"a", [b"b", b"c"], b"d", [b"e"]]
    result = flatten_units(mixed_context)
    assert result == [b"a", b"b", b"c", b"d", b"e"]


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_type_error():
    """Test TypeError when subunit_sampler is not a TokenSampler."""
    with pytest.raises(TypeError, match="subunit_sampler must be a TokenSampler"):
        MultiTokenUnitSampler(
            subunit_sampler="not a sampler",
            boundary_predicate=boundary_token_set({b" "}),
        )


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_start_weight():
    """Test start_weight method from subunit_sampler"""
    vocab = [b"hello", b" ", b"world"]
    logws = np.log([0.4, 0.2, 0.3, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" ", EOS})

    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
    )
    start_w = await unit_sampler.start_weight()
    expected_start_w = await subunit_sampler.start_weight()
    assert start_w == expected_start_w


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_cleanup():
    """Test cleanup"""
    vocab = [b"hello", b" ", b"world"]
    logws = np.log([0.4, 0.2, 0.3, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" ", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
    )
    await unit_sampler.cleanup()


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_exception_handling():
    """Test exception handling in sample method."""
    vocab = [b"a", b"b"]
    logws = np.log([0.5, 0.5])
    mock_potential = MockPotential(vocab, logws)
    # Create a subunit sampler that will raise an exception
    class FailingSampler(DirectTokenSampler):
        async def sample(self, context, draw=None):
            # Fail after first token
            if len(context) > 0:
                raise RuntimeError("Test exception")
            return await super().sample(context, draw)
    subunit_sampler = FailingSampler(mock_potential)
    boundary = boundary_token_set({b" ", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
    )
    # Should handle exception and return -inf weight
    unit, weight, logp = await unit_sampler.sample([], draw=None)
    assert weight == float("-inf")
    assert isinstance(unit, list)


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_eos_in_unit():
    """Test EOS handling when EOS appears in the middle of a unit."""
    vocab = [b"hello", b" ", b"world"]
    # Make EOS highly likely
    logws = np.log([0.1, 0.1, 0.1, 0.7])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" ", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    # Sample multiple times to get EOS
    eos_found = False
    for _ in range(20):
        unit, weight, _ = await unit_sampler.sample([], draw=None)
        if unit and unit[-1] is EOS:
            eos_found = True
            break
    assert eos_found, "Should eventually sample EOS"


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_flatten_to_subunits():
    """Test _flatten_to_subunits internal method."""
    vocab = [b"a", b"b", b"c"]
    logws = np.log([0.4, 0.3, 0.2, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = boundary_token_set({b" ", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
    )
    # Test with mixed unit context
    unit_context = [[b"a", b"b"], b"c", [b"a"]]
    flattened = unit_sampler._flatten_to_subunits(unit_context)
    assert flattened == [b"a", b"b", b"c", b"a"]
