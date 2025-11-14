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
from genlm.control.sampler.sequence import SMC
from genlm.control.constant import EOS
from conftest import MockPotential


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
    unit, weight, logp = await unit_sampler.sample([], draw=None)
    
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
    
    # Boundary: exactly 3 tokens per unit
    boundary = boundary_fixed_length(3)
    
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    # Sample unit
    unit, weight, logp = await unit_sampler.sample([], draw=None)
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
        [b"hello", b" "],  # First unit (word "hello ")
        [b"world", b" "],  # Second unit (word "world ")
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
    unit, weight, logp = await unit_sampler.sample([], draw=None)
    
    # Unit should NOT end with boundary token (if boundary was reached)
    if len(unit) > 0:
        assert unit[-1] not in {b" "}


@pytest.mark.asyncio  
async def test_multi_token_unit_sampler_timeout():
    """Test that timeout prevents infinite loops."""
    # Create a potential where boundary is never reached
    vocab = [b"a", b"b", b"c"]  # No boundary tokens in vocab
    logws = np.log([0.4, 0.3, 0.2, 0.1])  # vocab + EOS
    
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    
    # Boundary that's never satisfied
    boundary = boundary_token_set({b"NEVER"})
    
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=5,  # Small timeout
    )
    
    # Sample unit
    unit, weight, logp = await unit_sampler.sample([], draw=None)
    assert len(unit) == 5  # Exactly max_subunits_per_unit


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
    
    unit, weight, logp = await unit_sampler.sample([], draw=None)
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
    
    unit2, weight2, logp2 = await unit_sampler2.sample([], draw=None)
    if EOS not in unit2:
        assert len(unit2) == 3
    
    # Test repr
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
    

    print("unit_sampler", unit_sampler)
    # Run SMC with unit sampler
    sequences = await SMC(unit_sampler, critic=None)(
        n_particles=5,
        ess_threshold=0.5,
        max_tokens=10,  # Max 10 units
    )
    
    # Each sequence should be a list of units
    for seq, logw in sequences:
        assert isinstance(seq, list)
        # Each unit should be a list of tokens
        for unit in seq:
            assert isinstance(unit, list) or unit is EOS

