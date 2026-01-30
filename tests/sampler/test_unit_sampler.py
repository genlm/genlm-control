import pytest
import numpy as np
import unittest.mock as mock

from genlm.control.sampler import (
    DirectTokenSampler,
    MultiTokenUnitSampler,
    TokenSetBoundary,
    FixedLengthBoundary,
    BoundaryPredicate,
)
from genlm.control.sampler.unit import flatten_units
from genlm.control.sampler import CFGBoundary
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
    boundary = TokenSetBoundary({b" ", b"!", EOS})
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
    # By default, TokenSetBoundary includes boundary token (preserves context)
    # Unit should end with boundary token or EOS
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
    boundary = FixedLengthBoundary(3)
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
    boundary = TokenSetBoundary({b" ", EOS})

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
    flat_context = [token for unit in unit_context for token in unit]
    unit, weight, logp = await unit_sampler.sample(
        flat_context, unit_context=unit_context, draw=None
    )
    assert isinstance(unit, list)
    assert len(unit) > 0


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_timeout():
    """Test that timeout prevents infinite loops."""
    # Create potential where boundary is never reached
    vocab = [b"a", b"b", b"c"]
    logws = np.log([0.4, 0.3, 0.2, 0.1])

    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)

    # Boundary that's never satisfied
    boundary = TokenSetBoundary({b"NEVER"})

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

    # Test TokenSetBoundary (default: includes boundary)
    boundary = TokenSetBoundary({b" ", b"!", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )

    unit, _, _ = await unit_sampler.sample([], draw=None)
    assert isinstance(unit, list)
    assert len(unit) > 0
    # Default behavior: boundary token is included
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
async def test_boundary_predicate_finalize_unit():
    """Test finalize_unit behavior for different boundary predicates."""
    # TokenSetBoundary always keeps boundary tokens
    token_boundary = TokenSetBoundary({b" ", b"\n"})
    buffer = [b"hello", b" "]
    finalized = token_boundary.finalize_unit(buffer)
    assert finalized == [b"hello", b" "]

    # FixedLengthBoundary keeps all tokens
    fixed_boundary = FixedLengthBoundary(10)
    buffer = [b"a"] * 10
    finalized = fixed_boundary.finalize_unit(buffer)
    assert finalized == buffer
    assert len(finalized) == 10

    # CFGBoundary keeps all tokens (complete parsed unit)
    grammar = 'start: "x"+'
    cfg_boundary = CFGBoundary(grammar, min_length=1)
    buffer = [b"x", b"x", b"x"]
    finalized = cfg_boundary.finalize_unit(buffer)
    assert finalized == buffer


@pytest.mark.asyncio
async def test_sequence_model_with_multi_token_units():
    """Test SequenceModel with MultiTokenUnitSampler."""
    vocab = [b"hello", b" ", b"world", b"!"]
    logws = np.log([0.3, 0.2, 0.2, 0.2, 0.1])

    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = TokenSetBoundary({b" ", b"!", EOS})

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
            boundary_predicate=TokenSetBoundary({b" "}),
        )


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_start_weight():
    """Test start_weight method from subunit_sampler"""
    vocab = [b"hello", b" ", b"world"]
    logws = np.log([0.4, 0.2, 0.3, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = TokenSetBoundary({b" ", EOS})

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
    boundary = TokenSetBoundary({b" ", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
    )
    await unit_sampler.cleanup()


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_exception_handling():
    """Test exception handling in sample method for expected errors."""
    vocab = [b"a", b"b"]
    logws = np.log([0.499, 0.499, 0.002])  # EOS very unlikely, won't be sampled first
    mock_potential = MockPotential(vocab, logws)

    # Create a subunit sampler that will raise an expected exception
    class FailingSampler(DirectTokenSampler):
        async def sample(self, context, draw=None):
            # Fail after first token with a runtime error (expected failure type)
            if len(context) > 0:
                raise RuntimeError("Simulated sampling failure")
            return await super().sample(context, draw)

    subunit_sampler = FailingSampler(mock_potential)
    # Boundary that's never hit (no b" " in vocab, EOS won't be sampled)
    boundary = TokenSetBoundary({b" ", EOS})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=3,
    )
    # Should handle RuntimeError gracefully and return -inf weight
    unit, weight, _ = await unit_sampler.sample([], draw=None)
    assert weight == float("-inf")
    assert isinstance(unit, list)

    # Verify TypeError
    class BuggySampler(DirectTokenSampler):
        def __init__(self, potential):
            super().__init__(potential)
            self.call_count = 0

        async def sample(self, context, draw=None):
            self.call_count += 1
            raise TypeError("Programming error: wrong type")

    buggy_subunit_sampler = BuggySampler(mock_potential)
    buggy_unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=buggy_subunit_sampler,
        boundary_predicate=boundary,
    )
    with pytest.raises(TypeError, match="Programming error"):
        await buggy_unit_sampler.sample([], draw=None)


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_eos_in_unit():
    """Test EOS handling when EOS appears in the middle of a unit."""
    vocab = [b"hello", b" ", b"world"]
    # Make EOS highly likely
    logws = np.log([0.1, 0.1, 0.1, 0.7])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = TokenSetBoundary({b" ", EOS})
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


def test_cfg_boundary_import():
    """Test that CFGBoundary is available."""
    from genlm.control.sampler import CFGBoundary

    assert CFGBoundary is not None


@pytest.mark.asyncio
async def test_cfg_boundary_simple_arithmetic():
    """Test CFGBoundary with simple arithmetic grammar."""
    # Simple arithmetic grammar
    grammar = """
        start: expr
        expr: NUMBER
            | expr "+" NUMBER
        NUMBER: /[0-9]+/
    """
    boundary = CFGBoundary(grammar, complete_rules={"start"}, min_length=1)
    assert boundary([], [b"5"]) is True
    assert boundary([], [b"1", b"+", b"2"]) is True
    assert boundary([], [b"1", b"+"]) is False
    assert boundary([], [b"+"]) is False
    assert boundary([], []) is False


@pytest.mark.asyncio
async def test_cfg_boundary_min_length():
    """Test CFGBoundary min_length parameter."""
    grammar = """
        start: "a"+
    """
    boundary = CFGBoundary(grammar, min_length=3)
    assert boundary([], [b"a"]) is False
    assert boundary([], [b"a", b"a"]) is False
    assert boundary([], [b"a", b"a", b"a"]) is True
    assert boundary([], [b"a", b"a", b"a", b"a"]) is True


@pytest.mark.asyncio
async def test_cfg_boundary_complete_rules_none():
    """Test CFGBoundary with complete_rules=None (any parse is complete)."""
    grammar = """
        start: word+
        word: /[a-z]+/
    """
    boundary = CFGBoundary(grammar, complete_rules=None, min_length=1)
    assert boundary([], [b"hello"]) is True
    assert boundary([], [b"hello", b"world"]) is True
    assert boundary([], [b"123"]) is False


@pytest.mark.asyncio
async def test_cfg_boundary_with_eos():
    """Test CFGBoundary handles EOS tokens correctly."""
    grammar = """
        start: "x"+
    """
    boundary = CFGBoundary(grammar, min_length=1)
    assert boundary([], [b"x", EOS]) is True
    assert boundary([], [b"x", b"x", EOS]) is True
    assert boundary([], [EOS]) is False


@pytest.mark.asyncio
async def test_cfg_boundary_integration():
    """Test CFGBoundary integrated with MultiTokenUnitSampler."""
    # Grammar: parentheses must be balanced
    grammar = """
        start: atom
        atom: "(" atom ")"
            | "x"
    """
    boundary = CFGBoundary(grammar, complete_rules={"start"}, min_length=1)
    vocab = [b"(", b")", b"x"]
    logws = np.log([0.3, 0.3, 0.3, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=20,
    )
    unit, _, _ = await unit_sampler.sample([], draw=None)
    assert isinstance(unit, list)
    assert len(unit) > 0
    text = b"".join(t for t in unit if isinstance(t, bytes) and t is not EOS).decode(
        "utf-8"
    )
    is_complete = boundary([], unit)
    is_at_limit = len(unit) >= 20 or (unit and unit[-1] is EOS)
    assert is_complete or is_at_limit, f"Unit should be complete or at limit: {text!r}"


@pytest.mark.asyncio
async def test_cfg_boundary_lalr_parser():
    """Test CFGBoundary with LALR parser (faster but no ambiguity support)."""
    grammar = """
        start: item+
        item: "a" | "b"
    """
    boundary = CFGBoundary(grammar, parser_type="lalr", min_length=1)
    assert boundary([], [b"a"]) is True
    assert boundary([], [b"a", b"b"]) is True
    assert boundary([], [b"c"]) is False


def test_cfg_boundary_invalid_grammar():
    """Test CFGBoundary raises error with invalid grammar."""
    invalid_grammar = """
        start: undefined_rule
    """
    with pytest.raises(ValueError, match="Failed to create Lark parser"):
        CFGBoundary(invalid_grammar)


def test_cfg_boundary_get_parse_tree():
    """Test CFGBoundary.get_parse_tree helper method."""
    grammar = """
        start: "a"+
    """
    boundary = CFGBoundary(grammar, min_length=1)
    tree = boundary.get_parse_tree("aaa")
    assert tree is not None
    assert tree.data == "start"
    tree = boundary.get_parse_tree("bbb")
    assert tree is None


def test_cfg_boundary_repr():
    """Test CFGBoundary string representation."""
    grammar = 'start: "x"'
    boundary1 = CFGBoundary(grammar, start_rule="start", complete_rules={"start"})
    assert "CFGBoundary" in repr(boundary1)
    assert "start" in repr(boundary1)
    assert "complete_rules" in repr(boundary1)
    boundary2 = CFGBoundary(grammar, complete_rules=None)
    assert "CFGBoundary" in repr(boundary2)
    assert "complete_rules" not in repr(boundary2)


@pytest.mark.asyncio
async def test_multi_token_unit_sampler_max_subunits_reached():
    """Test MultiTokenUnitSampler when max_subunits_per_unit is reached without boundary."""

    # Create a boundary that never returns True
    class NeverCompleteBoundary(BoundaryPredicate):
        def __call__(self, unit_context, subunit_buffer):
            return False

        def __repr__(self):
            return "NeverCompleteBoundary()"

    vocab = [b"a", b"b", b"c"]
    logws = np.log([0.4, 0.3, 0.2, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = NeverCompleteBoundary()
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=5,
    )
    unit, weight, _ = await unit_sampler.sample([], draw=None)
    assert len(unit) <= 5
    if EOS in unit:
        assert unit[-1] is EOS
    assert weight <= 1e-10


def test_cfg_boundary_exception_handling():
    """Test CFGBoundary handles LarkError."""
    from lark.exceptions import LarkError

    grammar = 'start: "x"'
    boundary = CFGBoundary(
        grammar, start_rule="start", complete_rules={"start"}, min_length=1
    )
    result = boundary([], [bytes([ord("x")])])
    assert result is True
    result = boundary([], [])
    assert result is False
    with mock.patch.object(
        boundary.parser, "parse", side_effect=LarkError("Parse failed")
    ):
        result = boundary([], [bytes([ord("x")])])
        assert result is False

    # Other exceptions
    with mock.patch.object(
        boundary.parser, "parse", side_effect=ValueError("Unexpected error")
    ):
        with pytest.raises(ValueError, match="Unexpected error"):
            boundary([], [bytes([ord("x")])])


@pytest.mark.asyncio
async def test_weight_accumulation_single_token_unit():
    """Test that a single-token unit has the correct weight.

    When a unit consists of exactly one token (immediate boundary hit),
    the unit weight should equal that token's individual weight.
    """
    # Vocabulary with space as boundary token
    vocab = [b"a", b" "]
    # Weights: a=0.3, space=0.6, EOS=0.1
    logws = np.log([0.3, 0.6, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = TokenSetBoundary({b" "})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )

    # Create deterministic draw function to always pick space (index 1)
    def draw_space(probs):
        return vocab[1]

    unit, logw, logp = await unit_sampler.sample([], draw=draw_space)
    # Unit should be just the space token
    assert unit == [b" "]
    expected_logw = np.log(0.3 + 0.6 + 0.1)
    assert np.isclose(logw, expected_logw, atol=1e-10)
    expected_logp = np.log(0.6)
    assert np.isclose(logp, expected_logp, atol=1e-10)


@pytest.mark.asyncio
async def test_weight_accumulation_two_token_unit():
    """Test that a two-token unit has weight = product of individual weights.

    When sampling [token1, token2], the unit weight should be w1 * w2.
    """
    vocab = [b"h", b" "]
    # Weights chosen for easy verification: h=0.4, space=0.5, EOS=0.1
    logws = np.log([0.4, 0.5, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)

    boundary = TokenSetBoundary({b" "})

    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    sample_sequence = iter([b"h", b" "])

    def draw_sequence(probs):
        return next(sample_sequence)

    unit, logw, logp = await unit_sampler.sample([], draw=draw_sequence)
    # Unit should be [h, space]
    assert unit == [b"h", b" "]
    Z = 0.4 + 0.5 + 0.1
    expected_logw = 2 * np.log(Z)
    assert np.isclose(logw, expected_logw, atol=1e-10)
    # logp = log(p(h)) + log(p(space)) = log(0.4) + log(0.5)
    expected_logp = np.log(0.4) + np.log(0.5)
    assert np.isclose(logp, expected_logp, atol=1e-10)


@pytest.mark.asyncio
async def test_weight_accumulation_three_token_unit():
    """Test weight accumulation for a three-token unit with non-uniform weights."""
    vocab = [b"a", b"b", b" "]
    # Non-uniform weights that don't sum to 1: a=0.2, b=0.3, space=0.4, EOS=0.1
    logws = np.log([0.2, 0.3, 0.4, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = TokenSetBoundary({b" "})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    sample_sequence = iter([b"a", b"b", b" "])

    def draw_sequence(probs):
        return next(sample_sequence)

    unit, logw, logp = await unit_sampler.sample([], draw=draw_sequence)
    # Unit should be [a, b, space]
    assert unit == [b"a", b"b", b" "]
    Z = 0.2 + 0.3 + 0.4 + 0.1
    expected_logw = 3 * np.log(Z)
    assert np.isclose(logw, expected_logw, atol=1e-10)
    # logp = log(0.2) + log(0.3) + log(0.4)
    expected_logp = np.log(0.2) + np.log(0.3) + np.log(0.4)
    assert np.isclose(logp, expected_logp, atol=1e-10)


@pytest.mark.asyncio
async def test_weight_accumulation_with_non_unit_normalizing_constant():
    """Test weight accumulation when Z != 1."""
    vocab = [b"x", b" "]
    # Weights that sum to 10: x=3, space=5, EOS=2
    logws = np.log([3.0, 5.0, 2.0])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = TokenSetBoundary({b" "})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    # Sample x twice, then space
    sample_sequence = iter([b"x", b"x", b" "])

    def draw_sequence(probs):
        return next(sample_sequence)

    unit, logw, logp = await unit_sampler.sample([], draw=draw_sequence)
    assert unit == [b"x", b"x", b" "]
    Z = 3.0 + 5.0 + 2.0
    expected_logw = 3 * np.log(Z)
    assert np.isclose(logw, expected_logw, atol=1e-10)
    expected_logp = 2 * np.log(3.0 / Z) + np.log(5.0 / Z)
    assert np.isclose(logp, expected_logp, atol=1e-10)


@pytest.mark.asyncio
async def test_weight_accumulation_eos_terminates():
    """Test that EOS terminates the unit and weight is correctly accumulated."""
    vocab = [b"a", b"b"]
    logws = np.log([0.3, 0.3, 0.4])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    # Boundary that won't be hit by a or b
    boundary = TokenSetBoundary({b" "})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    # Sample a, then EOS
    sample_sequence = iter([b"a", EOS])

    def draw_sequence(probs):
        return next(sample_sequence)

    unit, logw, logp = await unit_sampler.sample([], draw=draw_sequence)
    assert unit == [b"a", EOS]
    Z = 0.3 + 0.3 + 0.4
    expected_logw = 2 * np.log(Z)
    assert np.isclose(logw, expected_logw, atol=1e-10)
    expected_logp = np.log(0.3) + np.log(0.4)
    assert np.isclose(logp, expected_logp, atol=1e-10)


@pytest.mark.asyncio
async def test_weight_accumulation_fixed_length_boundary():
    """Test weight accumulation with FixedLengthBoundary."""
    vocab = [b"1", b"2", b"3"]
    # Weights: 1=0.2, 2=0.3, 3=0.4, EOS=0.1
    logws = np.log([0.2, 0.3, 0.4, 0.1])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    boundary = FixedLengthBoundary(3)
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=10,
    )
    sample_sequence = iter([b"1", b"2", b"3"])

    def draw_sequence(probs):
        return next(sample_sequence)

    unit, logw, logp = await unit_sampler.sample([], draw=draw_sequence)
    assert unit == [b"1", b"2", b"3"]
    Z = 0.2 + 0.3 + 0.4 + 0.1
    expected_logw = 3 * np.log(Z)
    assert np.isclose(logw, expected_logw, atol=1e-10)
    expected_logp = np.log(0.2) + np.log(0.3) + np.log(0.4)
    assert np.isclose(logp, expected_logp, atol=1e-10)


@pytest.mark.asyncio
async def test_weight_is_negative_infinity_on_max_subunits():
    """Test that weight is -inf when max_subunits_per_unit is exceeded without boundary."""
    vocab = [b"a", b"b"]
    logws = np.log([0.4, 0.4, 0.2])
    mock_potential = MockPotential(vocab, logws)
    subunit_sampler = DirectTokenSampler(mock_potential)
    # Boundary that will never be satisfied
    boundary = TokenSetBoundary({b"NEVER"})
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=subunit_sampler,
        boundary_predicate=boundary,
        max_subunits_per_unit=3,
    )

    def draw_a(probs):
        return b"a"

    unit, logw, logp = await unit_sampler.sample([], draw=draw_a)
    assert len(unit) == 3
    assert all(t == b"a" for t in unit)
    assert logw == float("-inf")
    # logp should still be accumulated correctly
    Z = 0.4 + 0.4 + 0.2
    expected_logp = 3 * np.log(0.4 / Z)
    assert np.isclose(logp, expected_logp, atol=1e-10)
