import re
import pytest
import graphviz
import numpy as np
from genlm.grammar import WFSA as BaseWFSA, Float, Log, Boolean
from genlm.control.potential.built_in import WFSA, BoolFSA
from hypothesis import strategies as st, given, settings


@pytest.fixture
def float_wfsa():
    """Creates a simple WFSA in float semiring"""
    m = BaseWFSA(Float)
    m.add_I(0, 1.0)
    m.add_arc(0, b"a"[0], 1, 2)
    m.add_arc(1, b"b"[0], 2, 1)
    m.add_arc(1, b"c"[0], 2, 1)
    m.add_arc(1, b"d"[0], 3, 1)  # dead end
    m.add_F(2, 1.0)
    return m


@pytest.fixture
def log_wfsa():
    """Creates a simple WFSA in float semiring"""
    m = BaseWFSA(Log)
    m.add_I(0, Log(0.0))
    m.add_arc(0, b"a"[0], 1, Log(0.0))
    m.add_arc(1, b"b"[0], 2, Log(np.log(0.6)))
    m.add_arc(1, b"c"[0], 2, Log(np.log(0.4)))
    m.add_arc(1, b"d"[0], 3, Log(-float("inf")))  # dead end
    m.add_F(2, Log(0.0))
    return m


@pytest.mark.asyncio
async def test_wfsa(float_wfsa):
    pot = WFSA(float_wfsa)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"ac")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, np.log(4))

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(2))

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


@pytest.mark.asyncio
async def test_wfsa_regex():
    pot = WFSA.from_regex("a(b|c)")

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b"ac")
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, 0)

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(0.5))

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


@pytest.mark.asyncio
async def test_bool_fsa(float_wfsa):
    pot = BoolFSA(float_wfsa)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"ac")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert log_weight == 0

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert log_weight == 0

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


@pytest.mark.asyncio
async def test_wfsa_long_ctx():
    # Test that we don't underflow when the context is long.
    pot = BoolFSA.from_regex(r".*")
    long_ctx = b"a" * 1000

    log_weight = await pot.complete(long_ctx)
    assert log_weight == 0

    log_weight = await pot.prefix(long_ctx)
    assert log_weight == 0


@st.composite
def regex_pattern(draw, max_depth=3):
    """Composite strategy to generate nested regex patterns"""

    def pattern_strategy(depth):
        if depth <= 0:
            # Base case: single escaped character
            char = draw(st.characters(blacklist_categories=("Cs",)))
            return re.escape(char)

        # Choose which type of pattern to generate
        pattern_type = draw(
            st.sampled_from(
                [
                    "simple",
                    "alternation",
                    "concatenation",
                    "optional",
                    "kleene",
                    "plus",
                    "quantified",
                ]
            )
        )

        if pattern_type == "simple":
            return pattern_strategy(0)

        # Generate sub-pattern(s)
        if pattern_type in ("alternation", "concatenation"):
            num_patterns = draw(st.integers(min_value=2, max_value=3))
            patterns = [pattern_strategy(depth - 1) for _ in range(num_patterns)]

            if pattern_type == "alternation":
                return f"({'|'.join(patterns)})"
            else:  # concatenation
                return f"({''.join(patterns)})"

        # Single sub-pattern with operator
        sub_pattern = pattern_strategy(depth - 1)
        if pattern_type == "optional":
            return f"({sub_pattern})?"
        elif pattern_type == "kleene":
            return f"({sub_pattern})*"
        elif pattern_type == "plus":
            return f"({sub_pattern})+"
        else:  # quantified
            quantifier = draw(st.sampled_from(["+", "*", "?", "{1,3}"]))
            return f"({sub_pattern}){quantifier}"

    return pattern_strategy(max_depth)


@pytest.mark.asyncio
@settings(deadline=None)
@given(regex_pattern(max_depth=3), st.data())
async def test_bool_fsa_with_generated_regex(pattern, data):
    """Test that BoolFSA accepts strings that match its regex pattern"""
    pot = BoolFSA.from_regex(pattern)

    matching_str = data.draw(st.from_regex(pattern, fullmatch=True))
    byte_string = matching_str.encode("utf-8")

    log_weight = await pot.complete(byte_string)
    assert log_weight == 0, [matching_str, pattern]

    for prefix in range(len(byte_string)):
        log_weight = await pot.prefix(byte_string[:prefix])
        assert log_weight == 0, [matching_str, byte_string[:prefix]]


def test_wfsa_init_wrong_semiring():
    # Test initialization with unsupported semiring
    wfsa = BaseWFSA(Boolean)  # TODO: support this semiring
    with pytest.raises(ValueError, match="Unsupported semiring"):
        WFSA(wfsa=wfsa)


def test_wfsa_init_float_conversion(log_wfsa):
    # Test that Float semiring is converted to Log
    pot = WFSA(wfsa=log_wfsa)
    assert pot.wfsa.R is Log


def test_wfsa_init_log_no_conversion(log_wfsa):
    # Test that Log semiring is not converted
    pot = WFSA(wfsa=log_wfsa)
    assert pot.wfsa.R is Log
    assert pot.wfsa is log_wfsa


def test_wfsa_repr(log_wfsa):
    pot = WFSA(wfsa=log_wfsa)
    repr(pot)

    try:
        pot._repr_svg_()
    except graphviz.backend.execute.ExecutableNotFound:
        pytest.skip("Graphviz not installed")


def test_bool_fsa_repr(log_wfsa):
    pot = BoolFSA(wfsa=log_wfsa)
    repr(pot)

    try:
        pot._repr_svg_()
    except graphviz.backend.execute.ExecutableNotFound:
        pytest.skip("Graphviz not installed")


def test_wfsa_spawn(log_wfsa):
    pot = WFSA(wfsa=log_wfsa)
    spawned = pot.spawn()
    assert isinstance(spawned, WFSA)


def test_wfsa_clear_cache(log_wfsa):
    pot = WFSA(wfsa=log_wfsa)
    # The empty-prefix base chart lives outside the LRU (``_start_chart``), so the
    # ``_consume`` cache holds only non-empty prefixes and is empty after a clear.
    pot._consume(b"a")
    assert len(pot.cache) > 0
    pot.clear_cache()
    assert len(pot.cache) == 0
    # `_consume(())` still returns the base chart after a clear (held separately).
    assert pot._consume(()) is pot._start_chart


def test_wfsa_consume_lru_evicts(log_wfsa):
    # The LRU bounds the chart cache: with a tiny cap, consuming more distinct
    # prefixes than the cap keeps the cache at the cap (no unbounded growth).
    pot = WFSA(wfsa=log_wfsa, cache_maxsize=3)
    for bs in [b"a", b"ab", b"abc", b"abcd", b"abcde"]:
        pot._consume(bs)
    assert len(pot.cache) <= 3


@pytest.mark.asyncio
async def test_zero_weight_context():
    pot = WFSA.from_regex(r"a")
    with pytest.raises(ValueError, match="Context.*has zero weight."):
        await pot.logw_next(b"b")

    pot = BoolFSA.from_regex(r"a")
    with pytest.raises(ValueError, match="Context.*has zero weight."):
        await pot.logw_next(b"b")
