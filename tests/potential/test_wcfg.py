import pytest
import numpy as np
from genlm.grammar import CFG, Float, Boolean
from genlm.control.potential.built_in import WCFG, BoolCFG


@pytest.fixture
def byte_wcfg():
    c = CFG(Float, S="S", V={b"a"[0], b"b"[0]})
    c.add(3.0, "S", "A", "B")
    c.add(2.0, "S", "A", "B", "B")
    c.add(1.0, "A", b"a"[0])
    c.add(1.0, "B", b"b"[0])
    return c


def test_wcfg_init_wrong_semiring():
    # Test initialization with non-Float semiring
    c = CFG(Boolean, S="S", V={b"a"[0], b"b"[0]})
    with pytest.raises(ValueError):
        WCFG(c)


WCFG_GRAMMAR_STR = """
3.0: S -> A B
2.0: S -> A B B
1.0: A -> a
1.0: B -> b
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_pot",
    [
        pytest.param(lambda byte_wcfg: WCFG(byte_wcfg), id="direct_cfg"),
        pytest.param(
            lambda byte_wcfg: WCFG.from_string(WCFG_GRAMMAR_STR), id="from_string"
        ),
    ],
)
async def test_wcfg_complete(byte_wcfg, make_pot):
    pot = make_pot(byte_wcfg)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(3))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    pot_spawned = pot.spawn()
    assert pot_spawned.cfg == pot.cfg


@pytest.mark.asyncio
async def test_wcfg_prefix(byte_wcfg):
    pot = WCFG(byte_wcfg)

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(5))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, np.log(5))


BCFG_LARK_GRAMMAR = """
start: A B | A B B
A: "a"
B: "b"
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_pot",
    [
        pytest.param(lambda byte_wcfg: BoolCFG(byte_wcfg), id="direct_cfg"),
        pytest.param(
            lambda byte_wcfg: BoolCFG.from_lark(BCFG_LARK_GRAMMAR), id="from_lark"
        ),
    ],
)
async def test_bcfg_complete(byte_wcfg, make_pot):
    pot = make_pot(byte_wcfg)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    pot_spawned = pot.spawn()
    assert pot_spawned.cfg == pot.cfg


@pytest.mark.asyncio
async def test_bcfg_prefix(byte_wcfg):
    pot = BoolCFG(byte_wcfg)

    # Test empty string handling
    log_weight = await pot.prefix(b"")
    assert log_weight == 0

    log_weight = await pot.prefix(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.prefix(b"a")
    assert log_weight == 0


@pytest.mark.asyncio
async def test_properties(byte_wcfg):
    pot = WCFG(byte_wcfg)

    await pot.assert_logw_next_consistency(b"ab")
    await pot.assert_autoreg_fact(b"ab")
    await pot.assert_batch_consistency([b"a", b"ab"])

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")
    await pot.assert_batch_consistency([b""])

    pot = BoolCFG(byte_wcfg)

    await pot.assert_logw_next_consistency(b"ab")
    await pot.assert_autoreg_fact(b"ab")
    await pot.assert_batch_consistency([b"a", b"ab"])

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")
    await pot.assert_batch_consistency([b""])


@pytest.mark.parametrize(
    "semiring, cls",
    [(Float, WCFG), (Boolean, BoolCFG)],
    ids=["wcfg", "bcfg"],
)
def test_wcfg_repr(semiring, cls):
    c = CFG(semiring, S="S", V={b"a"[0]})
    pot = cls(c)
    repr(pot)
    pot._repr_html_()
