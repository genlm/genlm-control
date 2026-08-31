import pytest
import numpy as np
from genlm.control.potential.base import Potential


class MockPotential(Potential):
    def __init__(self, has_errors=False):
        self.has_errors = has_errors
        super().__init__([1, 2, 3])

    async def complete(self, context):
        return 1.0

    async def prefix(self, context):
        if self.has_errors:
            return float("-inf")
        return 1.0

    async def logw_next(self, context):
        weights = np.array([0] * (len(self.vocab_eos)))
        if self.has_errors:
            weights[0] = 100.0  # Create inconsistency
        return self.make_lazy_weights(weights)


@pytest.mark.asyncio
async def test_assert_logw_next_consistency():
    pot = MockPotential()
    await pot.assert_logw_next_consistency([], verbosity=1)
    await pot.assert_logw_next_consistency([1], verbosity=1)

    pot = MockPotential(has_errors=True)
    with pytest.raises(AssertionError) as exc:
        await pot.assert_logw_next_consistency([])
    assert "logw_next consistency" in str(exc.value)

    pot = MockPotential()
    await pot.assert_logw_next_consistency([], top=2)


@pytest.mark.asyncio
async def test_assert_autoreg_fact():
    pot = MockPotential()
    await pot.assert_autoreg_fact([], verbosity=1)
    await pot.assert_autoreg_fact([1], verbosity=1)

    pot = MockPotential(has_errors=True)
    with pytest.raises(AssertionError) as exc:
        await pot.assert_autoreg_fact([])
    assert "Factorization not satisfied" in str(exc.value)


@pytest.mark.asyncio
async def test_assert_batch_consistency():
    pot = MockPotential()
    await pot.assert_batch_consistency([[1], [2]], verbosity=1)

    class ScoreErrorPotential(MockPotential):
        async def score(self, context, *args):
            return 100.0

        async def batch_score(self, contexts, *args):
            return [1000.0] * len(contexts)

    pot = ScoreErrorPotential()
    with pytest.raises(AssertionError) as exc:
        await pot.assert_batch_consistency([[]], verbosity=1)
    assert "Batch score mismatch" in str(exc.value)

    class LogwNextErrorPotential(MockPotential):
        async def logw_next(self, context, *args):
            return self.make_lazy_weights([0.0] * len(self.vocab_eos))

        async def batch_logw_next(self, contexts, *args):
            return self.make_lazy_weights(
                np.stack([[1.0] * len(self.vocab_eos)] * len(contexts))
            )

    pot = LogwNextErrorPotential()
    with pytest.raises(AssertionError) as exc:
        await pot.assert_batch_consistency([[]], verbosity=1)
    assert "Batch logw_next mismatch" in str(exc.value)
