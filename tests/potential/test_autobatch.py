import pytest
import asyncio
import time
import numpy as np
from genlm.control.potential import Potential


class MockPotential(Potential):
    """Mock potential for testing with controlled delays"""

    def __init__(self):
        super().__init__(list(range(256)))
        self.delay = 0.1  # 100ms delay per operation

    async def complete(self, context):
        time.sleep(self.delay)
        return np.log(len(context))

    async def prefix(self, context):
        time.sleep(self.delay)
        return np.log(len(context) / 2)

    async def batch_complete(self, contexts):
        time.sleep(self.delay)  # Single delay for batch
        return np.array([np.log(len(context)) for context in contexts])

    async def batch_prefix(self, contexts):
        time.sleep(self.delay)  # Single delay for batch
        return np.array([np.log(len(context) / 2) for context in contexts])

    def spawn(self):
        return MockPotential()


@pytest.mark.asyncio
async def test_correctness():
    """Test that autobatched results match sequential results"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    want = await asyncio.gather(*(potential.complete(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.complete(seq) for seq in sequences))
    assert want == have, [want, have]

    want = await asyncio.gather(*(potential.prefix(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.prefix(seq) for seq in sequences))
    assert want == have, [want, have]

    want = await asyncio.gather(*(potential.score(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.score(seq) for seq in sequences))
    assert want == have, [want, have]

    wants = await asyncio.gather(*(potential.logw_next(seq) for seq in sequences))
    haves = await asyncio.gather(*(autobatched.logw_next(seq) for seq in sequences))
    for have, want in zip(haves, wants):
        have.assert_equal(want)

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_batch_methods():
    """Test that batch methods return expected results (they shouldn't change)"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    want_complete = await potential.batch_complete(sequences)
    have_complete = await autobatched.batch_complete(sequences)
    np.testing.assert_array_equal(want_complete, have_complete)

    want_prefix = await potential.batch_prefix(sequences)
    have_prefix = await autobatched.batch_prefix(sequences)
    np.testing.assert_array_equal(want_prefix, have_prefix)

    want_score = await potential.batch_score(sequences)
    have_score = await autobatched.batch_score(sequences)
    np.testing.assert_array_equal(want_score, have_score)

    want_logw_next = await potential.batch_logw_next(sequences)
    have_logw_next = await autobatched.batch_logw_next(sequences)
    have_logw_next.assert_equal(want_logw_next)  # both batched LazyWeights [N, V+1]

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_performance():
    """Test that autobatched operations are faster than sequential"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    start = time.perf_counter()
    await asyncio.gather(*(potential.complete(seq) for seq in sequences))
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    await asyncio.gather(*(autobatched.complete(seq) for seq in sequences))
    autobatched_time = time.perf_counter() - start

    print(sequential_time, autobatched_time)

    assert autobatched_time < sequential_time / 2

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_error_handling():
    """Test that errors in batch processing are properly propagated"""

    class ErrorPotential(MockPotential):
        async def batch_complete(self, contexts):
            raise ValueError("Test error")

    potential = ErrorPotential()
    autobatched = potential.to_autobatched()

    with pytest.raises(ValueError, match="Test error"):
        await autobatched.complete(b"test")

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_spawn_and_repr():
    """Test spawn method creates new instance and repr works correctly"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    # Test spawn
    spawned = autobatched.spawn()
    assert isinstance(spawned, type(autobatched))
    assert spawned is not autobatched
    assert spawned.potential is not autobatched.potential

    # Test repr
    expected_repr = f"AutoBatchedPotential({potential!r})"
    assert repr(autobatched) == expected_repr

    await autobatched.cleanup()
    await spawned.cleanup()


@pytest.mark.asyncio
async def test_close_and_cleanup():
    """Test close() and cleanup() methods"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    # Test that the background loop is running
    assert autobatched.background_loop.task is not None
    assert not autobatched.background_loop.task.done()

    # Test close()
    autobatched.background_loop.close()
    assert autobatched.background_loop.task is None

    # Test cleanup()
    autobatched = potential.to_autobatched()  # Create new instance
    await autobatched.cleanup()
    assert autobatched.background_loop.task is None


@pytest.mark.asyncio
async def test_del_cleanup():
    """Test __del__ cleanup"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    # Get reference to background loop
    loop = autobatched.background_loop
    assert loop.task is not None

    # Delete the autobatched instance
    del autobatched

    # Verify the background loop was cleaned up
    assert loop.task is None
