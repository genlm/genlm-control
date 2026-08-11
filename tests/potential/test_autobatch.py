import pytest
import asyncio
import time
import numpy as np
from genlm.control.potential import Potential
from genlm.control.potential.autobatch import AutoBatchedPotential, autobatched
from genlm.control.sampler.token import DirectTokenSampler, AWRS
from genlm.control.sampler.sequence import SMC
from genlm.control.sampler.set import TrieSetSampler


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
async def test_cleanup_is_a_safe_noop():
    """There is no background task to stop -- the window lives and dies with its
    callers -- so cleanup() is a no-op, safe to call repeatedly and after use."""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    await autobatched.cleanup()
    await asyncio.gather(*(autobatched.complete(seq) for seq in [b"a", b"bb"]))
    await autobatched.cleanup()
    await autobatched.cleanup()  # idempotent


class SmallPotential(Potential):
    """Vocab of 4 int tokens with a real (context-length-dependent) logw_next, so
    concurrent requests produce distinguishable rows."""

    def __init__(self):
        super().__init__([0, 1, 2, 3])

    async def complete(self, context):
        return -float(len(context))

    async def prefix(self, context):
        return -float(len(context))

    async def logw_next(self, context):
        base = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # len(vocab) + 1 (EOS)
        return self.make_lazy_weights(base * (len(context) + 1))

    async def batch_logw_next(self, contexts):
        base = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rows = np.stack([base * (len(c) + 1) for c in contexts])
        return self.make_lazy_weights(rows)

    def spawn(self):
        return SmallPotential()


class AllowAllPotential(Potential):
    """Vocab of 4 int tokens, zero log-weight everywhere. Doubles as a boolean
    AWRS condition (0 in log-space) and as a trivial SMC critic; only used to
    exercise sampler construction, never `.sample()`."""

    def __init__(self):
        super().__init__([0, 1, 2, 3])

    async def complete(self, context):
        return 0.0

    async def prefix(self, context):
        return 0.0


# Constructed at module scope -- before any event loop exists -- to pin that
# AutoBatchedPotential and the samplers wrapping it never need one at construction.
_module_potential = SmallPotential()
_module_critic = AllowAllPotential()
_module_sampler = DirectTokenSampler(_module_potential, autobatch=True)
_module_smc = SMC(_module_sampler, critic=_module_critic, autobatch=True)


def test_autobatched_memoized():
    """autobatched() returns the same wrapper for the same potential, passes None
    through unchanged, and does not double-wrap an already-wrapped potential."""
    potential = SmallPotential()

    w1 = autobatched(potential)
    w2 = autobatched(potential)
    assert w1 is w2
    assert isinstance(w1, AutoBatchedPotential)

    assert autobatched(None) is None
    assert autobatched(w1) is w1

    other = autobatched(SmallPotential())
    assert other is not w1


def test_direct_token_sampler_seat_default():
    """DirectTokenSampler wraps its potential/proposal seats by default;
    autobatch=False leaves them bare."""
    sampler = DirectTokenSampler(SmallPotential(), proposal=SmallPotential())
    assert isinstance(sampler.potential, AutoBatchedPotential)
    assert isinstance(sampler.proposal, AutoBatchedPotential)

    potential = SmallPotential()
    bare = DirectTokenSampler(potential, autobatch=False)
    assert bare.potential is potential
    assert not isinstance(bare.potential, AutoBatchedPotential)


def test_awrs_seat_default():
    """AWRS wraps its potential/condition/proposal seats by default;
    autobatch=False leaves them bare."""
    sampler = AWRS(SmallPotential(), AllowAllPotential(), proposal=SmallPotential())
    assert isinstance(sampler.potential, AutoBatchedPotential)
    assert isinstance(sampler.condition, AutoBatchedPotential)
    assert isinstance(sampler.proposal, AutoBatchedPotential)

    potential = SmallPotential()
    condition = AllowAllPotential()
    bare = AWRS(potential, condition, autobatch=False)
    assert bare.potential is potential
    assert bare.condition is condition
    assert not isinstance(bare.potential, AutoBatchedPotential)
    assert not isinstance(bare.condition, AutoBatchedPotential)


def test_smc_seat_default():
    """SMC wraps its critic seat by default; autobatch=False leaves it bare."""
    sampler = DirectTokenSampler(SmallPotential(), autobatch=False)

    smc = SMC(sampler, critic=AllowAllPotential())
    assert isinstance(smc.critic, AutoBatchedPotential)

    critic = AllowAllPotential()
    bare_smc = SMC(sampler, critic=critic, autobatch=False)
    assert bare_smc.critic is critic
    assert not isinstance(bare_smc.critic, AutoBatchedPotential)


@pytest.mark.asyncio
async def test_trie_set_sampler_wraps_iter_seat_only():
    """TrieSetSampler wraps only iter_potential by default (one concurrent ask
    per particle per step); item_potential is never wrapped -- the trie walk
    asks it sequentially. autobatch=False leaves both seats bare."""

    class TinyBytes(Potential):
        def __init__(self, vocab):
            super().__init__(vocab)

        async def complete(self, context):
            return 0.0

        async def prefix(self, context):
            return 0.0

    iter_potential = TinyBytes([b"a", b"b"])
    item_potential = TinyBytes([97, 98])
    sampler = TrieSetSampler(iter_potential, item_potential)
    assert isinstance(sampler.iter_potential, AutoBatchedPotential)
    assert sampler.iter_potential.potential is iter_potential
    assert sampler.item_potential is item_potential
    assert not isinstance(sampler.item_potential, AutoBatchedPotential)
    await sampler.cleanup()

    bare_sampler = TrieSetSampler(
        TinyBytes([b"a", b"b"]), TinyBytes([97, 98]), autobatch=False
    )
    assert not isinstance(bare_sampler.iter_potential, AutoBatchedPotential)
    assert not isinstance(bare_sampler.item_potential, AutoBatchedPotential)
    await bare_sampler.cleanup()


@pytest.mark.asyncio
async def test_construct_outside_event_loop():
    """The wrapper and the samplers seated on it were constructed at module scope,
    before any event loop existed; running them later must still work."""
    seqs = await _module_smc(n_particles=4, ess_threshold=0.5, max_tokens=3)
    assert len(seqs) == 4
    await _module_smc.cleanup()


@pytest.mark.asyncio
async def test_concurrent_logw_next_one_batched_call_and_correct_rows():
    """Concurrent logw_next calls through the wrapper meet in the window and hit
    the underlying batch_logw_next as ONE call over the whole cohort; the batched
    [N, V+1] LazyWeights splits back so row i matches the unbatched result for
    request i (not some other row)."""
    potential = SmallPotential()
    wrapped = autobatched(potential)

    calls = []
    orig_batch_logw_next = potential.batch_logw_next

    async def counting(contexts):
        calls.append(len(contexts))
        return await orig_batch_logw_next(contexts)

    potential.batch_logw_next = counting

    contexts = [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    want = await asyncio.gather(*(potential.logw_next(c) for c in contexts))
    have = await asyncio.gather(*(wrapped.logw_next(c) for c in contexts))

    assert calls == [len(contexts)]  # exactly one batched call, whole cohort
    for h, w in zip(have, want):
        h.assert_equal(w)

    await wrapped.cleanup()


@pytest.mark.asyncio
async def test_concurrent_error_fails_whole_cohort():
    """An exception raised by the underlying batch method fails every caller
    in the cohort with that exception -- never silence for some and not others."""

    class ErrorPotential(SmallPotential):
        async def batch_logw_next(self, contexts):
            raise ValueError("boom")

    wrapped = autobatched(ErrorPotential())
    contexts = [[], [0], [0, 1], [0, 1, 2]]

    results = await asyncio.gather(
        *(wrapped.logw_next(c) for c in contexts), return_exceptions=True
    )
    assert len(results) == len(contexts)
    for r in results:
        assert isinstance(r, ValueError)
        assert str(r) == "boom"

    await wrapped.cleanup()


@pytest.mark.asyncio
async def test_spawn_rewraps():
    """spawn() on the wrapper re-wraps a fresh spawn of the underlying potential,
    not the same wrapper or the same underlying instance."""
    potential = SmallPotential()
    wrapped = autobatched(potential)

    spawned = wrapped.spawn()
    assert isinstance(spawned, AutoBatchedPotential)
    assert spawned is not wrapped
    assert isinstance(spawned.potential, SmallPotential)
    assert spawned.potential is not potential

    await wrapped.cleanup()
    await spawned.cleanup()
