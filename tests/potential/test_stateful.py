from genlm.control.potential.stateful import make_immutable, StatefulPotential
from genlm.control.potential.streaming import (
    AsyncStreamingPotential,
    StreamingPotential,
    Timeout,
    PING_TOKEN,
    KEEP_ALIVE_SET,
)
import pytest
import asyncio
import time
from threading import Thread
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    invariant,
    precondition,
    Bundle,
    consumes,
)
import hypothesis.strategies as st
from hypothesis import assume
from genlm.control.potential.stateful import PriorityMap


def test_make_immutable_converts_non_bytes_to_tuple():
    assert make_immutable([257]) == (257,)


def test_make_immutable_converts_to_bytes_if_possible():
    assert make_immutable([]) == b""
    assert make_immutable([0]) == b"\x00"


class DummyPotential(StreamingPotential):
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))

    def calculate_score_from_stream(self, stream) -> float:
        size = 0
        for s in stream:
            size += len(s)
        if size >= 10:
            return 0.0


async def no_sleep(time):
    pass


def no_start(*args, **kwargs):
    raise RuntimeError()


tock = 0


def fast_clock():
    global tock
    tock += 1
    return tock


@pytest.mark.asyncio
async def test_will_time_out_if_too_many_threads_start(monkeypatch):
    (monkeypatch.setattr(asyncio, "sleep", no_sleep),)
    monkeypatch.setattr(Thread, "start", no_start)
    monkeypatch.setattr(time, "time", fast_clock)
    potential = DummyPotential()
    with pytest.raises(Timeout):
        await potential.prefix(b"hi")


@pytest.mark.asyncio
async def test_finished_clone_is_no_op():
    potential = DummyPotential()
    state = potential.new_state()
    await state.finish()
    assert state.finished
    assert (await state.clone()) is state


def test_must_specify_state_class_or_implement_new_state():
    potential = StatefulPotential(vocabulary=[0, 1])
    with pytest.raises(NotImplementedError):
        potential.new_state()


def test_tokens_have_right_repr():
    assert repr(PING_TOKEN) == "PING_TOKEN"


class DummyAsyncPotential(AsyncStreamingPotential):
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))

    async def calculate_score_from_stream(self, stream) -> float:
        size = 0
        while True:
            try:
                size += await stream.more()
            except StopAsyncIteration:
                break
        if size >= 10:
            return 0.0


@pytest.mark.asyncio
async def test_cleanup_clears_up_async_tasks():
    initial = len(KEEP_ALIVE_SET)
    potential = DummyAsyncPotential()
    await potential.prefix(b"hello")
    assert len(KEEP_ALIVE_SET) > initial
    await potential.cleanup()
    assert len(KEEP_ALIVE_SET) <= initial


@pytest.mark.asyncio
async def test_operations_after_finish_are_ignored():
    potential = DummyAsyncPotential()
    state = potential.new_state()
    await state.update_context([0])
    await state.finish()
    assert state.finished
    await state.update_context([0])
    assert len(state.context) == 1
    await state.finish()
    assert state.finished


class PriorityMapTest(RuleBasedStateMachine):
    keys = Bundle("keys")

    def __init__(self):
        super().__init__()
        self.priority_map = PriorityMap()
        self.model = {}

    @invariant()
    def check_soundness(self):
        assert len(self.priority_map) == len(self.model)
        for k, v in self.model.items():
            assert self.priority_map[k] == v
        if len(self.model) > 0:
            k, v = self.priority_map.peek()
            assert k in self.model
            assert v == min(self.model.values())

    @rule(x=st.integers(), y=st.integers(), target=keys)
    def set_item(self, x, y):
        self.priority_map[x] = y
        assert self.priority_map[x] == y
        self.model[x] = y

    @precondition(lambda self: len(self.model) > 0)
    @rule(x=consumes(keys))
    def del_item(self, x):
        # The key can be added to the bundle multiple times, so the
        # `consumes` will not always ensure that the key is in the model.
        assume(x in self.model)
        del self.priority_map[x]
        assert x not in self.priority_map
        del self.model[x]

    @precondition(lambda self: len(self.model) > 0)
    @rule()
    def pop(self):
        expected_v = min(self.model.values())
        k, v = self.priority_map.pop()
        assert v == expected_v
        assert self.model.pop(k) == v


TestPriorityMap = PriorityMapTest.TestCase


def test_peek_after_set():
    x = PriorityMap()
    x[0] = 0
    x[0] = 1
    assert x.peek() == (0, 1)


def test_validation_errors_are_raised():
    x = PriorityMap()
    with pytest.raises(ValueError):
        x.peek()
    with pytest.raises(ValueError):
        x.pop()


def test_priority_map_repr():
    x = PriorityMap()
    x[0] = 0
    x[1] = 1
    assert repr(x) == "PriorityMap({0: 0, 1: 1})"


@pytest.mark.asyncio
async def test_error_on_startup_is_correctly_handled():
    class ErrorPotential(StreamingPotential):
        def calculate_score_from_stream(self, stream) -> float:
            raise Exception("Oh no")

    potential = ErrorPotential(vocabulary=list(range(256)))
    assert await potential.prefix(b"") == -float("inf")
