import bisect
from collections import defaultdict
import heapq
import asyncio
from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from genlm.control.potential.base import Potential


def make_immutable(context):
    if isinstance(context, (str, bytes, tuple)):
        return context
    try:
        return bytes(context)
    except (ValueError, TypeError):
        return tuple(context)


class ParticleState(ABC):
    def __init__(self, owner):
        self.owner = owner
        self.finished = False
        self.context = []

    async def update_context(self, incremental_context):
        """Update the context with more data that has come in."""
        if self.finished:
            return
        self.context.extend(incremental_context)
        await self.impl_update_context(incremental_context)

    async def finish(self):
        """Mark this state as finished, clearing up any associated
        state, and updating the current score to reflect whether
        this is a valid string in the associated language."""
        if self.finished:
            return
        self.finished = True
        await self.impl_finish()

    @abstractproperty
    def current_score(self):
        """The current score associated with this potential, which
        will reflect whether the current context is a suitable member
        of the language if this has been finished, or whether it is a
        suitable prefix if it has not."""

    @abstractmethod
    async def impl_update_context(self, incremental_context): ...

    @abstractmethod
    async def impl_finish(self): ...

    async def clone(self):
        if self.finished:
            return self
        result = self.owner.new_state()
        await result.update_context(self.context)
        assert self.context == result.context
        assert self.current_score == result.current_score
        return result


class StatefulPotential(Potential):
    def __init__(
        self, vocabulary, token_type=None, eos=None, state_class=None, cache_size=100
    ):
        super().__init__(vocabulary=vocabulary, token_type=token_type, eos=eos)
        self.__state_class = state_class

        self.__cache_size = cache_size
        self.__state_count = 0

        self.__state_pool = defaultdict(list)
        self.__known_contexts = []

        self.__eviction_heap = []
        self.__ages = dict()
        self.__epoch = 0

    def __tick(self):
        self.__epoch += 1
        return self.__epoch

    def new_state(self) -> ParticleState:
        if self.__state_class is None:
            raise NotImplementedError()
        return self.__state_class(self)

    async def cleanup(self):
        await asyncio.gather(
            *[state.finish() for pool in self.__state_pool.values() for state in pool]
        )
        self.__state_pool.clear()
        self.__known_contexts.clear()
        self.__ages.clear()
        self.__eviction_heap.clear()

    async def __look_up_state(self, context):
        context = make_immutable(context)

        state = None

        i = bisect.bisect_left(self.__known_contexts, context)
        if i < len(self.__known_contexts):
            existing = self.__known_contexts[i]
            if context[: len(existing)] == existing:
                pool = self.__state_pool[existing]
                if pool:
                    state = pool.pop()
                    if not pool:
                        del self.__known_contexts[i]
                    self.__state_count -= 1
                    assert self.__state_count >= 0
        if state is None:
            state = self.new_state()
        if len(context) > len(state.context):
            await state.update_context(context[len(state.context) :])
        assert len(state.context) == len(context)
        assert list(state.context) == list(context)
        return state

    def __return_state(self, state):
        assert not state.finished
        context = make_immutable(state.context)
        i = bisect.bisect_left(self.__known_contexts, context)
        if i >= len(self.__known_contexts):
            self.__known_contexts.append(context)
        elif self.__known_contexts[i] != context:
            self.__known_contexts.insert(i, context)
        self.__state_pool[context].append(state)
        self.__state_count += 1
        age = self.__tick()
        heapq.heappush(self.__eviction_heap, (age, context))
        self.__ages[context] = age
        while self.__state_count > self.__cache_size:
            assert len(self.__eviction_heap) >= self.__state_count
            seen_age, to_evict = heapq.heappop(self.__eviction_heap)
            if seen_age == self.__ages[to_evict]:
                i = bisect.bisect_left(self.__known_contexts, to_evict)
                assert self.__known_contexts[i] == to_evict
                pool = self.__state_pool.pop(to_evict, ())
                assert pool
                self.__state_count -= len(pool)
                del self.__known_contexts[i]

    async def prefix(self, context):
        state = await self.__look_up_state(context)
        result = state.current_score
        self.__return_state(state)
        return result

    async def complete(self, context):
        state = await self.__look_up_state(context)
        await state.finish()
        return state.current_score

    async def logw_next(self, context):
        """Compute the next-token weights of each token in `self.vocab_eos` given `context`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.
        """
        state = await self.__look_up_state(context)
        assert not state.finished
        ctx_log_w = state.current_score

        if ctx_log_w == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")

        async def step_score(x):
            local_state = await state.clone()
            await local_state.update_context([x])

            if x == self.eos:
                await local_state.finish()
                return local_state.current_score
            else:
                result = local_state.current_score
                await local_state.finish()
                return result

        scores = np.array(
            await asyncio.gather(*[step_score(x) for x in self.vocab_eos])
        )

        logws = scores - ctx_log_w

        return self.make_lazy_weights(logws)
