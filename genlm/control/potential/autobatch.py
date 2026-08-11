import asyncio
import weakref
from collections import defaultdict

from genlm.control.potential.base import Potential, VocabTables
from genlm.control.util import LazyWeights, window_stats


class AutoBatchedPotential(Potential):
    """
    AutoBatchedPotential is a wrapper around a Potential that enables automatic batching of concurrent requests.

    Concurrent calls to instance methods (`complete`, `prefix`, `score`,
    `logw_next`) meet in a per-event-loop window and execute as one call to the
    corresponding batch method of the underlying potential (`batch_complete`,
    `batch_prefix`, `batch_score`, `batch_logw_next`). The window is held open
    by its first caller until a full event-loop pass adds no new request, then
    flushed in that caller's own coroutine — no background task, nothing bound
    to a loop at construction time.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].

    Attributes:
        potential (Potential): The underlying potential instance that is being wrapped.
    """

    def __init__(self, potential):
        self.potential = potential
        self._windows = weakref.WeakKeyDictionary()  # event loop -> _Window
        # The wrapped potential's own tables: a wrapper indexes the same vocabulary,
        # so rebuilding them would cost O(len(vocab)) per wrap for an identical result.
        super().__init__(
            potential.vocab,
            tables=VocabTables(
                potential.token_type,
                potential.eos,
                potential.vocab_eos,
                potential.lookup,
            ),
        )

    async def _queued(self, batch_method_name, context):
        loop = asyncio.get_running_loop()
        window = self._windows.get(loop)
        if window is None:
            window = self._windows[loop] = _Window()
        future = loop.create_future()
        window.queue.append((batch_method_name, context, future))
        if not window.armed:
            window.armed = True
            try:
                # Callers reach their request at different depths of a `gather`
                # tree, and each level is another scheduler turn; yield until a
                # turn adds nothing, so the whole cohort lands in one flush.
                while True:
                    n = len(window.queue)
                    await asyncio.sleep(0)
                    if len(window.queue) == n:
                        break
                queue, window.queue = window.queue, []
            finally:
                window.armed = False
            await self._flush(queue)
        return await future

    async def _flush(self, queue):
        """One call per batch method for the whole cohort. Every future gets its
        result or the exception -- never silence."""
        groups = defaultdict(list)
        for method_name, context, future in queue:
            groups[method_name].append((context, future))
        for method_name, requests in groups.items():
            window_stats[("autobatch", method_name, len(requests))] += 1
            try:
                results = await getattr(self.potential, method_name)(
                    [context for context, _ in requests]
                )
                # batch_logw_next returns ONE batched LazyWeights [N, V+1]; split it
                # back into per-request rows (other batch methods return [N] arrays).
                if isinstance(results, LazyWeights):
                    results = [
                        results.spawn(results.weights[i])
                        for i in range(len(requests))
                    ]
                assert len(results) == len(requests)
                for (_, future), result in zip(requests, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                for _, future in requests:
                    if not future.done():
                        future.set_exception(exc)

    async def complete(self, context):
        return await self._queued("batch_complete", context)

    async def prefix(self, context):
        return await self._queued("batch_prefix", context)

    async def score(self, context):
        return await self._queued("batch_score", context)

    async def logw_next(self, context):
        return await self._queued("batch_logw_next", context)

    async def logw_eos(self, context):
        # No batch form to queue against, and the wrapped potential may answer it far
        # more cheaply than the default read off a whole `logw_next` row.
        return await self.potential.logw_eos(context)

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts)

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts)

    async def batch_score(self, contexts):
        return await self.potential.batch_score(contexts)

    async def batch_logw_next(self, contexts):
        return await self.potential.batch_logw_next(contexts)

    def spawn(self, *args, **kwargs):
        return AutoBatchedPotential(self.potential.spawn(*args, **kwargs))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.potential!r})"

    async def cleanup(self):
        # Nothing of the window's to stop (it lives and dies with its callers);
        # forward like every wrapper, or the seat flag would break the chain.
        await self.potential.cleanup()


class _Window:
    """Per-event-loop request meeting point; must not outlive its loop."""

    __slots__ = ("queue", "armed")

    def __init__(self):
        self.queue = []
        self.armed = False


_WRAPPERS = weakref.WeakKeyDictionary()  # potential -> its AutoBatchedPotential


def autobatched(potential):
    """THE autobatched view of ``potential`` -- memoized, so every call site
    (and every sampler sharing the potential) resolves to the same wrapper and
    therefore the same batching window. ``None`` passes through; an
    already-wrapped potential is not wrapped twice."""
    if potential is None or isinstance(potential, AutoBatchedPotential):
        return potential
    wrapper = _WRAPPERS.get(potential)
    if wrapper is None:
        wrapper = _WRAPPERS[potential] = AutoBatchedPotential(potential)
    return wrapper
