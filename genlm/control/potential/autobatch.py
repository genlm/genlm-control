import asyncio
from typing import NamedTuple, Callable
from collections import defaultdict

from genlm.control.potential.base import Potential
from genlm.control.util import LazyWeights


class Request(NamedTuple):
    batch_method_name: str
    args_accumulator: Callable
    future: asyncio.Future


class AutoBatchedPotential(Potential):
    """
    AutoBatchedPotential is a wrapper around a Potential that enables automatic batching of concurrent requests.

    This class manages a background loop that collects concurrent requests to instance methods
    (`complete`, `prefix`, `score`, `logw_next`) and batches them together before
    delegating to the corresponding batch methods of the underlying potential
    (`batch_complete`, `batch_prefix`, `batch_score`, `batch_logw_next`).

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].

    Attributes:
        potential (Potential): The underlying potential instance that is being wrapped.
        background_loop (AsyncBatchLoop): An asynchronous loop that manages batch requests.
    """

    def __init__(self, potential):
        self.potential = potential
        self.background_loop = AsyncBatchLoop(potential)
        self.background_loop.start()
        super().__init__(potential.vocab)

    async def complete(self, context):
        return await self.background_loop.queue_request(
            "batch_complete", lambda args: ([*args[0], context],)
        )

    async def prefix(self, context):
        return await self.background_loop.queue_request(
            "batch_prefix", lambda args: ([*args[0], context],)
        )

    async def score(self, context):
        return await self.background_loop.queue_request(
            "batch_score", lambda args: ([*args[0], context],)
        )

    async def logw_next(self, context):
        return await self.background_loop.queue_request(
            "batch_logw_next", lambda args: ([*args[0], context],)
        )

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts)

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts)

    async def batch_score(self, contexts):
        return await self.potential.batch_score(contexts)

    async def batch_logw_next(self, contexts):
        return await self.potential.batch_logw_next(contexts)

    def spawn(self, *args, **kwargs):
        # creates a new background loop.
        return AutoBatchedPotential(self.potential.spawn(*args, **kwargs))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.potential!r})"

    async def cleanup(self):
        """Async cleanup - preferred method"""
        await self.background_loop.cleanup()

    def __del__(self):
        if loop := getattr(self, "background_loop", None):
            loop.close()


class AsyncBatchLoop:
    """Asynchronous batch processing loop for potential methods."""

    def __init__(self, potential, history=None):
        self.potential = potential
        self.q = asyncio.Queue()
        self.task = None
        self.history = history

    def start(self):
        """Start the background processing task."""
        self.task = asyncio.create_task(self._background_loop())

    def queue_request(self, batch_method_name, arg_accumulator):
        """Queue a request for batch processing."""
        future = asyncio.Future()
        self.q.put_nowait(Request(batch_method_name, arg_accumulator, future))
        return future

    async def _background_loop(self):
        """Background task that processes queued requests."""
        while True:
            try:
                method_groups = defaultdict(list)
                req = await self.q.get()
                method_groups[req.batch_method_name].append(req)

                try:
                    while True:
                        req = self.q.get_nowait()
                        method_groups[req.batch_method_name].append(req)
                except asyncio.QueueEmpty:
                    pass

                for method_name, requests in method_groups.items():
                    try:
                        batch_args = ([],)
                        for req in requests:
                            batch_args = req.args_accumulator(batch_args)

                        results = await getattr(self.potential, method_name)(
                            *batch_args
                        )

                        # batch_logw_next returns ONE batched LazyWeights [N, V+1]; split it
                        # back into per-request rows (other batch methods return [N] arrays).
                        if isinstance(results, LazyWeights):
                            results = [
                                results.spawn(results.weights[i])
                                for i in range(len(requests))
                            ]

                        assert len(results) == len(requests)
                        for i, req in enumerate(requests):
                            req.future.set_result(results[i])

                    except Exception as e:
                        for req in requests:
                            if not req.future.done():
                                req.future.set_exception(e)

            except asyncio.CancelledError:
                break

    def close(self):
        """Stop the background processing task and cleanup resources."""
        if task := getattr(self, "task", None):
            try:
                task.cancel()
            except RuntimeError:  # pragma: no cover
                pass  # pragma: no cover
            self.task = None

    async def cleanup(self):
        """Async cleanup - preferred method"""
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def __del__(self):
        self.close()
