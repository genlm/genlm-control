import asyncio
import numpy as np
from multiprocessing import Pool
from genlm_control.potential.base import Potential


class MPPotential(Potential):
    """A Potential that adds parallel processing capabilities to any base Potential implementation."""

    def __init__(self, potential_factory, factory_args, num_workers=2):
        self.pool = Pool(
            num_workers,
            initializer=self._init_worker,
            initargs=(potential_factory, factory_args),
        )
        # maybe TODO: use shared memory to pass the weights to the main process
        decode = self.pool.apply(self._get_decode)
        super().__init__(decode)

    @staticmethod
    def _init_worker(factory, args):
        global _worker_potential, _worker_event_loop
        _worker_potential = factory(*args)
        _worker_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_event_loop)

    @staticmethod
    def _get_decode():
        return _worker_potential.decode

    @staticmethod
    def _run_coroutine(coroutine):
        global _worker_event_loop
        return _worker_event_loop.run_until_complete(coroutine)

    @staticmethod
    def _worker_logw_next(context):
        return MPPotential._run_coroutine(_worker_potential.logw_next(context)).weights

    @staticmethod
    def _worker_prefix(context):
        return MPPotential._run_coroutine(_worker_potential.prefix(context))

    @staticmethod
    def _worker_complete(context):
        return MPPotential._run_coroutine(_worker_potential.complete(context))

    @staticmethod
    def _worker_score(context):
        return MPPotential._run_coroutine(_worker_potential.score(context))

    @staticmethod
    def _worker_logw_next_seq(context, extension):
        return MPPotential._run_coroutine(
            _worker_potential.logw_next_seq(context, extension)
        )

    async def _run_in_pool(self, func, *args):
        loop = asyncio.get_running_loop()
        future = asyncio.Future()

        def _callback(result):
            loop.call_soon_threadsafe(future.set_result, result)

        def _error_callback(exc):
            loop.call_soon_threadsafe(future.set_exception, exc)

        self.pool.apply_async(
            func, args, callback=_callback, error_callback=_error_callback
        )

        return await future

    async def logw_next(self, context):
        result = await self._run_in_pool(self._worker_logw_next, context)
        return self.make_lazy_weights(result)

    async def prefix(self, context):
        return await self._run_in_pool(self._worker_prefix, context)

    async def complete(self, context):
        return await self._run_in_pool(self._worker_complete, context)

    async def logw_next_seq(self, context, extension):
        return await self._run_in_pool(self._worker_logw_next_seq, context, extension)

    async def batch_logw_next(self, contexts):
        results = await asyncio.gather(
            *(
                self._run_in_pool(self._worker_logw_next, context)
                for context in contexts
            )
        )
        return [self.make_lazy_weights(result) for result in results]

    async def batch_complete(self, contexts):
        results = await asyncio.gather(
            *(self._run_in_pool(self._worker_complete, context) for context in contexts)
        )
        return np.array(results)

    async def batch_prefix(self, contexts):
        results = await asyncio.gather(
            *(self._run_in_pool(self._worker_prefix, context) for context in contexts)
        )
        return np.array(results)

    def __del__(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
