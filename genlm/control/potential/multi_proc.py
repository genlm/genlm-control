import asyncio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from genlm.control.potential.base import Potential


class MultiProcPotential(Potential):
    """A Potential that adds parallel processing capabilities to any base Potential implementation.

    Creates a process pool of worker processes, each containing an instance of the potential.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].
    Each method delegates to a corresponding method of the potential instances running in the
    worker processes, distributing work across multiple processes for improved performance.
    """

    def __init__(self, potential_factory, factory_args, num_workers=2):
        """
        Initialize the MultiProcPotential.

        Args:
            potential_factory (callable): A factory function that creates a potential instance.
            factory_args (tuple): Arguments to pass to the potential factory.
            num_workers (int): The number of worker processes to spawn. Each will contain an instance of the potential.
        """
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=self._init_worker,
            initargs=(potential_factory, factory_args),
        )
        # Get vocab and eos from one of the workers
        vocab, eos = self.executor.submit(self._get_vocab_and_eos).result()
        super().__init__(vocab, eos=eos)

    @staticmethod
    def _init_worker(factory, args):
        global _worker_potential, _worker_event_loop
        _worker_potential = factory(*args)
        _worker_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_event_loop)

    @staticmethod
    def _get_vocab_and_eos():
        return _worker_potential.vocab, _worker_potential.eos

    @staticmethod
    def _run_coroutine(coroutine):
        global _worker_event_loop
        return _worker_event_loop.run_until_complete(coroutine)

    @staticmethod
    def _worker_logw_next(context):
        return MultiProcPotential._run_coroutine(
            _worker_potential.logw_next(context)
        ).weights

    @staticmethod
    def _worker_prefix(context):
        return MultiProcPotential._run_coroutine(_worker_potential.prefix(context))

    @staticmethod
    def _worker_complete(context):
        return MultiProcPotential._run_coroutine(_worker_potential.complete(context))

    # @staticmethod
    # def _worker_score(context):
    #    return MultiProcPotential._run_coroutine(_worker_potential.score(context))

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def logw_next(self, context):
        result = await self._run_in_executor(self._worker_logw_next, context)
        return self.make_lazy_weights(result)

    async def prefix(self, context):
        return await self._run_in_executor(self._worker_prefix, context)

    async def complete(self, context):
        return await self._run_in_executor(self._worker_complete, context)

    async def batch_logw_next(self, contexts):
        results = await asyncio.gather(
            *(
                self._run_in_executor(self._worker_logw_next, context)
                for context in contexts
            )
        )
        return self.make_lazy_weights(np.stack(results))  # one batched [N, V+1] LazyWeights

    async def batch_complete(self, contexts):
        results = await asyncio.gather(
            *(
                self._run_in_executor(self._worker_complete, context)
                for context in contexts
            )
        )
        return np.array(results)

    async def batch_prefix(self, contexts):
        results = await asyncio.gather(
            *(
                self._run_in_executor(self._worker_prefix, context)
                for context in contexts
            )
        )
        return np.array(results)

    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_workers=})"

    def spawn(self):
        raise ValueError("MultiProcPotentials are not spawnable.")
