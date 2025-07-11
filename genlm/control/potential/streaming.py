from genlm.control.potential.stateful import StatefulPotential, ParticleState
from abc import ABC, abstractmethod
from typing import Any, Iterable
from queue import SimpleQueue
from enum import Enum, auto
from threading import Thread
import asyncio
import random
import time


class Responses(Enum):
    INCOMPLETE = auto()
    COMPLETE = auto()
    ERROR = auto()


class UniqueIdentifier:
    def __init__(self, name):
        self.__name = name

    def __repr__(self):
        return self.__name


PING_TOKEN = UniqueIdentifier("PING_TOKEN")
SHUTDOWN_TOKEN = UniqueIdentifier("SHUTDOWN_TOKEN")


class Timeout(Exception):
    pass


def timeout_sequence():
    start = time.time()
    # Initially we just yield to the the event loop
    for _ in range(3):
        yield 0.0
    # Then we do a series of short sleeps
    for _ in range(3):
        yield random.random() * 0.01
    sleep = 0.015
    while time.time() < start + 30:
        yield random.random() * sleep
        sleep = min(sleep * 1.1, 1)
    raise Timeout(f"Timed out after {time.time() - start:.2f}s")


class RunningInThread:
    def __init__(self, function):
        self.incoming_data = SimpleQueue()
        self.responses = SimpleQueue()
        self.last_message = None
        self.running = False
        self.complete = False
        self.error = None
        self.function = function

    def __chunks(self):
        while True:
            self.last_message, chunk = self.incoming_data.get()
            if chunk is SHUTDOWN_TOKEN:
                break
            yield chunk
            self.responses.put((self.last_message, Responses.INCOMPLETE))

    def run(self):
        assert not self.running
        try:
            self.running = True
            self.last_message, chunk = self.incoming_data.get()
            assert chunk == PING_TOKEN
            self.responses.put((self.last_message, Responses.INCOMPLETE))
            result = self.function(self.__chunks())
        except Exception as e:
            self.error = e
            self.responses.put((self.last_message, Responses.ERROR, e))
        else:
            self.responses.put((self.last_message, Responses.COMPLETE, result))
        finally:
            self.running = False
            self.complete = True


class StreamingState(ParticleState):
    def __init__(self, owner):
        super().__init__(owner)
        self.__token = 0
        self.__background = None
        self.__background_thread = None
        self.__score = 0.0
        self.__shut_down = False
        self.diagnostics = {}

    def __new_token(self):
        self.__token += 1
        return self.__token

    async def __initialize_background(self):
        if self.__background is None:
            self.__background = RunningInThread(self.owner.calculate_score_from_stream)

            # Sometimes, especially in consistency check tests, we have too many threads
            # running and need to wait before we're able to start a new thread.
            for t in timeout_sequence():
                try:
                    self.__background_thread = Thread(
                        target=self.__background.run, daemon=True
                    )
                    self.__background_thread.start()
                    break
                except RuntimeError:
                    await asyncio.sleep(t)
            await self.__send_message(PING_TOKEN)
            assert self.__background.running or self.__background.complete
        assert self.__background is not None

    async def impl_update_context(self, incremental_context):
        await self.__initialize_background()
        finish = False
        if incremental_context and incremental_context[-1] == self.owner.eos:
            finish = True
            incremental_context = incremental_context[:-1]
        bytes(incremental_context)
        await self.__send_message(incremental_context)
        if finish:
            await self.finish()

    async def impl_finish(self):
        await self.__initialize_background()
        self.shutdown()

    @property
    def current_score(self):
        return self.__score

    async def __send_message(self, message):
        if self.__background.complete:
            self.__score = -float("inf")
            self.diagnostics["error"] = self.__background.error
            import traceback

            traceback.print_exception(self.__background.error)
            return
        token = self.__new_token()
        self.__background.incoming_data.put((token, message))

        for timeout in timeout_sequence():
            if not self.__background.responses.empty():
                break
            await asyncio.sleep(timeout)
        self.__receive_response(token)

    def __receive_response(self, token):
        while True:
            # In some error cases we can fail to acknowledge a response. We just silently
            # drop these.
            response_token, response_type, *payload = self.__background.responses.get()
            assert response_token <= token
            if token == response_token:
                break
        assert token == response_token, (token, response_token, response_type)
        match response_type:
            case Responses.INCOMPLETE:
                pass
            case Responses.COMPLETE:
                self.__score = payload[0] or 0.0
            case Responses.ERROR:
                self.__score = -float("inf")
                self.diagnostics["error"] = payload[0]

    def shutdown(self):
        if self.__shut_down:
            return
        self.__shut_down = True
        if self.__background_thread is not None and self.__background_thread.is_alive():
            token = self.__new_token()
            self.__background.incoming_data.put((token, SHUTDOWN_TOKEN))
            # Should in fact terminate very fast. Long timeout here for debugging purposes
            # only - we want a log if it hangs.
            self.__background_thread.join(timeout=1.0)
            self.__receive_response(token)

    def __del__(self):
        self.shutdown()


class StreamingPotential(StatefulPotential, ABC):
    def __init__(self, vocabulary, token_type=None, eos=None, **kwargs):
        super().__init__(
            vocabulary=vocabulary,
            token_type=token_type,
            eos=eos,
            state_class=StreamingState,
            **kwargs,
        )

    @abstractmethod
    def calculate_score_from_stream(self, stream: Iterable[Any]) -> float: ...


# This should be an async generator really but async generators
# are fundamentally broken. See https://peps.python.org/pep-0789/
# I kept running into problems with this during implementation, so
# ended up finding it easier to just hand roll implementations of
# this rather than trying to use yield based generators.
class AsyncSource(ABC):
    @abstractmethod
    async def more(self): ...


class Chunks(AsyncSource):
    def __init__(self, running_in_task):
        self.running_in_task = running_in_task
        self.__first = True

    async def more(self):
        if not self.__first:
            await self.running_in_task.responses.put(
                (self.running_in_task.last_message, Responses.INCOMPLETE)
            )
        self.__first = False
        (
            self.running_in_task.last_message,
            chunk,
        ) = await self.running_in_task.incoming_data.get()
        if chunk is SHUTDOWN_TOKEN:
            raise StopAsyncIteration()
        return chunk


class RunningInTask:
    def __init__(self, function):
        self.incoming_data = asyncio.Queue()
        self.responses = asyncio.Queue()
        self.last_message = None
        self.running = False
        self.complete = False
        self.function = function

    async def run(self):
        assert not self.running
        try:
            self.running = True
            self.last_message, chunk = await self.incoming_data.get()
            assert chunk == PING_TOKEN
            await self.responses.put((self.last_message, Responses.INCOMPLETE))
            chunks = Chunks(self)
            result = await self.function(chunks)
        except Exception as e:
            await self.responses.put((self.last_message, Responses.ERROR, e))
        else:
            await self.responses.put((self.last_message, Responses.COMPLETE, result))
        finally:
            self.running = False
            self.complete = True


# This is sortof insane, but asyncio will get *very* upset with you if your task
# objects are garbage collected before they're complete. This keeps a set of them
# around until they're completed.
KEEP_ALIVE_SET = set()


class AsyncStreamingState(ParticleState):
    def __init__(self, owner):
        super().__init__(owner)
        self.__token = 0
        self.__background = None
        self.__score = 0.0

    def __new_token(self):
        self.__token += 1
        return self.__token

    async def __initialize_background(self):
        if self.__background is None:
            self.__background = RunningInTask(self.owner.calculate_score_from_stream)
            self.__background_task = asyncio.create_task(self.__background.run())
            await self.__send_message(PING_TOKEN)
            KEEP_ALIVE_SET.add(self.__background_task)
            self.__background_task.add_done_callback(KEEP_ALIVE_SET.discard)
        assert self.__background is not None

    async def impl_update_context(self, incremental_context):
        await self.__initialize_background()
        finish = False
        if incremental_context and incremental_context[-1] == self.owner.eos:
            finish = True
            incremental_context = incremental_context[:-1]
        bytes(incremental_context)
        await self.__send_message(incremental_context)
        if finish:
            await self.finish()

    async def impl_finish(self):
        await self.__initialize_background()
        await self.shutdown()

    @property
    def current_score(self):
        return self.__score

    async def __send_message(self, message):
        if self.__background.complete:
            return
        token = (self.__new_token(), message)
        await self.__background.incoming_data.put((token, message))

        (
            response_token,
            response_type,
            *payload,
        ) = await self.__background.responses.get()

        assert token == response_token
        match response_type:
            case Responses.INCOMPLETE:
                pass
            case Responses.COMPLETE:
                self.__score = payload[0] or 0.0
            case Responses.ERROR:
                self.__score = -float("inf")

    async def shutdown(self):
        if self.__background is not None:
            await self.__send_message(SHUTDOWN_TOKEN)


class AsyncStreamingPotential(StatefulPotential, ABC):
    def __init__(self, vocabulary, token_type=None, eos=None):
        super().__init__(
            vocabulary=vocabulary,
            token_type=token_type,
            eos=eos,
            state_class=AsyncStreamingState,
        )

    @abstractmethod
    async def calculate_score_from_stream(self, stream: AsyncSource) -> float: ...
