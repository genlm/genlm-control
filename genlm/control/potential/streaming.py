from genlm.control.potential.base import StatefulPotential, ParticleState
from abc import ABC, abstractmethod
from typing import Any, Iterable
from queue import SimpleQueue
from enum import Enum, auto
from threading import Thread
import asyncio
from time import time


class Responses(Enum):
    INCOMPLETE = auto()
    COMPLETE = auto()
    ERROR = auto()


PING_TOKEN = object()
SHUTDOWN_TOKEN = object()


def timeout_sequence():
    start = time()
    # Initially we just yield to the the event loop
    for _ in range(3):
        yield 0.0
    # Then we do a series of short sleeps
    for _ in range(3):
        yield 0.01
    sleep = 0.015
    # Long timeout which we should never really hit,
    # this is just hang detection.
    while time() < start + 5:
        yield sleep
        sleep *= 1.1
    raise AssertionError("Timeout")


THREAD_COUNTER = 0


class RunningInThread:
    def __init__(self, function):
        self.incoming_data = SimpleQueue()
        self.responses = SimpleQueue()
        self.last_message = None
        self.running = False
        self.complete = False
        self.function = function

    def __chunks(self):
        while True:
            self.last_message, chunk = self.incoming_data.get()
            if chunk is SHUTDOWN_TOKEN:
                break
            yield chunk
            self.responses.put((self.last_message, Responses.INCOMPLETE))

    def run(self):
        global THREAD_COUNTER
        THREAD_COUNTER += 1
        # print("THREAD WENT UP", THREAD_COUNTER)
        assert not self.running
        try:
            self.running = True
            self.last_message, chunk = self.incoming_data.get()
            assert chunk == PING_TOKEN
            self.responses.put((self.last_message, Responses.INCOMPLETE))
            result = self.function(self.__chunks())
        except Exception as e:
            self.responses.put((self.last_message, Responses.ERROR, e))
        else:
            self.responses.put((self.last_message, Responses.COMPLETE, result))
        finally:
            THREAD_COUNTER -= 1
            # print("THREAD WENT DOWN", THREAD_COUNTER)
            self.running = False
            self.complete = True


class StreamingState(ParticleState):
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
            self.__background = RunningInThread(self.owner.calculate_score_from_stream)
            self.__background_thread = Thread(target=self.__background.run, daemon=True)
            self.__background_thread.start()
            await self.__send_message(PING_TOKEN)
            assert self.__background.running or self.__background.complete
        assert self.__background is not None

    async def impl_update_context(self, incremental_context):
        await self.__initialize_background()
        await self.__send_message(incremental_context)

    async def impl_finish(self):
        await self.__initialize_background()
        self.shutdown()

    @property
    def current_score(self):
        return self.__score

    async def __send_message(self, message):
        if self.__background.complete:
            return
        token = self.__new_token()
        self.__background.incoming_data.put((token, message))

        for timeout in timeout_sequence():
            if not self.__background.responses.empty():
                break
            await asyncio.sleep(timeout)
        response_token, response_type, *payload = self.__background.responses.get()
        assert token == response_token
        match response_type:
            case Responses.INCOMPLETE:
                pass
            case Responses.COMPLETE:
                self.__score = payload[0] or 0.0
            case Responses.ERROR:
                self.__score = payload[0] = -float("inf")

    def shutdown(self):
        if self.__background_thread is not None and self.__background_thread.is_alive():
            token = self.__new_token()
            self.__background.incoming_data.put((token, SHUTDOWN_TOKEN))
            # Should in fact terminate very fast. Long timeout here for debugging purposes
            # only - we want a log if it hangs.
            self.__background_thread.join(timeout=1.0)

    def __del__(self):
        self.shutdown()


class StreamingPotential(StatefulPotential, ABC):
    def __init__(self, vocabulary, token_type=None, eos=None):
        super().__init__(
            vocabulary=vocabulary,
            token_type=token_type,
            eos=eos,
            state_class=StreamingState,
        )

    @abstractmethod
    def calculate_score_from_stream(self, stream: Iterable[Any]) -> float: ...
