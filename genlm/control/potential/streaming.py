from genlm.control.potential.stateful import StatefulPotential, ParticleState
from abc import ABC, abstractmethod
from enum import Enum, auto
import asyncio


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
        await self.running_in_task.responses.put(
            (self.running_in_task.last_message, Responses.INCOMPLETE)
        )
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
        self.diagnostics = {}

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
                self.diagnostics["error"] = payload
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
