import json_stream
import json
import regex
from typing import Generic, TypeVar, Union, Any, Callable
from jsonschema import Draft7Validator
from jsonschema import _types
from typing import Iterable, AsyncIterator
from genlm.control.potential import Potential
from contextlib import contextmanager
from genlm.control.potential.streaming import (
    StreamingPotential,
    AsyncStreamingPotential,
    AsyncSource,
)
from array import array


def is_sequence(checker, instance):
    from collections.abc import Sequence, Mapping

    return isinstance(instance, Sequence) and not isinstance(
        instance, (str, bytes, bytearray, Mapping)
    )


def is_object(checker, instance):
    from json_stream.base import StreamingJSONObject
    from collections.abc import Mapping

    return isinstance(instance, (Mapping, StreamingJSONObject))


# We're using a streaming JSON library that doesn't return proper lists
# and dicts. In theory we could use jsonschema's custom typechecker logic
# here. In practice, this works until it encounters an explicitly specified
# schema type, at which point it creates a new validator that ignores the
# type checker. There is probably a sensible official way to fix this (I hope)
# but I couldn't figure it out and this was expedient and probably won't
# cause too many problems (I hope) - DRMacIver.
_types.is_array.__code__ = is_sequence.__code__
_types.is_object.__code__ = is_object.__code__


# Ideally we would be using Draft202012Validator for compatibility with
# jsonschemabench, but something about the way it's written makes it worse
# at lazy validation, so we're using an older draft for now.
LazyCompatibleValidator = Draft7Validator


UTF8_START_BYTE_MASKS = [
    (0b00000000, 0b10000000),
    (0b11000000, 0b11100000),
    (0b11100000, 0b11110000),
    (0b11110000, 0b11111000),
]


def is_utf8_start_byte(n: int) -> bool:
    """Checks if this is a byte that can appear at the
    start of a UTF-8 character."""
    assert 0 <= n < 256
    for prefix, mask in UTF8_START_BYTE_MASKS:
        if n & mask == prefix:
            return True
    return False


BAD_WHITESPACE = regex.compile(rb"(?:\n\s*\n)", regex.MULTILINE)


def chunk_to_complete_utf8(byte_blocks):
    for s in chunk_bytes_to_strings(byte_blocks):
        yield s.encode("utf-8")


def chunk_bytes_to_strings(byte_blocks):
    buffer = bytearray()
    for block in byte_blocks:
        buffer.extend(block)
        try:
            yield buffer.decode("utf-8")
            buffer.clear()
            continue
        except UnicodeDecodeError as e:
            if e.reason == "unexpected end of data":
                good_prefix = buffer[: e.start]
                if good_prefix:
                    yield good_prefix.decode("utf-8")
                    del buffer[: e.start]
            else:
                raise
        if buffer:
            assert is_utf8_start_byte(buffer[0])
    buffer.decode("utf-8")
    assert not buffer


class StreamingJsonSchema(StreamingPotential):
    def __init__(self, schema, **kwargs):
        super().__init__(
            vocabulary=list(range(256)),
            **kwargs,
        )
        self.schema = schema
        self.validator = LazyCompatibleValidator(
            self.schema, format_checker=Draft7Validator.FORMAT_CHECKER
        )
        self.parser = json_schema_parser(schema)

    def calculate_score_from_stream(self, stream: Iterable[Any]) -> float:
        buffer = bytearray()

        def buffer_stream():
            for s in stream:
                buffer.extend(s)
                yield bytes(s)

        buffered = buffer_stream()
        rechunked = chunk_to_complete_utf8(buffered)

        x = json_stream.load(rechunked, persistent=True)
        self.validator.validate(x)
        if hasattr(x, "read_all"):
            x.read_all()

        json.loads(buffer)
        for s in buffered:
            if s.strip():
                raise ValueError(f"Data after JSON: {s.decode('utf-8')}")
        return 0.0


VALID_JSON_START = regex.compile(rb'^\[|\{|"|(-?[0-9])|[nft]', regex.MULTILINE)


class ValidateJSON(Potential):
    """This is a dumping ground for any extra JSON validation we want to do
    to work around LLM weirdness.
    """

    def __init__(self):
        super().__init__(
            vocabulary=list(range(256)),
        )

    async def prefix(self, context):
        context = bytes(context)
        # Sometimes a model gets itself off to a bad start immediately.
        # We want to catch this early. Note that we forbid whitespace
        # at the start of the context. It seems to almost always be
        # a bad sign.
        if not VALID_JSON_START.match(context, partial=True):
            return float("-inf")

        # Sometimes a model can get itself into a position where it can't
        # generate any valid tokens, but it can keep generating whitespace
        # indefinitely.
        if BAD_WHITESPACE.search(context):
            return float("-inf")
        for c in context:
            # Forbid control characters other than newline.
            if c != ord(b"\n") and c < ord(b" "):
                return float("-inf")
        return 0.0

    async def complete(self, context):
        return await self.prefix(context)


def JsonSchema(schema):
    return (
        ValidateJSON()
        * StreamingJsonSchema(schema)
        * ParserPotential(json_schema_parser(schema))
    )


class StringSource(AsyncSource):
    def __init__(self, byte_source):
        self.byte_source = byte_source
        self.buffer = bytearray()

    async def more(self):
        while True:
            # Might raise but that's fine, we're done then.
            block = await self.byte_source.more()
            self.buffer.extend(block)
            try:
                result = self.buffer.decode("utf-8")
                self.buffer.clear()
                return result
            except UnicodeDecodeError:
                for i in range(1, min(5, len(self.buffer) + 1)):
                    if is_utf8_start_byte(self.buffer[-i]):
                        block = self.buffer[:-i]
                        if block:
                            del self.buffer[:-i]
                            return block.decode("utf-8")
                        break
                else:
                    raise


class ParserPotential(AsyncStreamingPotential):
    def __init__(self, parser):
        super().__init__(
            vocabulary=list(range(256)),
        )
        self.parser = parser

    async def calculate_score_from_stream(self, stream: AsyncSource) -> float:
        rechunked = StringSource(stream)
        input = Input(rechunked)
        await input.parse(self.parser)
        return 0.0


S = TypeVar("S")
T = TypeVar("T")


class ParseError(Exception):
    pass


class Incomplete(Exception):
    pass


class Input:
    """Convenience wrapper to provide a stateful stream-like interface
    that makes it easier to write parsers."""

    def __init__(self, incoming: AsyncIterator[str]):
        self.__incoming = incoming
        self.__finished = False
        # There's no textarray equivalent, so we store the growable
        # string as an array of integer codepoints.
        self.buffer = array("I")
        self.index = 0

    async def __read_more(self):
        if self.__finished:
            return False
        try:
            next_block = await self.__incoming.more()
            self.buffer.extend([ord(c) for c in next_block])
            return True
        except StopAsyncIteration:
            self.__finished = True
            return False

    async def __read_until(self, condition):
        while True:
            if condition():
                break
            if not await self.__read_more():
                raise Incomplete()

    async def read_pattern(self, pattern, group=0):
        await self.__read_until(lambda: self.index < len(self.buffer))
        while True:
            # Having to convert the whole thing to a string here is really
            # annoying, but in practice the inefficiency is dwarfed by the LLM
            # so hopefully we don't have to worry about it.
            buffer = "".join(chr(i) for i in self.buffer[self.index :])
            match = pattern.match(buffer, pos=0, partial=True)
            if match is None or (result := match.group(group)) is None:
                raise ParseError()
            elif match.partial:
                if not await self.__read_more():
                    raise Incomplete()
            else:
                self.index += match.end()
                return result

    async def current_char(self):
        await self.__read_until(lambda: self.index < len(self.buffer))
        return chr(self.buffer[self.index])

    async def read(self, n) -> str:
        await self.__read_until(lambda: self.index + n <= len(self.buffer))
        result = self.buffer[self.index : self.index + n]
        assert len(result) == n
        self.index += n
        return "".join(map(chr, result))

    async def expect(self, expected: str):
        actual = await self.read(len(expected))
        if actual != expected:
            raise ParseError(
                f"Expected: {expected} but got {actual} at index {self.index}"
            )

    @contextmanager
    def preserving_index(self):
        """Only advance the index if the operation in the context block does
        not error."""
        start = self.index
        try:
            yield
        except Exception:
            self.index = start
            raise

    async def parse(self, parser: "Parser[T]") -> T:
        with self.preserving_index():
            return await parser.parse(self)

    async def skip_whitespace(self):
        if self.index == len(self.buffer):
            if not await self.__read_more():
                return
        # TODO: Given inefficiencies with regex, maybe worth a more direct
        # implementation here?
        await self.parse(WHITESPACE_PARSER)


class TrivialSource(AsyncSource):
    def __init__(self, value):
        self.value = value
        self.__called = False

    async def more(self):
        if not self.__called:
            self.__called = True
            return self.value
        else:
            raise StopAsyncIteration()


class Parser(Generic[T]):
    """Very basic parser combinators for mostly unambiguous grammars."""

    async def parse(self, input: Input) -> T: ...

    async def parse_string(self, s: str) -> T:
        return await Input(TrivialSource(s)).parse(self)

    def __floordiv__(self, other: Generic[S]) -> "Parser[Union[T, S]]":
        return AltParser(self, other)

    def drop_result(self) -> "Parser[None]":
        return self.map(lambda x: None)

    def map(self, apply: Callable[[T], S]) -> "Parser[S]":
        return MapParser(self, apply)

    def filter(self, predicate: Callable[[T], bool]) -> "Parser[T]":
        return FilterParser(self, predicate)


class MapParser(Parser[T]):
    def __init__(self, base: Parser[S], apply: Callable[[S], T]):
        self.base = base
        self.apply = apply

    async def parse(self, input: Input) -> T:
        return self.apply(await input.parse(self.base))

    def __repr__(self):
        return f"{self.base}.map({self.apply})"


class FilterParser(Parser[T]):
    def __init__(self, base: Parser[S], predicate: Callable[[S], T]):
        self.base = base
        self.predicate = predicate

    async def parse(self, input: Input) -> T:
        result = await input.parse(self.base)
        if not self.predicate(result):
            raise ParseError(f"{result} did not satisfy {self.predicate}")
        return result

    def __repr__(self):
        return f"{self.base}.filter({self.predicate})"


R = TypeVar("R")


class AltParser(Parser[Union[S, T]]):
    def __init__(self, left: Parser[S], right: Parser[T]):
        self.left = left
        self.right = right

    async def parse(self, input: Input) -> Union[S, T]:
        try:
            with input.preserving_index():
                return await self.left.parse(input)
        except ParseError:
            return await self.right.parse(input)

    def __floordiv__(self, other: Parser[R]) -> "Parser[Union[T, S, R]]":
        if isinstance(other, AltParser):
            return self.left // (self.right // other)
        return AltParser(self, other)


class RegexParser(Parser[str]):
    def __init__(self, pattern, group=0, options=regex.MULTILINE | regex.UNICODE):
        self.pattern = regex.compile(pattern, options)
        self.group = group

    async def parse(self, input: Input) -> str:
        return await input.read_pattern(self.pattern, group=self.group)

    def __repr__(self):
        return f"RegexParser({self.pattern})"


FLOAT_REGEX_PARSER: Parser[float] = RegexParser(
    r"-?((0|([1-9][0-9]*))((\.[0-9]+)?)([eE][+-]?[0-9]+)?)"
).map(json.loads)


class FloatParser(Parser[float]):
    async def parse(self, input: Input) -> float:
        start = input.index
        preliminary_result = await input.parse(FLOAT_REGEX_PARSER)
        try:
            next_char = await input.read(1)
        except Incomplete:
            return preliminary_result

        if next_char == ".":
            await input.read(1)
        elif next_char in "eE":
            next_next_char = await input.read(1)
            if next_next_char in "-+":
                await input.read(1)

        try:
            while (await input.read(1)) in "0123456789":
                continue
        except Incomplete:
            pass

        input.index = start
        return await input.parse(FLOAT_REGEX_PARSER)


FLOAT_PARSER = FloatParser()

INTEGER_PARSER: Parser[float] = RegexParser(
    r"-?((0|([1-9][0-9]*))([eE]+?[0-9]+)?)"
).map(json.loads)


STRING_LITERAL_PARSER = RegexParser(r'"([^\\"]|\\"|\\[^"])*"').map(json.loads)

NULL_PARSER = RegexParser("null").drop_result()

BOOL_PARSER = RegexParser("false|true").map(json.loads)

# We restrict whitespace to be ASCII to avoid the model doing silly things
# to avoid being rejected. Note that unicode whitespace *inside a string*
# is still allowed. This parser is not used for that part, only whitespace
# between tokens.
WHITESPACE_PARSER = RegexParser(r"\s*").filter(lambda x: all(ord(c) < 256 for c in x))


class ObjectSchemaParser(Parser[Any]):
    def __init__(self, schema):
        self.schema = schema

        properties = self.schema.get("properties", {})
        self.child_parsers = {k: json_schema_parser(v) for k, v in properties.items()}

        # JSON schemas accept additional properties by default, but when
        # generating that's almost always not what we want. The approach
        # we take is to default to false, except in the case where no properties
        # are specified, which we take to mean that an arbitrary object is expected
        # here, so we default it to false. Where it is specified we always use
        # the explicit value.
        if "additionalProperties" in schema:
            allow_additional_properties = schema["additionalProperties"]
        else:
            allow_additional_properties = "properties" not in schema

        if allow_additional_properties:
            self.key_parser = STRING_LITERAL_PARSER
        else:
            # TODO: Something is going wrong here with regex escape codes
            self.key_parser = RegexParser(
                "|".join(
                    f"({regex.escape(json.dumps(k, ensure_ascii=b))})"
                    for k in properties
                    for b in [False, True]
                )
            ).map(json.loads)
        self.required_keys = frozenset(schema.get("required", ()))

    def __repr__(self):
        return f"ObjectSchemaParser({self.schema})"

    async def parse(self, input: Input):
        await input.skip_whitespace()

        await input.expect("{")

        result = {}

        keys_seen = set()

        first = True

        while True:
            await input.skip_whitespace()
            if await input.current_char() == "}":
                await input.read(1)
                break
            if not first:
                await input.expect(",")
                await input.skip_whitespace()
            first = False
            key = await input.parse(self.key_parser)
            assert isinstance(key, str)
            if key in keys_seen:
                raise ParseError(f"Duplicated key {repr(key)}")
            keys_seen.add(key)
            await input.skip_whitespace()
            await input.expect(":")
            await input.skip_whitespace()
            value_parser = self.child_parsers.get(key, ARBITRARY_JSON)
            result[key] = await input.parse(value_parser)
        return result


class ArraySchemaParser(Parser[Any]):
    def __init__(self, schema):
        self.schema = schema
        if "items" in schema:
            self.items_parser = json_schema_parser(schema["items"])
        else:
            self.items_parser = None

    def __repr__(self):
        return f"ArraySchemaParser({self.schema})"

    async def parse(self, input: Input):
        await input.skip_whitespace()

        await input.expect("[")

        if self.items_parser is None:
            items_parser = ARBITRARY_JSON
        else:
            items_parser = self.items_parser

        result = []

        first = True

        while True:
            await input.skip_whitespace()
            if await input.current_char() == "]":
                await input.read(1)
                break
            if not first:
                await input.expect(",")
                await input.skip_whitespace()
            first = False
            result.append(await input.parse(items_parser))
        return result


ARBITRARY_JSON = (
    NULL_PARSER
    // BOOL_PARSER
    // FLOAT_PARSER
    // STRING_LITERAL_PARSER
    // ArraySchemaParser({})
    // ObjectSchemaParser({"additionalProperties": True})
)


def json_schema_parser(schema):
    if "anyOf" in schema:
        *rest, base = schema["anyOf"]
        result = json_schema_parser(base)
        for schema in reversed(rest):
            result = json_schema_parser(schema) // result
        return result

    if "type" not in schema:
        return ARBITRARY_JSON
    elif schema["type"] == "number":
        return FLOAT_PARSER
    elif schema["type"] == "integer":
        return INTEGER_PARSER
    elif schema["type"] == "null":
        return NULL_PARSER
    elif schema["type"] == "boolean":
        return BOOL_PARSER
    elif schema["type"] == "string":
        return STRING_LITERAL_PARSER
    elif schema["type"] == "object":
        return ObjectSchemaParser(schema)
    elif schema["type"] == "array":
        return ArraySchemaParser(schema)
    else:
        return ARBITRARY_JSON
