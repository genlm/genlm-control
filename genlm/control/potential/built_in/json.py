import json_stream
import json
import regex
import math
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
import unicodedata
from dataclasses import dataclass, field


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


BAD_WHITESPACE = regex.compile(rb"(?:\n\s*\n)")
VALID_JSON_START = regex.compile(rb'^[ \n]{0,2}\[|\{|"|(-?[0-9])|[nft]')


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
        #
        # pos=1 because we specifically allow two newlines at the start,
        # as LLMs like doing that for tokenization reasons.
        if BAD_WHITESPACE.search(context, pos=1):
            return float("-inf")
        for c in context:
            # Forbid control characters other than newline.
            if c != ord(b"\n") and c < ord(b" "):
                return float("-inf")
        return 0.0

    async def complete(self, context):
        return await self.prefix(context)


def JsonSchema(schema):
    Draft7Validator.check_schema(schema)
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
        self.__index = 0
        self.__in_preserving_block = False

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, value):
        assert value >= self.__index
        self.__index = value

    def __repr__(self):
        buffer = "".join(chr(i) for i in self.buffer)
        i = self.index
        return f"Input({repr(buffer[:i])}, ||, {repr(buffer[i:])})"

    async def advance_input(self):
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
            if not await self.advance_input():
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
                if not await self.advance_input():
                    raise Incomplete()
            else:
                self.index += match.end()
                return result

    async def get_partial_pattern(self, pattern):
        """If the remainder of the buffer read so far could match a prefix
        of pattern, or start with a complete match for the pattern return it.

        Note: This is pure lookahead and does *not* advance the input."""

        await self.__read_until(lambda: self.index < len(self.buffer))
        buffer = "".join(chr(i) for i in self.buffer[self.index :])
        return pattern.match(buffer, pos=0, partial=True)

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
        for c in expected:
            actual = await self.read(1)
            if actual != c:
                raise ParseError(
                    f"Expected: {c} but got {actual} at index {self.index - 1}"
                )

    @contextmanager
    def preserving_index(self):
        """Only advance the index if the operation in the context block does
        not error."""
        start = self.index
        try:
            yield
        except Exception:
            self.__index = start
            raise

    @contextmanager
    def resetting_index(self):
        """Always reset the index to where it started at the end of this block."""
        start = self.index
        try:
            yield
        finally:
            self.__index = start

    async def parse(self, parser: "Parser[T]") -> T:
        with self.preserving_index():
            return await parser.parse(self)

    async def skip_whitespace(self):
        if self.index == len(self.buffer):
            if not await self.advance_input():
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
    def __init__(self, *parsers):
        flattened_parsers = []
        for parser in parsers:
            if isinstance(parser, AltParser):
                flattened_parsers.extend(parser.parsers)
            else:
                flattened_parsers.append(parser)
        self.parsers = flattened_parsers

    async def parse(self, input: Input) -> Union[S, T]:
        incomplete_parsers = []
        for parser in self.parsers:
            try:
                with input.preserving_index():
                    start = input.index
                    result = await parser.parse(input)
                    assert input.index > start
                    return result
            except Incomplete:
                incomplete_parsers.append(parser)
            except ParseError:
                continue
        if incomplete_parsers:
            raise Incomplete(f"{incomplete_parsers} have not yet completed.")
        else:
            raise ParseError(f"None of {self.parsers} successfully parsed.")


class ConstParser(Parser[None]):
    def __init__(self, value: Any):
        self.value = value
        self.literal = json.dumps(value)

    async def parse(self, input: Input) -> None:
        await input.skip_whitespace()
        await input.expect(self.literal)
        return self.value


class RegexParser(Parser[str]):
    def __init__(self, pattern, group=0, options=regex.UNICODE):
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
        with input.resetting_index():
            await input.parse(FLOAT_REGEX_PARSER)
            try:
                next_char = await input.read(1)
            except Incomplete:
                pass
            else:
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
        return await input.parse(FLOAT_REGEX_PARSER)


FLOAT_PARSER = FloatParser()

INTEGER_REGEX = regex.compile(r"-?((0|([1-9][0-9]*))([eE]+?[0-9]+)?)")


class IntegerParser(Parser[int]):
    async def parse(self, input: Input) -> float:
        with input.resetting_index():
            await input.read_pattern(INTEGER_REGEX)

            while True:
                try:
                    c = await input.read(1)
                except Incomplete:
                    break
                if c == ".":
                    raise ParseError()
                elif c in "Ee":
                    # Might raise Incomplete, but if so it's
                    # correct to raise Incomplete here.
                    d = await input.read(1)
                    if d == "-":
                        raise ParseError()
                elif c not in "0123456789":
                    break
        return json.loads(await input.read_pattern(INTEGER_REGEX))


INTEGER_PARSER = IntegerParser()

STRING_REGEX = r'"([^\\"]|\\"|\\[^"])*"'

STRING_LITERAL_PARSER = RegexParser(STRING_REGEX).map(json.loads)

NULL_PARSER = RegexParser("null").drop_result()

BOOL_PARSER = RegexParser("false|true").map(json.loads)

# We restrict whitespace to be ASCII to avoid the model doing silly things
# to avoid being rejected. Note that unicode whitespace *inside a string*
# is still allowed. This parser is not used for that part, only whitespace
# between tokens.
WHITESPACE_PARSER = RegexParser(r"\s*").filter(lambda x: all(ord(c) < 256 for c in x))

STRING_PATTERN = regex.compile(STRING_REGEX)


class StringLiteralMatchingPatternParser(Parser[str]):
    def __init__(self, pattern):
        self.pattern = regex.compile(pattern, regex.UNICODE)

    async def parse(self, input: Input):
        prev = None
        while True:
            # We check whether whatever we've read so far of the
            # available data is the start of or starts with a string
            # literal.
            #
            # If it's not, the pattern is irrelevant, we've got the
            # wrong type (or bad JSON) here.

            match = await input.get_partial_pattern(STRING_PATTERN)
            if match is None:
                raise ParseError()
            literal = match.group(0)
            # We advance the input on each loop, so this literal should always
            # increase in length on each iteration.
            assert literal != prev
            prev = literal
            if not match.partial:
                # We have a complete string literal and we just need to
                # parse the whole thing.
                try:
                    decoded = json.loads(literal)
                except json.JSONDecodeError:
                    raise ParseError()
            else:
                # We have the start of a string literal. Try to read it
                # interpret it as a valid string.
                try:
                    decoded = json.loads(literal + '"')
                except json.JSONDecodeError:
                    # This might be because there's an escaped character at the
                    # end that hasn't been finished. We could try to repair that,
                    # but we'll advance by one character each loop, so it doesn't
                    # seem worth the effort.
                    if not await input.advance_input():
                        raise Incomplete()
                    continue

            # If we've seen the string halfway through a surrogate pair, drop the
            # surrogate, as it will throw off the match.
            if decoded and unicodedata.category(decoded[-1]) == "Cs":
                if not match.partial:
                    raise ParseError()
                else:
                    decoded = decoded[:-1]

            # The pattern applies to the decoded string. If we have a complete
            # string then we don't want to allow partial matches, because the
            # pattern has to match the whole thing, but if we've only got a
            # partial string then we only want a partial match.
            #
            # Note search rather than match here, because a pattern constraint
            # in JSON schema applies if the pattern matches anywhere in the string.
            match_decoded = self.pattern.search(decoded, partial=match.partial)
            if match_decoded is None:
                raise ParseError()

            if not match.partial:
                advance = await input.read(len(literal))
                assert advance == literal
                return decoded

            # If we're here, then the entire buffer read so far is a partial
            # match for the pattern. We can't make progress until more
            # data has arrived.
            if not await input.advance_input():
                raise Incomplete()


@dataclass()
class PatriciaTrieNode:
    # All strings accepted by this node start with prefix
    prefix: str
    # Is the prefix accepted
    accepting: bool

    children: "dict[int, PatriciaTrieNode]" = field(default_factory=dict)


def split_node(node, i):
    assert i < len(node.prefix)
    new_prefix = node.prefix[:i]
    c = node.prefix[i]
    new_child = PatriciaTrieNode(
        accepting=node.accepting, children=node.children, prefix=node.prefix[i + 1 :]
    )
    node.accepting = False
    node.prefix = new_prefix
    node.children = {c: new_child}


class PatriciaTrie:
    def __init__(self, values=()):
        self.root = PatriciaTrieNode(prefix="", accepting=False)
        for v in values:
            self.add_string(v)

    def add_string(self, value):
        node = self.root
        while True:
            if node.prefix == value:
                node.accepting = True
                return
            elif value.startswith(node.prefix):
                c = value[len(node.prefix)]
                tail = value[len(node.prefix) + 1 :]
                try:
                    node = node.children[c]
                    value = tail
                except KeyError:
                    node.children[c] = PatriciaTrieNode(
                        accepting=True,
                        prefix=tail,
                    )
                    return
            elif node.prefix.startswith(value):
                split_node(node, len(value))
                assert node.prefix == value
            else:
                for i in range(len(value)):
                    if node.prefix[i] != value[i]:
                        break
                else:  # pragma: no cover
                    assert False, (
                        f"{value} and {node.prefix} should have a different char at this point."
                    )
                split_node(node, i)
                assert value.startswith(node.prefix)


class FixedSetParser(Parser[str]):
    """Parser that matches a precise set of strings, some of which might
    be prefixes of each other, always returning the longest matching one."""

    def __init__(self, values):
        super().__init__()
        if not values:
            raise ValueError("values for FixedSetParser cannot be empty")
        self.trie = PatriciaTrie(values)
        self.values = values

    def __repr__(self):
        return f"FixedSetParser({self.values})"

    async def parse(self, input: Input) -> str:
        start = input.index
        match_length = -1
        node = self.trie.root

        with input.resetting_index():
            while True:
                assert node.accepting or node.children or node is self.trie.root

                try:
                    await input.expect(node.prefix)
                except (Incomplete, ParseError):
                    if match_length < 0:
                        raise
                    else:
                        break
                if node.accepting:
                    match_length = input.index - start
                    if not node.children:
                        break
                try:
                    c = await input.read(1)
                except Incomplete:
                    if match_length < 0:
                        raise
                    else:
                        break
                try:
                    node = node.children[c]
                except KeyError:
                    if match_length < 0:
                        raise ParseError(
                            f"Unexpected character {c}. Expected one of {repr(''.join(node.children))}"
                        )
                    break

        # Should have errored in the loop if not
        assert match_length >= 0
        result = await input.read(match_length)
        assert input.index == start + match_length
        return result


def possible_representations(value):
    if isinstance(value, int) and not isinstance(value, bool):
        values = (value, float(value))
    elif isinstance(value, float) and value == math.floor(value):
        values = (value, int(value))
    else:
        values = (value,)

    return {json.dumps(v, ensure_ascii=b) for v in values for b in [False, True]}


def EnumParser(values):
    reps = {s for v in values for s in possible_representations(v)}
    if len(reps) == 1:
        return ConstParser(values[0])
    return FixedSetParser(reps).map(json.loads)


class ObjectSchemaParser(Parser[Any]):
    def __init__(self, schema):
        self.schema = schema

        if not schema.get("additionalProperties", True) and not schema.get(
            "properties"
        ):
            self.empty_object = True
            return
        else:
            self.empty_object = False

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
            self.key_parser = EnumParser(list(properties.keys()))
        self.required_keys = frozenset(schema.get("required", ()))

    def __repr__(self):
        return f"ObjectSchemaParser({self.schema})"

    async def parse(self, input: Input):
        await input.skip_whitespace()

        await input.expect("{")
        if self.empty_object:
            await input.skip_whitespace()
            await input.expect("}")
            return {}

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
            start = input.index
            result[key] = await input.parse(value_parser)
            assert input.index > start, (input.index, start)
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
    if "const" in schema:
        return ConstParser(schema["const"])

    if "enum" in schema:
        return EnumParser(schema["enum"])

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
        pattern = schema.get("pattern")
        if pattern is not None:
            return StringLiteralMatchingPatternParser(pattern)
        else:
            return STRING_LITERAL_PARSER
    elif schema["type"] == "object":
        return ObjectSchemaParser(schema)
    elif schema["type"] == "array":
        return ArraySchemaParser(schema)
    else:
        return ARBITRARY_JSON
