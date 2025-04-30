import json_stream
import json
import regex
from typing import Generic, TypeVar, Union, Any, Callable
from jsonschema import Draft7Validator, ValidationError
from jsonschema import _types


from genlm.control.potential.base import Potential


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


class OutOfBytes(Exception):
    pass


class JustOneBlockIterable:
    """Provides a single value (intended to be bytes from a context)
    and then signals if the reader tried to read past it. This allows
    us to distinguish invalid JSON from incomplete JSON by seeing if
    the reader tried to read more than it had or failed early."""

    def __init__(self, block):
        self.block = block
        self.read_past_first_block = False

    def __iter__(self):
        yield self.block
        self.read_past_first_block = True


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


BAD_WHITESPACE = regex.compile(rb"(?:\n\s+\n)|(?:\n\n\n)", regex.MULTILINE)


def remove_incomplete_trailing_utf8(context: bytes) -> tuple[bool, bytes, str]:
    context = bytes(context)

    # JSON documents have to be valid UTF-8, but we might be
    # in the middle of generating a UTF-8 character. If so, we
    # only consider the prefix that is valid UTF-8, but need
    # to signal at the end that this is a valid prefix and not
    # a valid complete document.
    incomplete_utf8_at_end = False
    try:
        try:
            context_as_string = context.decode("utf-8")
        except UnicodeDecodeError:
            for i in range(1, min(5, len(context))):
                if is_utf8_start_byte(context[-i]):
                    context = context[:-i]
                    context_as_string = context.decode("utf-8")
                    incomplete_utf8_at_end = True
                    break
            else:
                raise
    except UnicodeDecodeError:
        raise ValueError("Invalid UTF-8")

    return (incomplete_utf8_at_end, context, context_as_string)


class JsonSchema(Potential):
    def __init__(self, schema):
        super().__init__(
            list(range(256)),
        )
        self.schema = schema
        self.validator = LazyCompatibleValidator(
            self.schema, format_checker=Draft7Validator.FORMAT_CHECKER
        )
        self.parser = json_schema_parser(schema)

    def __check_context(self, context):
        context = bytes(context)

        incomplete_utf8_at_end, context, context_as_string = (
            remove_incomplete_trailing_utf8(context)
        )

        # Sometimes a model can get itself itno a position where it can't
        # generate any valid tokens, but it can keep generating whitespace
        # indefinitely.
        if BAD_WHITESPACE.search(context):
            raise ValueError("Improper JSON formatting.")

        # Feeding just whitespace to json-stream causes it to raise
        # StopIteration, and this is always a valid start to a JSON
        # document of any schema, and never a valid JSON value.
        if not context.strip():
            raise OutOfBytes()

        iterable = JustOneBlockIterable(context)
        try:
            x = json_stream.load(iterable, persistent=True)
            self.validator.validate(x)
            if hasattr(x, "read_all"):
                x.read_all()
        except ValueError:
            if iterable.read_past_first_block:
                raise OutOfBytes()
            else:
                raise
        if incomplete_utf8_at_end:
            raise OutOfBytes()

        # json-stream will just read a JSON object off the start of
        # the stream and then stop, so we reparse the whole string
        # with the normal JSON parser to validate it at the end, or
        # we will allow JSON values to be followed by arbitrary nonsense.
        # This should only fire when we've successfully created a valid
        # JSON value and want to terminate the sequence.
        try:
            json.loads(context)
        except json.JSONDecodeError as e:
            raise ValueError(*e.args)

    async def complete(self, context) -> float:
        # TODO:
        # 1. Create some sort of caching for the validator, so
        #    we can reuse ones from previous calls.
        # 2. Use a Lark JSON grammar as a prefilter to rule out any
        #    bytes that can't be included next in valid JSON.

        try:
            self.__check_context(context)
        except (ValueError, ValidationError, OutOfBytes):
            return -float("inf")

        return 0.0

    async def prefix(self, context) -> float:
        # TODO:
        # 1. Create some sort of caching for the validator, so
        #    we can reuse ones from previous calls.
        # 2. Use a Lark JSON grammar as a prefilter to rule out any
        #    bytes that can't be included next in valid JSON.
        try:
            self.__check_context(context)
        except (ValueError, ValidationError):
            return -float("inf")
        except OutOfBytes:
            pass

        # There are a number of cases where the approach we use in check_context
        # will fail to catch an error early enough because it only operates on
        # completed values. The biggest problem here is that if there is no way to
        # close a string that conforms to the schema, it will force the LLM to
        # just keep extending the string. In these cases what we do is use a very
        # rough approximate parser that accepts a superset of valid JSON strings
        # for this schema but is able to reject some cases earlier than the full
        # validation.
        #
        # We only need to do this in prefix, because the full document is guaranteed
        # to be checkable exactly by the JSONSchema validator.
        context_as_string = remove_incomplete_trailing_utf8(context)[-1]
        try:
            self.parser.parse(context_as_string, 0)
        except ParseError:
            return -float("inf")
        except Incomplete:
            pass

        return 0.0


S = TypeVar("S")
T = TypeVar("T")


class ParseError(Exception):
    pass


class Incomplete(Exception):
    pass


class Parser(Generic[T]):
    """Very basic parser combinators for mostly unambiguous grammars."""

    def parse(self, buffer: str, start: int) -> tuple[int, T]: ...

    def __floordiv__(self, other: Generic[S]) -> "Parser[Union[T, S]]":
        return AltParser(self, other)

    def __add__(self, other: Generic[S]) -> "Parser[tuple[T, S]]":
        return CatParser(self, other)

    def drop_result(self) -> "Parser[None]":
        return self.map(lambda x: None)

    def map(self, apply: Callable[[T], S]) -> "Parser[S]":
        return MapParser(self, apply)


class Input:
    """Convenience wrapper to provide a stateful stream-like interface
    that makes it easier to write parsers."""

    def __init__(self, buffer, index):
        self.buffer = buffer
        self.index = index

    def current_char(self):
        if self.index >= len(self.buffer):
            raise Incomplete()
        else:
            return self.buffer[self.index]

    def read(self, n) -> str:
        result = self.buffer[self.index : self.index + n]
        if len(result) < n:
            raise Incomplete()
        else:
            self.index += n
            return result

    def expect(self, expected: str):
        actual = self.read(len(expected))
        if actual != expected:
            raise ParseError(
                f"Expected: {expected} but got {actual} at index {self.index}"
            )

    def parse(self, parser: Parser[T]) -> T:
        try:
            self.index, result = parser.parse(self.buffer, self.index)
            return result
        except Incomplete:
            self.index = len(self.buffer)
            raise

    def skip_whitespace(self):
        self.parse(WHITESPACE_PARSER)


class MapParser(Parser[T]):
    def __init__(self, base: Parser[S], apply: Callable[[S], T]):
        self.base = base
        self.apply = apply

    def parse(self, buffer: str, start: int) -> tuple[int, T]:
        end, result = self.base.parse(buffer, start)
        return (end, self.apply(result))

    def __repr__(self):
        return f"{self.base}.map({self.apply})"


class AltParser(Parser[Union[S, T]]):
    def __init__(self, left: Parser[S], right: Parser[T]):
        self.left = left
        self.right = right

    def parse(self, buffer: str, start: int) -> tuple[int, Union[S, T]]:
        try:
            return self.left.parse(buffer, start)
        # NB it's correct that we don't catch incomplete here. If
        # the first parser needs more characters to tell whether it matches
        # then we can't yet try the second.
        except ParseError:
            return self.right.parse(buffer, start)


class CatParser(Parser[tuple[S, T]]):
    def __init__(self, left: Parser[S], right: Parser[T]):
        self.left = left
        self.right = right

    def parse(self, buffer: str, start: int) -> Parser[Union[S, T]]:
        next_start, result_left = self.left.parse(buffer, start)
        end, result_right = self.right.parse(buffer, next_start)
        return (end, (result_left, result_right))


class RegexParser(Parser[str]):
    def __init__(self, pattern, group=0, options=regex.MULTILINE | regex.UNICODE):
        self.pattern = regex.compile(pattern, options)
        self.group = group

    def parse(self, buffer: str, start: int) -> tuple[int, str]:
        match = self.pattern.match(buffer, pos=start, partial=True)
        if match is None or (result := match.group(self.group)) is None:
            raise ParseError()
        elif match.partial:
            raise Incomplete()
        else:
            return (match.end(), result)

    def __repr__(self):
        return f"RegexParser({self.pattern})"


FLOAT_REGEX_PARSER: Parser[float] = RegexParser(
    r"-?((0|([1-9][0-9]*))((\.[0-9]+)?)([eE][+-]?[0-9]+)?)"
).map(json.loads)


class FloatParser(Parser[float]):
    def parse(self, buffer: str, start: int) -> tuple[int, float]:
        i, result = FLOAT_REGEX_PARSER.parse(buffer, start)
        # We need to do a tiny bit of lookahead here so that we
        # can guarantee that if we're in the middle of a float
        # we always either return a valid value or raise Incomplete.
        # Otherwise we end up in situations like "[0." raising
        # ParseError because the float completes and then the
        # list parser looks for a comma and gets a dot.
        if i < len(buffer) and buffer[i] in ".eE":
            raise Incomplete()
        return (i, result)


FLOAT_PARSER = FloatParser()

INTEGER_PARSER: Parser[float] = RegexParser(
    r"-?((0|([1-9][0-9]*))([eE]+?[0-9]+)?)"
).map(json.loads)


STRING_LITERAL_PARSER = RegexParser(r'"([^\\"]|\\"|\\[^"])*"').map(json.loads)

NULL_PARSER = RegexParser("null").drop_result()

BOOL_PARSER = RegexParser("false|true").map(json.loads)

WHITESPACE_PARSER = RegexParser(r"\s*")


class ObjectSchemaParser(Parser[Any]):
    def __init__(self, schema):
        self.schema = schema

        properties = self.schema.get("properties", {})
        self.child_parsers = {k: json_schema_parser(v) for k, v in properties.items()}
        if schema.get("additionalProperties", False):
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

    def parse(self, buffer: str, start: int):
        input = Input(buffer, start)
        input.skip_whitespace()

        input.expect("{")

        result = {}

        keys_seen = set()

        first = True

        while True:
            input.skip_whitespace()
            if input.current_char() == "}":
                input.read(1)
                break
            if not first:
                input.expect(",")
                input.skip_whitespace()
            first = False
            key = input.parse(self.key_parser)
            assert isinstance(key, str)
            if key in keys_seen:
                raise ParseError(f"Duplicated key {repr(key)}")
            keys_seen.add(key)
            input.skip_whitespace()
            input.expect(":")
            input.skip_whitespace()
            value_parser = self.child_parsers.get(key, ARBITRARY_JSON)
            result[key] = input.parse(value_parser)
        return (input.index, result)


class ArraySchemaParser(Parser[Any]):
    def __init__(self, schema):
        self.schema = schema
        if "items" in schema:
            self.items_parser = json_schema_parser(schema["items"])
        else:
            self.items_parser = None

    def __repr__(self):
        return f"ArraySchemaParser({self.schema})"

    def parse(self, buffer: str, start: int):
        input = Input(buffer, start)
        input.skip_whitespace()

        input.expect("[")

        if self.items_parser is None:
            items_parser = ARBITRARY_JSON
        else:
            items_parser = self.items_parser

        result = []

        first = True

        while True:
            input.skip_whitespace()
            if input.current_char() == "]":
                input.read(1)
                break
            if not first:
                input.expect(",")
                input.skip_whitespace()
            first = False
            result.append(input.parse(items_parser))
        return (input.index, result)


ARBITRARY_JSON = (
    NULL_PARSER
    // BOOL_PARSER
    // FLOAT_PARSER
    // STRING_LITERAL_PARSER
    // ArraySchemaParser({})
    // ObjectSchemaParser({"additionalProperties": True})
)


def json_schema_parser(schema):
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
    elif schema["type"] == "object" and schema.get("properties"):
        return ObjectSchemaParser(schema)
    elif schema["type"] == "array":
        return ArraySchemaParser(schema)
    else:
        return ARBITRARY_JSON
