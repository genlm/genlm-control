import pytest
from genlm.control.potential.built_in.json import (
    JsonSchema,
    json_schema_parser,
    ARBITRARY_JSON,
    Incomplete,
    FLOAT_PARSER,
    chunk_to_complete_utf8,
    ParseError,
    StreamingJsonSchema,
    ValidateJSON,
    ParserPotential,
    StringSource,
    Input,
    FloatParser,
    WHITESPACE_PARSER,
    StringLiteralMatchingPatternParser,
)
from genlm.control.potential.streaming import AsyncSource
import json
from typing import Any
from dataclasses import dataclass
from hypothesis import given, strategies as st, assume, example, settings, reject
from hypothesis_jsonschema import from_schema
import asyncio
from jsonschema import SchemaError
import regex


@pytest.mark.asyncio
async def test_validates_a_list_of_integers():
    potential = JsonSchema({"type": "array", "items": {"type": "integer"}})

    assert await potential.prefix(b"[1,2,3") == 0.0
    assert await potential.prefix(b'["hello world"') == -float("inf")
    assert await potential.prefix(b"{") == -float("inf")


@pytest.mark.asyncio
async def test_rejects_as_prefix_when_no_valid_continuation():
    potential = JsonSchema({"type": "object"})

    assert await potential.prefix(b"}") == -float("inf")


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [{"type": "array", "items": {"type": "integer"}}])
@pytest.mark.parametrize(
    "context",
    [
        b"[1,2,3",
        b"[0]",
    ],
)
async def test_consistency_properties(schema, context):
    potential = JsonSchema(schema)
    await potential.assert_autoreg_fact(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "potential",
    [
        StreamingJsonSchema({"type": "array", "items": {"type": "integer"}}),
        ValidateJSON(),
        ParserPotential(
            json_schema_parser({"type": "array", "items": {"type": "integer"}})
        ),
    ],
)
async def test_logw_next_has_results(potential):
    logs = await potential.logw_next(b"")
    assert logs[b"["[0]] == 0.0


@pytest.mark.asyncio
async def test_will_error_on_impossible_unicode_prefixes():
    potential = JsonSchema({"type": "object"})
    assert await potential.prefix([190] * 5) == -float("inf")


@st.composite
def basic_schema(draw):
    dtype = draw(
        st.sampled_from(
            [
                "null",
                "boolean",
                "integer",
                "number",
                "string",
                "enum",
            ]
        )
    )

    # Note: We do not currently implement patterns here, because it's too hard to do this
    # without hitting https://github.com/mrabarnett/mrab-regex/issues/571

    if dtype == "enum":
        return {
            "enum": draw(
                st.lists(
                    st.none()
                    | st.booleans()
                    # We don't currently handle the semantics of numbers very well
                    # in enums, because the correct semantics depends on them being
                    # equal to floats. Integer enums will still work, but they will
                    # reject things that are technically fine. e.g. {"enum": [0]}
                    # should match the string "0.0" and doesn't.
                    #
                    # Hypothesis is very good at pointing this out, so we solve this
                    # by not letting it do that, as we mostly only care about
                    # enums of strings anyway.
                    | st.integers()
                    # | st.floats(allow_nan=False, allow_infinity=False)
                    | st.text(),
                    min_size=1,
                    unique_by=lambda x: (type(x), x),
                )
            )
        }

    return {"type": dtype}


@st.composite
def composite_schema(draw, sub_schema):
    match draw(
        st.sampled_from(
            [
                "array",
                "anyOf",
                "object",
            ]
        )
    ):
        case "array":
            result = {"type": "array", "items": draw(sub_schema)}
            min_contains = draw(st.integers(0, 10))
            if min_contains > 0:
                result["minContains"] = min_contains
            if draw(st.booleans()):
                max_contains = draw(st.integers(min_contains, 20))
                result["maxContains"] = max_contains
            return result
        case "anyOf":
            return {"anyOf": draw(st.lists(sub_schema, min_size=2))}
        case "object":
            result = {"type": "object"}
            result["properties"] = draw(
                st.dictionaries(
                    st.from_regex("[A-Za-z0-9_]+"),
                    sub_schema,
                )
            )
            if result["properties"]:
                result["required"] = draw(
                    st.lists(st.sampled_from(sorted(result["properties"])), unique=True)
                )
            result["additionalProperties"] = draw(st.booleans())
            return result


json_schema = st.recursive(
    base=basic_schema(),
    extend=composite_schema,
)


@dataclass(frozen=True)
class JSONSchemaPotentialProblem:
    schema: Any
    document: bytes
    prefix: bytes

    @property
    def value(self):
        return json.loads(self.document)


@st.composite
def json_schema_potential_problem(draw):
    schema = draw(json_schema)
    value = draw(from_schema(schema))
    text = json.dumps(
        value,
        # Inverted so that this shrinks to True, as ascii-only
        # JSON is simpler.
        ensure_ascii=not draw(st.booleans()),
        # Similarly inverted so as to shrink to True, on the
        # theory that this means that if keys are out of
        # order in a shrunk example then it really matters.
        sort_keys=not draw(st.booleans()),
        indent=draw(st.one_of(st.none(), st.integers(0, 4))),
    )

    document = text.encode("utf-8")
    assert document
    assume(len(document) > 1)
    i = draw(st.integers(1, len(document) - 1))
    prefix = document[:i]
    assume(prefix.strip())

    return JSONSchemaPotentialProblem(schema=schema, document=document, prefix=prefix)


@pytest.mark.asyncio
@example(
    JSONSchemaPotentialProblem(
        schema={"type": "string"},
        document=b'"0\xc2\x80\xc2\x80"',
        prefix=b'"0\xc2\x80\xc2',
    )
)
@example(
    JSONSchemaPotentialProblem(
        schema={
            "type": "string",
        },
        document=b'"000000000\\u001f\xc2\x80\xc2\x80"',
        prefix=b'"000000000\\u001f\xc2\x80\xc2\x80',
    ),
)
@example(
    JSONSchemaPotentialProblem(
        schema={
            "type": "string",
        },
        document=b'"000\\u001f\xc2\x80\xc2\x80\xc2\x80"',
        prefix=b'"000\\u001f\xc2\x80\xc2\x80\xc2',
    ),
)
@example(
    JSONSchemaPotentialProblem(
        schema={"type": "array", "items": {"type": "integer"}},
        document=b"[0]",
        prefix=b"[",
    ),
)
@example(
    JSONSchemaPotentialProblem(
        schema={
            "anyOf": [
                {"type": "object", "properties": {}, "additionalProperties": False},
                {
                    "type": "object",
                    "properties": {"0": {"type": "null"}},
                    "required": ["0"],
                    "additionalProperties": False,
                },
            ]
        },
        document=b'{"0": null}',
        prefix=b"{",
    ),
)
@given(json_schema_potential_problem())
@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
async def test_always_returns_correctly_on_valid_documents(problem):
    parser = json_schema_parser(problem.schema)

    await parser.parse_string(problem.document.decode("utf-8"))
    try:
        await parser.parse_string(problem.prefix.decode("utf-8"))
    except (Incomplete, UnicodeDecodeError):
        pass

    potential = JsonSchema(problem.schema)

    assert await potential.prefix(problem.prefix) == 0.0
    assert await potential.prefix(problem.document) == 0.0
    if await potential.complete(problem.prefix) > -float("inf"):
        # This can sometimes happen because e.g. numeric literals can have
        # a prefix that is also a valid JSON value. We check here that the
        # prefix is actually valid JSON and if so allow it.
        json.loads(problem.prefix)
    assert await potential.complete(problem.document) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "format",
    [
        "ipv4",
        "date-time",
        "date",
        "date-time",
        # duration not present in Draft 7 which we're currently using.
        # "duration",
        "email",
        "hostname",
        "idn-hostname",
        "ipv4",
        "ipv6",
        "json-pointer",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
    ],
)
async def test_validates_formats(format):
    potential = JsonSchema({"format": format, "type": "string"})
    assert await potential.prefix(b'"hello world"') == -float("inf")


@pytest.mark.asyncio
async def test_validates_regex_format():
    potential = JsonSchema({"format": "regex", "type": "string"})
    assert await potential.prefix(b'"["') == -float("inf")


@pytest.mark.asyncio
async def test_will_not_allow_nonsense_after_json():
    potential = JsonSchema({"type": "object"})
    assert await potential.prefix(b"{} hello world") == -float("inf")
    assert await potential.complete(b"{} hello world") == -float("inf")


@pytest.mark.asyncio
async def test_valid_prefix_for_schema_eg1():
    potential = JsonSchema(
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "array",
            "items": {
                "$schema": "http://json-schema.org/draft-04/schema#",
                "type": "object",
                "properties": {
                    "time": {"type": "string", "format": "date-time"},
                    "relayId": {"type": "string"},
                    "data": {
                        "type": "object",
                        "patternProperties": {
                            "^[0-9a-zA-Z_-]{1,255}$": {
                                "type": ["number", "string", "boolean"]
                            }
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        }
    )

    assert await potential.prefix(b"[{") == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ws",
    [
        b"\n\n\n",
        b"\n    \n",
    ],
)
async def test_forbids_weird_whitespace(ws):
    potential = JsonSchema({})
    assert await potential.prefix(ws) == -float("inf")


@pytest.mark.asyncio
async def test_rejects_as_prefix_when_invalid_key_has_been_started():
    potential = JsonSchema(
        {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                }
            },
            "required": ["data"],
            "additionalProperties": False,
        }
    )

    assert await potential.prefix(b'{"fo') == -float("inf")


@pytest.mark.asyncio
async def test_rejects_when_value_is_invalid_before_object_is_complete():
    potential = JsonSchema(
        {
            "type": "object",
            "properties": {
                "stuff": {
                    "type": "string",
                },
                "data": {
                    "type": "string",
                },
            },
            "additionalProperties": False,
        }
    )

    assert await potential.prefix(b'{"data": 1.0, ') == -float("inf")


@pytest.mark.asyncio
async def test_rejects_duplicated_key():
    potential = JsonSchema(
        {
            "type": "object",
        }
    )

    assert await potential.prefix(b'{"data": 1.0, "data"') == -float("inf")


@pytest.mark.asyncio
async def test_rejects_string_as_invalid_integer_before_complete():
    potential = JsonSchema(
        {
            "type": "integer",
        }
    )

    assert await potential.prefix(b'"') == -float("inf")


@pytest.mark.asyncio
async def test_accepts_basic_integer_list():
    potential = JsonSchema({"type": "array", "items": {"type": "integer"}})

    assert await potential.prefix(b"[0]") == 0.0
    assert await potential.complete(b"[0]") == 0.0

    logs = dict(await potential.logw_next(b"[0]"))
    for k, v in logs.items():
        # Forbid all ascii characters other than newline and space.
        if isinstance(k, int) and k < 128 and k not in b" \n":
            assert v == -float("inf")
    assert logs[potential.eos] == 0.0


@pytest.mark.asyncio
async def test_rejects_string_as_invalid_integer_inside_list():
    potential = JsonSchema({"type": "array", "items": {"type": "integer"}})

    assert await potential.prefix(b'["') == -float("inf")


@pytest.mark.asyncio
async def test_can_extend_zero_to_integer_list():
    schema = {"type": "array", "items": {"type": "integer"}}
    potential = JsonSchema(schema)
    assert await potential.prefix(b"[0,") == 0


@dataclass(frozen=True)
class SchemaAndDocument:
    schema: Any
    document: Any


@st.composite
def json_schema_and_document(draw):
    schema = draw(json_schema)
    document = draw(from_schema(schema))
    return SchemaAndDocument(schema, document)


@pytest.mark.asyncio
@settings(report_multiple_bugs=False, deadline=None)
@given(json_schema_and_document())
async def test_parser_for_schema_always_returns_document(sad):
    parser = json_schema_parser(sad.schema)
    text = json.dumps(sad.document)
    result = await parser.parse_string(text)
    assert result == sad.document


@pytest.mark.asyncio
@example(
    JSONSchemaPotentialProblem(schema={"type": "integer"}, document=b"-1", prefix=b"-"),
)
@example(
    JSONSchemaPotentialProblem(
        schema={"type": "string"}, document=b'"\xc2\x80"', prefix=b'"'
    )
)
@example(
    JSONSchemaPotentialProblem(
        schema={
            "type": "object",
            "properties": {
                "0": {"type": "null"},
                "0\x7f": {"type": "null"},
                "1": {"type": "null"},
            },
            "required": ["0", "0\x7f", "1"],
            "additionalProperties": False,
        },
        document=b'{"0": null, "0\x7f": null, "1": null}',
        prefix=b"{",
    ),
)
@example(
    JSONSchemaPotentialProblem(
        schema={"type": "array", "items": {"type": "number"}},
        document=b"[\n1.3941332551795901e+28\n]",
        prefix=b"[\n1.3941332551795901e+",
    ),
)
@settings(report_multiple_bugs=False, deadline=None)
@given(json_schema_potential_problem())
async def test_parser_for_schema_prefix_can_only_raise_incomplete(problem):
    parser = json_schema_parser(problem.schema)

    # Just to get coverage on the repr methods.
    repr(parser)

    whole_text = problem.document.decode("utf-8")
    result = await parser.parse_string(whole_text)
    assert result == problem.value

    try:
        text = problem.prefix.decode("utf-8")
    except UnicodeDecodeError:
        reject()
    try:
        await parser.parse_string(text)
    except Incomplete:
        pass


@st.composite
def json_object(draw):
    return draw(
        st.one_of(
            st.none(),
            st.booleans(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(),
            st.lists(json_object()),
            st.dictionaries(st.text(), json_object()),
        )
    )


@pytest.mark.asyncio
@example(False)
@settings(report_multiple_bugs=False, deadline=None)
@given(json_object())
async def test_parser_for_arbitrary_json_can_parse_arbitrary_json(obj):
    text = json.dumps(obj)
    await ARBITRARY_JSON.parse_string(text)


@pytest.mark.asyncio
@settings(report_multiple_bugs=False, deadline=None)
@given(st.sets(st.text()))
async def test_correctly_handles_fixed_object_keys(keys):
    parser = json_schema_parser(
        {
            "type": "object",
            "properties": {key: {"type": "null"} for key in keys},
            "additionalProperties": False,
        }
    )

    x = {key: None for key in keys}
    s = json.dumps(x)
    result = await parser.parse_string(s)
    assert result == x


@pytest.mark.asyncio
async def test_float_parser_incomplete_literal():
    with pytest.raises(Incomplete):
        await FLOAT_PARSER.parse_string("0.")


@st.composite
def chunked_utf8(draw, base=None):
    if base is None:
        base = draw(st.text(min_size=1)).encode("utf-8")
    assume(len(base) > 1)
    offsets = draw(st.sets(st.integers(1, len(base) - 1)))
    offsets.update((0, len(base)))
    offsets = sorted(offsets)
    chunks = [base[u:v] for u, v in zip(offsets, offsets[1:])]
    assert b"".join(chunks) == base
    return chunks


@given(chunked_utf8())
@settings(report_multiple_bugs=False, deadline=None)
def test_utf8_chunking_always_splits_utf8(chunks):
    rechunked = list(chunk_to_complete_utf8(chunks))
    assert b"".join(rechunked) == b"".join(chunks)
    for chunk in rechunked:
        assert chunk
        chunk.decode("utf-8")


class BasicSource(AsyncSource):
    def __init__(self, blocks):
        self.__blocks = iter(blocks)

    async def more(self):
        try:
            return next(self.__blocks)
        except StopIteration:
            raise StopAsyncIteration()


@pytest.mark.asyncio
@given(chunked_utf8())
@settings(report_multiple_bugs=False, deadline=None)
async def test_utf8_chunking_always_splits_utf8_async(chunks):
    source = BasicSource(chunks)
    string_source = StringSource(source)

    buffer = bytearray()

    while True:
        try:
            chunk = await string_source.more()
        except StopAsyncIteration:
            break
        buffer.extend(chunk.encode("utf-8"))

    assert bytes(buffer) == b"".join(chunks)


@pytest.mark.asyncio
async def test_parser_raises_incomplete_on_empty_string():
    with pytest.raises(Incomplete):
        await FLOAT_PARSER.parse_string("")


@pytest.mark.asyncio
async def test_validates_a_list_of_integers_parser_only():
    parser = json_schema_parser({"type": "array", "items": {"type": "integer"}})

    with pytest.raises(Incomplete):
        await parser.parse_string("[1,2,3")

    with pytest.raises(ParseError):
        assert await parser.parse_string('["hello world"')

    with pytest.raises(ParseError):
        await parser.parse_string("{")


@pytest.mark.asyncio
async def test_can_calculate_many_prefixes():
    potential = JsonSchema({"type": "object"})

    for i in range(100):
        prefix = b'{ "' + str(i).encode("utf-8")
        pot = await potential.prefix(prefix)
        assert pot == 0.0


@pytest.mark.asyncio
async def test_raises_value_error_for_logw_next_of_bad_prefix():
    potential = JsonSchema({"type": "object"})
    with pytest.raises(ValueError):
        await potential.logw_next(b"[")


@pytest.mark.asyncio
async def test_basic_json_validator_rejects_silly_whitespace():
    potential = ValidateJSON()
    assert await potential.prefix(b"\n\n\n") == -float("inf")
    assert await potential.complete(b"\n\n\n") == -float("inf")


@pytest.mark.asyncio
async def test_float_parser_can_continue_parsing_across_boundaries():
    source = BasicSource(["2", ".", "0", "1"])

    input = Input(source)

    parser = FloatParser()

    f = await input.parse(parser)

    assert f == 2.01


@dataclass(frozen=True)
class JSONSchemaPotentialProblemMulti:
    schema: Any
    document: bytes
    values: list[bytes]

    @property
    def value(self):
        return json.loads(self.document)


@st.composite
def json_schema_potential_problem_multi(draw):
    schema = draw(json_schema)
    value = draw(from_schema(schema))
    text = json.dumps(
        value,
        # Inverted so that this shrinks to True, as ascii-only
        # JSON is simpler.
        ensure_ascii=not draw(st.booleans()),
        # Similarly inverted so as to shrink to True, on the
        # theory that this means that if keys are out of
        # order in a shrunk example then it really matters.
        sort_keys=not draw(st.booleans()),
        indent=draw(st.one_of(st.none(), st.integers(0, 4))),
    )

    document = text.encode("utf-8")
    assert document
    assume(len(document) > 1)

    values = []

    for _ in range(draw(st.integers(1, 10))):
        offsets = draw(st.sets(st.integers(1, len(document) - 1), min_size=1))
        offsets = sorted(offsets)
        prefixes = [document[:v] for v in offsets]
        values.extend(prefixes)

    values = draw(st.permutations(values))
    values = values[: draw(st.integers(1, len(values)))]

    return JSONSchemaPotentialProblemMulti(
        schema=schema, document=document, values=values
    )


@pytest.mark.asyncio
@example(
    problem=JSONSchemaPotentialProblemMulti(
        schema={"type": "boolean"},
        document=b"false",
        values=[b"f", b"fa", b"fal", b"f"],
    ),
    cache_size=1,
)
@example(
    problem=JSONSchemaPotentialProblemMulti(
        schema={"type": "boolean"},
        document=b"false",
        values=[b"f", b"fa", b"f", b"fa", b"fal", b"f", b"fa", b"fal", b"fals"],
    ),
    cache_size=5,
)
@given(json_schema_potential_problem_multi(), st.integers(1, 100))
@settings(report_multiple_bugs=False, deadline=None)
async def test_cache_eviction_with_many_prefixes(problem, cache_size):
    potential = StreamingJsonSchema(problem.schema, cache_size=cache_size)

    results = list(
        await asyncio.gather(*[potential.prefix(value) for value in problem.values])
    )
    assert all(result == 0.0 for result in results)

    assert await potential.complete(problem.document) == 0.0


@pytest.mark.asyncio
async def test_can_reject_wrong_type_inside_any_of():
    schema = {
        "anyOf": [
            {
                "anyOf": [
                    {
                        "type": "object",
                    },
                ]
            },
        ]
    }

    parser = json_schema_parser(schema)
    potential = ParserPotential(parser)

    assert await potential.prefix(b'"') == -float("inf")


@pytest.mark.asyncio
async def test_can_reject_early_in_any_of():
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                },
                "required": ["a"],
            },
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                },
                "required": ["a"],
            },
        ]
    }

    parser = json_schema_parser(schema)
    potential = ParserPotential(parser)

    assert await potential.prefix(b'"') == -float("inf")
    assert await potential.prefix(b'{"a":') == 0
    assert await potential.prefix(b'{"a": "') == 0
    assert await potential.prefix(b'{"a": 1') == 0
    assert await potential.prefix(b'{"a": {') == -float("inf")


@pytest.mark.asyncio
async def test_will_reject_invalid_unicode_at_end():
    potential = StreamingJsonSchema({"type": "object"})
    assert await potential.prefix(b"{ }\n\n    \xe2\x9d\x8d\xb0") == -float("inf")


def test_chunk_to_complete_utf8_will_error_on_invalid_unicode():
    with pytest.raises(UnicodeDecodeError):
        list(chunk_to_complete_utf8([b"{ }\n\n    \xe2\x9d\x8d\xb0"]))


@pytest.mark.asyncio
async def test_rejects_using_unicode_whitespace():
    pot = JsonSchema({"type": "object"})
    assert await pot.prefix("{ \u3000".encode("utf-8")) == -float("inf")


def test_chunking_immediately_rejects_invalid_utf8_bytes():
    def bad_bytes():
        yield b"\xc0"
        assert False

    with pytest.raises(UnicodeDecodeError):
        list(chunk_to_complete_utf8(bad_bytes()))


def test_chunking_bails_early_on_invalid_start_bytes():
    def bad_bytes():
        yield b"\xe3\x86\x8c\x80"
        assert False

    with pytest.raises(UnicodeDecodeError):
        list(chunk_to_complete_utf8(bad_bytes()))


@pytest.mark.asyncio
async def test_long_whitespace_at_start_is_rejected():
    assert await ValidateJSON().prefix(b"  ") == 0
    assert await ValidateJSON().prefix(b"\n\n") == 0
    assert await ValidateJSON().prefix(b"    ") == -float("inf")
    assert await ValidateJSON().prefix(b"\n\n  ") == -float("inf")


@pytest.mark.asyncio
async def test_no_double_newline_after_start():
    potential = JsonSchema({"type": "object"})
    assert await potential.prefix(b"{\n\n") == -float("inf")
    assert await potential.prefix(b"{\n  \n") == -float("inf")


def test_repr_of_filter():
    assert "filter" in repr(WHITESPACE_PARSER)


@pytest.mark.asyncio
async def test_const_fails_fast():
    potential = ParserPotential(json_schema_parser({"const": False}))
    assert await potential.prefix(b" ") == 0
    assert await potential.prefix(b" f") == 0
    assert await potential.prefix(b" false") == 0
    assert await potential.prefix(b" n") == -float("inf")


@pytest.mark.asyncio
async def test_const_fails_fast_in_string_literals():
    potential = ParserPotential(json_schema_parser({"const": "Hello world"}))
    assert await potential.prefix(b" ") == 0
    assert await potential.prefix(b'"Hello') == 0
    assert await potential.prefix(b'"Hi') == -float("inf")


@pytest.mark.asyncio
async def test_const_in_object():
    potential = ParserPotential(
        json_schema_parser({"type": "object", "properties": {"foo": {"const": None}}})
    )

    assert await potential.prefix(b'{"foo": nu') == 0
    assert await potential.complete(b'{"foo": null}') == 0
    assert await potential.prefix(b'{"foo": f') == -float("inf")


def test_errors_on_bad_types():
    with pytest.raises(SchemaError):
        JsonSchema({"type": "float"})


@pytest.mark.asyncio
async def test_can_validate_patterns_in_incomplete_strings():
    parser = json_schema_parser({"type": "string", "pattern": "^[0-9]*$"})
    potential = ParserPotential(parser)

    for prefix in [
        '"',
        '"01234',
        '"01234"',
    ]:
        try:
            await parser.parse_string(prefix)
        except Incomplete:
            pass

        assert await potential.prefix(prefix.encode("utf-8")) == 0.0

    for prefix in [
        '"A',
        '"0000A',
    ]:
        with pytest.raises(ParseError):
            await parser.parse_string(prefix)

        assert await potential.prefix(prefix.encode("utf-8")) == -float("inf")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix",
    [
        "{",
        '{"id1": "0',
        '{"id1": "0",',
        '{"id1": "0",',
        '{"id1": "0", "id2": "A',
    ],
)
async def test_can_validate_patterns_in_incomplete_strings_in_objects_good_prefix(
    prefix,
):
    parser = json_schema_parser(
        {
            "type": "object",
            "properties": {
                "id1": {"type": "string", "pattern": "[0-9]+"},
                "id2": {"type": "string", "pattern": "[A-Z]+"},
            },
        }
    )
    potential = ParserPotential(parser)

    with pytest.raises(Incomplete):
        await parser.parse_string(prefix)

    assert await potential.prefix(prefix.encode("utf-8")) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ['{"id2": "0", "id1": "A'])
async def test_can_validate_patterns_in_incomplete_strings_in_objects_bad_prefix(
    prefix,
):
    parser = json_schema_parser(
        {
            "type": "object",
            "properties": {
                "id1": {"type": "string", "pattern": "[0-9]+"},
                "id2": {"type": "string", "pattern": "[A-Z]+"},
            },
        }
    )
    potential = ParserPotential(parser)

    with pytest.raises(ParseError):
        await parser.parse_string(prefix)

    assert await potential.prefix(prefix.encode("utf-8")) == -float("inf")


@pytest.mark.asyncio
async def test_can_handle_incomplete_escape_sequences():
    parser = json_schema_parser({"type": "string", "pattern": ".*"})
    snowman = "\u2603"

    json_snowman = json.dumps(snowman, ensure_ascii=True)

    for i in range(len(json_snowman)):
        with pytest.raises(Incomplete):
            await parser.parse_string(json_snowman[:i])

    assert await parser.parse_string(json_snowman) == snowman


@pytest.mark.asyncio
async def test_errors_if_string_only_matches_a_prefix():
    parser = json_schema_parser({"type": "string", "pattern": "cabbages"})

    cab = json.dumps("cab")

    with pytest.raises(ParseError):
        assert await parser.parse_string(cab)


@pytest.mark.asyncio
async def test_patterns_apply_if_matching_anywhere():
    schema = {"type": "string", "pattern": "\\.(mp4|avi|mov|wmv|flv)$"}
    parser = json_schema_parser(schema)
    await parser.parse_string('"0.mp4"')


@pytest.mark.asyncio
async def test_patterns_reject_non_strings():
    schema = {"type": "string", "pattern": "\\.(mp4|avi|mov|wmv|flv)$"}
    parser = json_schema_parser(schema)

    with pytest.raises(ParseError):
        await parser.parse_string("0")


@pytest.mark.asyncio
async def test_patterns_apply_if_match_before_end_of_string():
    schema = {"type": "string", "pattern": "<[^>]+>"}
    parser = json_schema_parser(schema)

    await parser.parse_string('"<0>0"')


# List of 100 regular expressions for common data validation scenarios.
# Generated by Claude rather than mined from real data..
COMMON_PATTERNS = [
    # Email validation
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    r"^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$",
    # Phone numbers
    r"^\+?1?-?\.?\s?\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}$",
    r"^(\+\d{1,3}[- ]?)?\d{10}$",
    r"^\(\d{3}\)\s?\d{3}-\d{4}$",
    r"^\d{3}-\d{3}-\d{4}$",
    r"^\+\d{1,3}\s\d{1,4}\s\d{1,4}\s\d{1,9}$",
    # URLs
    r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.]*))?(?:\#(?:[\w.]*))?)?$",
    r"^(https?|ftp)://[^\s/$.?#].[^\s]*$",
    r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/.*)?$",
    # IP addresses
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
    r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
    r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$",  # IPv6
    # Credit card numbers
    r"^4[0-9]{12}(?:[0-9]{3})?$",  # Visa
    r"^5[1-5][0-9]{14}$",  # MasterCard
    r"^3[47][0-9]{13}$",  # American Express
    r"^3[0-9]{4,}$",  # Diners Club
    r"^6(?:011|5[0-9]{2})[0-9]{12}$",  # Discover
    # Dates
    r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
    r"^\d{2}/\d{2}/\d{4}$",  # MM/DD/YYYY
    r"^\d{2}-\d{2}-\d{4}$",  # MM-DD-YYYY
    r"^\d{1,2}/\d{1,2}/\d{4}$",  # M/D/YYYY
    r"^\d{4}/\d{2}/\d{2}$",  # YYYY/MM/DD
    # Time formats
    r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$",  # HH:MM
    r"^([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$",  # HH:MM:SS
    r"^(1[0-2]|0?[1-9]):[0-5][0-9]\s?(AM|PM)$",  # 12-hour format
    # Passwords (various strength requirements)
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$",  # Min 8 chars, uppercase, lowercase, digit
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",  # With special chars
    r"^.{8,}$",  # At least 8 characters
    r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$",  # At least 6 chars with letter and digit
    # Social Security Numbers
    r"^\d{3}-\d{2}-\d{4}$",
    r"^\d{9}$",
    # ZIP codes
    r"^\d{5}$",  # US ZIP
    r"^\d{5}-\d{4}$",  # US ZIP+4
    r"^[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d$",  # Canadian postal code
    # UUIDs
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    r"^[0-9a-fA-F]{32}$",  # UUID without hyphens
    # Numbers and currencies
    r"^\d+$",  # Positive integers
    r"^-?\d+$",  # Integers
    r"^\d*\.?\d+$",  # Positive decimal numbers
    r"^-?\d*\.?\d+$",  # Decimal numbers
    r"^\$\d{1,3}(,\d{3})*(\.\d{2})?$",  # Currency format
    r"^\d{1,3}(,\d{3})*(\.\d{2})?$",  # Number with commas
    # Usernames
    r"^[a-zA-Z0-9_]{3,20}$",
    r"^[a-zA-Z][a-zA-Z0-9_.-]{2,19}$",
    r"^[a-zA-Z0-9_.-]{3,30}$",
    # Names
    r"^[a-zA-Z\s]{2,50}$",
    r"^[a-zA-Z\'\-\s]{1,50}$",
    r"^[A-Z][a-z]+(\s[A-Z][a-z]+)*$",  # Proper case names
    # Alphanumeric strings
    r"^[a-zA-Z0-9]+$",
    r"^[a-zA-Z0-9\s]+$",
    r"^[a-zA-Z0-9_-]+$",
    # File extensions
    r"\.(jpg|jpeg|png|gif|bmp)$",  # Images
    r"\.(pdf|doc|docx|txt|rtf)$",  # Documents
    r"\.(mp3|wav|flac|aac)$",  # Audio
    r"\.(mp4|avi|mov|wmv|flv)$",  # Video
    # HTML tags
    r"<[^>]+>",
    r"<\/?[a-zA-Z][a-zA-Z0-9]*[^<>]*>",
    # Hex colors
    r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$",
    r"^#[0-9A-Fa-f]{6}$",
    # Base64
    r"^[A-Za-z0-9+/]*={0,2}$",
    # JSON strings (basic)
    r"^[\{\[].*[\}\]]$",
    # MAC addresses
    r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$",
    # Version numbers
    r"^\d+\.\d+\.\d+$",  # Semantic versioning
    r"^\d+\.\d+$",  # Major.minor
    # Stock symbols
    r"^[A-Z]{1,5}$",
    # License plates (US format examples)
    r"^[A-Z]{3}\d{4}$",
    r"^[A-Z]{2}\d{5}$",
    # Geographic coordinates
    r"^-?([1-8]?\d(\.\d+)?|90(\.0+)?),-?((1[0-7]\d)|([1-9]?\d))(\.\d+)?|180(\.0+)?$",  # Lat,Long
    # Binary strings
    r"^[01]+$",
    # Hashtags
    r"^#[a-zA-Z0-9_]+$",
    # Mention patterns
    r"^@[a-zA-Z0-9_]+$",
    # SQL injection patterns (for blacklisting)
    r".*(union|select|insert|delete|update|drop|create|alter|exec|execute).*",
    # XSS patterns (for blacklisting)
    r".*<script.*?>.*</script>.*",
    # Domain names
    r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$",
    # Port numbers
    r"^([1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$",
    # ISBN (10 and 13 digit)
    r"^(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]$",
    # US State abbreviations
    r"^(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)$",
    # Language codes (ISO 639-1)
    r"^[a-z]{2}$",
    # Country codes (ISO 3166-1 alpha-2)
    r"^[A-Z]{2}$",
    # MIME types
    r"^[a-zA-Z][a-zA-Z0-9][a-zA-Z0-9\!#\$&\-\^]*\/[a-zA-Z0-9][a-zA-Z0-9\!#\$&\-\^]*$",
    # JWT tokens (basic structure)
    r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$",
    # MongoDB ObjectId
    r"^[0-9a-fA-F]{24}$",
    # Credit card expiry (MM/YY)
    r"^(0[1-9]|1[0-2])\/([0-9]{2})$",
    # CVV codes
    r"^[0-9]{3,4}$",
    # Routing numbers (US banks)
    r"^[0-9]{9}$",
    # IBAN (basic format)
    r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}$",
    # Twitter handles
    r"^@[A-Za-z0-9_]{1,15}$",
    # Slack channels
    r"^#[a-z0-9_-]+$",
    # Discord user tags
    r"^.{3,32}#[0-9]{4}$",
    # YouTube video IDs
    r"^[a-zA-Z0-9_-]{11}$",
    # Bitcoin addresses
    r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$",
    # Ethereum addresses
    r"^0x[a-fA-F0-9]{40}$",
    # ISBN-10
    r"^(?:[0-9]{9}X|[0-9]{10})$",
    # ISBN-13
    r"^97[89][0-9]{10}$",
    # VAT numbers (EU format)
    r"^[A-Z]{2}[0-9A-Z]+$",
    # Strong password with all character types
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?])[A-Za-z\d!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]{12,}$',
    # Weak passwords (for blacklisting)
    r"^(password|123456|qwerty|abc123|letmein|admin|welcome)$",
    # Only letters
    r"^[a-zA-Z]+$",
    # Only digits
    r"^[0-9]+$",
    # Whitespace validation
    r"^\S+$",  # No whitespace
    r"^[^\s]+(\s[^\s]+)*$",  # No leading/trailing/multiple whitespace
]

COMMON_PATTERNS.sort(key=lambda s: (len(s), s))


@pytest.mark.asyncio
@given(st.data())
async def test_can_always_validate_prefixes_of_pattern_matching_strings(data):
    pattern = data.draw(st.sampled_from(COMMON_PATTERNS))

    string = data.draw(st.from_regex(pattern))

    # To avoid hitting https://github.com/mrabarnett/mrab-regex/issues/571
    assume(regex.search(pattern, string) is not None)

    literal = json.dumps(string)

    parser = StringLiteralMatchingPatternParser(pattern)

    await parser.parse_string(literal)

    prefix = literal[: data.draw(st.integers(0, len(literal) - 1))]

    with pytest.raises(Incomplete):
        await parser.parse_string(prefix)


@pytest.mark.asyncio
async def test_cutting_through_a_surrogate_pair():
    parser = StringLiteralMatchingPatternParser("^[A-Z]{2}\\d{5}$")
    string = json.dumps("AA𐒠0000")

    await parser.parse_string(string)

    for i in range(len(string)):
        with pytest.raises(Incomplete):
            await parser.parse_string(string[:i])


@pytest.mark.asyncio
async def test_reject_half_of_a_surrogate_pair():
    parser = StringLiteralMatchingPatternParser("^[A-Z]{2}\\d{5}$")
    string = json.dumps("AA\ud801")

    with pytest.raises(ParseError):
        await parser.parse_string(string)


@pytest.mark.asyncio
async def test_will_reject_invalid_json_string():
    parser = StringLiteralMatchingPatternParser(".*")

    with pytest.raises(ParseError):
        await parser.parse_string('"\\0"')


@pytest.mark.asyncio
async def test_will_not_reject_partial_string():
    parser = StringLiteralMatchingPatternParser(".*")

    with pytest.raises(Incomplete):
        await parser.parse_string('"\\')


@pytest.mark.asyncio
async def test_will_handle_string_split_midway_valid():
    parser = StringLiteralMatchingPatternParser('"*')

    assert await parser.parse(Input(BasicSource(['"\\', '""']))) == '"'


@pytest.mark.asyncio
async def test_will_reject_string_split_midway_invalid():
    parser = StringLiteralMatchingPatternParser('^"*$')

    with pytest.raises(ParseError):
        assert await parser.parse(Input(BasicSource(['"\\', '"A', '"'])))


@pytest.mark.asyncio
async def test_union_of_integer_and_number():
    parser = json_schema_parser(
        {
            "type": "array",
            "items": {"anyOf": [{"type": "integer"}, {"type": "number"}]},
        }
    )

    assert await parser.parse_string("[0]") == [0]
    assert await parser.parse_string("[0.0]") == [0.0]
    xs = await parser.parse_string("[0.0, 0, 0.0]")
    assert xs == [0.0, 0, 0.0]
    assert list(map(type, xs)) == [float, int, float]


@pytest.mark.asyncio
async def test_union_of_objects():
    schema = {
        "anyOf": [
            {"type": "object", "properties": {}, "additionalProperties": False},
            {
                "type": "object",
                "properties": {"0": {"type": "null"}},
                "required": ["0"],
                "additionalProperties": False,
            },
        ]
    }

    document = '{"0": null}'

    parser = json_schema_parser(schema)

    assert await parser.parse_string(document) == {"0": None}


@pytest.mark.asyncio
async def test_empty_object():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}

    parser = json_schema_parser(schema)

    assert await parser.parse_string("{}") == dict()

    with pytest.raises(ParseError):
        await parser.parse_string('{"')


@pytest.mark.asyncio
async def test_empty_object_allows_keys():
    parser = json_schema_parser({"type": "object"})

    for i in range(100):
        with pytest.raises(Incomplete):
            await parser.parse_string('{ "' + str(i))


@pytest.mark.asyncio
async def test_enum_parsing():
    values = [1, "two", "twin", 3.0]
    parser = json_schema_parser({"enum": values})

    for v in values:
        s = json.dumps(v)
        await parser.parse_string(s)
        for i in range(len(s)):
            with pytest.raises(Incomplete):
                await parser.parse_string(s[:i])

    for s in [
        "2",
        '"q',
        "30",
        "twa",
    ]:
        with pytest.raises(ParseError):
            await parser.parse_string(s)


@pytest.mark.asyncio
async def test_single_value_enum():
    values = [1]
    parser = json_schema_parser({"enum": values})
    await parser.parse_string("1")

    with pytest.raises(ParseError):
        await parser.parse_string("2")


@pytest.mark.asyncio
async def test_union_of_integer_and_number_with_e_notation():
    schema = {
        "type": "array",
        "items": {"anyOf": [{"type": "null"}, {"type": "integer"}, {"type": "number"}]},
    }

    parser = json_schema_parser(schema)

    await parser.parse_string("[1e-05]") == [1e-05]


@pytest.mark.asyncio
async def test_integer_accepts_only_positive_e_notation():
    schema = {
        "type": "array",
        "items": {"type": "integer"},
    }

    parser = json_schema_parser(schema)

    await parser.parse_string("[1e05]")

    with pytest.raises(ParseError):
        await parser.parse_string("[1e-05]")
