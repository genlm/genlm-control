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
)
from genlm.control.potential.streaming import AsyncSource
import json
from typing import Any
from dataclasses import dataclass
from hypothesis import given, strategies as st, assume, example, settings, reject
from hypothesis_jsonschema import from_schema
import asyncio


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
async def test_whitespace_is_valid_prefix_and_invalid_complete():
    potential = JsonSchema({"type": "object"})

    assert await potential.prefix(b" ") == 0.0
    assert await potential.complete(b" ") == -float("inf")


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
def json_schema(draw):
    type = draw(
        st.sampled_from(
            [
                "null",
                "boolean",
                "integer",
                "number",
                "string",
                "object",
                "array",
            ]
        )
    )

    # TODO: Add some bounds in for some of these?
    if type in ("null", "boolean", "integer", "number", "string"):
        return {"type": type}

    if type == "object":
        result = {"type": "object"}
        result["properties"] = draw(
            st.dictionaries(
                st.from_regex("[A-Za-z0-9_]+"),
                json_schema(),
            )
        )
        if result["properties"]:
            result["required"] = draw(
                st.lists(st.sampled_from(sorted(result["properties"])), unique=True)
            )
        result["additionalProperties"] = draw(st.booleans())
        return result

    assert type == "array"
    result = {"type": "array", "items": draw(json_schema())}
    min_contains = draw(st.integers(0, 10))
    if min_contains > 0:
        result["minContains"] = min_contains
    if draw(st.booleans()):
        max_contains = draw(st.integers(min_contains, 20))
        result["maxContains"] = max_contains
    return result


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
    schema = draw(json_schema())
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
@given(json_schema_potential_problem())
@settings(max_examples=200, deadline=None)
async def test_always_returns_correctly_on_valid_documents(problem):
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
    schema = draw(json_schema())
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
    schema = draw(json_schema())
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
