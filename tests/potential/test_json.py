import pytest
from genlm.control.potential.built_in.json import (
    JsonSchema,
    json_schema_parser,
    ARBITRARY_JSON,
    Incomplete,
    FLOAT_PARSER,
)
import json
from typing import Any
from dataclasses import dataclass
from hypothesis import given, strategies as st, assume, example, settings, reject
from hypothesis_jsonschema import from_schema


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

    assert await potential.prefix(b"\t") == 0.0
    assert await potential.complete(b"\t") == -float("inf")


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [{"type": "array", "items": {"type": "integer"}}])
@pytest.mark.parametrize(
    "context",
    [b"[1,2,3", json.dumps(list(range(20))).encode("utf-8")],
)
async def test_consistency_properties(schema, context):
    potential = JsonSchema(schema)
    await potential.assert_autoreg_fact(context)


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
        indent=draw(st.one_of(st.none(), st.integers(0, 4), st.text(alphabet=" \t"))),
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
    return
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


@settings(report_multiple_bugs=False)
@given(json_schema_and_document())
def test_parser_for_schema_always_returns_document(sad):
    parser = json_schema_parser(sad.schema)
    text = json.dumps(sad.document)
    _, result = parser.parse(text, 0)
    assert result == sad.document


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
@settings(report_multiple_bugs=False)
@given(json_schema_potential_problem())
def test_parser_for_schema_prefix_can_only_raise_incomplete(problem):
    parser = json_schema_parser(problem.schema)

    # Just to get coverage on the repr methods.
    repr(parser)

    whole_text = problem.document.decode("utf-8")
    end, result = parser.parse(whole_text, 0)
    assert end == len(whole_text)
    assert result == problem.value

    try:
        text = problem.prefix.decode("utf-8")
    except UnicodeDecodeError:
        reject()
    try:
        parser.parse(text, 0)
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


@example(False)
@settings(report_multiple_bugs=False)
@given(json_object())
def test_parser_for_arbitrary_json_can_parse_arbitrary_json(obj):
    text = json.dumps(obj)
    ARBITRARY_JSON.parse(text, 0)


@given(st.sets(st.text()))
def test_correctly_handles_fixed_object_keys(keys):
    parser = json_schema_parser(
        {
            "type": "object",
            "properties": {key: {"type": "null"} for key in keys},
            "additionalProperties": False,
        }
    )

    x = {key: None for key in keys}
    s = json.dumps(x)
    end, result = parser.parse(s, 0)
    assert end == len(s)
    assert result == x


def test_float_parser_incomplete_literal():
    with pytest.raises(Incomplete):
        FLOAT_PARSER.parse("0.", 0)
