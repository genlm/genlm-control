import io
import tokenize

import pytest
import numpy as np

from genlm.control.potential.built_in.python_parser import (
    PythonParser,
    _PythonLexer,
    _build_grammar,
)


@pytest.fixture(scope="module")
def P():
    # Byte-level vocabulary: lets tests drive the potential with arbitrary text.
    return PythonParser([bytes([i]) for i in range(256)])


@pytest.fixture(scope="module")
def lexer():
    larkstuff, _ = _build_grammar()
    return _PythonLexer(larkstuff)


def ctx(s):
    return [bytes([b]) for b in s.encode("utf-8")]


# Real DS-1000 reference solutions (verbatim), spanning libraries and structures:
# one-liners, augmented assignment, def-with-block, nested lambda, multi-statement body.
DS1000_SOLUTIONS = [
    "df['datetime'] = df['datetime'].dt.tz_localize(None)",
    "df.loc[df['product'].isin(products), 'score'] *= 10",
    "df.loc[~df['product'].isin(products), 'score'] *= 10",
    "result = g(df.copy())",
    "def g(df, List):\n    return df.iloc[List]\n\nresult = g(df.copy(), List)",
    "def g(df, List):\n"
    "    df2 = df.iloc[List].reindex().reset_index(drop=True)\n"
    "    return (df2.Type != df.Type).sum()\n\n"
    "result = g(df.copy(), List)",
    "def g(df):\n"
    '    return df.where(df.apply(lambda x: x.map(x.value_counts())) >= 2, "other")\n\n'
    "result = g(df.copy())",
    "result = np.where(a > 0, a, 0)",
    "result = [x ** 2 for x in range(5) if x > 1]",
]

# Synthetic but valid Python exercising structures DS-1000 solutions use.
VALID_COMPLETE = DS1000_SOLUTIONS + [
    "x = 1",
    "x = 1\n",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i)\n",
    "with open('f') as fh:\n    data = fh.read()\n",
    "try:\n    x = 1\nexcept Exception:\n    x = 2\n",
    "class A:\n    def m(self):\n        return 1\n",
    "f = lambda a, b=3: a + b",
    "d = {k: v for k, v in items}",
    "y = a if b else c",
    "s = 'a string with spaces and \\' escapes'",
    'm = """triple\nquoted\nstring"""',
    "r = r'\\d+raw'",
    "z = 0x1f + 0b101 + 3.14e-2 + 5j",
]

INVALID_COMPLETE = [
    "def def",
    "x = = 1",
    "return )(",
    "for i in :",
    "def g(:",
    "x = (1 + 2",  # unbalanced bracket
    "if x\n    pass\n",  # missing colon
    "  x = 1",  # unexpected indent at top level
    "$ = 1",  # illegal character
    "class",  # bare keyword
]

# Incomplete but extensible -> valid prefix, but NOT a complete program.
PREFIX_VALID = [
    "",
    "def g(df):",
    "def g(df):\n    ret",
    "x = (1 + ",
    "result = df.gro",  # partial identifier
    "x = 'abc",  # open single-line string
    'm = """open triple',  # open triple string
    "if x =",  # '=' can still become '=='
    "for i in range(10):\n    ",  # awaiting block body
    "data = [1, 2,",
]

PREFIX_INVALID = [
    "x = = ",
    "1 2 3",  # adjacent atoms with no operator
    "def 1",  # def must be followed by a name, not a number
    "return return ",  # trailing space forces the 2nd token to lex as the keyword
    ") = 1",
]


def finite(w):
    return np.isfinite(w)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", VALID_COMPLETE)
async def test_complete_accepts_valid_python(P, code):
    assert finite(await P.complete(ctx(code))), code


@pytest.mark.asyncio
@pytest.mark.parametrize("code", INVALID_COMPLETE)
async def test_complete_rejects_invalid_python(P, code):
    assert not finite(await P.complete(ctx(code))), code


@pytest.mark.asyncio
@pytest.mark.parametrize("code", PREFIX_VALID)
async def test_prefix_accepts_extensible(P, code):
    assert finite(await P.prefix(ctx(code))), code


@pytest.mark.asyncio
@pytest.mark.parametrize("code", PREFIX_INVALID)
async def test_prefix_rejects_doomed(P, code):
    assert not finite(await P.prefix(ctx(code))), code


@pytest.mark.asyncio
@pytest.mark.parametrize("code", VALID_COMPLETE)
async def test_complete_implies_prefix(P, code):
    # A complete valid program must also be a valid prefix (potential monotonicity).
    if finite(await P.complete(ctx(code))):
        assert finite(await P.prefix(ctx(code))), code


@pytest.mark.asyncio
async def test_incremental_prefix_never_spuriously_rejects(P):
    prog = "def g(df):\n    return df.iloc[0]\n\nresult = g(df.copy())"
    for k in range(1, len(prog) + 1):
        assert finite(await P.prefix(ctx(prog[:k]))), repr(prog[:k])


@pytest.mark.asyncio
async def test_multibyte_split_is_valid_prefix(P):
    # A context that splits a multi-byte UTF-8 char mid-token is a valid prefix.
    snowman = "x = '☃'"  # ☃
    full = list(snowman.encode("utf-8"))
    truncated = [bytes([b]) for b in full[:-1]]  # drop final continuation byte
    assert finite(await P.prefix(truncated))


# -- white-box lexer checks --------------------------------------------------- #


def test_lexer_emits_indent_dedent(lexer):
    res = lexer.lex("def g():\n    return 1\n", eof=True)
    assert res.ok
    assert "_INDENT" in res.terminals
    assert "_DEDENT" in res.terminals
    assert res.terminals.count("_INDENT") == res.terminals.count("_DEDENT")


def test_lexer_keyword_vs_name(lexer):
    res = lexer.lex("def define = return", eof=True)
    # 'def'/'return' are hard keywords; 'define' is an identifier.
    assert res.terminals[0] == "DEF"
    assert res.terminals[1] == "NAME"
    assert "RETURN" in res.terminals


def test_lexer_maximal_munch_operators(lexer):
    res = lexer.lex("a == b", eof=True)
    assert "__ANON_19" in res.terminals  # '==' as one token, not two '='
    assert "EQUAL" not in res.terminals


def test_lexer_no_newline_inside_brackets(lexer):
    res = lexer.lex("x = [\n1,\n2,\n]\n", eof=True)
    assert res.ok
    # implicit line joins inside [] -> the only _NEWLINE is the trailing one
    assert res.terminals.count("_NEWLINE") == 1
    assert "_INDENT" not in res.terminals


# -- Potential interface consistency (PotentialTests mixin) ------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["x = 1\n", "def g():\n    return 1\n", "a + b\n"])
async def test_autoregressive_factorization(P, code):
    await P.assert_autoreg_fact(ctx(code))


@pytest.mark.asyncio
async def test_logw_next_consistency(P):
    await P.assert_logw_next_consistency(ctx("x = "))


def test_grammar_build_is_cached():
    a = _build_grammar()
    b = _build_grammar()
    assert a is b  # lru_cache: the expensive Earley preprocessing is shared


# -- lexer memoization: must be identical to lexing from scratch ------------- #

_ALL_STRINGS = VALID_COMPLETE + PREFIX_VALID + INVALID_COMPLETE + PREFIX_INVALID


def test_memoization_matches_fresh_lexing():
    larkstuff, _ = _build_grammar()
    memo = _PythonLexer(larkstuff)  # warmed incrementally across growing prefixes
    fresh = _PythonLexer(larkstuff)  # cleared before each call -> always cold
    for s in _ALL_STRINGS:
        for k in range(len(s) + 1):
            prefix = s[:k]
            for eof in (False, True):
                fresh.clear_cache()
                rf = fresh.lex(prefix, eof)
                rm = memo.lex(prefix, eof)
                assert (rm.ok, rm.terminals, rm.open_classes) == (
                    rf.ok,
                    rf.terminals,
                    rf.open_classes,
                ), (prefix, eof)


# -- oracle: agree with CPython's tokenizer on complete valid programs ------- #


def _our_tags(lexer, src):
    """Coarse content-token tags + INDENT count from our lexer."""
    res = lexer.lex(src, eof=True)
    assert res.ok, src
    content, indents = [], 0
    for t in res.terminals:
        if t == "_INDENT":
            indents += 1
        elif t in ("_DEDENT", "_NEWLINE"):
            pass
        elif t == "NAME":
            content.append("NAME")
        elif "NUMBER" in t:
            content.append("NUMBER")
        elif t in ("STRING", "LONG_STRING"):
            content.append("STRING")
        else:  # keyword literals are identifier-shaped; operators/punct are not.
            content.append(
                "NAME" if lexer._literals.get(t, "").isidentifier() else "OP"
            )
    return content, indents


def _tokenize_tags(src):
    """The same coarse tags from CPython's `tokenize` (the reference lexer)."""
    content, indents = [], 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME:
            content.append("NAME")
        elif tok.type == tokenize.NUMBER:
            content.append("NUMBER")
        elif tok.type == tokenize.STRING:
            content.append("STRING")
        elif tok.type == tokenize.OP:
            content.append("OP")
        elif tok.type == tokenize.INDENT:
            indents += 1
    return content, indents


@pytest.mark.parametrize("code", DS1000_SOLUTIONS + VALID_COMPLETE)
def test_lexer_agrees_with_cpython_tokenize(lexer, code):
    assert _our_tags(lexer, code) == _tokenize_tags(code), code
