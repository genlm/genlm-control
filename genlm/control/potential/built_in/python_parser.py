"""Syntactic-validity potential for Python.

Constrains SMC decoding to byte sequences that parse as Python 3. The potential is
built from two layers:

- **parser**: a genlm-control `BoolCFG` over a *terminal-level* Python grammar
  (Lark's bundled `python.lark`, converted to a `genlm.grammar` CFG). This is the
  "classic grammar potential" applied to lexer-token *classes*
  (`NAME`/`NUMBER`/`STRING`/`_INDENT`/...) rather than to raw bytes.

- **lexer**: `_PythonLexer` (this module), an incremental, maximal-munch lexer that
  maps the byte/character stream to those terminal classes, with an indent stack
  and bracket-depth tracker that emits `_NEWLINE`/`_INDENT`/`_DEDENT`. This is the
  part a character-level CFG cannot express: Lark's `%declare`d indent terminals
  are dropped by the grammar conversion, and a char-level grammar has neither
  maximal munch nor significant-indentation handling.

The potential's vocabulary is the LM's tokens, so it composes (via `Product` / `*`)
with an LM potential or a task critic over the same particle sequence. `prefix`
returns `0` for a byte sequence that can still be extended to valid Python and
`-inf` otherwise; `complete` returns `0` iff the sequence is a complete valid
Python `file_input`.
"""

from __future__ import annotations

import abc
import functools
import keyword
import os
import re
from collections import OrderedDict
from typing import Any

import numpy as np

from genlm.control.constant import EndOfSequence
from genlm.control.potential.base import Potential
from genlm.control.potential.built_in.wcfg import BoolCFG

# A matcher state is an opaque, matcher-specific value (`None` denotes a dead state).
_State = Any


# --------------------------------------------------------------------------- #
# Grammar layer                                                               #
# --------------------------------------------------------------------------- #

# Lark declares the indent terminals as `%declare _INDENT _DEDENT` (normally
# produced by its PythonIndenter postlex). genlm-grammar's LarkStuff runs no
# postlex and Lark's plain compile() drops declared-but-patternless terminals --
# taking the whole `_NEWLINE _INDENT stmt+ _DEDENT` branch of `suite` with them.
# We parse at the terminal level and emit the terminal NAMES ourselves, so the
# pattern is never used for lexing; giving the two terminals concrete sentinel
# patterns simply makes Lark keep them (and the indented-suite rules).
_INDENT_PATCH = ("%declare _INDENT _DEDENT", '_INDENT: "\\x00I"\n_DEDENT: "\\x00D"')


@functools.lru_cache(maxsize=1)
def _lark_python_source() -> str:
    """Reads Lark's bundled `python.lark` and applies the indent-terminal patch.

    Returns:
        The grammar source with a `start: file_input` rule prepended and the
        `%declare`d indent terminals replaced by sentinel-pattern terminals.

    Raises:
        RuntimeError: If the expected `%declare` line is absent (the installed
            Lark grammar changed and the patch needs revisiting).
    """
    import lark

    path = os.path.join(os.path.dirname(lark.__file__), "grammars", "python.lark")
    with open(path) as f:
        body = f.read()
    if _INDENT_PATCH[0] not in body:
        raise RuntimeError(
            "Lark's python.lark no longer contains "
            f"{_INDENT_PATCH[0]!r}; the indent-terminal patch needs updating."
        )
    return "start: file_input\n" + body.replace(*_INDENT_PATCH)


@functools.lru_cache(maxsize=1)
def _build_grammar() -> tuple[Any, Any]:
    """Builds the terminal-level Python grammar.

    The result is cached: converting the grammar and (downstream) preprocessing it
    for Earley parsing is the one expensive step (~1-2s), shared across every
    `PythonParser` instance via the `BoolCFG` built from this CFG.

    Returns:
        A `(LarkStuff, CFG)` pair, where the CFG's terminals are Python lexer-token
        class names and the `LarkStuff` exposes the terminal metadata the lexer needs.
    """
    from genlm.grammar.lark_interface import LarkStuff

    larkstuff = LarkStuff(_lark_python_source())
    return larkstuff, larkstuff.convert()


# --------------------------------------------------------------------------- #
# Lexer layer                                                                 #
# --------------------------------------------------------------------------- #

# PythonIndenter.tab_len: a tab advances the indent column to the next multiple of 8.
_TAB = 8

# How far back to look for a cached lexer checkpoint to resume from. A single decode
# step appends one LM token (a handful of bytes), so the immediately-preceding prefix
# is found within a few characters; beyond this window we re-lex from scratch.
_RESUME_WINDOW = 64

# Default bound on the lexer's prefix -> checkpoint LRU cache.
_LEXER_CACHE_SIZE = 50_000

# interegular mis-compiles the `(?![1-9])` lookahead in python.lark's number
# terminals (it drops the single-digit accept). We strip simple lookarounds before
# compiling; the only effect is over-accepting e.g. `01` (which the grammar rejects
# anyway, as two adjacent number tokens).
_LOOKAROUND = re.compile(r"\(\?[!=][^()]*\)")

# Valid STRING/LONG_STRING prefixes per `([ubf]?r? | r[ubf])` (case-insensitive).
_STR_PREFIXES = frozenset({"", "U", "B", "F", "R", "UR", "BR", "FR", "RU", "RB", "RF"})


def _strip_lookarounds(regex: str) -> str:
    """Removes simple `(?!...)` / `(?=...)` lookarounds from a regex string."""
    return _LOOKAROUND.sub("", regex)


class _Matcher(abc.ABC):
    """Per-terminal incremental recogniser.

    Lets the munch loop drive interegular FSMs and the hand-written string matchers
    uniformly. A state is an opaque value; `None` always denotes a dead state.
    """

    @abc.abstractmethod
    def start(self) -> _State:
        """Returns the initial (pre-input) state."""

    @abc.abstractmethod
    def step(self, state: _State, ch: str) -> _State:
        """Advances `state` by one character; returns `None` if no transition exists."""

    @abc.abstractmethod
    def is_final(self, state: _State) -> bool:
        """Returns whether `state` is accepting (a complete token ends here)."""


class _FSMMatcher(_Matcher):
    """Adapts an interegular FSM to the `_Matcher` interface.

    interegular compiles a terminal's regex into a deterministic automaton exposing
    `.initial`, `.finals`, `.map` (a `{state: {symbol: state}}` transition table),
    and `.alphabet` (which maps a character to its transition symbol). This wrapper
    walks that automaton one character at a time so the munch loop can drive it like
    any other matcher.
    """

    def __init__(self, fsm: Any) -> None:
        """Stores the compiled automaton.

        Args:
            fsm: An `interegular.fsm.FSM` built from a terminal's regex.
        """
        from interegular.fsm import anything_else

        self.fsm = fsm
        self._anything_else = anything_else

    def start(self) -> _State:
        """Returns the automaton's initial state (an integer state id)."""
        return self.fsm.initial

    def step(self, state: _State, ch: str) -> _State:
        """Follows the transition for `ch` out of `state`.

        Args:
            state: The current automaton state, or `None` if already dead.
            ch: The next input character.

        Returns:
            The successor state, or `None` if no transition exists (the recogniser is
            now dead). A character outside the automaton's explicit alphabet is routed
            through interegular's `anything_else` catch-all symbol when the automaton
            defines one (this is how negated character classes such as `[^\\n]` are
            represented).
        """
        if state is None:
            return None
        alphabet = self.fsm.alphabet
        try:
            sym = alphabet[ch]
        except KeyError:
            sym = (
                alphabet[self._anything_else]
                if self._anything_else in alphabet
                else None
            )
        row = self.fsm.map.get(state)
        return None if row is None else row.get(sym)

    def is_final(self, state: _State) -> bool:
        """Returns whether `state` is an accepting state of the automaton."""
        return state in self.fsm.finals


class _StringMatcher(_Matcher):
    """Recogniser for single-line `STRING` literals.

    interegular cannot compile python.lark's STRING regex (its escape lookbehind),
    so the literal is recognised explicitly: an optional prefix, then a `'`/`"`
    body with backslash escapes, closed by the matching unescaped quote; an
    unescaped raw newline kills the match. State is `("p", chars)` while consuming
    the prefix, `("b", quote, escaped)` in the body, and `("f",)` once closed.
    """

    def start(self) -> _State:
        """Returns the initial state `("p", "")`: reading the optional string prefix."""
        return ("p", "")

    def step(self, state: _State, ch: str) -> _State:
        """Advances the single-line string recogniser by one character.

        Args:
            state: The current state, one of: `("p", prefix_chars)` while consuming
                the optional `r`/`b`/`f`/`u` prefix (at most two letters); `("b",
                quote, escaped)` inside the body, where `quote` is the opening `'`/`"`
                and `escaped` flags that the previous character was a backslash; or
                `("f",)` once the closing quote has been consumed.
            ch: The next input character.

        Returns:
            The successor state, or `None` (dead) if `ch` cannot continue a string
            literal: an unrecognised prefix letter, a quote that does not follow a
            valid prefix, an unescaped raw newline inside the body, or any character
            after the literal has already closed.
        """
        if state is None:
            return None
        phase = state[0]
        if phase == "p":
            chars = state[1]
            if ch in "uUbBfFrR" and len(chars) < 2:
                return ("p", chars + ch)
            if ch in "'\"" and chars.upper() in _STR_PREFIXES:
                return ("b", ch, False)
            return None
        if phase == "b":
            quote, escaped = state[1], state[2]
            if escaped:
                return ("b", quote, False)
            if ch == "\\":
                return ("b", quote, True)
            if ch == quote:
                return ("f",)
            if ch in "\r\n":
                return None
            return ("b", quote, False)
        return None

    def is_final(self, state: _State) -> bool:
        """Returns whether the literal has closed (state `("f",)`)."""
        return state is not None and state[0] == "f"


class _LongStringMatcher(_Matcher):
    """Recogniser for triple-quoted `LONG_STRING` literals.

    Like `_StringMatcher`, but the body spans newlines and closes on three
    consecutive unescaped quotes. The body state additionally carries the run of
    trailing quotes seen so far.
    """

    def start(self) -> _State:
        """Returns the initial state `("p", "")`: reading the optional string prefix."""
        return ("p", "")

    def step(self, state: _State, ch: str) -> _State:
        """Advances the triple-quoted string recogniser by one character.

        Args:
            state: The current state, one of: `("p", prefix_chars)` reading the
                optional prefix; `("q1", quote)` / `("q2", quote)` after the first /
                second of the three opening quotes; `("b", quote, escaped, run)` in
                the body, where `run` counts the consecutive closing quotes seen so
                far (0-2); or `("f",)` once all three closing quotes are consumed.
            ch: The next input character.

        Returns:
            The successor state, or `None` (dead) if `ch` cannot continue the literal
            (e.g. a non-quote where the second or third opening quote is expected).
            Unlike a single-line string, raw newlines are allowed in the body.
        """
        if state is None:
            return None
        phase = state[0]
        if phase == "p":
            chars = state[1]
            if ch in "uUbBfFrR" and len(chars) < 2:
                return ("p", chars + ch)
            if ch in "'\"" and chars.upper() in _STR_PREFIXES:
                return ("q1", ch)
            return None
        if phase == "q1":
            return ("q2", state[1]) if ch == state[1] else None
        if phase == "q2":
            return ("b", state[1], False, 0) if ch == state[1] else None
        if phase == "b":
            quote, escaped, run = state[1], state[2], state[3]
            if escaped:
                return ("b", quote, False, 0)
            if ch == "\\":
                return ("b", quote, True, 0)
            if ch == quote:
                return ("f",) if run + 1 == 3 else ("b", quote, False, run + 1)
            return ("b", quote, False, 0)
        return None

    def is_final(self, state: _State) -> bool:
        """Returns whether the literal has closed (state `("f",)`)."""
        return state is not None and state[0] == "f"


class _LexResult:
    """Outcome of lexing a (prefix or complete) character string.

    Attributes:
        ok: `False` if a hard lexical or indentation error was hit (no valid Python
            can have this string as a prefix).
        terminals: Committed terminal-class names, excluding any open trailing token.
        open_classes: Terminal classes the trailing in-progress token could still
            become (empty when the trailing boundary is clean). Only meaningful for
            prefix lexing.
    """

    __slots__ = ("ok", "terminals", "open_classes")

    def __init__(self, ok: bool, terminals: list[str], open_classes: set[str]) -> None:
        """Stores a lex outcome.

        Args:
            ok: Whether lexing succeeded (no hard lexical/indentation error).
            terminals: The committed terminal-class names.
            open_classes: Candidate classes for the trailing open token (prefix mode).
        """
        self.ok = ok
        self.terminals = terminals
        self.open_classes = open_classes


class _LexCheckpoint:
    """A resumable lexer state at a clean token boundary.

    The state at `pos` is fully determined by the input up to `pos`, so lexing a
    longer string that shares this prefix can resume here instead of restarting.
    Terminals and the indent stack are held as immutable tuples so the cache entry
    is never aliased by a caller's mutable working lists.

    Attributes:
        pos: The character index this state was captured at.
        terminals: Committed terminal-class names up to `pos`.
        indent: The indent-column stack at `pos`.
        depth: The bracket-nesting depth at `pos`.
        line_start: Whether `pos` begins a logical line (indentation pending).
    """

    __slots__ = ("pos", "terminals", "indent", "depth", "line_start")

    def __init__(
        self,
        pos: int,
        terminals: tuple[str, ...],
        indent: tuple[int, ...],
        depth: int,
        line_start: bool,
    ) -> None:
        """Stores a resumable lexer state; see the class attributes.

        Args:
            pos: The character index this state was captured at.
            terminals: Committed terminal-class names up to `pos` (immutable).
            indent: The indent-column stack at `pos` (immutable).
            depth: The bracket-nesting depth at `pos`.
            line_start: Whether `pos` begins a logical line.
        """
        self.pos = pos
        self.terminals = terminals
        self.indent = indent
        self.depth = depth
        self.line_start = line_start


class _PythonLexer:
    """Incremental maximal-munch lexer over the patched `python.lark` terminals.

    `lex` is a pure function of its input string. For speed it memoises a checkpoint
    at the trailing token boundary of every prefix it lexes, so extending a string
    by one decode step re-scans only the short tail rather than the whole prefix
    (turning a trajectory's lexing cost from O(L^2) into O(L) amortised). The cache
    is an LRU keyed by the input prefix and can be dropped with `clear_cache`.
    """

    def __init__(self, larkstuff: Any, cache_size: int = _LEXER_CACHE_SIZE) -> None:
        """Compiles a recogniser for every terminal of `larkstuff`.

        Args:
            larkstuff: The `LarkStuff` returned by `_build_grammar`, providing the
                terminal definitions, ignore terminals, and literal values.
            cache_size: Maximum number of prefix checkpoints to retain (LRU).
        """
        import interegular
        from lark.lexer import PatternStr

        self._cache: OrderedDict[str, _LexCheckpoint] = OrderedDict()
        self._cache_size = cache_size

        self._matchers: dict[str, _Matcher] = {}
        self._literals: dict[str, str] = {}
        for terminal in larkstuff.terminals:
            name = terminal.name
            if name in ("_INDENT", "_DEDENT"):
                continue  # Synthetic: emitted from the indent stack, never lexed.
            if name == "STRING":
                self._matchers[name] = _StringMatcher()
            elif name == "LONG_STRING":
                self._matchers[name] = _LongStringMatcher()
            else:
                regex = _strip_lookarounds(terminal.pattern.to_regexp())
                self._matchers[name] = _FSMMatcher(
                    interegular.parse_pattern(regex).to_fsm()
                )
            if isinstance(terminal.pattern, PatternStr):
                self._literals[name] = terminal.pattern.value
        # Literal value -> terminal name, for hard-keyword and operator resolution.
        self._value_to_name = {v: n for n, v in self._literals.items()}
        # Lark's `ignore` terminals (whitespace / comment) are skipped, not emitted.
        self._ignore = set(larkstuff.ignore_terms)
        # True/False/None are in kwlist; the soft keywords match/case are not.
        self._hard_kw = set(keyword.kwlist)

    def _classify(self, lexeme: str, finals: list[str]) -> str:
        """Resolves the terminal class of a fully-matched lexeme.

        Args:
            lexeme: The matched text.
            finals: The terminal names whose recogniser accepts `lexeme`.

        Returns:
            The single terminal class to emit. Identifier-shaped lexemes resolve to
            the keyword terminal when they are a hard keyword and to `NAME`
            otherwise; symbolic lexemes resolve to their exact literal terminal.
        """
        if "NAME" in finals:
            if lexeme in self._hard_kw:
                return self._value_to_name.get(lexeme, "NAME")
            return "NAME"
        exact = self._value_to_name.get(lexeme)
        if exact in finals:
            return exact
        return sorted(finals)[0]  # Number / string / comment regex terminals.

    def _munch(self, text: str, i: int, eof: bool) -> tuple[str, Any]:
        """Matches the single longest token starting at `text[i]`.

        Args:
            text: The full input string.
            i: The index at which the token starts.
            eof: Whether `text` is treated as complete (no further input can arrive).

        Returns:
            A `(kind, payload)` pair, one of:
            `("commit", (terminal_name, next_index))` for a matched token,
            `("open", open_classes)` when input ran out mid-token and `eof` is False, or
            `("error", None)` when no valid token starts here.
        """
        n = len(text)
        states = {name: m.start() for name, m in self._matchers.items()}
        last_final_pos = None
        last_finals: list[str] | None = None
        j = i
        while j < n:
            ch = text[j]
            nxt = {}
            for name, state in states.items():
                advanced = self._matchers[name].step(state, ch)
                if advanced is not None:
                    nxt[name] = advanced
            if not nxt:
                break
            states = nxt
            j += 1
            fin = [n for n, s in states.items() if self._matchers[n].is_final(s)]
            if fin:
                last_final_pos, last_finals = j, fin
        if j == n and not eof:
            # Trailing in-progress token: the classes it could still become. Any
            # non-dead recogniser can extend, and any currently-final one counts too.
            alive = set(states)
            if last_finals:
                alive.update(last_finals)
            return ("open", alive)
        if last_final_pos is None:
            return ("error", None)
        lexeme = text[i:last_final_pos]
        return ("commit", (self._classify(lexeme, last_finals), last_final_pos))

    def lex(self, text: str, eof: bool) -> _LexResult:
        """Lexes `text` into terminal-class names.

        Args:
            text: The input string.
            eof: If True, `text` is treated as a complete program: the trailing
                token is force-committed, a closing `_NEWLINE` is synthesised when
                the last logical line is unterminated, and the indent stack is
                flushed with `_DEDENT`s. If False, the trailing token is left open
                for prefix checking.

        Returns:
            A `_LexResult` describing the committed terminals and (for `eof=False`)
            the open trailing token's candidate classes.
        """
        n = len(text)
        resume = self._resume(text)
        if resume is None:
            terminals: list[str] = []
            indent = [0]
            depth = 0  # () [] {} nesting; newlines inside are implicit line joins.
            line_start = True  # At a logical-line start (indentation pending).
            i = 0
        else:
            terminals = list(resume.terminals)
            indent = list(resume.indent)
            depth = resume.depth
            line_start = resume.line_start
            i = resume.pos
        open_classes: set[str] = set()
        # The most recent loop-top state: the clean boundary to checkpoint at. What
        # is processed after it (an open token, pending indentation, an unterminated
        # comment, trailing whitespace) can grow with more input, so a resume must
        # restart *before* it, never after.
        safe = (i, len(terminals), tuple(indent), depth, line_start)

        while i < n:
            safe = (i, len(terminals), tuple(indent), depth, line_start)
            if line_start and depth == 0:
                i, line_start, ok = self._handle_indent(text, i, indent, terminals)
                if not ok:
                    return _LexResult(False, terminals, set())
                continue
            a = text[i]
            if a in " \t\f":
                i += 1
                continue
            if a == "\\" and i + 1 < n and text[i + 1] in "\r\n":  # Line continuation.
                i += 3 if text[i + 1 : i + 3] == "\r\n" else 2
                continue
            if a in "\r\n":
                i += 2 if text[i : i + 2] == "\r\n" else 1
                if depth == 0:
                    terminals.append("_NEWLINE")
                    line_start = True
                continue
            if a == "#":
                while i < n and text[i] not in "\r\n":
                    i += 1
                continue
            kind, _munch_output = self._munch(text, i, eof)
            if kind == "error":
                return _LexResult(False, terminals, set())
            if kind == "open":
                open_classes = _munch_output
                break
            name, next_i = _munch_output
            if name in self._ignore:
                i = next_i
                continue
            terminals.append(name)
            if name in ("LPAR", "LSQB", "LBRACE"):
                depth += 1
            elif name in ("RPAR", "RSQB", "RBRACE"):
                depth = max(0, depth - 1)
            i = next_i

        # Cache the clean boundary for the next (longer) call. Only `eof=False` is
        # cached: it is the incremental hot path, and the checkpoint (a loop-top
        # state, before any eof-only synthetic terminals) is reusable by both
        # `prefix` and `complete`.
        if not eof:
            s_i, s_m, s_indent, s_depth, s_line_start = safe
            self._store(
                text,
                _LexCheckpoint(
                    s_i, tuple(terminals[:s_m]), s_indent, s_depth, s_line_start
                ),
            )

        if eof:
            if depth > 0:
                return _LexResult(False, terminals, set())
            if terminals and terminals[-1] != "_NEWLINE":
                terminals.append("_NEWLINE")
            while len(indent) > 1:
                terminals.append("_DEDENT")
                indent.pop()
        return _LexResult(True, terminals, open_classes)

    def _resume(self, text: str) -> _LexCheckpoint | None:
        """Finds a cached checkpoint to resume lexing `text` from.

        Checks `text` itself and its prefixes down to `_RESUME_WINDOW` characters
        shorter (the most a single decode step can have appended), longest first,
        and promotes the hit to most-recently-used.

        Args:
            text: The string about to be lexed.

        Returns:
            The checkpoint of the longest cached prefix of `text`, or `None` if no
            prefix within the window is cached (lexing then starts from scratch).
        """
        for k in range(len(text), max(-1, len(text) - _RESUME_WINDOW) - 1, -1):
            key = text[:k]
            checkpoint = self._cache.get(key)
            if checkpoint is not None:
                self._cache.move_to_end(key)
                return checkpoint
        return None

    def _store(self, text: str, checkpoint: _LexCheckpoint) -> None:
        """Caches `checkpoint` under `text`, evicting the least-recently-used entry
        once the cache exceeds `cache_size`.

        Args:
            text: The lexed string (the cache key).
            checkpoint: The resumable boundary state captured while lexing it.
        """
        self._cache[text] = checkpoint
        self._cache.move_to_end(text)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Drops all memoised prefix checkpoints (e.g. to bound memory)."""
        self._cache.clear()

    def _handle_indent(
        self, text: str, i: int, indent: list[int], terminals: list[str]
    ) -> tuple[int, bool, bool]:
        """Processes the leading whitespace of a logical line.

        Blank and comment-only lines are skipped without touching the indent stack
        (Lark's `_NEWLINE+` collapses them). A real line emits `_INDENT` or the
        matching run of `_DEDENT`s. Mutates `indent` and `terminals` in place.

        Args:
            text: The input string.
            i: The index of the first character of the line.
            indent: The indent-column stack (mutated).
            terminals: The committed terminal list (mutated).

        Returns:
            A `(next_index, line_start, ok)` triple. `ok` is False when the line's
            indentation matches no enclosing level (an indentation error).
        """
        n = len(text)
        col = 0
        j = i
        while j < n and text[j] in " \t":
            col += (_TAB - col % _TAB) if text[j] == "\t" else 1
            j += 1
        if j >= n or text[j] in "\r\n" or text[j] == "#":
            while j < n and text[j] not in "\r\n":  # Consume any comment body.
                j += 1
            if j < n:  # Consume the newline; the next line is still a line start.
                j += 2 if text[j : j + 2] == "\r\n" else 1
            return j, True, True
        if col > indent[-1]:
            terminals.append("_INDENT")
            indent.append(col)
        else:
            while col < indent[-1]:
                terminals.append("_DEDENT")
                indent.pop()
            if col != indent[-1]:
                return j, False, False
        return j, False, True


# --------------------------------------------------------------------------- #
# Potential                                                                   #
# --------------------------------------------------------------------------- #


class PythonParser(Potential):
    """Potential that scores a token sequence by Python syntactic validity.

    `prefix(context)` is `0` if the decoded bytes can still be extended to a valid
    Python `file_input` and `-inf` otherwise; `complete(context)` is `0` iff the
    decoded bytes are a complete valid `file_input`.

    The vocabulary is the LM's token vocabulary, so the potential composes with an
    LM or task critic over the same particle sequence (e.g.
    `Product(python_parser, critic)`). Tokens are decoded with `b"".join` + UTF-8,
    mirroring the DS-1000 code critic.
    """

    def __init__(self, vocabulary: list, eos: EndOfSequence | None = None) -> None:
        """Initialises the potential.

        Args:
            vocabulary: The token vocabulary (typically an LM's tokens, as bytes).
            eos: The end-of-sequence sentinel; defaults to the global `EOS`.
        """
        larkstuff, cfg = _build_grammar()
        self._parser = BoolCFG(cfg)
        self._lexer = _PythonLexer(larkstuff)
        super().__init__(vocabulary, eos=eos)

    @classmethod
    def from_llm(cls, llm: Any) -> "PythonParser":
        """Builds a `PythonParser` over the vocabulary of a `PromptedLLM`.

        Args:
            llm: The `PromptedLLM` whose vocabulary and EOS to adopt.

        Returns:
            A `PythonParser` sharing `llm`'s token vocabulary.
        """
        return cls(list(llm.vocab), eos=llm.eos)

    def _decode(self, context: list) -> tuple[str | None, bool]:
        """Decodes a token context to text.

        Args:
            context: A sequence of vocabulary tokens, possibly EOS-terminated.

        Returns:
            A `(text, ok)` pair. `ok` is False when the bytes do not form valid
            UTF-8 (a context that splits a multi-byte character), in which case
            `text` is None and the context should be treated as a valid prefix.
        """
        chunks = [t for t in context if not isinstance(t, EndOfSequence)]
        try:
            return b"".join(bytes(t) for t in chunks).decode("utf-8"), True
        except UnicodeDecodeError:
            return None, False

    async def prefix(self, context: list) -> float:
        """Scores `context` as a prefix: `0` if it can extend to valid Python.

        Decodes the tokens to text, lexes it with the trailing token left open, and
        accepts iff (a) the committed terminals are a valid grammar prefix and (b)
        the open trailing token can still resolve to some terminal the grammar allows
        next. Step (b) uses `logw_next` rather than `prefix` because it yields the
        allowed-next weight of every terminal in one Earley query, which we intersect
        with the open token's candidate classes (a token is viable if *any* class it
        could still become is grammatically allowed).

        Args:
            context: The token sequence to score (LM-vocabulary tokens, possibly
                EOS-terminated).

        Returns:
            `0.0` if `context` is a viable prefix of some valid Python `file_input`,
            else `float("-inf")`. A context that splits a multi-byte UTF-8 character
            decodes as incomplete and is treated as a valid prefix.
        """
        text, ok = self._decode(context)
        if not ok:
            return 0.0
        result = self._lexer.lex(text, eof=False)
        if not result.ok:
            return float("-inf")
        if await self._parser.prefix(result.terminals) == float("-inf"):
            return float("-inf")
        if not result.open_classes:
            return 0.0
        # The open trailing token is viable only if some class it could still become
        # is an allowed next terminal for the committed prefix.
        logw = await self._parser.logw_next(result.terminals)
        if any(np.isfinite(logw[t]) for t in result.open_classes):
            return 0.0
        return float("-inf")

    async def complete(self, context: list) -> float:
        """Scores `context` as a complete program: `0` iff it is valid Python.

        Decodes the tokens and lexes with `eof=True` (force-committing the trailing
        token and flushing the closing `_NEWLINE`/`_DEDENT`s), then defers the verdict
        to the grammar's `complete`.

        Args:
            context: The token sequence to score (LM-vocabulary tokens, possibly
                EOS-terminated).

        Returns:
            `0.0` if the decoded text is a complete valid Python `file_input`, else
            `float("-inf")` (including when the bytes are not valid UTF-8).
        """
        text, ok = self._decode(context)
        if not ok:
            return float("-inf")
        result = self._lexer.lex(text, eof=True)
        if not result.ok:
            return float("-inf")
        return await self._parser.complete(result.terminals)

    def clear_cache(self) -> None:
        """Drops the lexer and parser caches (e.g. to bound memory between E-steps)."""
        self._lexer.clear_cache()
        self._parser.clear_cache()

    def spawn(self) -> "PythonParser":
        """Returns a fresh instance with the same vocabulary (for multiprocessing)."""
        return PythonParser(self.vocab, eos=self.eos)

    def __repr__(self) -> str:
        return f"PythonParser(vocab_size={len(self.vocab)})"
