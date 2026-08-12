from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.sampler.token import TokenSampler
from genlm.control.util import flatten_units
from lark import Lark
from lark.exceptions import LarkError


class MultiTokenUnitSampler(TokenSampler):
    """Sampler that groups multiple tokens into larger units.

    This sampler enables generation at a coarser granularity than individual tokens
    by repeatedly sampling tokens until a boundary condition is met. Common use cases:

    - **Word-level sampling**: Group tokens until a word boundary (e.g., whitespace)
    - **Sentence-level sampling**: Group tokens until punctuation marks
    - **Grammar-based units**: Group tokens completing a grammar terminal

    The sampler delegates to a `subunit_sampler` (typically a token-level sampler)
    and accumulates samples until the `boundary_predicate` signals completion. The final
    weight is the product of weights from each individual token sample. This ensures that
    sampling remains properly weighted w.r.t. the target potential.

    **Weight calculation**: If sampling a unit requires $n$ token samples with weights
    $w_1, w_2, \\ldots, w_n$, the unit weight is $w = \\prod_{i=1}^{n} w_i$ (or
    $\\log w = \\sum_{i=1}^{n} \\log w_i$ in log-space).

    Args:
        subunit_sampler (TokenSampler): Sampler for subunits $s \\in \\mathcal{B}$
        boundary_predicate (BoundaryPredicate): Determines when a sequence of tokens forms
            a complete unit. Also controls how to finalize the unit via `finalize_unit()`.
        max_subunits_per_unit (int): Safety timeout to prevent non-termination. Default: 100.

    Example:
        >>> # Sample word-level units (multi-token)
        >>> llm = PromptedLLM.from_name("gpt2")
        >>> subunit_sampler = DirectTokenSampler(llm)
        >>>
        >>> # Word boundaries at whitespace
        >>> boundary = TokenSetBoundary({b" ", b"\\n"})
        >>> unit_sampler = MultiTokenUnitSampler(
        ...     subunit_sampler=subunit_sampler,
        ...     boundary_predicate=boundary,
        ...     max_subunits_per_unit=50
        ... )
        >>> # Units will be words WITH trailing space: [b"hello", b" "]
    """

    def __init__(
        self,
        subunit_sampler,
        boundary_predicate,
        max_subunits_per_unit=100,
    ):
        if not isinstance(subunit_sampler, TokenSampler):
            raise TypeError(
                f"subunit_sampler must be a TokenSampler, got {type(subunit_sampler)}"
            )
        if max_subunits_per_unit < 1:
            raise ValueError(
                f"max_subunits_per_unit must be >= 1, got {max_subunits_per_unit}"
            )

        # Initialized with subunit sampler's target
        # We may want to add support for different samplers in the future
        super().__init__(target=subunit_sampler.target)

        self.subunit_sampler = subunit_sampler
        self.boundary_predicate = boundary_predicate
        self.max_subunits_per_unit = max_subunits_per_unit

    async def start_weight(self):
        """Return $\\overrightarrow{\\psi}(\\epsilon)$ (prefix weight of empty sequence)."""
        return await self.subunit_sampler.start_weight()

    async def logw_eos(self, context):
        """EOS log-weight at the ``max_tokens`` boundary: defer to the subunit sampler
        on the flattened context."""
        return await self.subunit_sampler.logw_eos(flatten_units(context))

    async def sample(self, context, draw=None):
        """Sample a multi-token unit by running sequence sampling for $\\varphi_{\\bm{x}}$.
        SIS for the localized potential:

        1. Repeatedly sample $(s_i, w_i) \\sim q_{\\text{sub}}(\\cdot \\mid \\bm{s}_{<i})$ until boundary
        2. Accumulate weights: $w = \\overrightarrow{\\psi}_{\\bm{x}}(\\epsilon) \\prod_i w_i$
        3. Return $(\\bm{s}, w)$ where $\\bm{s} \\in \\mathcal{B}^*$ forms unit $x \\in \\mathcal{A}$

        Args:
            context (list): The particle's structured (possibly nested) unit context.
                It is flattened here so the subunit sampler sees a flat token list;
                the structured form feeds the boundary predicate.
            draw (callable, optional): Sampling function passed to subunit_sampler

        Returns:
            (unit, weight, logp):
                - unit: List of subunits $[s_1, \\ldots, s_n]$ forming $x \\in \\mathcal{A}$
                - weight: Importance weight $w$ such that $(\\text{unit}, w)$ is properly
                    weighted w.r.t. $\\psi(x \\mid \\bm{x})$
                - logp: Sum of log-probabilities of sampling choices
        """
        flat_context = flatten_units(context)

        buffer, logw, logp = [], 0.0, 0.0
        for _ in range(self.max_subunits_per_unit):
            subunit, logw_i, logp_i = await self.subunit_sampler.sample(
                flat_context, draw=draw
            )
            flat_context.append(subunit)
            buffer.append(subunit)
            logw += logw_i
            logp += logp_i
            if subunit is EOS:
                return buffer, logw, logp
            if self.boundary_predicate(context, buffer):
                return self.boundary_predicate.finalize_unit(buffer), logw, logp
        # max subunits without a boundary: reject the unit.
        return buffer, float("-inf"), logp

    async def cleanup(self):
        """Clean up resources."""
        await self.subunit_sampler.cleanup()


class BoundaryPredicate(ABC):
    """Abstract base class for boundary predicates.

    A boundary predicate determines when a sequence of subunits $\\bm{s} \\in \\mathcal{B}^*$
    forms a complete unit $x \\in \\mathcal{A}$.

    `__call__` method receives unit context and subunit buffer, allowing predicates
    to be stateless and context-aware.

    `finalize_unit` method transforms the buffer into the final unit after boundary
    detection, allowing predicates to control what tokens are included (e.g., removing
    delimiter tokens).
    """

    @abstractmethod
    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        """Check if subunit buffer forms a complete unit.

        Args:
            unit_context (list): Sequence of completed units $\\bm{x} \\in \\mathcal{A}^*$
            subunit_buffer (list): Current sequence of subunits $\\bm{s} \\in \\mathcal{B}^*$

        Returns:
            bool: True if $\\bm{s}$ forms a complete unit $x \\in \\mathcal{A}$
        """
        pass  # pragma: no cover

    def finalize_unit(self, subunit_buffer: list) -> list:
        """Transform buffer into final unit after boundary detected.

        Called after `__call__` returns True. Override to customize which tokens
        are included in the final unit (e.g., to remove delimiter tokens).

        Args:
            subunit_buffer (list): The buffer that triggered the boundary

        Returns:
            list: The final unit to return

        Note:
            Default implementation returns the entire buffer unchanged.
        """
        return subunit_buffer


class TokenSetBoundary(BoundaryPredicate):
    """Stateless boundary predicate based on token membership.

    A unit is complete when the last subunit is in a specified set of boundary tokens.

    Args:
        boundary_tokens: Set or iterable of tokens that mark unit boundaries

    Example:
        >>> boundary = TokenSetBoundary({b" ", b"\\n"})
        >>> boundary([], [b"hello", b" "])  # True (ends with whitespace)
        >>> # Unit will be [b"hello", b" "] - boundary token included
    """

    def __init__(self, boundary_tokens: Iterable):
        self.boundary_tokens = set(boundary_tokens)
        # Match by byte content, not hash-set membership: a real-LLM ``Token``
        # (bytes subclass hashing by token_id) has ``Token(13, b" ") in {b" "}``
        # False despite matching bytes, so the boundary would silently never fire
        # on that grain. Precompute a plain-bytes set (``bytes(t)`` works for both
        # ``Token`` and ``bytes``) plus an EOS flag (EOS matches by identity).
        self._eos_boundary = any(
            isinstance(t, EndOfSequence) for t in self.boundary_tokens
        )
        self._byte_boundaries = {
            bytes(t) for t in self.boundary_tokens if not isinstance(t, EndOfSequence)
        }

    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        """Check boundary (ignore unit_context for stateless predicate). Matches by
        byte content so it fires identically on ``bytes`` and real-LLM ``Token``
        subunits; EOS is matched by identity."""
        if not subunit_buffer:
            return False
        last = subunit_buffer[-1]
        if isinstance(last, EndOfSequence):
            return self._eos_boundary
        return bytes(last) in self._byte_boundaries

    def __repr__(self) -> str:
        return f"TokenSetBoundary({self.boundary_tokens!r})"


class FixedLengthBoundary(BoundaryPredicate):
    """Stateless boundary predicate based on fixed unit length.
    A unit is complete when it reaches a specified number of subunits.

    Args:
        length (int): Number of subunits per unit

    Example:
        >>> boundary = FixedLengthBoundary(10)
        >>> boundary([], [b"a"] * 9)   # False
        >>> boundary([], [b"a"] * 10)  # True
    """

    def __init__(self, length: int):
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        self.length = length

    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        """Check boundary (ignores unit_context for stateless predicate)."""
        return len(subunit_buffer) >= self.length

    def __repr__(self) -> str:
        return f"FixedLengthBoundary({self.length})"


class CFGBoundary(BoundaryPredicate):
    """Boundary predicate using Lark parser for context-free grammar-based boundaries.

    This uses Lark's parser to determine when a sequence of subunits forms a
    syntactically complete unit according to a context-free grammar.

    A unit can be marked as complete when:
    - The subunit buffer parses successfully
    - The parse tree's root matches one of the complete_rules

    Args:
        grammar_text (str): Lark grammar specification
        start_rule (str): Starting rule for parsing (default: "start")
        complete_rules (set or None): Set of rule names that constitute complete units.
                                      If None, any successful parse is complete.
                                      If provided, only parses with matching root are complete.
        min_length (int): Minimum buffer length before attempting to parse (default: 2)
        parser_type (str): Lark parser type: 'earley' (default, supports ambiguity) or 'lalr' (faster)
        ambiguity (str): How to handle ambiguous grammars: 'explicit' (default) or 'resolve'
        encoding (str): Text encoding for token decoding (default: "utf-8")
        decode_errors (str): How to handle decode errors (default: "ignore")

    Example:
        >>> # Simple arithmetic grammar
        >>> grammar = '''
        ...     start: expr
        ...     expr: term | expr "+" term
        ...     term: NUMBER
        ...     NUMBER: /[0-9]+/
        ... '''
        >>> boundary = CFGBoundary(grammar, complete_rules={"start"})
        >>> boundary([], [b"1", b"+", b"2"])  # True (complete expression)
        >>> boundary([], [b"1", b"+"])        # False (incomplete)
    """

    def __init__(
        self,
        grammar_text,
        start_rule="start",
        complete_rules=None,
        min_length=2,
        parser_type="earley",
        ambiguity="explicit",
        encoding="utf-8",
        decode_errors="ignore",
    ):
        self.grammar_text = grammar_text
        self.start_rule = start_rule
        self.complete_rules = set(complete_rules) if complete_rules else None
        self.min_length = min_length
        self.encoding = encoding
        self.decode_errors = decode_errors
        try:
            if parser_type == "earley":
                self.parser = Lark(
                    grammar_text,
                    start=start_rule,
                    parser=parser_type,
                    ambiguity=ambiguity,
                )
            else:
                self.parser = Lark(grammar_text, start=start_rule, parser=parser_type)
        except Exception as e:
            raise ValueError(f"Failed to create Lark parser: {e}") from e

    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        """Check if buffer forms a complete syntactic unit.

        Args:
            unit_context: Previous completed units (ignored)
            subunit_buffer: Current sequence of subunits to check

        Returns:
            bool: True if buffer parses successfully and meets criteria
        """
        if not subunit_buffer or len(subunit_buffer) < self.min_length:
            return False

        try:
            text = self._tokens_to_text(subunit_buffer)

            if not text or len(text) < self.min_length:
                return False

            tree = self.parser.parse(text)
            if self.complete_rules is None:
                return True

            root_rule = tree.data
            return root_rule in self.complete_rules

        except LarkError:
            # Parse failed: not a complete unit
            return False

    def _tokens_to_text(self, tokens: list) -> str:
        """Convert token buffer to text string.

        Args:
            tokens: List of tokens (bytes objects or lists)

        Returns:
            str: Decoded text
        """
        # Join byte tokens, filtering out EOS
        token_bytes = b"".join(
            t for t in tokens if isinstance(t, bytes) and t is not EOS
        )
        return token_bytes.decode(self.encoding, errors=self.decode_errors)

    def get_parse_tree(self, text: str) -> Optional[Any]:
        """Get the parse tree for a given text.

        Args:
            text (str): String to parse

        Returns:
            Lark Tree object or None if parsing fails
        """
        try:
            return self.parser.parse(text)
        except LarkError:
            return None

    def __repr__(self) -> str:
        rules_str = (
            f", complete_rules={self.complete_rules}" if self.complete_rules else ""
        )
        return f"CFGBoundary(start={self.start_rule!r}{rules_str})"
