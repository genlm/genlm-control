from abc import ABC, abstractmethod

from genlm.control.constant import EOS
from genlm.control.sampler.token import TokenSampler
from lark import Lark
from lark.exceptions import LarkError


def flatten_units(context):
    """
    Flatten nested unit context to a flat token list. When using MultiTokenUnitSampler, token_ctx becomes nested [[...], [...], ...].
    This helper flattens it for use with coercion functions like b"".join.

    Usage:
        potential.coerce(LLM, f=lambda ctx: b"".join(flatten_units(ctx)))
    Args:
        context: Either a flat list [token1, token2, ...] or nested [[token1, token2], [token3], ...]
    Returns:
        list: Flattened list of tokens
    """
    flattened = []
    for item in context:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


class MultiTokenUnitSampler(TokenSampler):
    """Unit sampler for multi-token units $x \\in \\mathcal{A}$ where $\\mathcal{A} \\subseteq \\mathcal{B}^*$.
    implements unit sampling by running a sequence sampler for localized potential:
    $\\varphi_{\\bm{x}} = (\\psi_{\\bm{x}}, \\overrightarrow{\\psi}_{\\bm{x}})$ over subunits $\\mathcal{B}$.
    Args:
        subunit_sampler (TokenSampler): Sampler for subunits $s \\in \\mathcal{B}$
        boundary_predicate (callable): Function for determining EOT
        max_subunits_per_unit (int): Safety timeout to prevent non-termination. Default: 100.
        include_boundary_in_unit (bool): Whether to include the boundary token
            in the returned unit. Default: True.

    Example:
        >>> # Sample word-level units (multi-token)
        >>> llm = PromptedLLM.from_name("gpt2")
        >>> subunit_sampler = DirectTokenSampler(llm)
        >>> def word_boundary(buf):
                return buf and buf[-1] in {b" ", b"\\n"}
        >>> unit_sampler = MultiTokenUnitSampler(
        ...     subunit_sampler=subunit_sampler,
        ...     boundary_predicate=word_boundary,
        ...     max_subunits_per_unit=50
        ... )
    """

    def __init__(
        self,
        subunit_sampler,
        boundary_predicate,
        max_subunits_per_unit=100,
        include_boundary_in_unit=True,
    ):
        if not isinstance(subunit_sampler, TokenSampler):
            raise TypeError(
                f"subunit_sampler must be a TokenSampler, got {type(subunit_sampler)}"
            )

        # TODO: Currently init with subunit sampler's target
        super().__init__(target=subunit_sampler.target)

        self.subunit_sampler = subunit_sampler
        self.boundary_predicate = boundary_predicate
        self.max_subunits_per_unit = max_subunits_per_unit
        self.include_boundary_in_unit = include_boundary_in_unit

    async def start_weight(self):
        """Return $\\overrightarrow{\\psi}(\\epsilon)$ (prefix weight of empty sequence)."""
        return await self.subunit_sampler.start_weight()

    async def forward(self):
        """Called by LLaMPPL Model.call() to sample one multi-token unit.

        Called by SequenceModel.step() when it calls self.call(unit_sampler).
        """
        parent = self.parent

        # Flatten parent.token_ctx before passing to sample
        # This ensures sample() always works with a flat list
        flat_context = self._flatten_to_subunits(parent.token_ctx)

        # Sample multi-token unit using flat context
        unit, logw, logp = await self.sample(flat_context, draw=None)

        # Update parent's weight and logp
        parent.score(logw)
        parent.logp += logp

        # If the unit ends with EOS, return EOS directly so SequenceModel can detect completion
        # SequenceModel.step() checks `token_ctx[-1] is EOS` to finish generation
        if unit and unit[-1] is EOS:
            # Keep the unit content before EOS in token_ctx, then return EOS separately
            if len(unit) > 1:
                parent.token_ctx.append(unit[:-1])  # Add unit without EOS
            return EOS  # Return EOS directly for SequenceModel to detect

        return unit

    async def sample(self, flat_token_context, draw=None):
        """Sample a multi-token unit by running sequence sampling for $\\varphi_{\\bm{x}}$.
        SIS for the localized potential:

        1. Repeatedly sample $(s_i, w_i) \\sim q_{\\text{sub}}(\\cdot \\mid \\bm{s}_{<i})$ until boundary
        2. Accumulate weights: $w = \\overrightarrow{\\psi}_{\\bm{x}}(\\epsilon) \\prod_i w_i$
        3. Return $(\\bm{s}, w)$ where $\\bm{s} \\in \\mathcal{B}^*$ forms unit $x \\in \\mathcal{A}$

        Args:
            flat_token_context (list): Flat sequence of all previously sampled tokens.
                This is pre-flattened by forward() to ensure compatibility with potentials.
            draw (callable, optional): Sampling function passed to subunit_sampler

        Returns:
            (unit, weight, logp):
                - unit: List of subunits $[s_1, \\ldots, s_n]$ forming $x \\in \\mathcal{A}$
                - weight: Importance weight $w$ such that $(\\text{unit}, w)$ is properly
                    weighted w.r.t. $\\psi(x \\mid \\bm{x})$
                - logp: Sum of log-probabilities of sampling choices
        """
        subunit_context = flat_token_context
        subunit_buffer = []

        # Accumulate weights
        cumulative_logw = 0.0
        cumulative_logp = 0.0

        # Sequential sampling until EOT
        for _ in range(self.max_subunits_per_unit):
            # Sample next subunit $(s_i, w_i) \\sim q_{\\text{sub}}(\\cdot \\mid \\bm{s}_{<i})$
            full_context = subunit_context + subunit_buffer
            try:
                subunit, logw_i, logp_i = await self.subunit_sampler.sample(
                    full_context, draw
                )
            except Exception:
                return subunit_buffer, float("-inf"), cumulative_logp
            # Accumulate weight and logp
            cumulative_logw += logw_i
            cumulative_logp += logp_i
            subunit_buffer.append(subunit)

            # Check for EOS
            if subunit is EOS:
                return subunit_buffer, cumulative_logw, cumulative_logp

            # Check boundary: is $\\bm{s} \\in \\mathcal{A}$ (complete unit)?
            if self.boundary_predicate(flat_token_context, subunit_buffer):
                unit = subunit_buffer
                # Exclude boundary token from unit if requested
                if not self.include_boundary_in_unit and unit:
                    unit = unit[:-1]
                return unit, cumulative_logw, cumulative_logp

        return subunit_buffer, cumulative_logw, cumulative_logp

    def _flatten_to_subunits(self, unit_context):
        """Convert unit sequence $\\bm{x} \\in \\mathcal{A}^*$ to subunit sequence $\\in \\mathcal{B}^*$.

        Args:
            unit_context (list): List of units, where each unit is either:
                - A list of subunits (for multi-token units)
                - A single token (for single-token units)

        Returns:
            list: Flattened sequence of subunits
        """
        subunits = []
        for unit in unit_context:
            if isinstance(unit, list):
                # Multi-token unit: extend with all subunits
                subunits.extend(unit)
            else:
                # Single-token unit: append directly
                subunits.append(unit)
        return subunits

    async def cleanup(self):
        """Clean up resources."""
        await self.subunit_sampler.cleanup()


class BoundaryPredicate(ABC):
    """Abstract base class for boundary predicates.

    A boundary predicate determines when a sequence of subunits $\\bm{s} \\in \\mathcal{B}^*$
    forms a complete unit $x \\in \\mathcal{A}$.

    `__call__` method receives unit context and subunit buffer, allowing predicates
    to be stateless and context-aware
    """

    @abstractmethod
    def __call__(self, unit_context, subunit_buffer):
        """Check if subunit buffer forms a complete unit.

        Args:
            unit_context (list): Sequence of completed units $\\bm{x} \\in \\mathcal{A}^*$
            subunit_buffer (list): Current sequence of subunits $\\bm{s} \\in \\mathcal{B}^*$

        Returns:
            bool: True if $\\bm{s}$ forms a complete unit $x \\in \\mathcal{A}$
        """
        pass  # pragma: no cover


class TokenSetBoundary(BoundaryPredicate):
    """Stateless boundary predicate based on token membership.

    A unit is complete when the last subunit is in a specified set of boundary tokens.

    Args:
        boundary_tokens: Set or iterable of tokens that mark unit boundaries

    Example:
        >>> boundary = TokenSetBoundary({b" ", b"\\n", b".", b","})
        >>> boundary([], [b"hello", b" "])  # True (ends with whitespace)
        >>> boundary([], [b"hello"])         # False (no boundary token)
    """

    def __init__(self, boundary_tokens):
        self.boundary_tokens = set(boundary_tokens)

    def __call__(self, unit_context, subunit_buffer):
        """Check boundary (ignore unit_context for stateless predicate)."""
        return subunit_buffer and subunit_buffer[-1] in self.boundary_tokens

    def __repr__(self):
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

    def __init__(self, length):
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        self.length = length

    def __call__(self, unit_context, subunit_buffer):
        """Check boundary (ignores unit_context for stateless predicate)."""
        return len(subunit_buffer) >= self.length

    def __repr__(self):
        return f"FixedLengthBoundary({self.length})"


def boundary_token_set(tokens):
    """Create a boundary predicate from a set of boundary tokens.

    Args:
        tokens: Set or iterable of tokens that mark unit boundaries

    Returns:
        TokenSetBoundary: Boundary predicate instance

    Example:
        >>> boundary = boundary_token_set({b" ", b"\\n", b".", b","})
        >>> boundary([b"hello", b" "])  # True (ends with whitespace)
    """
    return TokenSetBoundary(tokens)


def boundary_fixed_length(length):
    """Create a boundary predicate for fixed-length units.

    Args:
        length (int): Number of subunits per unit

    Returns:
        FixedLengthBoundary: Boundary predicate instance

    Example:
        >>> boundary = boundary_fixed_length(10)
        >>> boundary([b"a"] * 9)   # False
        >>> boundary([b"a"] * 10)  # True
    """
    return FixedLengthBoundary(length)


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

    def __call__(self, unit_context, subunit_buffer):
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
            return False  # not a complete unit
        except Exception:
            return False  # catch other errors

    def _tokens_to_text(self, tokens):
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

    def get_parse_tree(self, text):
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

    def __repr__(self):
        rules_str = (
            f", complete_rules={self.complete_rules}" if self.complete_rules else ""
        )
        return f"CFGBoundary(start={self.start_rule!r}{rules_str})"
