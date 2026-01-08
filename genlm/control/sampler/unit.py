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
        boundary_predicate (BoundaryPredicate): Predicate for determining unit completion.
            Controls both when a unit is complete and how to finalize it via `finalize_unit()`.
        max_subunits_per_unit (int): Safety timeout to prevent non-termination. Default: 100.

    Example:
        >>> # Sample word-level units (multi-token)
        >>> llm = PromptedLLM.from_name("gpt2")
        >>> subunit_sampler = DirectTokenSampler(llm)
        >>>
        >>> # Include boundary tokens (default - preserves context)
        >>> boundary = TokenSetBoundary({b" ", b"\\n"})
        >>> unit_sampler = MultiTokenUnitSampler(
        ...     subunit_sampler=subunit_sampler,
        ...     boundary_predicate=boundary,
        ...     max_subunits_per_unit=50
        ... )
        >>> # Units will be words WITH trailing space: [b"hello", b" "]
        >>>
        >>> # Exclude boundary tokens if you want clean semantic units
        >>> boundary = TokenSetBoundary({b" "}, include_boundary=False)
        >>> # Units will be words WITHOUT space: [b"hello"]
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

        # Initialized with subunit sampler's target
        # We may want to add support for different samplers in the future
        super().__init__(target=subunit_sampler.target)

        self.subunit_sampler = subunit_sampler
        self.boundary_predicate = boundary_predicate
        self.max_subunits_per_unit = max_subunits_per_unit

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
        flat_context = flatten_units(parent.token_ctx)

        # Sample multi-token unit, passing both flat context and structured unit context
        unit, logw, logp = await self.sample(
            flat_context, unit_context=parent.token_ctx, draw=None
        )

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

    async def sample(self, flat_token_context, unit_context=None, draw=None):
        """Sample a multi-token unit by running sequence sampling for $\\varphi_{\\bm{x}}$.
        SIS for the localized potential:

        1. Repeatedly sample $(s_i, w_i) \\sim q_{\\text{sub}}(\\cdot \\mid \\bm{s}_{<i})$ until boundary
        2. Accumulate weights: $w = \\overrightarrow{\\psi}_{\\bm{x}}(\\epsilon) \\prod_i w_i$
        3. Return $(\\bm{s}, w)$ where $\\bm{s} \\in \\mathcal{B}^*$ forms unit $x \\in \\mathcal{A}$

        Args:
            flat_token_context (list): Flat sequence of all previously sampled tokens.
                This is pre-flattened by forward() to ensure compatibility with potentials.
            unit_context (list, optional): Structured sequence of previously sampled units.
                Used by boundary predicates that need context. Defaults to [].
            draw (callable, optional): Sampling function passed to subunit_sampler

        Returns:
            (unit, weight, logp):
                - unit: List of subunits $[s_1, \\ldots, s_n]$ forming $x \\in \\mathcal{A}$
                - weight: Importance weight $w$ such that $(\\text{unit}, w)$ is properly
                    weighted w.r.t. $\\psi(x \\mid \\bm{x})$
                - logp: Sum of log-probabilities of sampling choices
        """
        if unit_context is None:
            unit_context = []

        subunit_buffer = []
        current_context = list(flat_token_context)

        # Accumulate weights
        cumulative_logw = 0.0
        cumulative_logp = 0.0

        # Sequential sampling until EOT
        for _ in range(self.max_subunits_per_unit):
            # Sample next subunit $(s_i, w_i) \\sim q_{\\text{sub}}(\\cdot \\mid \\bm{s}_{<i})$
            try:
                subunit, logw_i, logp_i = await self.subunit_sampler.sample(
                    current_context, draw
                )
            except (RuntimeError, OSError, TimeoutError):
                # Expected failures (network, timeout, system errors)
                # Return current buffer with -inf weight to discard this sample
                return subunit_buffer, float("-inf"), cumulative_logp

            # Accumulate weight and logp
            cumulative_logw += logw_i
            cumulative_logp += logp_i

            # Add to both buffer and context
            subunit_buffer.append(subunit)
            current_context.append(subunit)

            # Check for EOS
            if subunit is EOS:
                return subunit_buffer, cumulative_logw, cumulative_logp

            # Check boundary: is $\\bm{s} \\in \\mathcal{A}$ (complete unit)?
            if self.boundary_predicate(unit_context, subunit_buffer):
                # Let the predicate finalize the unit (e.g., remove delimiter tokens)
                unit = self.boundary_predicate.finalize_unit(subunit_buffer)
                return unit, cumulative_logw, cumulative_logp

        # Max subunits exceeded: we return -inf weight to reject incomplete/invalid unit
        return subunit_buffer, float("-inf"), cumulative_logp

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
    def __call__(self, unit_context, subunit_buffer):
        """Check if subunit buffer forms a complete unit.

        Args:
            unit_context (list): Sequence of completed units $\\bm{x} \\in \\mathcal{A}^*$
            subunit_buffer (list): Current sequence of subunits $\\bm{s} \\in \\mathcal{B}^*$

        Returns:
            bool: True if $\\bm{s}$ forms a complete unit $x \\in \\mathcal{A}$
        """
        pass  # pragma: no cover

    def finalize_unit(self, subunit_buffer):
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
        include_boundary: Whether to include the boundary token in the final unit.
            Default: True (keeps boundary token for proper context conditioning)

    Example:
        >>> # Include boundary (default - keeps context for next unit)
        >>> boundary = TokenSetBoundary({b" ", b"\\n"})
        >>> boundary([], [b"hello", b" "])  # True (ends with whitespace)
        >>> # After finalize_unit: [b"hello", b" "]
        >>>
        >>> # Exclude boundary (for semantic units without delimiters)
        >>> boundary = TokenSetBoundary({b" "}, include_boundary=False)
        >>> # After finalize_unit: [b"hello"]
    """

    def __init__(self, boundary_tokens, include_boundary=True):
        self.boundary_tokens = set(boundary_tokens)
        self.include_boundary = include_boundary

    def __call__(self, unit_context, subunit_buffer):
        """Check boundary (ignore unit_context for stateless predicate)."""
        return subunit_buffer and subunit_buffer[-1] in self.boundary_tokens

    def finalize_unit(self, subunit_buffer):
        """Finalize the unit, optionally removing the boundary token.

        Args:
            subunit_buffer (list): Buffer ending with a boundary token

        Returns:
            list: Buffer with or without the boundary token, based on include_boundary
        """
        if self.include_boundary or not subunit_buffer:
            return subunit_buffer
        return subunit_buffer[:-1]

    def __repr__(self):
        if not self.include_boundary:
            return f"TokenSetBoundary({self.boundary_tokens!r}, include_boundary=False)"
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
            # Parse failed: not a complete unit
            return False

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
