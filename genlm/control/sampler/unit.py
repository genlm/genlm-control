import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.sampler.token import TokenSampler
from genlm.control.sampler.burst import BurstDraw
from genlm.control.potential.base import burst_logw_next
from genlm.control.util import draw_key, draw_ordinal
from lark import Lark
from lark.exceptions import LarkError


@dataclass
class _UnitAccum:
    """Per-row, per-burst accumulator for an in-progress unit: the subunits drawn
    so far this unit round and their summed importance weight / log-prob.

    Lives in the burst's ``scratch`` dict keyed by the row's handle, and is discarded
    the moment the unit completes (one unit per burst, so nothing carries across)."""

    buffer: list
    logw: float
    logp: float


def flatten_units(context):
    """Flatten a (possibly nested) unit context to a flat token list. A
    MultiTokenUnitSampler nests units as sub-lists, and a nested unit sampler nests
    deeper -- so this recurses (matching the engine-prompt flatten in
    ``_Burst.context_ids``); a one-level flatten would leave deeper nesting for the
    subunit sampler / coercion to choke on.

    Usage:
        potential.coerce(LLM, f=lambda ctx: b"".join(flatten_units(ctx)))
    """
    flattened = []
    for item in context:
        if isinstance(item, list):
            flattened.extend(flatten_units(item))
        else:
            flattened.append(item)
    return flattened


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

    def supports_burst(self) -> bool:
        # Rides the fast lane iff its subunit sampler can (the subunits ARE the burst draws).
        return self.subunit_sampler.supports_burst()

    def burst_draw_sampler(self):
        # The subunit does the per-step draw, so recurse to its draw sampler.
        return self.subunit_sampler.burst_draw_sampler()

    def burst_free_running(self) -> bool:
        # Synchronized (unit grain): one SMC step is one whole unit, with ESS tested once
        # per unit round at the synced boundary (not per subunit decode step).
        return False

    def burst_max_steps(self, live) -> int:
        # One unit's worth of subunit decode steps (+1 margin); the control-side reject at
        # ``max_subunits_per_unit`` fires before this engine cap.
        return self.max_subunits_per_unit + 1

    async def burst_draw_batch(self, warm_batch, contexts, handles, burst):
        """Burst draw for one unit round: one subunit per decode step, completing an
        SMC step only at the unit boundary. Per particle, slice the batched warm
        (``{view: [N, V+1] LazyWeights}``) into a per-row injection and run the subunit
        sampler's REAL ``sample`` (one subunit); accumulate it in ``burst.scratch`` (keyed by
        particle ``handle``); then apply the SAME EOS-split / boundary / max-subunit logic as
        the slow ``sample``/``transition``:

        * EOS subunit -> split the content off so ``context[-1]`` is EOS (terminate).
        * boundary fires -> finalize the unit; the row pops out at the synced boundary.
        * max subunits without a boundary -> reject (``-inf`` weight, finishes).
        * otherwise -> mid-unit: emit the subunit, bank nothing (``step=None``)."""
        accums = burst.scratch  # handle -> _UnitAccum, fresh each burst

        async def one(i, context, handle):
            injection = self._row_injection(warm_batch, i)
            accum = accums.get(handle)
            buf = accum.buffer if accum is not None else []
            # Subunit context: completed units flattened to subunits + in-progress
            # subunits this unit, so the subunit sampler's factor scores statelessly.
            sub_context = flatten_units(context) + list(buf)
            # ordinal = subunits committed (leaf count) + those drawn this unit so far,
            # matching the slow loop's per-subunit ordinal under one transition scope.
            with burst_logw_next(injection), draw_key(handle, draw_ordinal(context) + len(buf)):
                subunit, sub_logw, sub_logp = await self.subunit_sampler.sample(sub_context)

            if accum is None:
                accum = accums[handle] = _UnitAccum([], 0.0, 0.0)
            status, unit = self._feed(accum, subunit, sub_logw, sub_logp, context)
            if status == "mid":
                return BurstDraw(token=subunit, step=None, pop=False)  # emit subunit, bank nothing
            accums.pop(handle, None)
            weight = float("-inf") if status == "max" else accum.logw
            step = (self._to_append(unit), weight, accum.logp)
            return BurstDraw(token=subunit, step=step, pop=status == "boundary")

        return await asyncio.gather(
            *(one(i, c, h) for i, (c, h) in enumerate(zip(contexts, handles)))
        )

    async def start_weight(self):
        """Return $\\overrightarrow{\\psi}(\\epsilon)$ (prefix weight of empty sequence)."""
        return await self.subunit_sampler.start_weight()

    async def transition(self, context):
        """The controller-facing per-step transition for multi-token units.

        Samples one multi-token unit and returns the list of items to append to
        the particle context, the importance-weight increment, and the
        log-probability of the random choices.

        The context passed by the controller is the structured (possibly nested) unit
        context; it is flattened before sampling so the subunit sampler sees a flat token
        list. The trailing-EOS split (when a unit ends with EOS) is done by ``_to_append``.

        Args:
            context (list): The particle's structured unit context.

        Returns:
            (to_append, logw, logp): items to extend the context with, the
                weight increment, and the log-probability of random choices.
        """
        flat_context = flatten_units(context)
        unit, logw, logp = await self.sample(
            flat_context, unit_context=context, draw=None
        )
        return self._to_append(unit), logw, logp

    @staticmethod
    def _to_append(unit):
        """Controller ``to_append`` from a completed unit, shared by the slow
        ``transition`` and the burst: if the unit ends with EOS, split the content off
        and append EOS separately so ``context[-1] is EOS`` (the terminal check fires);
        otherwise the unit is a single item."""
        if unit and unit[-1] is EOS:
            return ([unit[:-1]] if len(unit) > 1 else []) + [EOS]
        return [unit]

    def _feed(self, accum, subunit, sub_logw, sub_logp, unit_context):
        """Accumulate one subunit into ``accum`` and classify the unit -- the single
        per-subunit step shared by the slow loop and the burst (they differ only in how
        the subunit is drawn). Returns ``(status, unit)``: ``status`` in
        ``{"eos","boundary","max","mid"}``; ``unit`` is the completed unit (``None``
        mid-unit) -- the raw buffer for eos/max, the finalized unit for a boundary."""
        accum.buffer.append(subunit)
        accum.logw += sub_logw
        accum.logp += sub_logp
        if subunit is EOS:
            return "eos", accum.buffer
        if self.boundary_predicate(unit_context, accum.buffer):
            return "boundary", self.boundary_predicate.finalize_unit(accum.buffer)
        if len(accum.buffer) >= self.max_subunits_per_unit:
            return "max", accum.buffer
        return "mid", None

    async def sample(self, flat_token_context, unit_context=None, draw=None):
        """Sample a multi-token unit by running sequence sampling for $\\varphi_{\\bm{x}}$.
        SIS for the localized potential:

        1. Repeatedly sample $(s_i, w_i) \\sim q_{\\text{sub}}(\\cdot \\mid \\bm{s}_{<i})$ until boundary
        2. Accumulate weights: $w = \\overrightarrow{\\psi}_{\\bm{x}}(\\epsilon) \\prod_i w_i$
        3. Return $(\\bm{s}, w)$ where $\\bm{s} \\in \\mathcal{B}^*$ forms unit $x \\in \\mathcal{A}$

        Args:
            flat_token_context (list): Flat sequence of all previously sampled tokens.
                This is pre-flattened by transition() to ensure compatibility with potentials.
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

        accum = _UnitAccum([], 0.0, 0.0)
        current_context = list(flat_token_context)

        # Draw subunits until the unit completes; ``_feed`` (shared with the burst)
        # accumulates each and decides eos / boundary / max.
        for _ in range(self.max_subunits_per_unit):
            try:
                subunit, logw_i, logp_i = await self.subunit_sampler.sample(
                    current_context, draw
                )
            except (RuntimeError, OSError, TimeoutError):
                # Expected failures (network/timeout/system): reject with -inf weight.
                return accum.buffer, float("-inf"), accum.logp

            current_context.append(subunit)
            status, unit = self._feed(accum, subunit, logw_i, logp_i, unit_context)
            if status == "mid":
                continue
            weight = float("-inf") if status == "max" else accum.logw
            return unit, weight, accum.logp
        # No fall-through: max_subunits_per_unit >= 1 (checked in __init__), so the last
        # iteration always returns via _feed's "max" branch.

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
        # Compare by BYTE CONTENT, not by hash-set membership. A real-LLM token is a
        # ``Token`` (subclass of ``bytes`` that hashes by ``token_id``), so
        # ``Token(13, b" ") in {b" "}`` is False even though the bytes match -- the
        # boundary would silently never fire on the real-LLM grain. Precompute a
        # plain-bytes set (``bytes(t)`` is the content for both ``Token`` and
        # ``bytes``) plus an EOS flag (EOS is matched by identity, having no bytes).
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
