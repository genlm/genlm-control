import inspect
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

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

    **Weight- and critic-aware boundaries**: the predicate receives the running unit
    log-weight via a ``cumulative_logw`` keyword, so boundaries can react to the
    incremental importance weights (see `SurpriseBoundary`). Predicates may also be
    ``async`` to ``await`` a critic potential mid-unit (see `CriticBoundary`). Both are
    opt-in; the original ``(unit_context, subunit_buffer)`` signature keeps working.

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

        # Initialized with subunit sampler's target
        # We may want to add support for different samplers in the future
        super().__init__(target=subunit_sampler.target)

        self.subunit_sampler = subunit_sampler
        self.boundary_predicate = boundary_predicate
        self.max_subunits_per_unit = max_subunits_per_unit

        # Whether to forward the running unit log-weight to the predicate. Cached
        # here so we don't re-introspect the signature on every subunit.
        self._boundary_accepts_cumulative_logw = (
            self._predicate_accepts_cumulative_logw(boundary_predicate)
        )

    @staticmethod
    def _predicate_accepts_cumulative_logw(predicate) -> bool:
        """Whether ``predicate.__call__`` can receive a ``cumulative_logw`` keyword.

        True if it declares ``**kwargs`` or a ``cumulative_logw`` parameter. Predicates
        using the original ``(unit_context, subunit_buffer)`` signature return False, so
        the sampler falls back to the two-argument call (backward compatible).
        """
        try:
            sig = inspect.signature(predicate.__call__)
        except (TypeError, ValueError):  # pragma: no cover - C/builtin callables
            return False  # can't introspect; be conservative and don't pass it
        for param in sig.parameters.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return True
            if param.name == "cumulative_logw" and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False

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
            # Opt-in predicates also receive the running unit log-weight; async
            # predicates (e.g. ones that query a critic) return a coroutine we await.
            if self._boundary_accepts_cumulative_logw:
                is_boundary = self.boundary_predicate(
                    unit_context, subunit_buffer, cumulative_logw=cumulative_logw
                )
            else:
                is_boundary = self.boundary_predicate(unit_context, subunit_buffer)

            if inspect.iscoroutine(is_boundary):
                is_boundary = await is_boundary

            if is_boundary:
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

    `__call__` receives unit context and subunit buffer plus keyword signals from the
    sampler (currently ``cumulative_logw``); subclasses should accept ``**kwargs`` to
    stay forward-compatible. The original two-argument signature still works — the
    sampler detects it and omits the extra keyword.

    `__call__` may be sync (returning ``bool``) or ``async`` (returning an awaitable
    ``bool``); `MultiTokenUnitSampler` awaits coroutine results, so a predicate can
    ``await`` a critic potential mid-unit before deciding.

    `finalize_unit` method transforms the buffer into the final unit after boundary
    detection, allowing predicates to control what tokens are included (e.g., removing
    delimiter tokens).
    """

    @abstractmethod
    def __call__(self, unit_context: list, subunit_buffer: list, **kwargs) -> bool:
        """Check if subunit buffer forms a complete unit.

        Args:
            unit_context (list): Sequence of completed units $\\bm{x} \\in \\mathcal{A}^*$
            subunit_buffer (list): Current sequence of subunits $\\bm{s} \\in \\mathcal{B}^*$
            **kwargs: Extra signals supplied by `MultiTokenUnitSampler`. Currently
                includes ``cumulative_logw`` (float): the running log-weight
                $\\sum_i \\log w_i$ accumulated over the subunits sampled so far for
                the *current* unit. Predicates that don't need it should ignore it
                via ``**kwargs``.

        Returns:
            bool: True if $\\bm{s}$ forms a complete unit $x \\in \\mathcal{A}$. May also
                return an awaitable resolving to ``bool`` (see class docstring).
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

    def __call__(self, unit_context: list, subunit_buffer: list, **kwargs) -> bool:
        """Check boundary (ignore unit_context for stateless predicate)."""
        return bool(subunit_buffer and subunit_buffer[-1] in self.boundary_tokens)

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

    def __call__(self, unit_context: list, subunit_buffer: list, **kwargs) -> bool:
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

    def __call__(self, unit_context: list, subunit_buffer: list, **kwargs) -> bool:
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


class _ThresholdBoundary(BoundaryPredicate):
    """Base for boundaries that fire when a scalar signal crosses a threshold.

    Concrete subclasses implement `__call__` and obtain their signal differently
    (the cheap ``cumulative_logw``, or an expensive critic delta), but share the
    argument validation, the length-based gating (`_gate`), and the signed
    threshold test (`_crosses`).

    Args:
        threshold (float): Signal magnitude that fires a boundary. Must be > 0.
        min_subunits (int): Minimum subunits before firing.
        max_subunits (int): Force a boundary after this many subunits.
        sign (str): ``"abs"`` fires on ``|signal| >= threshold``, ``"down"`` on
            ``signal <= -threshold``, ``"up"`` on ``signal >= +threshold``.
    """

    def __init__(self, threshold, min_subunits, max_subunits, sign):
        if sign not in ("abs", "down", "up"):
            raise ValueError(f"sign must be 'abs', 'down', or 'up', got {sign!r}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if min_subunits < 1:
            raise ValueError(f"min_subunits must be >= 1, got {min_subunits}")
        if max_subunits < min_subunits:
            raise ValueError(
                f"max_subunits ({max_subunits}) must be >= min_subunits "
                f"({min_subunits})"
            )
        self.threshold = threshold
        self.min_subunits = min_subunits
        self.max_subunits = max_subunits
        self.sign = sign

    def _gate(self, n_subunits):
        """Decide from length alone: True (force), False (too short), or None
        (defer to the signal)."""
        if n_subunits >= self.max_subunits:
            return True
        if n_subunits < self.min_subunits:
            return False
        return None

    def _crosses(self, value):
        """Whether ``value`` crosses the threshold in ``sign``'s direction. NaN and
        non-numeric values never cross."""
        if not isinstance(value, (int, float)) or value != value:
            return False
        if self.sign == "abs":
            return abs(value) >= self.threshold
        if self.sign == "down":
            return value <= -self.threshold
        return value >= self.threshold  # sign == "up"


class SurpriseBoundary(_ThresholdBoundary):
    """Boundary placed when the unit's accumulated log-weight crosses a threshold.

    Reacts to the incremental importance weights (``cumulative_logw``) rather than the
    surface tokens: a large deviation means the target potential is informative about
    the current tokens, so a boundary here lets downstream SMC resample where the
    weight signal is strongest. Domain-agnostic and requires no pilot run.

    The ``sign`` controls which direction counts, which matters when the weight is
    one-sided: grammar-only weights are bounded above by 0, so ``sign="down"`` fires
    only on accumulated penalties and ignores positive noise.

    Args:
        threshold (float): Log-weight magnitude that fires a boundary. Default 1.5. > 0.
        min_subunits (int): Minimum subunits before firing. Default 1.
        max_subunits (int): Force a boundary after this many subunits. Default 50.
        sign (str): ``"abs"`` (default) fires on ``|cumulative_logw| >= threshold``,
            ``"down"`` on ``cumulative_logw <= -threshold``, ``"up"`` on
            ``cumulative_logw >= +threshold``.

    Example:
        >>> boundary = SurpriseBoundary(threshold=1.0, sign="down", min_subunits=2)
        >>> boundary([], [b"a", b"b"], cumulative_logw=2.0)   # wrong direction
        False
        >>> boundary([], [b"a", b"b"], cumulative_logw=-1.2)  # penalty crossed
        True
    """

    def __init__(self, threshold=1.5, min_subunits=1, max_subunits=50, sign="abs"):
        super().__init__(threshold, min_subunits, max_subunits, sign)

    def __call__(self, unit_context, subunit_buffer, **kwargs):
        gate = self._gate(len(subunit_buffer))
        if gate is not None:
            return gate
        return self._crosses(kwargs.get("cumulative_logw", 0.0))

    def __repr__(self):
        return (
            f"SurpriseBoundary(threshold={self.threshold}, sign={self.sign!r}, "
            f"min_subunits={self.min_subunits}, max_subunits={self.max_subunits})"
        )


class CriticBoundary(_ThresholdBoundary):
    """Async boundary placed on a critic potential's per-unit log-weight change.

    Reacts to an expensive ``critic`` rather than the surface tokens or the cheap
    subunit weight. At each subunit it measures how much the critic's prefix score has
    moved over the *current* (in-progress) unit::

        delta = critic.prefix(ctx + buffer) - critic.prefix(ctx)

    where ``ctx`` is the flattened completed `unit_context`. The baseline is derived
    from ``unit_context`` (cached per unit), so the predicate is stateless across SMC
    resampling. Fires when ``delta`` crosses ``threshold`` per ``sign``; a critic
    returning ``-inf`` (dead particle) fires under ``"abs"``/``"down"``.

    ``__call__`` is ``async`` and requires the sampler's await support.

    Args:
        critic: a `Potential` whose ``prefix(context)`` returns a float log-weight.
            Called once per subunit, so it must be safe to evaluate mid-stream.
        threshold, min_subunits, max_subunits, sign: as in `SurpriseBoundary`.
        coalesce_grammar (bool): if True, add the subunit-side ``cumulative_logw`` to
            the critic delta before thresholding. Default False (critic delta only).
    """

    def __init__(
        self,
        critic,
        threshold=1.5,
        min_subunits=1,
        max_subunits=50,
        sign="abs",
        coalesce_grammar=False,
    ):
        super().__init__(threshold, min_subunits, max_subunits, sign)
        self.critic = critic
        self.coalesce_grammar = coalesce_grammar
        # Per-unit baseline critic value, keyed by a hash of the completed context.
        # Cleared by reset(); grows with the number of distinct contexts seen.
        self._baseline_cache = {}

    @staticmethod
    def _critic_context(tokens):
        """Flatten units and drop EOS for feeding the critic's ``prefix``."""
        return [t for t in flatten_units(tokens) if t is not EOS]

    def reset(self):
        """Clear cached baselines. Call between independent generations."""
        self._baseline_cache.clear()

    async def _baseline(self, flat_ctx):
        try:
            key = hash(tuple(flat_ctx))
        except TypeError:  # pragma: no cover - non-hashable tokens
            key = hash(tuple(repr(t) for t in flat_ctx))
        if key not in self._baseline_cache:
            self._baseline_cache[key] = float(await self.critic.prefix(flat_ctx))
        return self._baseline_cache[key]

    async def __call__(self, unit_context, subunit_buffer, **kwargs):
        gate = self._gate(len(subunit_buffer))
        if gate is not None:
            return gate

        baseline_ctx = self._critic_context(unit_context)
        current_ctx = baseline_ctx + [t for t in subunit_buffer if t is not EOS]
        try:
            base = await self._baseline(baseline_ctx)
            delta = float(await self.critic.prefix(current_ctx)) - base
        except Exception:
            # Critic failure: don't fire, don't poison the baseline cache.
            return False

        if self.coalesce_grammar:
            cw = kwargs.get("cumulative_logw", 0.0)
            if isinstance(cw, (int, float)) and cw == cw:  # ignore NaN/non-numeric
                delta += cw

        return self._crosses(delta)

    def __repr__(self):
        return (
            f"CriticBoundary(threshold={self.threshold}, sign={self.sign!r}, "
            f"coalesce_grammar={self.coalesce_grammar}, "
            f"min_subunits={self.min_subunits}, max_subunits={self.max_subunits})"
        )
