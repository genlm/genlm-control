import string
import warnings

import numpy as np
from arsenal.maths import logsumexp

from genlm.grammar import Float, Log, WFSA as BaseWFSA
from genlm.grammar.semiring import Boolean
from genlm.grammar.lark_interface import interegular_to_wfsa

from genlm.control.potential.base import Potential


def _float_to_boolean(wfsa):
    """Convert a Float-semiring WFSA to Boolean: every non-zero weight
    becomes ``Boolean.one``. Same accepted language."""
    assert wfsa.R is Float
    new = BaseWFSA(Boolean)
    for i, w in wfsa.I:
        if w != 0:
            new.add_I(i, Boolean.one)
    for i, w in wfsa.F:
        if w != 0:
            new.add_F(i, Boolean.one)
    for i, a, j, w in wfsa.arcs():
        if w != 0:
            new.add_arc(i, a, j, Boolean.one)
    return new


class WFSA(Potential):
    """
    A weighted finite state automaton (WFSA) potential.

    This class wraps a `genlm_grammar.WFSA` and provides methods for computing the log-weight of a context,
    the prefix log-weight of a context, and the log-weights of the next token given a context.

    Attributes:
        wfsa (genlm_grammar.WFSA): The weighted finite state automaton used for potential calculations.
    """

    def __init__(self, wfsa):
        """
        Initializes the WFSA potential.

        Args:
            wfsa (genlm_grammar.WFSA): The weighted finite state automaton.

        Raises:
            ValueError: If the semiring of the provided WFSA is not Float or Log.

        Note:
            The WFSA will be converted to the Log semiring to avoid underflow if the semiring is Float.
        """
        if wfsa.R not in (Float, Log):
            raise ValueError(f"Unsupported semiring: {wfsa.R}")

        if wfsa.R is Float:
            self.wfsa = self._convert_to_log(wfsa)
        else:
            self.wfsa = wfsa

        self.cache = {(): self.wfsa.epsremove.start}
        super().__init__(vocabulary=list(self.wfsa.alphabet))

    @classmethod
    def from_regex(cls, pattern, charset=None, to_bytes=True):
        """
        Create a WFSA from a regex pattern.

        Args:
            pattern (str): The regex pattern to convert into a WFSA.
            charset (set): The character set to use for negative character classes.
                Defaults to characters in string.printable.
            to_bytes (bool): Whether to convert the WFSA transitions to bytes.
                Defaults to True. When set to False, the WFSA transitions will be strings.

        Returns:
            (WFSA): An instance of the WFSA class.

        Note:
            The transition weights are automatically normalized to form a probability distribution.
            For each state, the weights of all outgoing transitions (including final state transitions)
            sum to 1.0. This means if a state has n possible transitions, each transition will have
            weight 1/n. To create a WFSA from a regex with non-probabilistic transitions, use `BoolFSA`.
        """
        charset = charset or set(string.printable)
        wfsa = interegular_to_wfsa(pattern, charset=charset)
        if to_bytes:
            wfsa = wfsa.to_bytes()
        return cls(wfsa=wfsa)

    @staticmethod
    def _convert_to_log(wfsa):
        """Convert a WFSA from the Float semiring to the Log semiring."""
        assert wfsa.R is Float
        assert isinstance(wfsa, BaseWFSA)
        new = BaseWFSA(Log)

        for i, w in wfsa.I:
            new.add_I(i, Log(np.log(w)))

        for i, w in wfsa.F:
            new.add_F(i, Log(np.log(w)))

        for i, a, j, w in wfsa.arcs():
            new.add_arc(i, a, j, Log(np.log(w)))

        return new

    def _consume(self, bs):
        # XXX implement cache eviction
        bs = tuple(bs)

        try:
            return self.cache[bs]
        except KeyError:
            pass

        wfsa = self.wfsa.epsremove
        curr = wfsa.R.chart()
        prev = self._consume(bs[:-1])
        for i in prev:
            for j, w in wfsa.arcs(i, bs[-1]):
                curr[j] += prev[i] * w

        self.cache[bs] = curr

        return curr

    async def complete(self, context):
        """
        Computes the log weight of the context under the weighted language represented by the WFSA.

        For example, if the WFSA accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `complete("c")` returns $-\\infty$ since this sequence is not accepted by the WFSA\n
        - `complete("cat")` returns $\\log(w_{cat})$\n
        - `complete("d")` returns $-\\infty$ since this sequence is not accepted by the WFSA

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): Log weight of context under the WFSA.
        """
        # TODO: optimize to use _consume cache
        return self.wfsa(context).score

    def _prefix(self, context):
        curr = self._consume(context)

        if not curr:
            return float("-inf"), curr

        bkwd = self.wfsa.epsremove.backward
        log_ctx_w = logsumexp([(curr[i] * bkwd[i]).score for i in curr])

        if np.isnan(log_ctx_w):
            return float("-inf"), curr

        return log_ctx_w, curr

    async def prefix(self, context):
        """
        Computes the prefix log weight of `context` under the WFSA.

        This corresponds to the log of the sum of the weights of all sequences with prefix `context`.

        For example, if the WFSA accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `prefix("c")` returns $\\log(w_{cat} + w_{car})$\n
        - `prefix("ca")` returns $\\log(w_{cat})$\n
        - `prefix("d")` returns $-\\infty$ since the WFSA does not accept any sequences with prefix "d"

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): Log weight of `context` as a prefix under the WFSA.
        """
        return self._prefix(context)[0]

    async def logw_next(self, context):
        """Returns next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (LazyWeights): Log-weights for next token and EOS.
        """
        log_ctx_w, curr = self._prefix(context)

        if log_ctx_w == float("-inf"):
            raise ValueError(f"Context {context!r} has zero weight.")

        bkwd = self.wfsa.epsremove.backward

        ws = self.wfsa.R.chart()
        for i in curr:
            for b, j, w in self.wfsa.epsremove.arcs(i=i):
                ws[b] += curr[i] * w * bkwd[j]

        ws[self.eos] = self.wfsa.R.zero
        for j, w in self.wfsa.epsremove.F:
            ws[self.eos] += curr[j] * w

        log_ws = np.array([ws[b].score for b in self.vocab_eos]) - log_ctx_w

        return self.make_lazy_weights(log_ws)

    def _repr_svg_(self):
        return self.wfsa._repr_svg_()

    def __repr__(self):
        return f"WFSA(wfsa={self.wfsa!r})"

    def spawn(self):
        cls = type(self)
        return cls(wfsa=self.wfsa)

    def clear_cache(self):
        self.cache = {(): self.wfsa.epsremove.start}


class BoolFSA(WFSA):
    """Boolean FSA potential. Returns ``0`` for accepted, ``-inf`` for rejected.

    ``from_regex`` constructs a Boolean-semiring WFSA by default; the
    ``semiring="log"`` path is deprecated (the Log-semiring FSA closure is
    unsound because ``Log.star`` is partial).
    """

    def __init__(self, wfsa):
        """Initialize the BoolFSA from a WFSA.

        Args:
            wfsa (genlm_grammar.WFSA): A Float, Log, or Boolean WFSA.
                Float is converted to Log; Boolean is kept as-is.
        """
        if wfsa.R is Boolean:
            self.wfsa = wfsa
            self.cache = {(): self.wfsa.epsremove.start}
            Potential.__init__(self, vocabulary=list(self.wfsa.alphabet))
        else:
            super().__init__(wfsa)

    @classmethod
    def from_regex(cls, pattern, charset=None, to_bytes=True, semiring="boolean"):
        """Create a BoolFSA from a regex pattern.

        Args:
            pattern (str): regex pattern.
            charset (set): see ``WFSA.from_regex``.
            to_bytes (bool): see ``WFSA.from_regex``.
            semiring (str): ``"boolean"`` (default) or ``"log"``. ``"log"``
                is deprecated; its closure is unsound (``Log.star`` is partial).

        Returns:
            (BoolFSA): An instance of the BoolFSA class.
        """
        if semiring not in ("log", "boolean"):
            raise ValueError(
                f"semiring must be 'log' or 'boolean', got {semiring!r}"
            )
        if semiring == "log":
            warnings.warn(
                "semiring='log' is deprecated: the FSA closure is unsound on "
                "the Log semiring because Log.star is partial. "
                "Use semiring='boolean'.",
                DeprecationWarning,
                stacklevel=2,
            )
        charset = charset or set(string.printable)
        wfsa = interegular_to_wfsa(pattern, charset=charset)
        if to_bytes:
            wfsa = wfsa.to_bytes()
        if semiring == "boolean":
            wfsa = _float_to_boolean(wfsa)
        return cls(wfsa)

    def _prefix(self, context):
        if self.wfsa.R is Boolean:
            curr = self._consume(context)
            if not curr:
                return float("-inf"), curr
            bkwd = self.wfsa.epsremove.backward
            for i in curr:
                if bkwd[i].score:  # i is co-accessible
                    return 0.0, curr
            return float("-inf"), curr
        return super()._prefix(context)

    async def prefix(self, context):
        """Tests whether `context` is accepted as a prefix.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): ``0`` if accepted as a prefix, ``-inf`` otherwise.
        """
        if self.wfsa.R is Boolean:
            return self._prefix(context)[0]
        prefix_w = await super().prefix(context)
        if prefix_w > float("-inf"):
            return 0
        return float("-inf")

    async def complete(self, context):
        """Tests whether `context` is accepted.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): ``0`` if accepted, ``-inf`` otherwise.
        """
        if self.wfsa.R is Boolean:
            curr = self._consume(context)
            if not curr:
                return float("-inf")
            finals = dict(self.wfsa.epsremove.F)
            for i in curr:
                if i in finals and finals[i].score:
                    return 0.0
            return float("-inf")
        complete_w = await super().complete(context)
        if complete_w > float("-inf"):
            return 0
        return float("-inf")

    async def logw_next(self, context):
        """Per-token admissibility log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (LazyWeights): ``0`` for admissible next tokens (incl. EOS),
                ``-inf`` for rejected.
        """
        if self.wfsa.R is Boolean:
            curr = self._consume(context)
            if not curr:
                raise ValueError(f"Context {context!r} has zero weight.")
            bkwd = self.wfsa.epsremove.backward
            finals = dict(self.wfsa.epsremove.F)
            out = self.alloc_logws()
            for i in curr:
                for b, j, w in self.wfsa.epsremove.arcs(i=i):
                    if w.score and bkwd[j].score and b in self.lookup:
                        out[self.lookup[b]] = 0.0
            for i in curr:
                if i in finals and finals[i].score:
                    out[self.lookup[self.eos]] = 0.0
                    break
            return self.make_lazy_weights(out)
        logw_next = await super().logw_next(context)
        return logw_next.spawn(
            new_weights=np.where(
                logw_next.weights > float("-inf"), 0, logw_next.weights
            )
        )

    async def batch_logw_next(self, contexts):
        """Per-token admissibility log weights for a batch of contexts.

        Args:
            contexts (list): The list of contexts.

        Returns:
            (list): List of ``LazyWeights``, one per context.
        """
        if self.wfsa.R is Boolean:
            return [await self.logw_next(c) for c in contexts]
        logw_nexts = await super().batch_logw_next(contexts)
        return [
            logw_next.spawn(
                new_weights=np.where(
                    logw_next.weights > float("-inf"), 0, logw_next.weights
                )
            )
            for logw_next in logw_nexts
        ]

    def __repr__(self):
        return f"BoolFSA(wfsa={self.wfsa!r})"
