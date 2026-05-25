import string
import numpy as np
from arsenal.maths import logsumexp

from genlm.grammar import Float, Log, WFSA as BaseWFSA
from genlm.grammar.lark_interface import interegular_to_wfsa

from genlm.control.potential.base import Potential


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

    # -- stateful interface (the WFSA's state is its chart, what `_consume`
    #    already maintains): carry + advance it instead of replaying `context` --

    def state0(self):
        """Carried state for the empty context: the epsilon-removed start chart.

        This is a shared chart object (also cached by `_consume(())`); `advance`
        and the readers only *read* it and allocate a fresh chart, so callers
        must likewise treat a carried state as immutable.
        """
        return self.wfsa.epsremove.start

    def advance(self, state, token):
        """Advance the chart by one WFSA-alphabet symbol (one `_consume` step)."""
        wfsa = self.wfsa.epsremove
        curr = wfsa.R.chart()
        for i in state:
            for j, w in wfsa.arcs(i, token):
                curr[j] += state[i] * w
        return curr

    async def logw_next_from_state(self, state):
        """Next-token log weights from a carried chart `state`.

        Mirrors `logw_next` but reads the chart from `state` instead of
        `_consume`-ing the whole context. Bit-identical to `logw_next(context)`
        for the state reached by advancing `state0()` through `context` -- the
        parity contract. (Body intentionally duplicates `logw_next`; DRY-ing the
        shared chart->weights step is a housekeeping task, gated by that parity.)
        """
        if not state:
            raise ValueError("Context has zero weight (dead state).")
        bkwd = self.wfsa.epsremove.backward
        log_ctx_w = logsumexp([(state[i] * bkwd[i]).score for i in state])
        if np.isnan(log_ctx_w) or log_ctx_w == float("-inf"):
            raise ValueError("Context has zero weight.")

        ws = self.wfsa.R.chart()
        for i in state:
            for b, j, w in self.wfsa.epsremove.arcs(i=i):
                ws[b] += state[i] * w * bkwd[j]

        ws[self.eos] = self.wfsa.R.zero
        for j, w in self.wfsa.epsremove.F:
            ws[self.eos] += state[j] * w

        log_ws = np.array([ws[b].score for b in self.vocab_eos]) - log_ctx_w

        return self.make_lazy_weights(log_ws)

    def prefix_logw(self, state):
        """Log prefix weight from a carried chart `state` (sync; `-inf` if dead).

        Sync stateful analog of `prefix` -- no `_consume`, no async. Lets a
        coercion evaluate many candidate extensions from a carried chart without
        an `asyncio.gather` over the vocabulary (the dominant per-step cost)."""
        if not state:
            return float("-inf")
        bkwd = self.wfsa.epsremove.backward
        log_ctx_w = logsumexp([(state[i] * bkwd[i]).score for i in state])
        return float("-inf") if np.isnan(log_ctx_w) else float(log_ctx_w)

    def complete_logw(self, state):
        """Log complete weight from a carried chart `state` (sync). Sync stateful
        analog of `complete` (the EOS column)."""
        acc = self.wfsa.R.zero
        for j, w in self.wfsa.epsremove.F:
            acc += state[j] * w
        return float(acc.score)

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
    """Boolean FSA potential."""

    async def prefix(self, context):
        """
        Computes whether the context is accepted as a prefix by the FSA.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): `0` if the context is accepted as a prefix, `-inf` otherwise.
        """
        prefix_w = await super().prefix(context)
        if prefix_w > float("-inf"):
            return 0
        return float("-inf")

    async def complete(self, context):
        """
        Computes whether the context is accepted by the FSA.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): `0` if the context is accepted, `-inf` otherwise.
        """
        complete_w = await super().complete(context)
        if complete_w > float("-inf"):
            return 0
        return float("-inf")

    @staticmethod
    def _booleanize(logw_next):
        """Map a weighted `LazyWeights` to its boolean indicator (0 where alive,
        -inf elsewhere). The single definition shared by every BoolFSA next-token
        method so they cannot drift."""
        return logw_next.spawn(
            new_weights=np.where(
                logw_next.weights > float("-inf"), 0, logw_next.weights
            )
        )

    @staticmethod
    def _bool(w):
        """Scalar analogue of `_booleanize`: 0 if alive, -inf otherwise."""
        return 0.0 if w > float("-inf") else float("-inf")

    async def logw_next(self, context):
        """
        Returns next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (LazyWeights): Boolean log-weights for next token.
        """
        return self._booleanize(await super().logw_next(context))

    async def logw_next_from_state(self, state):
        """Boolean next-token log weights from a carried chart `state` (mirrors
        `logw_next` over the stateful path)."""
        return self._booleanize(await super().logw_next_from_state(state))

    def prefix_logw(self, state):
        """Boolean prefix weight from a carried chart (0 if alive, else -inf)."""
        return self._bool(super().prefix_logw(state))

    def complete_logw(self, state):
        """Boolean complete weight from a carried chart (0 if accepting, else -inf)."""
        return self._bool(super().complete_logw(state))

    async def batch_logw_next(self, contexts):
        """
        Returns next token log weights for a batch of contexts.

        Args:
            contexts (list): The list of contexts.

        Returns:
            (list): List of log-weights for next token, one per context.
        """
        return [self._booleanize(lw) for lw in await super().batch_logw_next(contexts)]

    def __repr__(self):
        return f"BoolFSA(wfsa={self.wfsa!r})"
