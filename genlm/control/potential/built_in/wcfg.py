import numpy as np
from genlm.grammar import CFG, Earley, Float, Boolean
from genlm.grammar.lark_interface import LarkStuff
from genlm.grammar.cfglm import _gen_nt

from genlm.control.constant import EOS
from genlm.control.potential.base import Potential


def _add_eos(cfg, eos):
    S = _gen_nt("<START>")
    cfg_eos = cfg.spawn(S=S)
    cfg_eos.V.add(eos)
    cfg_eos.add(cfg.R.one, S, cfg.S, eos)
    for r in cfg:
        cfg_eos.add(r.w, r.head, *r.body)
    return cfg_eos


class WCFG(Potential):
    """
    A weighted context-free grammar potential.

    This class wraps a `genlm_grammar.CFG` and provides methods for computing the log-weight of a sequence,
    the prefix log-weight of a sequence, and the log-weights of the next token given a sequence.
    """

    def __init__(self, cfg):
        """
        Initialize the WCFG potential.

        Args:
            cfg (genlm_grammar.CFG): The context-free grammar configuration to use.
                The CFG must in the Float semiring.
        """
        # TODO: convert to LogSemiring to handle underflow
        if cfg.R is not Float:
            raise ValueError("cfg semiring must be Float")
        self.cfg = cfg  # cfg before prefix transform
        self.cfg_eos = _add_eos(cfg, EOS)  # augmented with eos
        self.model = Earley(self.cfg_eos.prefix_grammar)
        super().__init__(vocabulary=list(cfg.V))

    @classmethod
    def from_string(cls, grammar, to_bytes=True, **kwargs):
        """Create a WCFG from a string.

        Args:
            grammar (str): The string grammar specification to create the WCFG from.
            to_bytes (bool, optional): Whether to convert the WCFG terminals to indivudual bytes.
                Defaults to True.
            **kwargs (dict): Additional arguments passed to the WCFG constructor.

        Returns:
            (WCFG): The created WCFG.
        """
        cfg = CFG.from_string(grammar, Float)
        if to_bytes:
            cfg = cfg.to_bytes()
        return cls(cfg, **kwargs)

    async def complete(self, context):
        """
        Compute the log weight of `context` under the WCFG.

        For example, if the WCFG accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `complete("c")` returns $-\\infty$ since this sequence is not accepted by the WCFG\n
        - `complete("cat")` returns $\\log(w_{cat})$\n
        - `complete("d")` returns $-\\infty$ since this sequence is not accepted by the WCFG

        Args:
            context (list): A sequence of tokens in the WCFG's alphabet.

        Returns:
            (float): The log weight of `context` under the WCFG.
        """
        w = self.model([*context, EOS])
        return np.log(w) if w > 0 else float("-inf")

    async def prefix(self, context):
        """
        Compute the log prefix weight of `context` under the WCFG.

        This corresponds to the log of the sum of the weights of all sequences with prefix `context`.

        For example, if the WCFG accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `prefix("c")` returns $\\log(w_{cat} + w_{car})$\n
        - `prefix("cat")` returns $\\log(w_{cat})$\n
        - `prefix("d")` returns $-\\infty$ since the WCFG does not accept any sequences with prefix "d"

        Args:
            context (list): A sequence of tokens in the WCFG's alphabet.

        Returns:
            (float): The log prefix weight of `context` under the WCFG.
        """
        w = self.model(context)
        return np.log(w) if w > 0 else float("-inf")

    async def logw_next(self, context):
        """
        Compute the next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WCFG's alphabet.

        Returns:
            (LazyWeights): The log weights for the next tokens and EOS given `context`.
        """
        ws = self.model.next_token_weights(self.model.chart(context))
        ws = ws.trim().normalize()

        ws_array = np.array([ws[x] for x in self.vocab_eos])
        mask = ws_array > 0
        log_ws = np.full_like(ws_array, float("-inf"), dtype=np.float64)
        log_ws[mask] = np.log(ws_array[mask])

        return self.make_lazy_weights(log_ws)

    def clear_cache(self):
        """Clear the internal cache of the parser."""
        self.model.clear_cache()

    def __repr__(self):
        return f"WCFG(cfg={self.cfg!r})"

    def _repr_html_(self):
        return self.cfg._repr_html_()

    def spawn(self):
        """Spawn a new WCFG."""
        return WCFG(self.cfg)


class BoolCFG(Potential):
    """BoolCFG represents a boolean context-free grammar."""

    def __init__(self, cfg):
        if cfg.R != Boolean:
            cfg = cfg.map_values(lambda x: Boolean(x > 0), Boolean)
        self.cfg = cfg  # cfg before prefix transform
        self.cfg_eos = _add_eos(cfg, EOS)  # augmented with eos
        self.model = Earley(self.cfg_eos.prefix_grammar)
        super().__init__(vocabulary=list(cfg.V))

    @classmethod
    def from_lark(cls, lark_string, charset="core"):
        """
        Create a BoolCFG instance from a Lark grammar string.

        The output grammar will be defined at the byte-level.

        Args:
            lark_string (str): The Lark grammar string to parse. See Lark documentation for correct syntax.
            charset (str): The character set to use. Defaults to "core".
                See `genlm-grammar` documentation for more details.

        Returns:
            (BoolCFG): An instance of BoolCFG created from the provided Lark grammar.
        """
        byte_cfg = LarkStuff(lark_string).byte_cfg(charset=charset)
        return cls(byte_cfg)

    async def complete(self, context):
        """
        Checks whether the context is accepted by the CFG.

        Args:
            context (list): A sequence of tokens in the CFG's alphabet.

        Returns:
            (float): Log weight for whether `context` is accepted by the CFG.
        """
        w = self.model([*context, EOS])
        return 0 if w.score else float("-inf")

    async def prefix(self, context):
        """
        Checks whether `context` is accepted as a prefix by the CFG, i.e.,
        whether there exists a completion to `context` that is accepted by the CFG.

        Args:
            context (list): A sequence of tokens in the CFG's alphabet.

        Returns:
            (float): Log weight for whether `context` is accepted as a prefix by the CFG.
        """
        if not context:  # FIX: this is a hack to handle the empty string because genlm-grammar doesn't support it
            return 0
        w = self.model(context)
        return 0 if w.score else float("-inf")

    async def logw_next(self, context):
        """
        Compute the next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the CFG's alphabet.

        Returns:
            (LazyWeights): The log weights for the next tokens and EOS given `context`.
        """
        ws = self.model.next_token_weights(self.model.chart(context))
        log_ws = np.array([0 if ws[x].score else float("-inf") for x in self.vocab_eos])
        return self.make_lazy_weights(log_ws)

    async def batch_logw_next(self, contexts):
        """
        Batch version of `logw_next`.

        Args:
            contexts (list): A list of sequences of tokens in the CFG's alphabet.

        Returns:
            (LazyWeights): one batched `LazyWeights`, `.weights` shape `[N, V+1]`.
        """
        rows = []
        for context in contexts:
            ws = self.model.next_token_weights(self.model.chart(context))
            rows.append(
                np.array([0 if ws[x].score else float("-inf") for x in self.vocab_eos])
            )
        return self.make_lazy_weights(np.stack(rows))

    def spawn(self):
        """Spawn a new BoolCFG."""
        return BoolCFG(self.cfg)

    def clear_cache(self):
        """Clear the internal cache of the parser."""
        self.model.clear_cache()

    def __repr__(self):
        return f"BoolCFG(cfg={self.cfg!r})"

    def _repr_html_(self):
        return self.cfg._repr_html_()
