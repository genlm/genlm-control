from genlm.control.potential import Potential
from genlm.backend.tokenization import Token
from itertools import chain


class Coerced(Potential):
    """
    Coerce a potential to operate on another vocabulary.

    This class allows a potential to be adapted to work with a different set of tokens,
    defined by a target vocabulary and coersion function.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].
    Each method delegates to the corresponding method of the underlying potential, but first
    maps any input token sequences from the target vocabulary to the original potential's vocabulary
    using the coercion function.

    Formally, if $f$ is the coercion function, then for any sequence $x_1, \\ldots, x_n$ of tokens from the target vocabulary,
    $$
    \\textsf{Coerced.prefix}(x_1, \\ldots, x_n) = \\textsf{Coerced.potential.prefix}(f(x_1, \\ldots, x_n))
    $$

    $$
    \\textsf{Coerced.complete}(x_1, \\ldots, x_n) = \\textsf{Coerced.potential.complete}(f(x_1, \\ldots, x_n))
    $$

    Attributes:
        potential (Potential): The original potential instance that is being coerced.
        f (callable): A function that maps sequences of tokens from the target vocabulary to sequences of tokens from
            the original potential's vocabulary.

    Note:
        The coerced potential's vocabulary will by default be pruned to only include tokens that can be mapped to the original potential's vocabulary
        via the coercion function (i.e. `set(f([x])) <= set(potential.vocab)`). If no such tokens are found, a `ValueError` is raised.
        This behavior can be overridden by setting `prune=False`, in which case the coerced potential's vocabulary will include all tokens from the target vocabulary.

        When the wrapped potential exposes a memoized chart (`_consume`, i.e. a
        WFSA/BoolFSA) AND `f` distributes over concatenation at the context in hand
        (`f(xs+[t]) == f(xs)+f([t])`, so each target token maps to a fixed symbol
        path), `logw_next` takes the shared-prefix-trie fast path
        (:meth:`live_logws`). Any other `f` -- which the contract permits -- falls
        back to the per-extension `batch_prefix`, which makes no such assumption, so
        a non-distributing `f` is correct, just not accelerated.
    """

    def __init__(
        self,
        potential,
        target_vocab,
        f,
        prune=True,
        homomorphic=None,
        trie=None,
        tables=None,
    ):
        """
        Initialize a Coerced potential.

        Args:
            potential (Potential): The original potential instance that is being coerced.
            target_vocab (list): The target vocabulary that the potential will operate on.
                Each element of `target_vocab` must be hashable.
            f (callable): A function that maps iterables of tokens from the target vocabulary
                to the original potential's vocabulary.
            prune (bool): Whether to prune the coerced potential's vocabulary to only include tokens that can be mapped to the original potential's vocabulary.
                If `False`, the coerced potential's vocabulary will include all tokens from the target vocabulary.
            homomorphic (bool | None): Whether `f` distributes over concatenation
                (`f(xs+[t]) == f(xs)+f([t])`), which enables the `live_logws` fast
                path. `None` (default) checks the identity at each context it is
                asked about, falling back to the assumption-free path wherever it
                does not hold; `True`/`False` declare it and skip the check.
            trie (dict | None): The symbol trie over `(target_vocab, f)`, as built by
                :meth:`build_trie`. It is a function of those two alone, so coercions
                sharing a vocabulary should build it once and pass it here rather than
                each paying for its own. `None` (default) builds one lazily.
            tables (VocabTables | None): Prebuilt vocabulary tables for `target_vocab`
                (`Potential.build_tables`), shared for the same reason as `trie`.
                Like `trie`, incompatible with `prune=True`, which narrows the
                vocabulary the tables describe.

        Raises:
            ValueError: If no valid tokens are found in the target vocabulary that can be mapped to the original potential's vocabulary.
        """
        self.potential = potential
        self.f = f
        if prune and (trie is not None or tables is not None):
            # Both index the coerced vocabulary, which pruning is about to shrink --
            # injected ones would silently describe the wrong tokens.
            raise ValueError(
                "an injected `trie`/`tables` indexes the coerced vocabulary, which "
                "`prune=True` narrows; pass `prune=False` or build them over the "
                "pruned vocab"
            )
        self._sym_trie_cache = trie

        if prune:
            # When vocab contains Token objects (bytes subclass), the coercion
            # function f (typically b"".join) produces bytes. set(bytes) yields
            # int byte values, so we need potential_items to also be int byte
            # values for the subset check to work.
            if potential.vocab and isinstance(potential.vocab[0], Token):
                potential_items = set(
                    byte_val for tok in potential.vocab for byte_val in tok.byte_string
                )
            else:
                potential_items = set(potential.vocab)

            tokens = []
            for target_token in target_vocab:
                base_token = f([target_token])
                if set(base_token) <= potential_items:
                    tokens.append(target_token)
        else:
            tokens = target_vocab

        if not tokens:
            raise ValueError("No valid tokens found in target vocabulary")

        super().__init__(tokens, tables=tables)

        # `None` = check the identity at each context (see `_distributes_at`); a
        # bool is the caller's declaration and is taken as given.
        self._f_homomorphic = None if homomorphic is None else bool(homomorphic)

    def _distributes_at(self, context, ctx_syms):
        """Whether `f(context + [t]) == f(context) + f([t])` at THIS context, over a
        couple of probe tokens -- the identity `live_logws` keys the trie on.

        Checked per call rather than once at construction because the ways `f` can
        fail to distribute are length- and position-dependent, so no fixed sample of
        contexts settles it. One extra `f` per probe token, against a walk over the
        whole vocabulary. Any error counts as non-distributing."""
        try:
            return all(
                tuple(self.f([*context, t])) == ctx_syms + tuple(self.f([t]))
                for t in self.vocab[:2]
            )
        except Exception:
            return False

    def _batch_f(self, contexts):
        return [self.f(context) for context in contexts]

    async def complete(self, context):
        return await self.potential.complete(context=self.f(context))

    async def prefix(self, context):
        return await self.potential.prefix(context=self.f(context))

    async def logw_eos(self, context):
        """EOS log-weight via ``complete - prefix`` on the coerced context."""
        return float(await self.complete(context) - await self.prefix(context))

    async def _logw_next_dense(self, context):
        # The assumption-free fallback to the `live_logws` trie walk: one coerced
        # extension prefix-ed PER vocab token, assuming nothing about `f`.
        Ws = self.alloc_logws()
        ctx = self.f(context)
        ctx_w = await self.potential.prefix(ctx)
        if ctx_w == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")
        Ws[-1] = await self.potential.complete(ctx) - ctx_w
        exts = [self.f(chain(context, [x])) for x in self.vocab]  # slow!!
        Ws[:-1] = await self.potential.batch_prefix(exts) - ctx_w
        return self.make_lazy_weights(Ws)

    # -- the trie fast path: a shared-prefix trie over the target vocab, scored from
    #    the wrapped potential's MEMOIZED chart (``_consume``), no per-vocab replay --

    @staticmethod
    def build_trie(vocab, f):
        """Prefix trie over the symbol sequences ``f([t])`` of `vocab`. A node is a
        dict ``{sym: child}``; tokens that END at a node are recorded under the
        sentinel key ``()`` as a list of vocab indices (a list because distinct
        target tokens can share an ``f``-image). Sharing common prefixes lets
        :meth:`live_logws` score each shared prefix ONCE instead of re-prefixing
        every token's full symbol path.

        A function of `(vocab, f)` alone -- build it once per vocabulary and pass it
        to every coercion over that vocabulary via `trie=`.
        """
        trie = {}
        for idx, tok in enumerate(vocab):
            node = trie
            for sym in f([tok]):
                node = node.setdefault(sym, {})
            node.setdefault((), []).append(idx)
        return trie

    @property
    def _sym_trie(self):
        """This coercion's symbol trie -- the injected one, else built and held."""
        if self._sym_trie_cache is None:
            self._sym_trie_cache = self.build_trie(self.vocab, self.f)
        return self._sym_trie_cache

    async def live_logws(self, context):
        """The vocab tokens the wrapped potential's MEMOIZED chart (``_consume``, i.e.
        a WFSA/BoolFSA) admits, scored by one shared-prefix trie walk over that chart
        rather than one coerced extension prefix-ed PER vocab token. ``None`` when the
        lane is unavailable, leaving ``logw_next`` its assumption-free fallback: the
        trie keys on ``f(context)+f([t])``, equal to ``f(context+[t])`` only when ``f``
        distributes (`_f_homomorphic`, asserted by `homomorphic=` or probed per call).

        The wrapped potential may offer ``_advance(chart, sym) -> chart | None``, the
        incremental step the walk is already shaped for: the chart threads down the
        trie instead of every node re-deriving and re-consuming its full symbol path,
        and a ``None`` prunes that subtree -- sound because the potential declares the
        branch dead. Without it each node is scored from ``_consume(ctx_syms + path)``.
        """
        p = self.potential
        if self._f_homomorphic is False or not hasattr(p, "_consume"):
            return None
        ctx_syms = tuple(self.f(context))
        if self._f_homomorphic is None and not self._distributes_at(context, ctx_syms):
            return None
        ctx_chart = p._consume(ctx_syms)
        ctx_w = p.prefix_logw(ctx_chart)
        if ctx_w == float("-inf"):
            # A zero-weight context has no live row -- scoring it would divide the
            # walk by `-inf` and hand back `+inf` weights under a `nan` EOS. The one
            # per-context `None`, and the batch it drops to the dense path raises there.
            return None
        # Read before the walk: a chart the walk mutates must not move EOS under it.
        eos = p.complete_logw(ctx_chart) - ctx_w
        indices, values = [], []
        advance = getattr(p, "_advance", None)
        if advance is None:
            root = ()

            def advance(path, sym):
                return path + (sym,)

            def chart_of(path):
                return p._consume(ctx_syms + path)
        else:
            root = ctx_chart

            def chart_of(chart):
                return chart

        stack = [(self._sym_trie, root)]
        while stack:
            node, key = stack.pop()
            ends = node.get(())
            if ends is not None:
                w = p.prefix_logw(chart_of(key)) - ctx_w
                if w != float("-inf"):  # dead tokens are the row's default
                    indices.extend(ends)
                    values.extend([w] * len(ends))
            for sym, child in node.items():
                if sym != ():
                    nxt = advance(key, sym)
                    if nxt is not None:
                        stack.append((child, nxt))
        return indices, values, eos

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts=self._batch_f(contexts))

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts=self._batch_f(contexts))

    # batch_logw_next inherited from Potential (stacks per-context logw_next -> [N, V+1]).

    def __repr__(self):
        return f"{self.__class__.__name__}({self.potential!r})"
