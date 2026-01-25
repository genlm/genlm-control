import html
import numpy as np
from arsenal import Integerizer, colors
from arsenal.maths import sample, logsumexp
from graphviz import Digraph
from genlm.grammar import Float
from genlm.control.potential import Potential
from hypothesis import strategies as st


class MockPotential(Potential):
    def __init__(self, vocab, next_token_logws):
        self.next_token_logws = np.array(next_token_logws)
        super().__init__(vocab)

    def _logw(self, context):
        return sum([self.next_token_logws[self.lookup[i]] for i in context])

    async def prefix(self, context):
        return self._logw(context)

    async def complete(self, context):
        return self._logw(context) + self.next_token_logws[-1]

    async def logw_next(self, context):
        return self.make_lazy_weights(self.next_token_logws)


@st.composite
def mock_vocab(draw):
    item_strategy = draw(
        st.sampled_from(
            (
                st.text(min_size=1),
                st.binary(min_size=1),
            )
        )
    )

    # Sample vocabulary of iterables.
    vocab = draw(st.lists(item_strategy, min_size=1, max_size=10, unique=True))
    return vocab


@st.composite
def mock_vocab_and_ws(draw, max_w=1e3):
    vocab = draw(mock_vocab())
    ws = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    return vocab, ws


@st.composite
def mock_params(draw, max_w=1e3):
    iter_vocab, iter_next_token_ws = draw(mock_vocab_and_ws())

    # Sample context from iter_vocab
    context = draw(st.lists(st.sampled_from(iter_vocab), min_size=0, max_size=10))

    return (iter_vocab, iter_next_token_ws, context)


@st.composite
def iter_item_params(draw, max_iter_w=1e3, max_item_w=1e3):
    iter_vocab, iter_next_token_ws, context = draw(mock_params(max_iter_w))

    # Convert iter_vocab to bytes for trie
    iter_vocab_bytes = []
    for item in iter_vocab:
        if isinstance(item, str):
            iter_vocab_bytes.append(item.encode("utf-8"))
        elif isinstance(item, bytes):
            iter_vocab_bytes.append(item)
        else:
            iter_vocab_bytes.append(bytes(item))
    iter_vocab = iter_vocab_bytes

    # Convert context to bytes
    context = [
        item.encode("utf-8") if isinstance(item, str) else item for item in context
    ]
    item_vocab = set()
    for items in iter_vocab:
        item_vocab.update(items)
    item_vocab = list(item_vocab)

    # Sample weights over item vocabulary and EOS.
    item_next_token_ws = draw(
        st.lists(
            st.floats(1e-5, max_item_w),
            min_size=len(item_vocab) + 1,
            max_size=len(item_vocab) + 1,
        )
    )

    return (iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context)


class WeightedSet(Potential):
    def __init__(self, sequences, weights):
        self.complete_logws = {
            tuple(seq): np.log(w) if w != 0 else float("-inf")
            for seq, w in zip(sequences, weights)
        }

        prefix_ws = {}
        for seq, w in zip(sequences, weights):
            for i in range(0, len(seq) + 1):
                prefix = tuple(seq[:i])
                if prefix not in prefix_ws:
                    prefix_ws[prefix] = 0.0
                prefix_ws[prefix] += w

        self.prefix_log_ws = {
            prefix: np.log(w) if w != 0 else float("-inf")
            for prefix, w in prefix_ws.items()
        }
        total_weight = sum(weights)
        assert np.isclose(
            self.prefix_log_ws[()],
            np.log(total_weight) if total_weight != 0 else float("-inf"),
        )

        super().__init__(list(set(t for seq in sequences for t in seq)))

    async def complete(self, context):
        return self.complete_logws.get(tuple(context), float("-inf"))

    async def prefix(self, context):
        return self.prefix_log_ws.get(tuple(context), float("-inf"))


@st.composite
def weighted_sequence(draw, max_seq_len=5):
    sequence = draw(st.text(min_size=1, max_size=max_seq_len))
    weight = draw(st.floats(min_value=1e-3, max_value=1e3))
    return sequence, weight


@st.composite
def double_weighted_sequence(draw, max_seq_len=5):
    # We use the second weight as the weight assigned to the sequence
    # by the critic.
    sequence = draw(st.text(min_size=1, max_size=max_seq_len))
    weight1 = draw(st.floats(min_value=1e-3, max_value=1e3))
    weight2 = draw(st.floats(min_value=0, max_value=1e3))
    return sequence, weight1, weight2


@st.composite
def weighted_set(draw, item_sampler, max_seq_len=5, max_size=5):
    return draw(
        st.lists(
            item_sampler(max_seq_len),
            min_size=1,
            max_size=max_size,
            unique_by=lambda x: x[0],
        )
    )


def separate_keys_vals(x):
    from genlm.control.util import LazyWeights

    if isinstance(x, LazyWeights):
        return x.keys(), x.values()
    elif isinstance(x, np.ndarray):
        return range(len(x)), x
    else:
        return list(x.keys()), np.array(list(x.values()))


class Tracer:
    """
    This class lazily materializes the probability tree of a generative process by program tracing.
    """

    def __init__(self):
        self.root = Node(idx=-1, mass=1.0, parent=None)
        self.cur = None

    def __call__(self, p, context=None):
        "Sample an action while updating the trace cursor and tree data structure."

        keys, p = separate_keys_vals(p)
        cur = self.cur

        if cur.child_masses is None:
            cur.child_masses = cur.mass * p
            cur.context = context

        if context != cur.context:
            print(colors.light.red % "ERROR: trace divergence detected:")
            print(colors.light.red % "trace context:", self.cur.context)
            print(colors.light.red % "calling context:", context)
            raise ValueError((p, cur))

        a = cur.sample()
        if a not in cur.active_children:
            cur.active_children[a] = Node(
                idx=a,
                mass=cur.child_masses[a],
                parent=cur,
                token=keys[a],
            )
        self.cur = cur.active_children[a]
        return keys[a]


class Node:
    __slots__ = (
        "idx",
        "mass",
        "parent",
        "token",
        "child_masses",
        "active_children",
        "context",
        "_mass",
    )

    def __init__(
        self,
        idx,
        mass,
        parent,
        token=None,
        child_masses=None,
        context=None,
    ):
        self.idx = idx
        self.mass = mass
        self.parent = parent
        self.token = token  # used for visualization
        self.child_masses = child_masses
        self.active_children = {}
        self.context = context
        self._mass = mass  # bookkeeping: remember the original mass

    def sample(self):
        return sample(self.child_masses)

    def p_next(self):
        return Float.chart((a, c.mass / self.mass) for a, c in self.children.items())

    # TODO: untested
    def sample_path(self):
        curr = self
        path = []
        P = 1
        while True:
            p = curr.p_next()
            a = curr.sample()
            P *= p[a]
            curr = curr.children[a]
            if not curr.children:
                break
            path.append(a)
        return (P, path, curr)

    def update(self):
        # TODO: Fennwick tree alternative, sumheap
        # TODO: optimize this by subtracting from masses, instead of resumming
        "Restore the invariant that self.mass = sum children mass."
        if self.parent is not None:
            self.parent.child_masses[self.idx] = self.mass
            self.parent.mass = np.sum(self.parent.child_masses)
            self.parent.update()

    def graphviz(
        self,
        fmt_edge=lambda x, a, y: f"{html.escape(str(a))}/{y._mass / x._mass:.2g}",
        # fmt_node=lambda x: ' ',
        fmt_node=lambda x: (
            f"{x.mass}/{x._mass:.2g}" if x.mass > 0 else f"{x._mass:.2g}"
        ),
    ):
        "Create a graphviz instance for this subtree"
        g = Digraph(
            graph_attr=dict(rankdir="LR"),
            node_attr=dict(
                fontname="Monospace",
                fontsize="10",
                height=".05",
                width=".05",
                margin="0.055,0.042",
            ),
            edge_attr=dict(arrowsize="0.3", fontname="Monospace", fontsize="9"),
        )
        f = Integerizer()
        xs = set()
        q = [self]
        while q:
            x = q.pop()
            xs.add(x)
            if x.child_masses is None:
                continue
            for a, y in x.active_children.items():
                a = y.token if y.token is not None else a
                g.edge(str(f(x)), str(f(y)), label=f"{fmt_edge(x, a, y)}")
                q.append(y)
        for x in xs:
            if x.child_masses is not None:
                g.node(str(f(x)), label=str(fmt_node(x)), shape="box")
            else:
                g.node(str(f(x)), label=str(fmt_node(x)), shape="box", fillcolor="gray")
        return g

    def downstream_nodes(self):
        q = [self]
        while q:
            x = q.pop()
            yield x
            if x.child_masses is None:
                continue
            for y in x.active_children.values():
                q.append(y)


class TraceSWOR(Tracer):
    """
    Sampling without replacement 🤝 Program tracing.
    """

    def __enter__(self):
        self.cur = self.root

    def __exit__(self, *args):
        self.cur.mass = 0  # we will never sample this node again.
        self.cur.update()  # update invariants

    def _repr_svg_(self):
        return self.root.graphviz()._repr_image_svg_xml()

    def sixel_render(self):
        try:
            from sixel import converter
            import sys
            from io import BytesIO

            c = converter.SixelConverter(
                BytesIO(self.root.graphviz()._repr_image_png())
            )
            c.write(sys.stdout)
        except ImportError:
            import warnings

            warnings.warn("Install imgcat or sixel to enable rendering.")
            print(self)

    def __repr__(self):
        return self.root.graphviz().source


async def trace_swor(sampler, context):
    tracer = TraceSWOR()
    logP = sampler.target.alloc_logws()
    while tracer.root.mass > 0:
        with tracer:
            token, logw, logp = await sampler.sample(context, draw=tracer)
            token_id = sampler.target.lookup[token]
            logP[token_id] = logsumexp([logP[token_id], logw + logp])

    return sampler.target.make_lazy_weights(logP)


async def trace_swor_set(sampler, context):
    tracer = TraceSWOR()
    logws = sampler.target.alloc_logws()
    while tracer.root.mass > 0:
        with tracer:
            set_logws, logp = await sampler.sample_set(context, draw=tracer)
            for token_id, logw in enumerate(set_logws.weights):
                if logw == float("-inf"):
                    continue
                logws[token_id] = logsumexp([logws[token_id], logw + logp])

    return sampler.target.make_lazy_weights(logws)
