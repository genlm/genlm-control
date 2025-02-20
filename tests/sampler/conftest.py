import numpy as np
from genlm_control.potential import Potential
from hypothesis import strategies as st


class MockPotential(Potential):
    def __init__(self, vocab, next_token_logws):
        self.next_token_logws = np.array(next_token_logws)
        super().__init__(vocab)

    def score(self, context):
        return sum([self.next_token_logws[self.encode[i]] for i in context])

    async def prefix(self, context):
        return self.score(context)

    async def complete(self, context):
        return self.score(context) + self.next_token_logws[-1]

    async def logw_next(self, context):
        return self.make_lazy_weights(self.next_token_logws)


@st.composite
def mock_params(draw, max_w=1e3):
    # Sample bytes or strings as iterables.
    item_strategy = draw(
        st.sampled_from(
            (
                st.text(min_size=1),
                st.binary(min_size=1),
            )
        )
    )

    # Sample vocabulary of iterables.
    iter_vocab = draw(st.lists(item_strategy, min_size=1, max_size=10, unique=True))

    # Sample weights over iterables vocabulary and EOS.
    iter_next_token_ws = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(iter_vocab) + 1,
            max_size=len(iter_vocab) + 1,
        )
    )

    # Sample context from iter_vocab
    context = draw(st.lists(st.sampled_from(iter_vocab), min_size=0, max_size=10))

    return (iter_vocab, iter_next_token_ws, context)


@st.composite
def iter_item_params(draw, max_iter_w=1e3, max_item_w=1e3):
    iter_vocab, iter_next_token_ws, context = draw(mock_params(max_iter_w))

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
