"""Control's `LazyWeights` seam onto the backend draw window.

The window's own behaviour (cohort formation, failure isolation, picker family)
is covered in genlm-backend's `tests/test_draw.py`.
"""

import pytest
import torch
from genlm.backend.batching import take_batch_stats

from genlm.control.util import LazyWeights, draw_from

VOCAB4 = [b"a", b"b", b"c", b"d"]


def lw(weights, vocab):
    return LazyWeights(
        weights=weights,
        encode={t: i for i, t in enumerate(vocab)},
        decode=vocab,
        log=True,
    )


@pytest.fixture(autouse=True)
def _clean_batch_stats():
    # Drain any residue from a previous test/loop before and after, so a
    # test's ``take_batch_stats()`` snapshot is its own draws, nothing else.
    take_batch_stats()
    yield
    take_batch_stats()


@pytest.mark.asyncio
async def test_window_draw_decodes_token_and_forwards_target():
    row = torch.log_softmax(torch.tensor([0.0, 0.0, 20.0, 0.0]), -1)  # ~surely "c"
    target = torch.log_softmax(torch.tensor([1.0, 2.0, 5.0, 0.0]), -1)

    tok, logw, logp = await draw_from(lw(row, VOCAB4), target=lw(target, VOCAB4))
    assert take_batch_stats()[("draw", 1)] == 1

    assert tok == b"c"  # decoded through the LazyWeights vocabulary
    assert logp == pytest.approx(row[2].item(), abs=1e-5)
    assert logw == pytest.approx(target[2].item() - logp, abs=1e-5)


@pytest.mark.asyncio
async def test_custom_draw_takes_solo_path():
    row = torch.tensor([1.0, 2.0, 3.0, 0.0])  # unnormalized logits

    def picker(chart):
        return max(chart, key=chart.__getitem__)  # highest-prob token

    tok, logw, logp = await draw_from(lw(row, VOCAB4), draw=picker)
    assert not take_batch_stats()  # solo path never touches the window

    assert tok == b"c"  # argmax of the raw logits
    idx = VOCAB4.index(tok)
    logZ = torch.logsumexp(row, 0).item()
    expected_logp = (row[idx] - logZ).item()
    assert logw == pytest.approx(logZ, abs=1e-6)
    assert logp == pytest.approx(expected_logp, abs=1e-6)

    target = torch.tensor([0.0, 0.0, 0.0, 10.0])
    tok2, logw2, logp2 = await draw_from(
        lw(row, VOCAB4), draw=picker, target=lw(target, VOCAB4)
    )
    assert not take_batch_stats()

    idx2 = VOCAB4.index(tok2)
    expected_logw2 = target[idx2].item() - logp2
    assert logw2 == pytest.approx(expected_logw2, abs=1e-6)
