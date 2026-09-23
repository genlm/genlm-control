import asyncio
from collections import Counter

import numpy as np
import pytest
import torch

from genlm.control import util
from genlm.control.util import (
    DRAW_METHODS,
    LazyWeights,
    draw_from,
    picker_indices,
    set_draw_method,
    BatchAbandoned,
)

VOCAB4 = [b"a", b"b", b"c", b"d"]


def lw(weights, vocab):
    return LazyWeights(
        weights=weights,
        encode={t: i for i, t in enumerate(vocab)},
        decode=vocab,
        log=True,
    )


@pytest.fixture
def cohorts(monkeypatch):
    """Cohort sizes reduced by the batched flush; the solo path bypasses the picker
    and records nothing."""
    seen = []
    picker = util._picker

    def spy(logps):
        seen.append(logps.shape[0])
        return picker(logps)

    monkeypatch.setattr(util, "_picker", spy)
    return seen


@pytest.mark.asyncio
async def test_nested_gather_depths_land_in_one_cohort(cohorts):
    # Peel ``depth`` extra layers of asyncio.gather before the actual draw_from
    # call -- each layer defers that call by one event-loop tick, staggering
    # arrivals at the window without opening a gap wide enough to flush early.
    async def nested(row, depth):
        if depth == 0:
            return await draw_from(lw(row, VOCAB4))
        return (await asyncio.gather(nested(row, depth - 1)))[0]

    rows = [torch.log_softmax(torch.randn(4), -1) for _ in range(9)]
    depths = [0, 1, 2] * 3
    await asyncio.gather(*[nested(r, d) for r, d in zip(rows, depths)])
    assert cohorts == [9]


@pytest.mark.asyncio
async def test_mixed_shape_and_backend_split_into_separate_groups():
    vocab3 = [b"x", b"y", b"z"]
    torch_rows = [torch.log_softmax(torch.randn(4), -1) for _ in range(4)]
    np_rows = [np.log(np.full(3, 1 / 3)) for _ in range(3)]

    out = await asyncio.gather(
        *[draw_from(lw(r, VOCAB4)) for r in torch_rows],
        *[draw_from(lw(r, vocab3)) for r in np_rows],
    )
    # Merged groups would raise in torch.stack on mismatched vocab sizes, so
    # answering at all is the proof they stayed split.
    assert all(tok in VOCAB4 for tok, _, _ in out[:4])
    assert all(tok in vocab3 for tok, _, _ in out[4:])


@pytest.mark.asyncio
async def test_importance_draw_matches_hand_computed_logw():
    vocab = [b"a", b"b", b"c"]
    proposal = torch.log_softmax(torch.tensor([5.0, 0.0, 0.0]), -1)  # peaked at "a"
    target = torch.log_softmax(torch.tensor([1.0, 2.0, 5.0]), -1)  # peaked at "c"
    plain_rows = [torch.log_softmax(torch.randn(3), -1) for _ in range(2)]

    out = await asyncio.gather(
        draw_from(lw(proposal, vocab), target=lw(target, vocab)),
        *[draw_from(lw(r, vocab)) for r in plain_rows],
    )

    tok, logw, logp = out[0]
    idx = vocab.index(tok)
    expected_logp = (proposal[idx] - torch.logsumexp(proposal, 0)).item()
    expected_logw = target[idx].item() - expected_logp
    assert logp == pytest.approx(expected_logp, abs=1e-5)
    assert logw == pytest.approx(expected_logw, abs=1e-5)

    for _, plain_logw, _ in out[1:]:
        assert abs(plain_logw) < 1e-5  # plain draw: logZ of an already-softmaxed row


@pytest.mark.asyncio
async def test_custom_draw_takes_solo_path():
    row = torch.tensor([1.0, 2.0, 3.0, 0.0])  # unnormalized logits

    def picker(chart):
        return max(chart, key=chart.__getitem__)  # highest-prob token

    tok, logw, logp = await draw_from(lw(row, VOCAB4), draw=picker)

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

    idx2 = VOCAB4.index(tok2)
    expected_logw2 = target[idx2].item() - logp2
    assert logw2 == pytest.approx(expected_logw2, abs=1e-6)


@pytest.mark.asyncio
async def test_exception_fails_its_group_without_poisoning_others():
    vocab3 = [b"x", b"y", b"z"]
    good_rows = [torch.log_softmax(torch.randn(4), -1) for _ in range(3)]
    bad_rows = [
        torch.log_softmax(torch.randn(3), -1),
        torch.zeros(2, 3),  # wrong shape: torch.stack over this group raises
    ]

    good_task = asyncio.gather(*[draw_from(lw(r, VOCAB4)) for r in good_rows])
    bad_task = asyncio.gather(
        *[draw_from(lw(r, vocab3)) for r in bad_rows], return_exceptions=True
    )
    good_out, bad_out = await asyncio.gather(good_task, bad_task)

    for tok, _, _ in good_out:
        assert tok in VOCAB4  # unaffected by the other group's failure

    assert len(bad_out) == 2
    assert all(isinstance(e, RuntimeError) for e in bad_out)
    assert bad_out[0] is bad_out[1]  # same exception instance on every future in the group


@pytest.fixture(params=sorted(DRAW_METHODS))
def picker(request, monkeypatch):
    """Each registered picker in turn. The setting is process-wide, so the prior one is
    registered for restore before it is changed."""
    monkeypatch.setattr(util, "_picker", util._picker)
    set_draw_method(request.param)


@pytest.mark.asyncio
async def test_distribution_matches_known_categorical(picker):
    vocab = [b"a", b"b", b"c"]
    probs = torch.tensor([0.2, 0.3, 0.5])
    row = probs.log()
    n = 2000

    torch.manual_seed(0)
    out = await asyncio.gather(*[draw_from(lw(row, vocab)) for _ in range(n)])

    counts = Counter(tok for tok, _, _ in out)
    for tok, p in zip(vocab, probs.tolist()):
        assert abs(counts[tok] / n - p) < 0.04


def test_picker_indices_shape_follows_the_row(picker):
    """`[V]` reduces to a scalar index, `[N, V]` to one index per row."""
    assert picker_indices(torch.log_softmax(torch.tensor([1.0, 2.0, 3.0, 0.5]), -1)).ndim == 0
    assert picker_indices(torch.log_softmax(torch.randn(5, 4), -1)).shape == (5,)


@pytest.mark.asyncio
async def test_abandoned_batch_fails_co_callers():
    """A cancelled batch holder must fail its co-callers, not orphan them.

    The failure is never the holder's own `CancelledError`: handed to a caller who
    never asked for one, that leaves their task cancelled and skips their
    `except Exception`.
    """
    rows = [torch.log_softmax(torch.randn(4), -1) for _ in range(3)]
    tasks = [asyncio.ensure_future(draw_from(lw(r, VOCAB4))) for r in rows]
    await asyncio.sleep(0)  # everyone is queued; tasks[0] holds the batch
    tasks[0].cancel()

    results = await asyncio.gather(*tasks[1:], return_exceptions=True)
    for result in results:
        assert isinstance(result, BatchAbandoned)
        assert not isinstance(result, asyncio.CancelledError)
    assert all(not t.cancelled() for t in tasks[1:])
    assert tasks[0].cancelled()
