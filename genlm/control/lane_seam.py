"""Lane-serving seam: one ContextVar binds a row's step to its engine lanes.

``RowBinding`` maps LM leaves to lanes. Reads pull the leaf's log-probability row
from its lane; feeds are lazy — ``commit`` records the step's token, and each
lane flushes it on its next read, so a row that never reads again (unit end, EOS)
closes without feeding and the engine never forwards the dead step.

The collector batches draws emergently: picks submitted in one loop pass fire as
one keyed device op per vocabulary, and everything scalar leaves the device in a
single crossing per fire.
"""

import asyncio
import contextlib
import contextvars

import torch

from genlm.control.constant import EndOfSequence

# The RowBinding in scope for one row's step; None outside a lane run.
_row_binding: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_row_binding", default=None
)


@contextlib.contextmanager
def row_binding(binding):
    token = _row_binding.set(binding)
    try:
        yield binding
    finally:
        _row_binding.reset(token)


def current_binding():
    return _row_binding.get()


class RowBinding:
    """One particle's ``{leaf: lane}`` map plus its lazy feed state.

    ``commit(item)`` records the step's drawn token; each lane feeds it on that
    lane's next read. ``commit(EOS)`` (or ``close()``) discards the pending feed:
    the engine only needs a feed to compute the step after it.
    """

    def __init__(self, lanes):
        self._lanes = dict(lanes)  # id(leaf) -> Lane
        self._pending: dict[int, int] = {}  # rid -> engine token id
        self.closing = False

    def has(self, leaf) -> bool:
        return id(leaf) in self._lanes

    def lane(self, leaf):
        return self._lanes[id(leaf)]

    @property
    def lanes(self):
        return list(self._lanes.values())

    async def read(self, leaf, engine_context_ids):
        """The leaf's log-probability row for this step: flush the lane's pending
        feed, verify the context, await the warm."""
        lane = self._lanes[id(leaf)]
        pend = self._pending.pop(lane.rid, None)
        if pend is not None:
            lane.feed(pend)
        lane.verify(engine_context_ids)
        return await lane.next()

    def commit(self, item):
        """Record the step's committed item. EOS marks the row closing instead of
        pending a feed."""
        if isinstance(item, EndOfSequence):
            self.closing = True
            return
        token_id = getattr(item, "token_id", item)
        for lane in self._lanes.values():
            self._pending[lane.rid] = int(token_id)

    def close(self):
        self._pending.clear()
        for lane in self._lanes.values():
            lane.close()


class DrawCollector:
    """Emergent batched pick: submissions landing in one loop pass fire together,
    one keyed op per vocabulary, one device->host crossing per fire.

    A submission is ``(logws, slot, step, companion)``; the future resolves to
    ``(token, logZ, logp)`` or ``(token, logZ, logp, comp)`` with a companion.
    The threefry key ``(slot, step)`` makes results batch-composition-independent,
    so a straggler splitting a cohort changes no draw.
    """

    def __init__(self):
        self._parked: list = []
        self._armed = False

    def submit(self, logws, slot, step, companion=None):
        fut = asyncio.get_running_loop().create_future()
        self._parked.append((logws, slot, step, companion, fut))
        if not self._armed:
            self._armed = True
            asyncio.get_running_loop().call_soon(self._fire)
        return fut

    def _fire(self):
        parked, self._parked, self._armed = self._parked, [], False
        by_vocab: dict[int, list] = {}
        for sub in parked:
            by_vocab.setdefault(id(sub[0].decode), []).append(sub)
        for batch in by_vocab.values():
            self._draw_batch(batch)

    def _draw_batch(self, batch):
        from genlm.control.util import draw_key, picker_indices

        W = torch.stack(
            [torch.as_tensor(lw.weights) for lw, _, _, _, _ in batch]
        )
        slots = torch.tensor([s for _, s, _, _, _ in batch], dtype=torch.int64)
        steps = torch.tensor([k for _, _, k, _, _ in batch], dtype=torch.int64)
        logZ = torch.logsumexp(W, dim=-1)
        logps = W - logZ[:, None]
        with draw_key(slots, steps):
            idx = picker_indices(logps)
        ar = torch.arange(len(batch), device=W.device)
        comp_vals = [None] * len(batch)
        comps = [c for _, _, _, c, _ in batch]
        if any(c is not None for c in comps):
            # Companion gathers ride the same crossing; a missing companion
            # gathers a zero row it never reads.
            C = torch.stack(
                [
                    torch.as_tensor(c.weights) if c is not None else W[i]
                    for i, (_, _, _, c, _) in enumerate(batch)
                ]
            )
            comp_vals = C[ar, idx].tolist()
        picked, zs, ids = logps[ar, idx].tolist(), logZ.tolist(), idx.tolist()
        decode = batch[0][0].decode
        for j, (lw, _, _, comp, fut) in enumerate(batch):
            if fut.cancelled():
                continue
            token = decode[ids[j]]
            if comp is None:
                fut.set_result((token, zs[j], picked[j]))
            else:
                fut.set_result((token, zs[j], picked[j], comp_vals[j]))


_collector = DrawCollector()


def collector():
    return _collector
