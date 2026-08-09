"""Lane-serving seam: ContextVars bind a row's step to its engine lanes and a
group boundary to its banked sums.

``RowBinding`` maps LM leaves to lanes. Reads pull the leaf's log-probability
row; the feed is the context delta between reads, so a row that never reads
again (EOS, a crossing) closes without feeding and the engine never forwards
the dead step.

The collector batches draws emergently: picks submitted in one loop pass fire as
one keyed device op per vocabulary, and everything scalar leaves the device in a
single crossing per fire.
"""

import asyncio
import contextlib
import contextvars

import torch

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
    """One particle's ``{leaf: lane}`` map. The feed is the context delta: each
    read carries the caller's engine-id context, and the tokens it holds beyond
    the lane's are fed before awaiting the warm — so a committed token reaches
    the engine only when the row reads again, and a row that never reads again
    (EOS, unit end into a crossing) closes without feeding its dead step."""

    def __init__(self, lanes):
        self._lanes = dict(lanes)  # id(leaf) -> Lane

    def has(self, leaf) -> bool:
        return id(leaf) in self._lanes

    def lane(self, leaf):
        return self._lanes[id(leaf)]

    @property
    def lanes(self):
        return list(self._lanes.values())

    async def read(self, leaf, engine_context_ids):
        """The leaf's raw log-probability row at ``engine_context_ids``: feed the
        delta beyond the lane's context (banking its processed log-prob into the
        lane — the leaf's own running ``prefix``), then await the warm. A context
        that is not the lane's plus at most one new token cannot be served — the
        lane stepped past it or never held it."""
        lane = self._lanes[id(leaf)]
        held = len(lane.context)
        delta = list(engine_context_ids[held:])
        lane.verify(engine_context_ids[:held])
        if len(delta) > 1:
            raise RuntimeError(
                f"lane {lane.rid} was read {len(delta)} tokens ahead of its "
                "context; a lane serves one step per read"
            )
        if delta:
            if lane.stash is not None:
                lane.bank += lane.stash[leaf.token_maps.decode[delta[0]]]
            lane.feed(delta[0])
        return await lane.next()

    def close(self):
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
