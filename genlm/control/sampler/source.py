"""Logit sources: where a SMC step's proposal LM logits come from.

The one lane-dependent piece of an otherwise shared SMC step. Two sources:

* :class:`PullSource` -- control PULLS the logits: ``await potential.logw_next``
  per row on the event loop (numpy). No engine coupling.
* :class:`PushSource` -- the vLLM engine PUSHES a batched ``[N, V+1]`` tensor into
  the Controller's ``draw`` callback while it drives its own decode loop. Owns ALL
  the engine-coupled state -- the live row<->request-id maps, abort/re-add, the
  warm-KV factor fold, the worker-thread event-loop hop -- so that coupling lives
  in the source, never in the model or the sampler.

The Controller keeps ``draw``/``drain_aborts``/``drain_adds`` (the backend's
``run_burst(control=...)`` calls them) but reads its engine state through the
``PushSource`` it was handed; resample/ESS/log_ml stay Controller-owned.
"""

import asyncio

import numpy as np
import torch

from genlm.control.constant import EndOfSequence
from genlm.control.sampler.util import _drive_or_hop


class PullSource:
    """Control-driven logit source (the slow lane): each row's proposal logits are
    ``await target.logw_next(context)`` on the event loop, in numpy.

    Carries no engine state; ``resample_realize`` is a no-op because the Scheduler
    has already reindexed the population arrays. Thin for now -- the slow
    ``StepLoop`` still draws via ``sampler.transition`` until the draw unforks
    (step 3); this is the object that draw will pull from."""

    def __init__(self, target):
        self.target = target

    async def logws(self, particles):
        return await asyncio.gather(
            *[self.target.logw_next(p.context) for p in particles]
        )

    def resample_realize(self):
        pass


class PushSource:
    """Engine-driven logit source (the fast lane): the vLLM engine runs its decode
    loop and pushes a batched ``[L, V+1]`` logit tensor into ``Controller.draw``
    once per step. This object owns everything engine-coupled.

    Run-scoped config (``__init__``): the engine LM, the optional additive factor
    and the product target, the gather maps onto the product vocab (the same maps
    ``Product.logw_next`` uses, resolved here against llm/factor), each group's
    prompt prefix, and the EOS placeholder id. The worker-thread event-loop hop is
    bound once per run via :meth:`bind_loop`.

    Per-burst mutable state (:meth:`begin_burst`): the live particle list, the
    external-id <-> population-row maps, the abort/re-add queues, the sampler
    scratch dict, and the exit reason. A burst is a stateless enter-run-exit unit;
    these reset each burst.

    The sampler-facing surface a ``burst_draw_batch`` uses is exposed directly
    (``factor``, ``scratch``, :meth:`product_logws`, :meth:`factor_logws`,
    :meth:`run_sync`) -- this replaces the old separate ``BurstContext``.
    """

    def __init__(self, controller, llm, factor, target, group_prefixes):
        self.controller = controller
        self.llm = llm
        self.factor = factor
        self.target = target
        # Snapshot of each group's prompt prefix (main thread). ``prompt_ids`` is a
        # thread-local ContextVar; the mid-burst re-add runs on the worker thread
        # where the override is invisible, so read the snapshot, not the live var.
        self.group_prefixes = group_prefixes

        # Gather maps onto the product vocab_eos (the same maps Product.logw_next
        # uses, resolved against llm/factor so p1/p2 order is irrelevant). None when
        # unconstrained.
        if factor is None:
            self.llm_idxs = None
            self.factor_idxs = None
        else:
            self.llm_idxs = np.array([llm.lookup[t] for t in target.vocab_eos])
            self.factor_idxs = np.array([factor.lookup[t] for t in target.vocab_eos])

        # The engine token id committed as the placeholder for an aborted / EOS row
        # (EOS carries no token_id). The row is aborted regardless, so any EOS id
        # serves; take the first. No EOS configured -> can't burst (fail loud).
        eos_idxs = list(llm.token_maps.eos_idxs)
        if not eos_idxs:
            raise ValueError(
                "Engine LM has no EOS token id; the burst path needs one as the "
                'committed placeholder for aborted rows. Use accelerate="off".'
            )
        self.eos_id = eos_idxs[0]

        # Bound by bind_loop / begin_burst.
        self._run_async = None
        self.particles = None
        self.abort_rows = set()
        self.add_rows = []
        self.ext_to_row = {}
        self.row_to_ext = {}
        self.next_ext = 0
        self.scratch = {}
        self.exit_reason = None

    # -- lifecycle --------------------------------------------------------------

    def bind_loop(self, loop):
        """Bind the worker-thread -> event-loop hop for this run. ``draw`` runs in a
        worker thread; an awaiting factor/critic call is scheduled back onto
        ``loop`` and blocked on."""
        self._run_async = lambda coro: asyncio.run_coroutine_threadsafe(
            coro, loop
        ).result()

    def begin_burst(self, live):
        """Reset the per-burst state for a fresh burst over ``live`` particles.

        Initial rows: external id ``i`` is the entry-order index, mapping to that
        live particle's population row (``p._i``). ``next_ext`` hands out a FRESH id
        per mid-burst re-add so an abort and a re-add of a slot never collide."""
        self.particles = live
        self.abort_rows = set()
        self.add_rows = []
        self.ext_to_row = {i: p._i for i, p in enumerate(live)}
        self.row_to_ext = {p._i: i for i, p in enumerate(live)}
        self.next_ext = len(live)
        self.scratch = {}
        self.exit_reason = None

    # -- engine LM fold + sampler-facing factor surface ------------------------

    def fold(self, logits):
        """The expensive 50k-vocab LM processing, ONCE for the whole batch on-device:
        the engine logits -> control-side ``[L, V+1]`` log-weights."""
        return self.llm._process_logw_next_batch(self.llm._maybe_temper(logits.float()))

    def factor_logws(self, context):
        """The factor's ``logw_next(context)`` on the driver's event loop -- the
        burst stand-in for the slow ``factor.logw_next(context)`` (the wrapped
        potential's ``_consume`` cache carries it incrementally across the burst).
        Inline for a non-suspending (FSA) factor; hop to the loop only if it awaits."""
        return _drive_or_hop(lambda: self.factor.logw_next(context), self._run_async)

    def product_logws(self, lm_weights, context):
        """Reconstruct ``Product(llm, factor).logw_next`` over the target vocab from
        the engine LM weights (one row of the folded batch) and
        ``factor.logw_next(context)``, gathering through the exact slow-path index
        maps so vocab narrowing matches."""
        factor_logws = self.factor_logws(context)
        return self.target.make_lazy_weights(
            lm_weights[self.llm_idxs] + factor_logws.weights[self.factor_idxs]
        )

    def run_sync(self, coro):
        """Run a sampler's async helper (e.g. AWRS's rejection) to completion on the
        driver's event loop and block -- one event-loop hop, no inner gather."""
        return self._run_async(coro)

    # -- engine prompt + row mapping -------------------------------------------

    def context_ids(self, p):
        """Engine prompt for a particle: its group's prompt prefix + drawn token ids.

        A token-grain context is a flat list of ``Token``s; a unit-grain context is
        a list of units (each a list of subunit ``Token``s). Both flatten the same:
        recurse into unit lists, take each token id, drop EOS sentinels. Reads the
        main-thread prefix snapshot (runs on the worker thread during re-add)."""
        ids = list(self.group_prefixes[self.controller.particles.group[p._i]])

        def _emit(item):
            if isinstance(item, EndOfSequence):
                return
            if isinstance(item, list):
                for sub in item:
                    _emit(sub)
            else:
                ids.append(item.token_id)

        for item in p.context:
            _emit(item)
        return ids

    def external_of(self, request_id):
        """The external particle index for a vLLM internal request id
        (``"{external}-{8 chars}"``); the BurstLoop assigned ``str(index)``."""
        return int(request_id.rsplit("-", 1)[0])

    def particle_of(self, request_id):
        """Map a vLLM request id to its particle via its external id -> population
        row (so re-added rows, which carry fresh ids, resolve to the right slot)."""
        return self.controller.particles[self.ext_to_row[self.external_of(request_id)]]

    # -- pop-out / re-add plumbing (consumed by run_burst) ---------------------

    def drain_aborts(self):
        """External row indices ``draw`` flagged to abort since the last call
        (consumed). ``run_burst`` issues ``abort_request`` for them -- the
        out-of-band pop-out that replaces the old forced-EOS."""
        rows = self.abort_rows
        self.abort_rows = set()
        return list(rows)

    def drain_adds(self):
        """Rows the controller asks ``run_burst`` to (re-)add this step, as
        ``(ext_id, prompt_ids)`` (consumed). The in-place resample queues a
        resampled group's surviving rows here; empty otherwise."""
        rows = self.add_rows
        self.add_rows = []
        return rows

    def resample_realize(self):
        """The burst's mid-stream per-group resample: reindex the groups that crossed
        ESS (Controller-owned), then -- WITHOUT draining the burst -- drop each
        resampled group's current engine rows and re-add its surviving rows at their
        resampled contexts (fresh ids). The engine re-prefills the re-added rows
        (prefix-cache-warm). Only translates a completed resample into engine
        abort/add plumbing.

        Resampling can flip a row done<->live, so re-adds are decided over each
        group's FULL post-reindex state: every still-live row is re-added."""
        c = self.controller
        c._maybe_resample()  # reindexes crossing groups; sets _last_resampled_groups
        for g in c._last_resampled_groups:
            rows = c._group_rows[g]
            # Drop the group's current engine rows -- their KV holds pre-resample
            # tokens. (A row already aborted on termination is harmless to re-abort.)
            for row in rows:
                ext = self.row_to_ext.pop(int(row), None)
                if ext is not None:
                    self.abort_rows.add(ext)
                    self.ext_to_row.pop(ext, None)
            # Re-add the still-live rows at their resampled context (fresh id).
            for row in rows:
                p = c.particles[int(row)]
                if p.done:
                    continue
                ext = self.next_ext
                self.next_ext += 1
                self.ext_to_row[ext] = int(row)
                self.row_to_ext[int(row)] = ext
                self.add_rows.append((ext, self.context_ids(p)))
