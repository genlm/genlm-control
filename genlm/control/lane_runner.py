"""The lane runner: opens engine lanes for a controller's rows and keeps them
true to the population across boundaries.

One lane per distinct engine leaf per row, opened under the row's handle with the
group as its cohort key. Lanes persist across rounds — the lazy feed
(:class:`~genlm.control.lane_seam.RowBinding`) holds each committed token until
the lane's next read, so a row that never reads again closes without feeding and
the engine never forwards its dead step. A resample crossing is the only event
that rewrites contexts, so it is the only event that closes + reopens lanes.
"""

from genlm.control.lane_seam import RowBinding
from genlm.control.potential.built_in.llm import lm_leaves


def lane_blocker(controller):
    """Why this controller cannot run with engine lanes, or ``None`` if it can:
    every group's draw path needs exactly one lane-capable engine leaf per view,
    every leaf must live on one shared server, and a critic the run consumes
    per step or at resamples must be lane-servable when it forwards an LM (a
    per-row LM round trip per token would serialize the decode loop)."""
    servers = set()
    for g, lanes in enumerate(controller.group_lanes):
        if not lanes or any(leaf is None for leaf in lanes):
            return f"group {g}: a draw view has no single engine LM leaf"
        for leaf in lanes:
            servers.add(id(leaf.model))
    consumed = controller.twist_with_critic or controller.ess_threshold > 0
    for g, critic in enumerate(controller.critics):
        if (
            critic is not None
            and consumed
            and lm_leaves(critic)
            and controller._critic_lane(critic) is None
        ):
            return f"group {g}: the consumed critic's LM leaves have no single lane"
    if len(servers) != 1:
        return "groups use different engines"
    leaf = controller.group_lanes[0][0]
    if not getattr(leaf.model, "supports_lanes", False):
        return f"engine {type(leaf.model).__name__} does not serve lanes"
    return None


class LaneRunner:
    """Owns the lane lifecycle for one controller run."""

    def __init__(self, controller):
        self.controller = controller
        self.server = controller.group_lanes[0][0].model
        self.bindings = {}  # row index -> RowBinding

    def open_row(self, p):
        """Open one lane per distinct leaf of ``p``'s group, bound to its row."""
        leaves = self.controller.group_lanes[p.group]
        row = self.server.ledger.row_handle()
        lanes = {}
        for leaf in dict.fromkeys(leaves):
            lanes[id(leaf)] = self.server.open_lane(
                leaf.prompt_ids + leaf.encode_tokens(_flat(p.context)),
                lora_name=leaf.lora_name,
                row=row,
                pool_key=int(p.group),
            )
        self.bindings[p.row] = RowBinding(lanes)

    def binding_of(self, p):
        return self.bindings.get(p.row)

    def close_row(self, p):
        binding = self.bindings.pop(p.row, None)
        if binding is not None:
            binding.close()

    def after_round(self, g, crossed):
        """Reconcile group ``g``'s lanes with the population after its boundary:
        close every done row's lanes; on a crossing, close + reopen every live
        row's (the resample rewrote their contexts — survivors included)."""
        for i in self.controller.group_rows(g):
            p = self.controller.particles[i]
            if p.done:
                self.close_row(p)
            elif crossed:
                self.close_row(p)
                self.open_row(p)

    def close_all(self):
        for binding in self.bindings.values():
            binding.close()
        self.bindings.clear()


def _flat(context):
    from genlm.control.util import flatten_units

    return flatten_units(context)
