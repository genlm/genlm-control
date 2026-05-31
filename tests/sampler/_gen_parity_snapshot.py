"""Generator for the per-token SMC parity snapshot.

Run while ``llamppl`` is still installed to capture the ground-truth outputs of
the original ``smc_standard`` path:

    python tests/sampler/_gen_parity_snapshot.py

This writes ``tests/sampler/parity_snapshot.json``, which ``test_per_token_parity.py``
loads and compares the new controller-driven ``SMC`` path against. Snapshotting the
reference path (rather than running llamppl live in the test) lets the parity
gate keep guarding the controller after the llamppl dependency is removed.

The reference ``_RefSequenceModel`` is a verbatim transcription of the deleted
``SequenceModel.step`` semantics; it calls ``unit_sampler.sample(context)``
directly instead of through the old llamppl ``SubModel.forward`` indirection,
which was a pure plumbing wrapper, so RNG-consumption order is preserved
step-for-step.
"""

import asyncio
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genlm.control.constant import EOS  # noqa: E402
from genlm.control.sampler.sequence import Sequences  # noqa: E402
from genlm.control.util import escape  # noqa: E402

from llamppl import Model, smc_standard  # noqa: E402

from _harness import seed_all, ctx_repr, num  # noqa: E402
from parity_cases import (  # noqa: E402
    SAMPLER_BUILDERS,
    matrix_combos,
    N_PARTICLES,
    MAX_TOKENS,
    SEED,
)


class _RefSequenceModel(Model):
    """Verbatim transcription of the deleted SequenceModel."""

    def __init__(
        self,
        unit_sampler,
        critic=None,
        max_tokens=float("inf"),
        twist_with_critic=True,
    ):
        assert max_tokens > 0
        super().__init__()
        self.token_ctx = []
        self.unit_sampler = unit_sampler
        self.max_tokens = max_tokens
        self.critic = critic
        self.logp = 0
        self.twist_with_critic = twist_with_critic

    async def start(self):
        start_w = await self.unit_sampler.start_weight()
        if start_w == float("-inf"):
            raise ValueError("Start weight is -inf (log(0)).")
        self.score(start_w)

    async def step(self):
        from genlm.control.sampler.unit import MultiTokenUnitSampler

        if isinstance(self.unit_sampler, MultiTokenUnitSampler):
            from genlm.control.sampler.unit import flatten_units

            flat_context = flatten_units(self.token_ctx)
            unit, logw, logp = await self.unit_sampler.sample(
                flat_context, unit_context=self.token_ctx, draw=None
            )
            self.score(logw)
            self.logp += logp
            if unit and unit[-1] is EOS:
                if len(unit) > 1:
                    self.token_ctx.append(unit[:-1])
                self.token_ctx.append(EOS)
            else:
                self.token_ctx.append(unit)
        else:
            token, logw, logp = await self.unit_sampler.sample(self.token_ctx)
            self.score(logw)
            self.logp += logp
            self.token_ctx.append(token)

        if self.weight == float("-inf"):
            if self.critic:
                assert self.twist_amount != float("-inf")
            self.finish()
            return

        if self.critic and self.twist_with_critic:
            twist_amt = await self.critic.score(self.token_ctx)
            if twist_amt != float("-inf"):
                self.twist(twist_amt)
            else:
                self.score(twist_amt)
                self.finish()
                return

        self.max_tokens -= 1
        if self.max_tokens == 0 or self.token_ctx[-1] is EOS:
            self.finish()
            if self.critic:
                if not self.twist_with_critic:
                    twist_amt = await self.critic.score(self.token_ctx)
                self.score(twist_amt)
            return

    def string_for_serialization(self):
        return "|".join(escape(y) for y in self.token_ctx)

    def immutable_properties(self):
        return set(["unit_sampler", "critic"])


async def _run_reference(unit_sampler, critic, ess_threshold, json_path):
    model = _RefSequenceModel(
        unit_sampler=unit_sampler,
        critic=critic,
        max_tokens=MAX_TOKENS,
        twist_with_critic=ess_threshold > 0,
    )
    particles = await smc_standard(
        model=model,
        n_particles=N_PARTICLES,
        ess_threshold=ess_threshold,
        json_file=json_path,
    )
    contexts, logws = map(
        list,
        zip(
            *[
                (p.token_ctx, float("-inf") if np.isnan(p.weight) else p.weight)
                for p in particles
            ]
        ),
    )
    seqs = Sequences(contexts, logws)
    return seqs


def _gen_case(sampler_name, use_critic, ess):
    """Generate one snapshot case in its own fresh event loop.

    Each case is run via its own ``asyncio.run`` so the cross-particle RNG
    scheduling matches ``test_per_token_parity.py`` (which also uses a fresh
    ``asyncio.run`` per case) exactly.
    """
    import tempfile

    seed_all(SEED)
    sampler, critic_pot = SAMPLER_BUILDERS[sampler_name]()
    critic = critic_pot if use_critic else None
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        json_path = tf.name
    seqs = asyncio.run(_run_reference(sampler, critic, ess, json_path))
    with open(json_path) as f:
        record = f.read()
    os.unlink(json_path)

    return {
        "contexts": [ctx_repr(c) for c, _ in seqs],
        "logws": [num(w) for _, w in seqs],
        "log_ml": num(seqs.log_ml),
        "record": record,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="regen only this builder (exact sampler name)")
    args = ap.parse_args()

    dst = os.path.join(os.path.dirname(__file__), "parity_snapshot.json")
    # Merge into the existing snapshot so a partial regen (e.g. --only awrs) adds new keys
    # without disturbing the frozen reference for the other samplers.
    try:
        with open(dst) as f:
            out = json.load(f)
    except FileNotFoundError:
        out = {}
    n_new = 0
    for sampler_name, use_critic, ess in matrix_combos():
        if args.only and sampler_name != args.only:
            continue
        key = f"{sampler_name}|critic={use_critic}|ess={ess}"
        out[key] = _gen_case(sampler_name, use_critic, ess)
        n_new += 1
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote/updated {n_new} cases ({len(out)} total) to {dst}")


if __name__ == "__main__":
    main()
