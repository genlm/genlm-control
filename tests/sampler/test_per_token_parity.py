"""Ground-truth per-token parity gate for the SMC hub.

This compares the new hub-driven slow path (``genlm.control.sampler.SMC``)
against a stored snapshot of the original llamppl ``smc_standard`` path
(``parity_snapshot.json``, produced by ``_gen_parity_snapshot.py`` while
llamppl was still installed). Under a fixed numpy+torch seed the per-particle
(context, logw), ``Sequences.log_ml``, and the serialized JSON record must
match the snapshot exactly: weights atol ~1e-9, contexts exact, JSON exact
modulo float formatting.

The matrix exercises {no critic, terminal critic} x ess in {0, 0.5, 1} over
DirectTokenSampler (on WeightedSet and MockPotential) and MultiTokenUnitSampler;
the ess=0.5 + critic cases exercise the twist/untwist-across-resample
interaction.

Regenerate the snapshot (only when the algorithm intentionally changes) with:
    python tests/sampler/_gen_parity_snapshot.py
"""

import json
import os

import numpy as np
import pytest

from genlm.control.constant import EOS
from genlm.control.sampler.sequence import SMC

from parity_cases import SAMPLER_BUILDERS, MATRIX, N_PARTICLES, MAX_TOKENS, SEED

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "parity_snapshot.json")

if not os.path.exists(SNAPSHOT_PATH):  # pragma: no cover
    pytest.skip(
        "parity_snapshot.json missing; run tests/sampler/_gen_parity_snapshot.py",
        allow_module_level=True,
    )

with open(SNAPSHOT_PATH) as _f:
    SNAPSHOT = json.load(_f)


def _seed(s=SEED):
    np.random.seed(s)
    import torch

    torch.manual_seed(s)


def _ctx_repr(ctx):
    """Mirror the generator's context serialization for exact comparison."""

    def one(t):
        if t is EOS or hasattr(t, "type_"):
            return f"<EOS:{getattr(t, 'type_', 'EOS')}>"
        if isinstance(t, list):
            return [one(x) for x in t]
        if isinstance(t, bytes):
            return "b:" + t.hex()
        return repr(t)

    return [one(t) for t in ctx]


def _num(x):
    if np.isnan(x):
        return "nan"
    if np.isneginf(x):
        return "-inf"
    if np.isposinf(x):
        return "inf"
    return float(x)


def _canonical_record(record_text):
    """Canonicalize a record JSON string so only structurally meaningful
    differences register: NaN compares equal, infinities normalized, finite
    floats round-tripped through float() to erase formatting noise."""
    history = json.loads(record_text)

    def fix(s):
        if s == "-Infinity":
            return "-inf"
        f = float(s)
        if np.isnan(f):
            return "nan"
        if np.isneginf(f):
            return "-inf"
        if np.isposinf(f):
            return "inf"
        return f

    out = []
    for entry in history:
        e = {"mode": entry["mode"], "step": entry["step"]}
        if "ancestors" in entry:
            e["ancestors"] = entry["ancestors"]
        e["particles"] = [
            {
                "contents": p["contents"],
                "logweight": fix(p["logweight"]),
                "weight_incr": fix(p["weight_incr"]),
            }
            for p in entry["particles"]
        ]
        out.append(e)
    return out


@pytest.mark.parametrize("sampler_name", list(SAMPLER_BUILDERS))
@pytest.mark.parametrize("use_critic", [False, True])
@pytest.mark.parametrize("ess_threshold", MATRIX)
def test_per_token_parity(sampler_name, use_critic, ess_threshold, tmp_path):
    # NOTE: this is intentionally a synchronous test that drives the hub with a
    # fresh ``asyncio.run`` event loop, matching the snapshot generator. For
    # potentials whose ``logw_next`` awaits (e.g. WeightedSet's base
    # implementation does an inner ``asyncio.gather``), the cross-particle
    # RNG-consumption order depends on event-loop task scheduling -- a property
    # the original llamppl path shares. pytest-asyncio's shared per-session loop
    # machinery can be perturbed by a preceding ``@given`` async test, shifting
    # that scheduling; a fresh ``asyncio.run`` per case removes the dependence
    # and keeps the gate order-independent.
    import asyncio

    key = f"{sampler_name}|critic={use_critic}|ess={ess_threshold}"
    snap = SNAPSHOT[key]

    _seed()
    sampler, critic_pot = SAMPLER_BUILDERS[sampler_name]()
    critic = critic_pot if use_critic else None
    json_path = str(tmp_path / "got.json")
    got = asyncio.run(
        SMC(sampler, critic=critic)(
            n_particles=N_PARTICLES,
            ess_threshold=ess_threshold,
            max_tokens=MAX_TOKENS,
            json_path=json_path,
        )
    )

    # Per-particle contexts (exact) and log weights (atol 1e-9).
    got_contexts = [_ctx_repr(c) for c, _ in got]
    assert got_contexts == snap["contexts"], "context mismatch"

    for (_, gw), sw in zip(got, snap["logws"]):
        if sw in ("nan", "-inf", "inf"):
            assert _num(gw) == sw
        else:
            assert abs(gw - sw) <= 1e-9, f"logw mismatch: {gw} vs {sw}"

    # log_ml.
    gml = _num(got.log_ml)
    if snap["log_ml"] in ("nan", "-inf", "inf"):
        assert gml == snap["log_ml"]
    else:
        assert abs(got.log_ml - snap["log_ml"]) <= 1e-9

    # JSON record (exact modulo float formatting).
    with open(json_path) as f:
        got_record = f.read()
    assert _canonical_record(got_record) == _canonical_record(snap["record"])
