"""Generate gate-2's slow reference from the ORIGINAL genlm-control (main +
llamppl ``smc_standard``), not our StepLoop.

The gate-2 reference is the genuine original genlm-control behavior -- the released
per-token SMC the downstream (lvar) actually uses -- cached once. The gate-2 test
then only LOADS this cache and compares the engine-native ``BurstLoop`` against it;
StepLoop is no longer the reference (it is the production fallback, pinned ==
original by gate-1). main and this branch share the potential/sampler/PromptedLLM
API verbatim -- the only difference is the SMC engine -- so the configs below are
the same builders as ``test_engine_native.py``, just run through main's llamppl SMC
(no ``accelerate=``).

Run on the box, shadowing our genlm.control with main's:

    PYTHONPATH=/root/genlm/genlm-control-main VLLM_USE_FLASHINFER_SAMPLER=0 \
        OMP_NUM_THREADS=1 /root/genlm-venv/bin/python \
        tests/sampler/gen_original_reference.py

Writes ``gate2_snapshot.json`` next to this file (the same path + key format the
gate-2 test loads). Pass ``--only <substr>`` to regenerate a subset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import numpy as np
import torch

from genlm.control.constant import EndOfSequence
from genlm.control.potential import Potential
from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.potential.built_in.wfsa import BoolFSA
from genlm.control.sampler.token import AWRS, DirectTokenSampler, SetTokenSampler
from genlm.control.sampler.unit import BoundaryPredicate, MultiTokenUnitSampler
from genlm.control.sampler import EagerSetSampler
from genlm.control.sampler.sequence import SMC

MODEL = "gpt2"
_PROMPT = "The"
_EOS_BYTES = [b"\n"]
_CONFIG = {"model": MODEL, "prompt": _PROMPT, "eos_hex": [b.hex() for b in _EOS_BYTES]}
_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "gate2_snapshot.json")


def _seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def _ctx_ids(ctx):
    out = []

    def emit(t):
        if isinstance(t, EndOfSequence):
            out.append("EOS")
        elif isinstance(t, list):
            for s in t:
                emit(s)
        else:
            out.append(t.token_id)

    for t in ctx:
        emit(t)
    return out


def _key(label, n_particles, ess_threshold, max_tokens, seed):
    return f"{label}|N={n_particles}|ess={ess_threshold}|mt={max_tokens}|seed={seed}"


# --- the gate-2 critics / boundary, verbatim from test_engine_native.py ---------


class _TerminalContainsCritic(Potential):
    def __init__(self, vocab):
        super().__init__(vocabulary=vocab)

    async def _indicator(self, context):
        bs = [t for t in context if not isinstance(t, EndOfSequence)]
        try:
            text = b"".join(bs).decode("utf-8")
        except UnicodeDecodeError:
            return float("-inf")
        return 0.0 if " " in text else float("-inf")

    async def complete(self, context):
        return await self._indicator(context)

    async def prefix(self, context):
        return 0.0

    async def score(self, context):
        return await self._indicator(context)


class _SoftVowelCritic(Potential):
    def __init__(self, vocab):
        super().__init__(vocabulary=vocab)

    def _pen(self, context):
        bs = [t for t in context if not isinstance(t, EndOfSequence)]
        try:
            text = b"".join(bs).decode("utf-8")
        except UnicodeDecodeError:
            return float("-inf")
        return -0.5 * sum(c in "aeiouAEIOU" for c in text)

    async def complete(self, context):
        return self._pen(context)

    async def prefix(self, context):
        return self._pen(context)


class _ByteLengthBoundary(BoundaryPredicate):
    def __init__(self, min_bytes):
        self.min_bytes = min_bytes

    def __call__(self, unit_context, subunit_buffer):
        return (
            sum(len(t) for t in subunit_buffer if not isinstance(t, EndOfSequence))
            >= self.min_bytes
        )


# --- config table: (label, N, ess, mt, seeds, build_sampler(llm), build_critic) -


def _build_configs(llm):
    """All gate-2 configs as (key-fields, sampler factory, critic factory). The
    factories take ``llm`` and return fresh objects per run (AWRS carries its own
    RNG, so a fresh one per seed keeps it seeded by ``_seed`` + its own seed)."""

    def boolfsa(regex):
        return llm * BoolFSA.from_regex(regex).coerce(llm, f=b"".join)

    # Build the trie set sampler once (heavy init), reuse across its seeds.
    set_sampler = EagerSetSampler(
        iter_potential=llm, item_potential=BoolFSA.from_regex(r"[a-z ]+")
    )

    S6 = (1234, 7, 99, 2024, 555, 31)
    S12 = (1234, 7, 99, 2024, 555, 31, 8, 17, 42, 123, 271, 314)

    configs = [
        ("unconstrained", 8, 0.0, 12, (1234,), lambda: DirectTokenSampler(llm), None),
        ("unconstrained", 8, 0.5, 12, (1234,), lambda: DirectTokenSampler(llm), None),
        ("boolfsa[a-z ]+", 16, 0.0, 12, (1234,), lambda: DirectTokenSampler(boolfsa(r"[a-z ]+")), None),
        ("boolfsa[a-z ]+", 16, 0.5, 12, (1234,), lambda: DirectTokenSampler(boolfsa(r"[a-z ]+")), None),
        ("boolfsa[aeiou ]+", 16, 0.5, 10, (1234, 7, 99, 2024, 555, 31, 808, 42, 17, 6, 71, 900),
            lambda: DirectTokenSampler(boolfsa(r"[aeiou ]+")), None),
        ("terminal-critic", 16, 0.0, 12, S6,
            lambda: DirectTokenSampler(llm), lambda: _TerminalContainsCritic(llm.vocab)),
        ("twist-critic", 16, 0.5, 12, S12,
            lambda: DirectTokenSampler(llm), lambda: _SoftVowelCritic(llm.vocab)),
        ("multitoken-boolfsa[a-z ]+", 8, 0.5, 6, S12,
            lambda: MultiTokenUnitSampler(
                DirectTokenSampler(boolfsa(r"[a-z ]+")), _ByteLengthBoundary(5), max_subunits_per_unit=6
            ), None),
        ("set[a-z ]+", 8, 0.0, 8, S6, lambda: SetTokenSampler(set_sampler), None),
        # 2d slow-lane base: the SAME constrained config the cadence test bursts,
        # run here WITHOUT any cadence (the original has no such concept) -- the
        # cadence is math-neutral, so the interleaved burst must match this.
        ("slowcadence-base[a-z ]+", 16, 0.5, 12, S6,
            lambda: DirectTokenSampler(boolfsa(r"[a-z ]+")), None),
    ]
    # AWRS needs a per-seed seed -> make the sampler factory close over the seed.
    for seed in S6:
        configs.append(
            ("awrs[a-z ]+", 16, 0.0, 12, (seed,),
                (lambda s: (lambda: AWRS(llm, BoolFSA.from_regex(r"[a-z ]+").coerce(llm, f=b"".join), seed=s)))(seed),
                None)
        )
    return configs, set_sampler


async def _run(make_sampler, n_particles, ess_threshold, max_tokens, seed, make_critic):
    _seed(seed)
    seqs = await SMC(make_sampler(), make_critic() if make_critic else None)(
        n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=max_tokens
    )
    return {
        "contexts": [_ctx_ids(c) for c in seqs.contexts],
        "logw": [float(w) for w in seqs.log_weights],
        "log_ml": float(seqs.log_ml),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="regen only configs whose label contains this")
    args = ap.parse_args()

    from genlm.backend.llm import AsyncVirtualLM

    model = AsyncVirtualLM.from_name(
        MODEL,
        engine_opts={"gpu_memory_utilization": 0.3, "max_model_len": 256, "enable_prefix_caching": True},
    )
    llm = PromptedLLM(model, eos_byte_strings=_EOS_BYTES)
    llm.set_prompt_from_str(_PROMPT)

    try:
        with open(_SNAPSHOT_PATH) as f:
            snap = json.load(f)
    except FileNotFoundError:
        snap = {}
    snap.pop("__config__", None)

    configs, set_sampler = _build_configs(llm)
    try:
        for label, N, ess, mt, seeds, make_s, make_c in configs:
            if args.only and args.only not in label:
                continue
            for seed in seeds:
                key = _key(label, N, ess, mt, seed)
                snap[key] = await _run(make_s, N, ess, mt, seed, make_c)
                print(f"  {key}: log_ml={snap[key]['log_ml']:.6f} n={len(snap[key]['contexts'])}")
                snap["__config__"] = _CONFIG
                with open(_SNAPSHOT_PATH, "w") as f:
                    json.dump(snap, f, indent=1, sort_keys=True)
    finally:
        await set_sampler.cleanup()
    print(f"wrote {_SNAPSHOT_PATH} ({len(snap) - 1} reference entries) from ORIGINAL genlm-control")


if __name__ == "__main__":
    asyncio.run(main())
