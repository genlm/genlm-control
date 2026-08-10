"""Shared harness for the unified speedup benchmark (``bench.py``).

This module owns everything that is NOT scenario-specific: engine build, seeding,
the smc/raw run matrix, timed trials, and a neat persistent results store
so the three control versions (main / speedup-old / speedup-now) are recorded once
and compared as the design iterates -- never re-run a baseline you already have.

Design constraint: the SAME file must run against any control version, so all
``genlm`` imports are LAZY (inside functions) and nothing here depends on an API
newer than ``SMC(...)``.
"""

from __future__ import annotations

import inspect
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- #
# Version / environment tagging                                                #
# --------------------------------------------------------------------------- #
def _sh(*cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def version_tag(label: str) -> dict:
    """Identify the control checkout under test. ``label`` is the user-facing key
    for cross-version compare (e.g. ``main`` / ``speedup-old`` / ``speedup-now``).
    git metadata is best-effort (the box checkouts are rsync'd without .git)."""
    import genlm.control as ctl

    here = Path(ctl.__file__).resolve().parent
    sha = _sh("git", "-C", str(here), "rev-parse", "--short", "HEAD")
    dirty = bool(_sh("git", "-C", str(here), "status", "--porcelain"))
    return {
        "label": label,
        "sha": sha or None,
        "dirty": dirty,
        "control_path": str(here),
        "version": getattr(ctl, "__version__", None),
    }


def env_tag() -> dict:
    import torch

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    try:
        import vllm

        vllm_v = vllm.__version__
    except Exception:
        vllm_v = None
    return {
        "host": platform.node(),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "vllm": vllm_v,
    }


# --------------------------------------------------------------------------- #
# Engine + seeding                                                             #
# --------------------------------------------------------------------------- #
def build_engine(model: str, engine_opts: dict, backend: str = "vllm"):
    """The engine under test. ``engine_opts`` are vLLM's; other backends take none."""
    from genlm.backend.llm import load_model_by_name

    opts = {"engine_opts": engine_opts} if backend == "vllm" else {}
    return load_model_by_name(model, backend=backend, llm_opts=opts)


def seed_all(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# The run matrix: smc (this checkout) / raw (engine decode ceiling)            #
# --------------------------------------------------------------------------- #
async def run_smc(sampler, critic, *, n_particles: int, max_tokens: int,
                  ess_threshold: float, seed: int):
    """One SMC run. How the engine serves it is a property of the checkout, so a
    speedup is read ACROSS versions of the same scenario, never across paths."""
    from genlm.control.sampler.sequence import SMC

    seed_all(seed)
    smc = SMC(sampler, critic=critic)
    t0 = time.perf_counter()
    seqs = await smc(n_particles=n_particles, ess_threshold=ess_threshold,
                     max_tokens=max_tokens)
    return time.perf_counter() - t0, seqs


def raw_ceiling(model, prompt_ids, *, n_particles: int, max_tokens: int) -> float:
    """Stock vLLM batch decode of N sequences -- no SMC at all. The engine's native
    decode floor; (smc - raw) is the residual control-side cost.

    vLLM only: no other backend exposes a batched no-control decode, so elsewhere the
    matrix is the smc run alone."""
    if not hasattr(model, "llm_engine"):
        raise _Unsupported(f"raw ceiling needs a vLLM engine, not {type(model).__name__}")
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(n=1, max_tokens=max_tokens, ignore_eos=True, detokenize=False)
    prompts = [TokensPrompt(prompt_token_ids=list(prompt_ids)) for _ in range(n_particles)]
    t0 = time.perf_counter()
    model.llm_engine.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0


class _Unsupported(Exception):
    """A (version, path) combination this checkout cannot run."""


# --------------------------------------------------------------------------- #
# Results store                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    scenario: str
    config: dict           # the compare key fields (model, sampler, constraint, N, ...)
    path: str              # smc | raw
    dt_median: float
    dt_all: list
    version: dict
    env: dict
    log_ml: Optional[float] = None
    mean_len: Optional[float] = None
    extra: dict = field(default_factory=dict)
    ts: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def store_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{name}.jsonl"


def record(name: str, result: Result) -> None:
    if not result.ts:
        result.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with store_path(name).open("a") as f:
        f.write(json.dumps(result.to_json()) + "\n")


def load(name: str) -> list[dict]:
    p = store_path(name)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _config_key(cfg: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in cfg.items()))


def compare_table(name: str) -> str:
    """Render the store as: one block per (scenario, config); rows = versions;
    columns = smc / raw + the derived control overhead. The whole point of the
    persisted store -- diff the current design against recorded baselines, and read
    the speedup DOWN a column (across versions), not across paths."""
    rows = load(name)
    if not rows:
        return f"(no results in {store_path(name)})"
    # group by scenario+config, then by version label, collecting per-path dt
    groups: dict = {}
    for r in rows:
        gk = (r["scenario"], _config_key(r["config"]))
        v = r["version"]["label"]
        groups.setdefault(gk, {}).setdefault(v, {})[r["path"]] = r["dt_median"]

    out: list[str] = []
    for (scenario, ckey), per_version in sorted(groups.items(), key=lambda x: x[0][0]):
        cfg = dict((k, v) for k, v in ckey)
        out.append("=" * 78)
        out.append(f"{scenario}  |  " + "  ".join(f"{k}={v}" for k, v in cfg.items()))
        out.append(f"  {'version':<16}{'smc':>12}{'raw(ceil)':>12}"
                    f"{'smc/raw':>12}{'vs first':>12}")
        base = None
        for v, paths in sorted(per_version.items()):
            smc = paths.get("smc")
            raw = paths.get("raw")
            if base is None:
                base = smc
            out.append(
                f"  {v:<16}"
                f"{_fmt(smc):>12}{_fmt(raw):>12}"
                f"{_fmtx(smc / raw if (smc and raw) else None):>12}"
                f"{_fmtx(base / smc if (base and smc) else None):>12}"
            )
    out.append("=" * 78)
    return "\n".join(out)


def _fmt(x: Optional[float]) -> str:
    return f"{x:.3f}s" if isinstance(x, (int, float)) else "-"


def _fmtx(x: Optional[float]) -> str:
    return f"{x:.2f}x" if isinstance(x, (int, float)) else "-"


# --------------------------------------------------------------------------- #
# Timed trials                                                                 #
# --------------------------------------------------------------------------- #
async def trials(run: Callable[[], Any], *, n_warmup: int, n_trials: int):
    """Median wall-clock over ``n_trials`` after ``n_warmup`` untimed runs. ``run``
    is a zero-arg coroutine factory returning (dt, payload); returns (median_dt,
    last_payload, all_dts)."""
    for _ in range(n_warmup):
        await run()
    dts, last = [], None
    for _ in range(n_trials):
        dt, last = await run()
        dts.append(dt)
    return float(np.median(dts)), last, dts
