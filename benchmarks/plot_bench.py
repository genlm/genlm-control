"""Per-scenario plots from a results store -> one SVG each.

Reads ``results/<store>.jsonl`` (written by bench.py), and for every scenario draws
a grouped bar chart: x = control version, bars = smc / raw wall-clock, with the
cross-version speedup annotated. Writes ``results/<store>__<scenario>.svg``.

    python benchmarks/plot_bench.py [store]   # default store: bench
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
VERSION_ORDER = ["main", "speedup-old", "speedup-now", "now-torch", "now-icdf",
                 "now-ondevice"]
PATHS = ["smc", "raw"]
PATH_LABEL = {"smc": "smc (engine-served)", "raw": "raw (vLLM ceiling)"}
PATH_COLOR = {"smc": "#2c7fb8", "raw": "#cccccc"}


def _cfgstr(cfg: dict) -> str:
    keys = [k for k in ("model", "N", "max_tokens", "constraint", "library", "lora")
            if k in cfg]
    return "  ".join(f"{k}={cfg[k]}" for k in keys)


def main() -> None:
    store = sys.argv[1] if len(sys.argv) > 1 else "bench"
    path = RESULTS / f"{store}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # scenario -> version -> path -> dt ; and keep a representative config per scenario.
    # rows are appended, so a re-run leaves stale lines -> last write wins (by ts).
    rows.sort(key=lambda r: r.get("ts", ""))
    data: dict = defaultdict(lambda: defaultdict(dict))
    cfg_of: dict = {}
    for r in rows:
        scenario = r["scenario"]
        data[scenario][r["version"]["label"]][r["path"]] = r["dt_median"]
        cfg_of[scenario] = r["config"]

    written = []
    for scenario, by_ver in sorted(data.items()):
        versions = ([v for v in VERSION_ORDER if v in by_ver]
                    + [v for v in by_ver if v not in VERSION_ORDER])
        x = np.arange(len(versions))
        w = 0.26
        fig, ax = plt.subplots(figsize=(1.8 * len(versions) + 3, 4.6))
        for i, p in enumerate(PATHS):
            vals = [by_ver[v].get(p, np.nan) for v in versions]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=PATH_LABEL[p],
                          color=PATH_COLOR[p])
            for rect, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(rect.get_x() + rect.get_width() / 2, val,
                            f"{val:.2f}s", ha="center", va="bottom", fontsize=7)
        # step/burst speedup annotation per version
        for xi, v in zip(x, versions):
            off, req = by_ver[v].get("off"), by_ver[v].get("require")
            if off and req:
                ax.text(xi, max(off, req) * 1.12, f"{off / req:.1f}× burst",
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color="#2c7fb8")
        ax.set_xticks(x)
        ax.set_xticklabels(versions)
        ax.set_ylabel("wall-clock (s, median of 3 trials)")
        ax.set_title(f"{scenario}   |   {_cfgstr(cfg_of[scenario])}", fontsize=10,
                     pad=30)
        # horizontal legend above the axes (below the title) -> no clipping/overlap
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=3, frameon=False)
        ax.margins(y=0.22)
        out = RESULTS / f"{store}__{scenario}.svg"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=120)
        plt.close(fig)
        written.append(out)

    print("wrote:")
    for o in written:
        print(" ", o)


if __name__ == "__main__":
    main()
