"""Diff two cProfile .pstats files -> the frames where the first spends MORE time.

Localizes a regression: ranks functions by self-time (tottime) delta (A - B), so
the rows at the top are where A's extra wall-clock actually went.

    python benchmarks/prof_diff.py <A.pstats> <B.pstats> [topN]   # e.g. now vs old
"""

from __future__ import annotations

import pstats
import sys


def _load(path: str) -> dict:
    st = pstats.Stats(path).stats  # {(file,line,func): (cc, nc, tt, ct, callers)}
    return {k: (v[2], v[3], v[1]) for k, v in st.items()}  # tottime, cumtime, ncalls


def _name(k) -> str:
    file, line, func = k
    return f"{func}  ({file.rsplit('/', 1)[-1]}:{line})"


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    a, b = _load(a_path), _load(b_path)

    rows = []
    for k in set(a) | set(b):
        at, ac, an = a.get(k, (0.0, 0.0, 0))
        bt, bc, bn = b.get(k, (0.0, 0.0, 0))
        rows.append((at - bt, k, at, bt, an, bn))
    rows.sort(reverse=True)

    a_tot = sum(t for t, _, _ in a.values())
    b_tot = sum(t for t, _, _ in b.values())
    print(f"A = {a_path}\nB = {b_path}")
    print(f"total self-time: A={a_tot:.3f}s  B={b_tot:.3f}s  (A-B={a_tot - b_tot:+.3f}s)\n")
    print(f"{'Δtottime':>10}  {'A_self':>8}  {'B_self':>8}  {'A_calls':>8}  {'B_calls':>8}  function")
    print("-" * 100)
    for dt, k, at, bt, an, bn in rows[:topn]:
        print(f"{dt:>+10.3f}  {at:>8.3f}  {bt:>8.3f}  {an:>8}  {bn:>8}  {_name(k)}")


if __name__ == "__main__":
    main()
