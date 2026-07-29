#!/usr/bin/env bash
# Canonical gprof call-graph viz (== the pyprof-callgraph `gprof-viz`):
#   gprof2dot -f pstats <p.pstats> | dot -Tsvg -o <p.svg>     (default node format)
# Usage: viz.sh <one-or-more .pstats files>
set -u
export PATH=/root/genlm/genlm-venv/bin:$PATH
for p in "$@"; do
  svg="${p%.pstats}.svg"
  gprof2dot -f pstats "$p" | dot -Tsvg -o "$svg" && echo "wrote $svg"
done
