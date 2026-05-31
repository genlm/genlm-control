#!/usr/bin/env bash
# Sync BOTH repos to the GPU box and run the SMC gates -- the dev loop as code, so you
# can't forget the backend (syncing only control leaves a stale run_burst running against
# new control = silently wrong results). The box is ephemeral: set BOX each session, and
# override the remote paths/venv below if they moved.
#
#   BOX=root@1.2.3.4 scripts/box.sh sync     # rsync both repos (control pkg + tests + backend)
#   BOX=root@1.2.3.4 scripts/box.sh gate1    # sync + byte-exact StepLoop parity (fast, foreground)
#   BOX=root@1.2.3.4 scripts/box.sh gate2     # sync + engine-native burst suite (slow, backgrounded)
#   BOX=root@1.2.3.4 scripts/box.sh tail      # follow the running gate2 log
set -euo pipefail

BOX="${BOX:?set BOX=root@<ip>}"
KEY="${KEY:-$HOME/.ssh/id_ed25519_verda}"
CTRL_LOCAL="${CTRL_LOCAL:-$(cd "$(dirname "$0")/.." && pwd)}"
BACK_LOCAL="${BACK_LOCAL:-$CTRL_LOCAL/../genlm-backend-engine-native}"
CTRL_REMOTE="${CTRL_REMOTE:-/root/genlm/genlm-control}"
BACK_REMOTE="${BACK_REMOTE:-/root/genlm/genlm-backend}"
VENV="${VENV:-/root/genlm-venv/bin/python}"
SSH="ssh -i $KEY -o ConnectTimeout=15"
PYTEST="VLLM_USE_FLASHINFER_SAMPLER=0 $VENV -m pytest"

sync() {
  rsync -az --exclude=__pycache__ -e "$SSH" "$CTRL_LOCAL/genlm/control/" "$BOX:$CTRL_REMOTE/genlm/control/"
  rsync -az --exclude=__pycache__ -e "$SSH" "$CTRL_LOCAL/tests/"        "$BOX:$CTRL_REMOTE/tests/"
  rsync -az --exclude=__pycache__ -e "$SSH" "$BACK_LOCAL/genlm/backend/" "$BOX:$BACK_REMOTE/genlm/backend/"
  echo "synced control + tests + backend -> $BOX"
}

case "${1:-gate2}" in
  sync)  sync ;;
  gate1) sync; $SSH "$BOX" "cd $CTRL_REMOTE && $PYTEST tests/sampler/test_per_token_parity.py -q" ;;
  gate2)
    sync
    $SSH "$BOX" "cd $CTRL_REMOTE && rm -f /tmp/gate2.log && nohup $PYTEST tests/sampler/test_engine_native.py -s -v --durations=0 > /tmp/gate2.log 2>&1 & echo gate2 PID=\$!"
    echo "running; follow with: BOX=$BOX scripts/box.sh tail" ;;
  tail)  $SSH "$BOX" "tail -f /tmp/gate2.log" ;;
  *) echo "usage: BOX=root@<ip> $0 {sync|gate1|gate2|tail}"; exit 1 ;;
esac
