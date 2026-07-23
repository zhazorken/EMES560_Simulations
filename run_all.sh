#!/bin/bash
# Run every simulation locally on the CPU (laptop). Do the one-time setup first:
#   julia setup.jl
# Override the Julia binary with e.g.  JULIA=~/bin/julia ./run_all.sh
set -e
cd "$(dirname "$0")"
JULIA="${JULIA:-julia}"
for s in [01][0-9]_*.jl; do
    echo "==================== $s ===================="
    time "$JULIA" --project "$s"
done
echo "Done — animations + snapshots are in ./output"
