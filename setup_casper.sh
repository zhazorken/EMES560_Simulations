#!/bin/bash -l
# setup_casper.sh — one-time env setup on Casper (NCAR). Run from the repo dir on a login node.
# Mirrors your Ovall26/sarqardleq_cg setup: uses the DEFAULT ~/.julia depot, so it REUSES the
# same Oceananigans 0.109 + CUDA stack your saqqar runs use (fast, and known to see the A100).
#
#   cd /glade/work/$USER && git clone <repo-url> EMES560_Simulations && cd EMES560_Simulations
#   JULIA=$HOME/.juliaup/bin/julia ./setup_casper.sh
#   qsub submit_casper.pbs
set -e
cd "$(dirname "$0")"
mkdir -p logs output          # PBS -o/-e write into logs/ at launch; missing dir => Held job

# Instantiating needs NO HPC modules (CUDA.jl bundles its toolkit; the driver is only needed at
# run time). Packages go to the default depot (~/.julia) unless you export JULIA_DEPOT_PATH.
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
command -v "$JULIA" >/dev/null 2>&1 || { echo "ERROR: '$JULIA' not found. Install juliaup or pass JULIA=/path/to/julia."; exit 1; }
echo "Julia:  $($JULIA --version)   depot: ${JULIA_DEPOT_PATH:-$HOME/.julia}"

# No Manifest is committed, so this resolves the pinned 0.109 stack fresh (reusing cached
# packages already in ~/.julia from your other runs, so it's quick).
$JULIA --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo
echo "Environment ready. Submit all 11 models on one A100:"
echo "    qsub submit_casper.pbs"
