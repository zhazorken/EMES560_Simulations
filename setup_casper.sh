#!/bin/bash -l
# setup_casper.sh — one-time environment setup on Casper (NCAR), run from the repo dir.
# Resolves + precompiles the Oceananigans 0.109 env (pinned in Project.toml).
#
#   cd /glade/work/$USER && git clone <your-repo-url> EMES560_Simulations
#   cd EMES560_Simulations
#   JULIA=$HOME/.juliaup/bin/julia ./setup_casper.sh
#   qsub submit_casper.pbs
set -e
cd "$(dirname "$0")"
mkdir -p logs output            # #PBS -o/-e write into logs/ at launch; missing dir ⇒ Held job

# Keep the Julia depot on /glade/work ($HOME quota is tiny). Instantiating needs NO HPC
# modules (CUDA.jl bundles its own toolkit; the GPU driver is only needed at run time).
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
JULIA="${JULIA:-julia}"
command -v "$JULIA" >/dev/null 2>&1 || {
  echo "ERROR: '$JULIA' not found. Install juliaup into /glade/work first:"
  echo "  curl -fsSL https://install.julialang.org | sh -s -- --yes && source ~/.bashrc"
  echo "  juliaup add 1.10 && juliaup default 1.10"
  exit 1; }
echo "Julia:  $($JULIA --version)   depot: $JULIA_DEPOT_PATH"

# Manifest.toml is .gitignored, so this resolves the pinned stack fresh for the cluster Julia.
$JULIA --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo
echo "Environment ready. Submit all 11 models on one A100:"
echo "    qsub submit_casper.pbs"
