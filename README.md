# EMES 560 — Oceananigans Teaching Simulations

A suite of **CPU-cheap** [Oceananigans.jl](https://clima.github.io/OceananigansDocumentation/stable/) simulations that illustrate the core fluid-dynamics and geophysical-fluid-dynamics concepts of EMES 560. Each script is self-contained, runs on a laptop CPU in seconds to a few minutes, and saves an animation (`.mp4`) plus a final snapshot (`.png`) into `output/`.

Drop the animations/frames straight into the lecture slides where the matching concept is taught.

---

## Requirements
- **Julia 1.10** (recommended; anything ≥ 1.9 should work).
- Packages: **Oceananigans**, **CairoMakie** (headless plotting), Printf, Statistics.

## One-time setup
From this folder:
```bash
julia setup.jl
```
This creates a local `Project.toml`/`Manifest.toml` here and precompiles everything (the first precompile of Oceananigans + CairoMakie takes a few minutes; after that, runs are fast).

## Running a simulation
```bash
julia --project=. 01_tracer_stirring.jl
```
Each script also self-activates this folder's environment, so `julia 01_tracer_stirring.jl` works too. Outputs land in `output/`.

To run everything (grab a coffee for the 3D ones):
```bash
for f in [0-1][0-9]_*.jl; do julia --project=. "$f"; done
```

---

## The simulations and where they fit

| # | Script | Concept illustrated | Lecture(s) | Dim | ~Cost |
|---|--------|---------------------|-----------|-----|-------|
| 01 | `01_tracer_stirring.jl` | Strain, stirring, filamentation of a passive tracer; streamlines vs particle paths | Wk 1–2 (kinematics) | 2D x–y | seconds |
| 02 | `02_kelvin_helmholtz.jl` | Shear instability, Ri < ¼, KH billows, mixing | Wk 6 & 12 | 2D x–z | ~1 min |
| 03 | `03_rayleigh_benard.jl` | Convection, plumes, buoyancy-driven turbulence | Wk 6 | 2D x–z | ~1–2 min |
| 04 | `04_two_dimensional_turbulence.jl` | 2D vorticity dynamics, vortex merger, inverse cascade | Wk 5 & 12 | 2D x–y | ~1 min |
| 05 | `05_inertial_oscillation.jl` | Inertial oscillations, the f-plane, hodograph | Wk 7 | column | seconds |
| 06 | `06_geostrophic_adjustment.jl` | Geostrophic adjustment, gravity-wave radiation, deformation radius | Wk 8–9 | 1D (shallow water) | seconds |
| 07 | `07_internal_wave.jl` | Internal gravity waves, phase ⟂ group, wave beams | Wk 10 | 2D x–z | ~1 min |
| 08 | `08_barotropic_instability_bickley.jl` | Barotropic instability of a jet, PV, vortex roll-up | Wk 9 & 12 | 2D x–y (shallow water) | ~1–2 min |
| 09 | `09_ekman_spiral.jl` | Wind-driven Ekman spiral & transport | Wk 11 | column | seconds |
| 10 | `10_baroclinic_instability.jl` | Baroclinic instability, mesoscale eddies, thermal wind | Wk 8 & 12 | 3D channel (coarse) | ~5–15 min |
| 11 | `11_rossby_wave.jl` | Rossby waves, β-effect, westward phase propagation | Wk 10 | 2D x–y (shallow water) | ~1 min |

All grids are deliberately coarse for speed. To make any run sharper, increase the `Nx, Nz` (etc.) at the top of the script — cost scales roughly with the number of grid points × number of steps.

---

## ⚠️ A note on the Oceananigans API
These scripts were written against **Oceananigans v0.9x** (Julia 1.10). Oceananigans's API evolves between minor versions. If a script errors on your installed version, the usual culprits and fixes are:

- **Initial-condition function signatures.** For a 2D grid with a `Flat` dimension, recent Oceananigans expects functions of the *non-flat* coordinates only, e.g. `bᵢ(x, z)` on an x–z grid. Older versions want the full `bᵢ(x, y, z)`. If you get a method error in `set!`, add/remove the `y` argument.
- **`ShallowWaterModel` fields.** With the default `ConservativeFormulation` the prognostic fields are `uh, vh, h` (so `u = uh/h`). These scripts use `VectorInvariantFormulation()` so the fields are simply `u, v, h`. If your version lacks it, switch to conservative and set `uh, vh, h`.
- **Output/animation.** We read fields directly from `model` during a callback and record with CairoMakie; if you prefer, use a `JLD2OutputWriter` and post-process.

Each script header links to the closest official Oceananigans example, which is the authoritative reference if you need to reconcile an API change.

Every script is **GPU-capable**: it reads `OCEAN_ARCH` and builds the grid on `GPU()` when that variable is `GPU` (the Casper job sets it), otherwise `CPU()`.

---

## Putting this on GitHub

A `.gitignore`, `LICENSE`, and `Project.toml` (pinned to Oceananigans 0.109, matching the Casper stack) are included; outputs, logs, and `Manifest.toml` are excluded. From this folder:

```bash
git init && git add -A && git commit -m "EMES 560 Oceananigans teaching simulations"
git branch -M main
# create an EMPTY repo on github.com first (no README), then:
git remote add origin git@github.com:<you>/EMES560_Simulations.git   # or the https URL
git push -u origin main
```

Or, with the GitHub CLI authenticated (`gh auth login`):

```bash
gh repo create EMES560_Simulations --public --source=. --push
```

(The repo has been `git init`-ed and committed for you locally — you just need to add your remote and push. I can't push from here without your GitHub credentials.)

---

## Running on Casper (NCAR)

Conventions match your working `Ovall26/sarqardleq_cg` run: account `UGIT0046`, `casper` queue, one A100, `juliaup`, the **default `~/.julia` depot** (so it reuses the same Oceananigans 0.109 + CUDA stack your saqqar runs use — which is known to see the A100), no `module load cuda`.

```bash
# on a Casper login node — code on /glade/work, packages in the default ~/.julia depot
cd /glade/work/$USER
git clone https://github.com/zhazorken/EMES560_Simulations.git
cd EMES560_Simulations

unset JULIA_DEPOT_PATH                               # IMPORTANT: use the default ~/.julia
JULIA=$HOME/.juliaup/bin/julia ./setup_casper.sh    # instantiate + precompile (once)

qsub submit_casper.pbs                              # ALL 11 models, one A100, one job
```

> If a previous session `export`ed `JULIA_DEPOT_PATH=/glade/work/$USER/.julia`, that separate depot gets a *fresh* CUDA install that may not detect the GPU. `unset JULIA_DEPOT_PATH` (above) makes both setup and the job use `~/.julia`, reusing your proven stack.

**Cost.** The models are tiny, so `submit_casper.pbs` runs the whole suite in a single Julia
session (CUDA kernels compile once, then all 11 run in seconds each) — well under **1 GPU-hour
total**, comfortably inside the ≤ 1 GPU-hour-per-model budget. The walltime directive is
`01:30:00` as a safety margin.

Other options:

```bash
qsub -v ONLY=08 submit_casper.pbs      # just one model (e.g. the Bickley jet)
qsub submit_casper_array.pbs           # one model per sub-job, ≤ 1 GPU-hr walltime EACH
                                       # (simpler scheduling, but recompiles CUDA per sub-job)
```

Outputs (`.mp4` + `.png`) land in `./output`. For a fast local sanity check before the cluster,
`julia setup.jl` then `./run_all.sh` runs everything on your laptop CPU.

