# EMES 560 — one-time environment setup for the Oceananigans simulation suite.
# Run once from this folder:   julia setup.jl
# (Uses whatever Julia you launch it with; Julia 1.10 recommended.)

using Pkg
Pkg.activate(@__DIR__)
println("Adding packages (Oceananigans + CairoMakie for headless plotting)...")
Pkg.add([
    "Oceananigans",   # the fluid-dynamics engine
    "CairoMakie",     # headless (CPU) plotting + mp4/gif animations
    "Printf",
    "Statistics",
])
Pkg.instantiate()
Pkg.precompile()
println("\nDone. Run any simulation with, e.g.:")
println("    julia --project=. 01_tracer_stirring.jl")
