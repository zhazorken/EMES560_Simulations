# ==============================================================================
# 06 — Geostrophic adjustment of an initial height bump (shallow water)
# Concept: an unbalanced height anomaly radiates inertia–gravity waves and
#          settles into a balanced geostrophic jet whose width is the Rossby
#          deformation radius L_d = √(gH)/f.
# Lecture: Week 8 (Adjustment) & Week 9 (Shallow Water / L_d). 1D. Cost: seconds.
# ==============================================================================
import Pkg; Pkg.activate(@__DIR__)
using Oceananigans

# CPU by default; set OCEAN_ARCH=GPU (done by the Casper PBS script) to run on a GPU.
arch = get(ENV, "OCEAN_ARCH", "CPU") == "GPU" ? GPU() : CPU()
using CairoMakie
using Printf

# non-dimensional units: g = H = f = 1  ⇒  wave speed c = √(gH) = 1, L_d = 1
Nx = 512
Lx = 40.0
grid = RectilinearGrid(arch; size = Nx, x = (-Lx/2, Lx/2), topology = (Bounded, Flat, Flat))

g, H, f = 1.0, 1.0, 1.0
model = ShallowWaterModel(; grid, coriolis = FPlane(f = f),
                          gravitational_acceleration = g)

# initial condition: a Gaussian height bump of width ≈ L_d, fluid at rest
Δh, Lb = 0.2, 1.0
hᵢ(x) = H + Δh * exp(-x^2 / (2Lb^2))
set!(model, h = hᵢ)

simulation = Simulation(model, Δt = 0.01, stop_time = 30)

frames = Tuple{Float64, Vector{Float64}, Vector{Float64}}[]  # (t, h, v)
function grab(sim)
    h  = Array(interior(sim.model.solution.h))[:, 1, 1]
    vh = Array(interior(sim.model.solution.vh))[:, 1, 1]
    push!(frames, (sim.model.clock.time, h, vh ./ h))
end
simulation.callbacks[:grab] = Callback(grab, TimeInterval(0.25))

@info "Running geostrophic-adjustment simulation..."
run!(simulation)

xc = LinRange(-Lx/2 + Lx/2Nx, Lx/2 - Lx/2Nx, Nx)
n  = Observable(1)
hline = @lift frames[$n][2]
vline = @lift frames[$n][3]

fig = Figure(size = (900, 620))
ax1 = Axis(fig[1, 1], ylabel = "layer depth h", title = "Geostrophic adjustment")
ylims!(ax1, H - 0.05, H + Δh + 0.02)
lines!(ax1, xc, hline, color = :navy)
ax2 = Axis(fig[2, 1], xlabel = "x (units of L_d)", ylabel = "along-jet velocity v")
lines!(ax2, xc, vline, color = :teal)

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "06_geostrophic_adjustment.mp4")
record(fig, outfile, 1:length(frames); framerate = 20) do i
    n[] = i
    ax1.title = @sprintf("Geostrophic adjustment — t = %.1f (units 1/f)", frames[i][1])
end
save(joinpath(@__DIR__, "output", "06_geostrophic_adjustment.png"), fig)
@info "Saved $outfile — gravity waves radiate out; a geostrophic v-jet remains over ~L_d."
