# ==============================================================================
# 01 — Stirring of a passive tracer by a cellular (Taylor–Green) flow
# Concept: kinematics — strain vs. rotation, stretching of material lines,
#          filamentation and stirring, streamlines vs. tracer evolution.
# Lecture: Week 1–2 (Fluid Kinematics). 2D x–y, incompressible. Cost: seconds.
# Closest Oceananigans example: "two_dimensional_turbulence".
# ==============================================================================
import Pkg; Pkg.activate(@__DIR__)
using Oceananigans

# Architecture: use the GPU only if OCEAN_ARCH=GPU AND a CUDA GPU is actually usable here;
# otherwise fall back to the CPU (these models are tiny, so CPU is perfectly fine). On
# Oceananigans >= 0.109 the zero-arg GPU() lives in the CUDA extension, so load CUDA first.
if get(ENV, "OCEAN_ARCH", "CPU") == "GPU"
    using CUDA
    if CUDA.functional()
        arch = GPU()
    else
        @warn "OCEAN_ARCH=GPU but no usable CUDA GPU on this node — falling back to CPU."
        arch = CPU()
    end
else
    arch = CPU()
end
@info "Architecture: $arch"
using Oceananigans.Units
using CairoMakie
using Printf

# ---- cheap 2D grid on a doubly-periodic box [0, 2π]² -------------------------
Nx = Ny = 128
L  = 2π
grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, L), y = (0, L),
                       topology = (Periodic, Periodic, Flat))

# a little viscosity/diffusivity keeps the run numerically clean
model = NonhydrostaticModel(grid;
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :c,
                            closure = ScalarDiffusivity(ν = 1e-3, κ = 1e-4))

# ---- initial condition ------------------------------------------------------
# Taylor–Green vortex velocity from streamfunction ψ = sin(x) sin(y):
#   u =  ∂ψ/∂y =  sin(x) cos(y),   v = -∂ψ/∂x = -cos(x) sin(y)
# Vortex centres are pure rotation; the cell edges are pure strain (stagnation).
uᵢ(x, y) =  sin(x) * cos(y)
vᵢ(x, y) = -cos(x) * sin(y)

# a compact tracer blob placed on a strain region so we see it stretched
cᵢ(x, y) = exp(-((x - π)^2 + (y - π/2)^2) / 0.1)

set!(model, u = uᵢ, v = vᵢ, c = cᵢ)

simulation = Simulation(model, Δt = 0.01, stop_time = 15)

# ---- collect snapshots for the animation ------------------------------------
frames = Tuple{Float64, Matrix{Float64}}[]
save_snapshot(sim) = push!(frames,
    (sim.model.clock.time, Array(interior(sim.model.tracers.c))[:, :, 1]))
simulation.callbacks[:grab] = Callback(save_snapshot, TimeInterval(0.2))

@info "Running tracer-stirring simulation..."
run!(simulation)

# ---- animate ----------------------------------------------------------------
xc = LinRange(L/2Nx, L - L/2Nx, Nx)
yc = LinRange(L/2Ny, L - L/2Ny, Ny)
n  = Observable(1)
field = @lift frames[$n][2]

fig = Figure(size = (620, 560))
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "y", aspect = 1,
           title = "Passive tracer stirred by a cellular flow")
hm  = heatmap!(ax, xc, yc, field, colormap = :thermal, colorrange = (0, 1))
Colorbar(fig[1, 2], hm, label = "tracer c")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "01_tracer_stirring.mp4")
record(fig, outfile, 1:length(frames); framerate = 20) do i
    n[] = i
    ax.title = @sprintf("Tracer stirring — t = %.1f", frames[i][1])
end
save(joinpath(@__DIR__, "output", "01_tracer_stirring.png"), fig)
@info "Saved $outfile"
