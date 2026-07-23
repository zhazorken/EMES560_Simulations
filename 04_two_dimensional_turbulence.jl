# ==============================================================================
# 04 — Decaying two-dimensional turbulence
# Concept: 2D vorticity dynamics, like-signed vortex merger, and the inverse
#          energy cascade (energy flows to LARGER scales) — the hallmark of
#          geostrophic turbulence.
# Lecture: Week 5 (Vorticity) and Week 12 (GFD Turbulence). 2D x–y. ~1 min.
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
using CairoMakie
using Printf

Nx = Ny = 128
L = 2π
grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, L), y = (0, L),
                       topology = (Periodic, Periodic, Flat))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = WENO(),
                            closure = ScalarDiffusivity(ν = 1e-4))

# random, small-scale initial velocity — it will self-organize into vortices
uᵢ(x, y) = randn()
vᵢ(x, y) = randn()
set!(model, u = uᵢ, v = vᵢ)

# vorticity ζ = ∂x v − ∂y u, recomputed each snapshot
u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))

simulation = Simulation(model, Δt = 0.02, stop_time = 30)

frames = Tuple{Float64, Matrix{Float64}}[]
function grab(sim)
    compute!(ζ)
    push!(frames, (sim.model.clock.time, Array(interior(ζ))[:, :, 1]))
end
simulation.callbacks[:grab] = Callback(grab, TimeInterval(0.4))

@info "Running 2D turbulence simulation..."
run!(simulation)

xc = LinRange(L/2Nx, L - L/2Nx, Nx)
yc = LinRange(L/2Ny, L - L/2Ny, Ny)
n  = Observable(1)
field = @lift frames[$n][2]
clim = 0.6 * maximum(abs, frames[1][2]) + eps()

fig = Figure(size = (620, 560))
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "y", aspect = 1,
           title = "2D turbulence — vorticity")
hm  = heatmap!(ax, xc, yc, field, colormap = :balance, colorrange = (-clim, clim))
Colorbar(fig[1, 2], hm, label = "vorticity ζ")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "04_two_dimensional_turbulence.mp4")
record(fig, outfile, 1:length(frames); framerate = 18) do i
    n[] = i
    ax.title = @sprintf("2D turbulence (vortex merger) — t = %.1f", frames[i][1])
end
save(joinpath(@__DIR__, "output", "04_two_dimensional_turbulence.png"), fig)
@info "Saved $outfile"
