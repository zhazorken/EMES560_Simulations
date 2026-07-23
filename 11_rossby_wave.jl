# ==============================================================================
# 11 — Rossby waves on a β-plane (shallow water)
# Concept: the β-effect (f varying with latitude) makes a large-scale
#          perturbation propagate its phase WESTWARD — the defining property of
#          Rossby waves.
# Lecture: Week 10 (Waves). 2D x–y shallow water. Cost: ~1 min.
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

Nx, Ny = 256, 128
Lx, Ly = 40.0, 20.0
grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, Lx), y = (-Ly/2, Ly/2),
                       topology = (Periodic, Bounded, Flat))

g, H, f₀, β = 1.0, 1.0, 1.0, 0.2      # L_d = √(gH)/f₀ = 1
model = ShallowWaterModel(grid; coriolis = BetaPlane(f₀ = f₀, β = β),
                          gravitational_acceleration = g,
                          momentum_advection = WENO())

# a large-scale, geostrophically balanced height pattern (2 wavelengths in x)
A, σ = 0.05, 3.0
k = 2π * 2 / Lx
E(y)  = exp(-y^2 / (2σ^2))
Ey(y) = -y / σ^2 * E(y)
hp(x, y)  = A * cos(k * x) * E(y)              # height perturbation
hᵢ(x, y)  = H + hp(x, y)
ug(x, y)  = -(g / f₀) * A * cos(k * x) * Ey(y) # geostrophic balance
vg(x, y)  =  (g / f₀) * (-A * k * sin(k * x)) * E(y)
uhᵢ(x, y) = ug(x, y) * hᵢ(x, y)
vhᵢ(x, y) = vg(x, y) * hᵢ(x, y)

set!(model, h = hᵢ, uh = uhᵢ, vh = vhᵢ)

simulation = Simulation(model, Δt = 0.03, stop_time = 120)

frames = Tuple{Float64, Matrix{Float64}}[]
grab(sim) = push!(frames,
    (sim.model.clock.time, Array(interior(sim.model.solution.h))[:, :, 1] .- H))
simulation.callbacks[:grab] = Callback(grab, TimeInterval(1.0))

@info "Running Rossby-wave simulation..."
run!(simulation)

xc = LinRange(0, Lx, Nx)
yc = LinRange(-Ly/2, Ly/2, Ny)
n  = Observable(1)
field = @lift frames[$n][2]
clim = A

fig = Figure(size = (860, 480))
ax  = Axis(fig[1, 1], xlabel = "x (units of L_d)", ylabel = "y",
           title = "Rossby wave — height anomaly (phase drifts west →)")
hm  = heatmap!(ax, xc, yc, field, colormap = :balance, colorrange = (-clim, clim))
Colorbar(fig[1, 2], hm, label = "h − H")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "11_rossby_wave.mp4")
record(fig, outfile, 1:length(frames); framerate = 20) do i
    n[] = i
    ax.title = @sprintf("Rossby wave (westward phase) — t = %.0f", frames[i][1])
end
save(joinpath(@__DIR__, "output", "11_rossby_wave.png"), fig)
@info "Saved $outfile — watch the crests march westward."
