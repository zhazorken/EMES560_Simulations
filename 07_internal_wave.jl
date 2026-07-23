# ==============================================================================
# 07 — Internal gravity waves radiated from an oscillating source
# Concept: internal waves in a stratified fluid; frequency set by angle
#          (ω = N cos θ); phase ⟂ group velocity → the "St Andrew's Cross" of
#          four wave beams.
# Lecture: Week 10 (Waves). 2D x–z, nonhydrostatic. Cost: ~1 min.
# Closest Oceananigans example: "internal_wave".
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

Nx = Nz = 160
L = 10.0
grid = RectilinearGrid(arch; size = (Nx, Nz), x = (-L/2, L/2), z = (-L/2, L/2),
                       topology = (Periodic, Flat, Bounded))

N  = 1.0                       # buoyancy frequency
N² = N^2
ω  = 0.61 * N                  # forcing frequency < N  →  cos θ = ω/N ≈ 0.61

# background stratification b̄ = N² z (waves are perturbations about it)
B̄(x, z, t) = N² * z

# a small, localized, oscillating vertical force at the origin
A, σ = 1e-3, 0.2
w_force(x, z, t) = A * exp(-(x^2 + z^2) / 2σ^2) * sin(ω * t)

model = NonhydrostaticModel(; grid,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b = BackgroundField(B̄)),
                            forcing = (; w = Forcing(w_force)),
                            closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4))

simulation = Simulation(model, Δt = 0.02, stop_time = 40)

frames = Tuple{Float64, Matrix{Float64}}[]
# w sits on z-faces (Nz+1 points); keep the first Nz so it matches zc
grab(sim) = push!(frames,
    (sim.model.clock.time, Array(interior(sim.model.velocities.w))[:, 1, 1:Nz]))
simulation.callbacks[:grab] = Callback(grab, TimeInterval(0.4))

@info "Running internal-wave simulation..."
run!(simulation)

xc = LinRange(-L/2, L/2, Nx)
zc = LinRange(-L/2, L/2, Nz)
n  = Observable(1)
field = @lift frames[$n][2]
clim = 0.3 * maximum(abs, frames[end][2]) + eps()

fig = Figure(size = (620, 560))
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "z", aspect = 1,
           title = "Internal-wave beams (w)")
hm  = heatmap!(ax, xc, zc, field, colormap = :balance, colorrange = (-clim, clim))
Colorbar(fig[1, 2], hm, label = "w")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "07_internal_wave.mp4")
record(fig, outfile, 1:length(frames); framerate = 18) do i
    n[] = i
    ax.title = @sprintf("Internal-wave beams (ω = 0.61 N) — t = %.1f", frames[i][1])
end
save(joinpath(@__DIR__, "output", "07_internal_wave.png"), fig)
@info "Saved $outfile — four beams at θ = acos(ω/N) ≈ 52° from vertical."
