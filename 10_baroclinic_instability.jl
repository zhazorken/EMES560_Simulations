# ==============================================================================
# 10 — Baroclinic instability of a front (coarse 3D channel)
# Concept: a thermal-wind-balanced front stores available potential energy;
#          baroclinic instability releases it into mesoscale eddies at ~L_d.
#          This is the ocean's "weather" and the atmosphere's cyclones.
# Lecture: Week 8 (Thermal Wind) & Week 12 (Instabilities). 3D. Cost: ~5–15 min.
# Closest Oceananigans example: "baroclinic_adjustment".
# NOTE: the only pricey run here — coarsen (Nx,Ny,Nz) further to go faster.
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

Nx, Ny, Nz = 64, 64, 16
Lx, Ly, Lz = 500kilometers, 500kilometers, 1kilometers
grid = RectilinearGrid(arch; size = (Nx, Ny, Nz),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))

f  = 1e-4
N² = 1e-5                    # stratification  → N = 3.2e-3, L_d = NH/f ≈ 32 km
Δb = 5e-3                    # buoyancy jump across the front
model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = FPlane(f = f),
                                    buoyancy = BuoyancyTracer(), tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    free_surface = ImplicitFreeSurface(),
                                    closure = ScalarDiffusivity(ν = 1, κ = 1))

# a tanh buoyancy front in y on top of stable stratification, plus noise
front(y) = Δb * tanh(y / (0.05Ly))
bᵢ(x, y, z) = N² * z + 0.5 * front(y) + 1e-2 * Δb * randn()
set!(model, b = bᵢ)

simulation = Simulation(model, Δt = 60.0, stop_time = 40days)
wizard = TimeStepWizard(cfl = 0.5, max_Δt = 20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

progress(sim) = @info @sprintf("t = %.1f days,  Δt = %.0f s",
                               sim.model.clock.time/86400, sim.Δt)
simulation.callbacks[:progress] = Callback(progress, TimeInterval(2days))

# surface relative vorticity (eddies are clearest in ζ)
u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))

frames = Tuple{Float64, Matrix{Float64}}[]
function grab(sim)
    compute!(ζ)
    # ζ is on corners (Ny+1 in bounded y); trim to Ny; top level = Nz
    push!(frames, (sim.model.clock.time, Array(interior(ζ))[:, 1:Ny, Nz]))
end
simulation.callbacks[:grab] = Callback(grab, TimeInterval(1day))

@info "Running baroclinic-instability simulation (this is the slow one)..."
run!(simulation)

xc = LinRange(0, Lx, Nx) ./ 1e3
yc = LinRange(-Ly/2, Ly/2, Ny) ./ 1e3
n  = Observable(1)
field = @lift frames[$n][2]
clim = 0.5 * maximum(abs, frames[end][2]) + eps()

fig = Figure(size = (620, 560))
ax  = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", aspect = 1,
           title = "Baroclinic eddies — surface vorticity")
hm  = heatmap!(ax, xc, yc, field, colormap = :balance, colorrange = (-clim, clim))
Colorbar(fig[1, 2], hm, label = "ζ (s⁻¹)")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "10_baroclinic_instability.mp4")
record(fig, outfile, 1:length(frames); framerate = 8) do i
    n[] = i
    ax.title = @sprintf("Baroclinic eddies — t = %.0f days", frames[i][1]/86400)
end
save(joinpath(@__DIR__, "output", "10_baroclinic_instability.png"), fig)
@info "Saved $outfile"
