# ==============================================================================
# 03 — Free convection: plumes driven by surface buoyancy loss
# Concept: buoyancy-driven convection, plumes, mixed-layer deepening, and the
#          onset of turbulence (an oceanographic Rayleigh–Bénard problem).
# Lecture: Week 6 (Boundary Layers & Turbulence). 2D x–z, Boussinesq. ~1–2 min.
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

Nx, Nz = 128, 64
Lx, Lz = 64.0, 32.0     # metres
grid = RectilinearGrid(arch; size = (Nx, Nz), x = (0, Lx), z = (-Lz, 0),
                       topology = (Periodic, Flat, Bounded))

# Cooling at the surface = a positive (upward) buoyancy flux out of the domain.
Qᵇ = 1e-7               # surface buoyancy flux  [m² s⁻³]
N² = 1e-5               # initial stratification [s⁻²]
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = NonhydrostaticModel(grid;
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4),
                            boundary_conditions = (; b = b_bcs))

# stable initial stratification + tiny noise to break symmetry
bᵢ(x, z) = N² * z + 1e-2 * N² * Lz * randn()
set!(model, b = bᵢ)

simulation = Simulation(model, Δt = 1.0, stop_time = 3hours)

# adaptive time step keeps the fast plumes stable and the run cheap
wizard = TimeStepWizard(cfl = 0.7, max_Δt = 20.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

frames = Tuple{Float64, Matrix{Float64}}[]
grab(sim) = push!(frames,
    (sim.model.clock.time, Array(interior(sim.model.velocities.w))[:, 1, 1:Nz]))
simulation.callbacks[:grab] = Callback(grab, TimeInterval(3minutes))

@info "Running free-convection simulation..."
run!(simulation)

xc = LinRange(Lx/2Nx, Lx - Lx/2Nx, Nx)
zf = LinRange(-Lz, 0, Nz)
n  = Observable(1)
field = @lift frames[$n][2]
clim = maximum(abs, frames[end][2]) + eps()

fig = Figure(size = (820, 460))
ax  = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)",
           title = "Convective plumes (vertical velocity)")
hm  = heatmap!(ax, xc, zf, field, colormap = :balance, colorrange = (-clim, clim))
Colorbar(fig[1, 2], hm, label = "w (m/s)")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "03_rayleigh_benard.mp4")
CairoMakie.record(fig, outfile, 1:length(frames); framerate = 12) do i
    n[] = i
    ax.title = @sprintf("Free convection — t = %.0f min", frames[i][1] / 60)
end
save(joinpath(@__DIR__, "output", "03_rayleigh_benard.png"), fig)
@info "Saved $outfile"
