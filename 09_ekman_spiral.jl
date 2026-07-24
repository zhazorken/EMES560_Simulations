# ==============================================================================
# 09 — The wind-driven Ekman spiral
# Concept: rotation + vertical friction → surface flow 45° to the right of the
#          wind (NH), turning and decaying with depth; net transport 90° to the
#          wind. A 1-D vertical column.
# Lecture: Week 11 (Ekman Layers & Wind-Driven Circulation). Cost: seconds.
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

Nz, Lz = 64, 80.0                      # metres
grid = RectilinearGrid(arch; size = Nz, z = (-Lz, 0), topology = (Flat, Flat, Bounded))

f  = 1e-4                              # Coriolis parameter
Aᵥ = 1e-2                             # vertical eddy viscosity  → δ_E = √(2Aᵥ/f) ≈ 14 m
ρ₀ = 1025.0
τx = 0.1                              # eastward wind stress [N/m²]

# eastward wind stress = a downward flux of eastward momentum: Qᵘ = −τx/ρ₀
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-τx / ρ₀))

model = NonhydrostaticModel(grid; coriolis = FPlane(f = f),
                            closure = ScalarDiffusivity(ν = Aᵥ),
                            buoyancy = nothing, tracers = (),
                            boundary_conditions = (; u = u_bcs))

simulation = Simulation(model, Δt = 30.0, stop_time = 6e4)   # ~1 inertial period

frames = Tuple{Float64, Vector{Float64}, Vector{Float64}}[]
grab(sim) = push!(frames, (sim.model.clock.time,
                           Array(interior(sim.model.velocities.u))[1, 1, :],
                           Array(interior(sim.model.velocities.v))[1, 1, :]))
simulation.callbacks[:grab] = Callback(grab, TimeInterval(600))

@info "Running Ekman-spiral simulation..."
run!(simulation)

zc = LinRange(-Lz, 0, Nz)

# final profiles + the hodograph (the spiral)
uf, vf = frames[end][2], frames[end][3]
fig = Figure(size = (900, 460))
ax1 = Axis(fig[1, 1], xlabel = "velocity (m/s)", ylabel = "z (m)",
           title = "Ekman profiles")
lines!(ax1, uf, zc, label = "u"); lines!(ax1, vf, zc, label = "v"); axislegend(ax1)
ax2 = Axis(fig[1, 2], xlabel = "u (m/s)", ylabel = "v (m/s)", aspect = DataAspect(),
           title = "Ekman spiral (hodograph)")
lines!(ax2, uf, vf, color = zc, colormap = :viridis)
scatter!(ax2, [uf[end]], [vf[end]], color = :orange, markersize = 12)  # surface
mkpath(joinpath(@__DIR__, "output"))
save(joinpath(@__DIR__, "output", "09_ekman_spiral.png"), fig)

# animation: the spiral filling in as the layer spins up
n = Observable(1)
uu = @lift frames[$n][2]; vv = @lift frames[$n][3]
figA = Figure(size = (500, 500))
axA  = Axis(figA[1, 1], xlabel = "u (m/s)", ylabel = "v (m/s)", aspect = DataAspect(),
            title = "Ekman spiral")
lines!(axA, uu, vv, color = zc, colormap = :viridis)
outfile = joinpath(@__DIR__, "output", "09_ekman_spiral.mp4")
CairoMakie.record(figA, outfile, 1:length(frames); framerate = 20) do i
    n[] = i
    axA.title = @sprintf("Ekman spiral — t = %.1f h", frames[i][1] / 3600)
end
@info "Saved $outfile — surface flow ~45° right of the eastward wind."
