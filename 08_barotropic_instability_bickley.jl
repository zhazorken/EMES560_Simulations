# ==============================================================================
# 08 — Barotropic instability of a Bickley jet (shallow water)
# Concept: a sheared jet (with a PV-gradient reversal) is barotropically
#          unstable; it meanders and rolls up into a vortex street. Illustrates
#          Rayleigh–Kuo instability and PV dynamics in one layer.
# Lecture: Week 9 (Shallow Water / PV) & Week 12 (Instabilities). 2D. ~1–2 min.
# Closest Oceananigans example: "shallow_water_Bickley_jet".
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

Nx, Ny = 128, 128
Lx, Ly = 10.0, 20.0
grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, Lx), y = (-Ly/2, Ly/2),
                       topology = (Periodic, Bounded, Flat))

g, H, f, U = 1.0, 1.0, 1.0, 0.5
model = ShallowWaterModel(grid; coriolis = FPlane(f = f),
                          gravitational_acceleration = g,
                          momentum_advection = WENO())

# Bickley jet in geostrophic balance:  Ū = U sech²(y),  h̄ = H − (fU/g) tanh(y)
ū(y)  = U * sech(y)^2
h̄(y)  = H - (f * U / g) * tanh(y)

# small random perturbation to seed the instability
ϵ = 1e-3
uhᵢ(x, y) = ū(y) * h̄(y) + ϵ * exp(-y^2) * randn()
vhᵢ(x, y) = ϵ * exp(-y^2) * randn()
hᵢ(x, y)  = h̄(y)

set!(model, uh = uhᵢ, vh = vhᵢ, h = hᵢ)

simulation = Simulation(model, Δt = 0.02, stop_time = 90)

# relative vorticity of the (u = uh/h, v = vh/h) field
uh, vh, h = model.solution.uh, model.solution.vh, model.solution.h
ζ = Field(∂x(vh / h) - ∂y(uh / h))

frames = Tuple{Float64, Matrix{Float64}}[]
function grab(sim)
    compute!(ζ)
    # ζ lives on cell corners (Ny+1 in the bounded y); trim to Ny to match yc
    push!(frames, (sim.model.clock.time, Array(interior(ζ))[:, 1:Ny, 1]))
end
simulation.callbacks[:grab] = Callback(grab, TimeInterval(1.0))

@info "Running Bickley-jet instability simulation..."
run!(simulation)

xc = LinRange(0, Lx, Nx)
yc = LinRange(-Ly/2, Ly/2, Ny)
n  = Observable(1)
field = @lift frames[$n][2]
clim = 0.6 * maximum(abs, frames[end][2]) + eps()

fig = Figure(size = (560, 720))
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "y",
           title = "Bickley jet — vorticity")
hm  = heatmap!(ax, xc, yc, field, colormap = :balance, colorrange = (-clim, clim))
Colorbar(fig[1, 2], hm, label = "ζ")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "08_barotropic_instability_bickley.mp4")
CairoMakie.record(fig, outfile, 1:length(frames); framerate = 18) do i
    n[] = i
    ax.title = @sprintf("Barotropic instability — t = %.0f", frames[i][1])
end
save(joinpath(@__DIR__, "output", "08_barotropic_instability_bickley.png"), fig)
@info "Saved $outfile"
