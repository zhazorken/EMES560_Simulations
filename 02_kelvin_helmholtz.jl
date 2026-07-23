# ==============================================================================
# 02 — Kelvin–Helmholtz instability of a stratified shear layer
# Concept: shear instability, the Richardson number (Ri < 1/4), billow roll-up,
#          and mixing of a stratified fluid.
# Lecture: Week 6 (Boundary Layers & Turbulence) and Week 12 (Instabilities).
# 2D x–z, nonhydrostatic, Boussinesq. Cost: ~1 min.
# ==============================================================================
import Pkg; Pkg.activate(@__DIR__)
using Oceananigans

# CPU by default; set OCEAN_ARCH=GPU (done by the Casper PBS script) to run on a GPU.
# On Oceananigans >= 0.109 the zero-arg GPU() lives in the CUDA extension, so load CUDA first.
if get(ENV, "OCEAN_ARCH", "CPU") == "GPU"
    using CUDA
    arch = GPU()
else
    arch = CPU()
end
using CairoMakie
using Printf

# ---- grid -------------------------------------------------------------------
Nx, Nz = 128, 96
Lx, Lz = 14.0, 10.0
grid = RectilinearGrid(arch; size = (Nx, Nz), x = (0, Lx), z = (-Lz/2, Lz/2),
                       topology = (Periodic, Flat, Bounded))

model = NonhydrostaticModel(; grid,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4))

# ---- background shear + stratification --------------------------------------
# shear layer  U(z) = U₀ tanh(z/h);  uniform stratification  b = N² z
# Minimum Richardson number  Ri = N² / (dU/dz)²  is set below 1/4 → unstable.
h  = 0.5           # shear-layer half thickness
U₀ = 0.5           # half the velocity jump
N² = 0.10          # buoyancy frequency squared  →  Ri_min = N²/(U₀/h)² = 0.1 < 1/4
λ  = 7.0           # perturbation wavelength (≈ most unstable mode)

Uᵢ(x, z) = U₀ * tanh(z / h)
Bᵢ(x, z) = N² * z
# a small vertical-velocity perturbation seeds a clean billow train
Wᵢ(x, z) = 1e-2 * sin(2π * x / λ) * exp(-(z / h)^2)

set!(model, u = Uᵢ, w = Wᵢ, b = Bᵢ)

simulation = Simulation(model, Δt = 0.02, stop_time = 45)

frames = Tuple{Float64, Matrix{Float64}}[]
grab(sim) = push!(frames,
    (sim.model.clock.time, Array(interior(sim.model.tracers.b))[:, 1, :]))
simulation.callbacks[:grab] = Callback(grab, TimeInterval(0.5))

@info "Running Kelvin–Helmholtz simulation..."
run!(simulation)

# ---- animate the buoyancy field (billows stir the density interface) --------
xc = LinRange(Lx/2Nx, Lx - Lx/2Nx, Nx)
zc = LinRange(-Lz/2 + Lz/2Nz, Lz/2 - Lz/2Nz, Nz)
n  = Observable(1)
field = @lift frames[$n][2]
clim = N² * Lz/2

fig = Figure(size = (820, 460))
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "z",
           title = "Kelvin–Helmholtz billows (Ri = 0.1)")
hm  = heatmap!(ax, xc, zc, field, colormap = :balance, colorrange = (-clim, clim))
Colorbar(fig[1, 2], hm, label = "buoyancy b")

mkpath(joinpath(@__DIR__, "output"))
outfile = joinpath(@__DIR__, "output", "02_kelvin_helmholtz.mp4")
record(fig, outfile, 1:length(frames); framerate = 15) do i
    n[] = i
    ax.title = @sprintf("Kelvin–Helmholtz (Ri = 0.1) — t = %.1f", frames[i][1])
end
save(joinpath(@__DIR__, "output", "02_kelvin_helmholtz.png"), fig)
@info "Saved $outfile"
