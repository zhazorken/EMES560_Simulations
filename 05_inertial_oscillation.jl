# ==============================================================================
# 05 — Inertial oscillation on the f-plane
# Concept: with no pressure gradient, a moving parcel loops in anticyclonic
#          inertial circles of period T = 2π/f. Shows the Coriolis force as a
#          pure deflection (does no work — speed is constant).
# Lecture: Week 7 (Rotation & Coriolis). Tiny grid. Cost: seconds.
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
using Statistics: mean

# A tiny box: the flow stays horizontally uniform, so there are no pressure
# gradients — the momentum balance is just  du/dt = fv,  dv/dt = −fu.
grid = RectilinearGrid(arch; size = (4, 4, 4), extent = (1, 1, 1),
                       topology = (Periodic, Periodic, Bounded))

f = 1e-4                                   # Coriolis parameter (mid-latitude)
model = NonhydrostaticModel(grid; coriolis = FPlane(f = f),
                            buoyancy = nothing, tracers = ())

set!(model, u = 0.1)                        # give it an initial eastward kick

T = 2π / f                                  # inertial period (~17.5 h)
simulation = Simulation(model, Δt = T/200, stop_time = 2.2T)

track = Tuple{Float64, Float64, Float64}[]  # (t, ū, v̄)
grab(sim) = push!(track, (sim.model.clock.time,
                          mean(interior(sim.model.velocities.u)),
                          mean(interior(sim.model.velocities.v))))
simulation.callbacks[:grab] = Callback(grab, IterationInterval(1))

@info "Running inertial-oscillation simulation..."
run!(simulation)

ts = getindex.(track, 1) ./ 3600            # hours
us = getindex.(track, 2)
vs = getindex.(track, 3)

# static figure: time series + hodograph (the inertial circle)
fig = Figure(size = (900, 400))
ax1 = Axis(fig[1, 1], xlabel = "time (hours)", ylabel = "velocity (m/s)",
           title = "Inertial oscillation: u, v")
lines!(ax1, ts, us, label = "u"); lines!(ax1, ts, vs, label = "v")
axislegend(ax1)
ax2 = Axis(fig[1, 2], xlabel = "u (m/s)", ylabel = "v (m/s)", aspect = 1,
           title = @sprintf("Hodograph — T = %.1f h", T/3600))
lines!(ax2, us, vs, color = :navy)
mkpath(joinpath(@__DIR__, "output"))
save(joinpath(@__DIR__, "output", "05_inertial_oscillation.png"), fig)

# animation: a dot tracing the inertial circle
n = Observable(2)
trace_u = @lift us[1:$n]; trace_v = @lift vs[1:$n]
head_u  = @lift [us[$n]]; head_v  = @lift [vs[$n]]
figA = Figure(size = (500, 500))
axA  = Axis(figA[1, 1], xlabel = "u (m/s)", ylabel = "v (m/s)", aspect = 1,
            title = "Inertial circle")
lines!(axA, trace_u, trace_v, color = :teal)
scatter!(axA, head_u, head_v, color = :orange, markersize = 14)
outfile = joinpath(@__DIR__, "output", "05_inertial_oscillation.mp4")
record(figA, outfile, 2:length(track); framerate = 25) do i
    n[] = i
end
@info "Saved $outfile"
