using OrdinaryDiffEq
using Trixi

include("common/lw_cfl.jl")
###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    r = sqrt(x[1]^2 + x[2]^2)
    γ = equations.gamma

    v1 = v2 = 0.0
    σ_ρ = 0.25
    ρ0 = 1.0
    rho = ρ0 + 0.25 / (π * σ_ρ^2) * exp(-0.5 * r^2 / σ_ρ^2)

    σ_p = 0.15
    p0 = 1.0e-5
    p = p0 + 0.25 * (γ - 1.0) / (π * σ_p^2) * exp(-0.5 * r^2 / (σ_p^2))

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_central
basis = LobattoLegendreBasis(4)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

trees_per_dimension = (64, 64)

coordinates_min = (-1.5, -1.5)
coordinates_max = (1.5, 1.5)
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = 100)

save_solution = SaveSolutionCallback(interval = 10000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

stepsize_callback = StepsizeCallback(cfl = trixi2lw(0.98, solver))

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-10, 5.0e-10),
                                                     variables = (Trixi.density, pressure))
###############################################################################
# run the simulation

sol = solve(ode, SSPRK54(stage_limiter!),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
