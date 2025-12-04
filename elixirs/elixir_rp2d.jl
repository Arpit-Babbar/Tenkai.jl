using Downloads: download
using OrdinaryDiffEq
using Trixi
include("common/lw_cfl.jl")

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_rp(x_, t, equations::CompressibleEulerEquations2D)
    x, y = x_[1], x_[2]

    if x >= 0.5 && y >= 0.5
        rho, v1, v2, p = (0.5313, 0.0, 0.0, 0.4)
    elseif x < 0.5 && y >= 0.5
        rho, v1, v2, p = (1.0, 0.7276, 0.0, 1.0)
    elseif x < 0.5 && y < 0.5
        rho, v1, v2, p = (0.8, 0.0, 0.0, 1.0)
    elseif x >= 0.5 && y < 0.5
        rho, v1, v2, p = (1.0, 0.0, 0.7276, 1.0)
    end

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_rp

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# The initial condition is 2-periodic
coordinates_min = (-1.5, -1.5) # minimum coordinates (min(x), min(y))
coordinates_max = (2.5, 2.5) # maximum coordinates (max(x), max(y))

mesh = TreeMesh(coordinates_min, coordinates_max,
                periodicity = true, n_cells_max = 1025^2 * 16,
                initial_refinement_level = 10)

volume_flux = flux_central
surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = 100)

save_solution = SaveSolutionCallback(interval = 10000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = joinpath(@__DIR__, "out"))
stepsize_callback = StepsizeCallback(cfl = trixi2lw(0.98, solver))

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

###############################################################################
# run the simulation
sol = solve(ode, SSPRK54(stage_limiter!),
            dt = 1,
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
