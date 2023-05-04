# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using SSFR
using Plots
# Submodules
Eq = SSFR.EqEuler1D
plotlyjs() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (neumann, neumann)
γ = 1.4

initial_value_ref, exact_solution, final_time, ic_name = Eq.toro5_data

boundary_value = exact_solution # dummy function

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = ceil(Int64,1200/(degree+1))
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 10.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.95

# blend parameters
indicator_model = "gassner"
cfl_safety_factor = 0.95
pure_fv = false
debug_blend = true

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value_ref, boundary_value,
                  boundary_condition, final_time, exact_solution)
equation = Eq.get_equation(γ)
limiter = setup_limiter_blend(
                              blend_type = fo_blend(equation),
                              indicating_variables = Eq.primitive_indicator!,
                              # indicating_variables = Eq.primitive_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = indicator_model,
                              debug_blend = debug_blend,
                              pure_fv = pure_fv
                             )
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(
                      grid_size, cfl, bounds, save_iter_interval,
                      save_time_interval, compute_error_interval;
                      animate = animate,
                      cfl_safety_factor = cfl_safety_factor,
                      time_scheme = "SSPRK54"
                  )
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);

return errors, plot_data
