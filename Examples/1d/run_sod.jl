# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqEuler1D
plotlyjs() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (neumann, neumann)
γ = 1.4

initial_value, exact_solution, final_time, ic_name = Eq.sod_data

boundary_value = exact_solution # dummy function

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 100
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 10.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

blend_type = "muscl"

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation(γ)
# limiter = setup_limiter_blend(
#                                  blend_type = Tenkai.mh_blend,
#                                  indicating_variables = Tenkai.rho_p_indicator!,
#                                  reconstruction_variables = Tenkai.conservative_reconstruction,
#                                  indicator_model = indicator_model,
#                                  debug_blend = debug_blend
#                                 )
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
limiter = setup_limiter_hierarchical(alpha = 1.0,
                                     reconstruction = characteristic_reconstruction)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK54")
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return errors, plot_data
