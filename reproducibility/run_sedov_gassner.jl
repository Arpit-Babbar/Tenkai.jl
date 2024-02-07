using Tenkai
Eq = Tenkai.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = -1.5, 1.5
ymin, ymax = -1.5, 1.5

boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

initial_value, exact_solution = Eq.sedov_data
boundary_value = exact_solution

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes"
bflux = evaluate
final_time = 20.0

nx, ny = 64, 64
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate_time_factor = 1 # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain,
                  initial_value,
                  boundary_value, boundary_condition,
                  final_time, exact_solution)
blend = setup_limiter_blend(blend_type = mh_blend(equation),
                            indicating_variables = Eq.rho_p_indicator!,
                            reconstruction_variables = conservative_reconstruction,
                            indicator_model = "gassner",
                            debug_blend = false,
                            pure_fv = false)
limiter = blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol
