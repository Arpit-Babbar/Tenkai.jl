import Roots.find_zero
using Tenkai
# Submodules
Eq = Tenkai.EqBurg1D

#------------------------------------------------------------------------------
xmin, xmax = -2.0, 2.0
initial_value = Eq.initial_value_hat
boundary_condition = (dirichlet, dirichlet)
final_time = 0.5

exact_solution = Eq.exact_solution_hat
boundary_value = exact_solution

degree = 4
solver = "lwfr"
solution_points = "gll"
correction_function = "radau"
bflux = evaluate
numerical_flux = Eq.rusanov
bound_limit = "no"

nx = 100
cfl = 0.0
bounds = ([-0.2], [0.2])
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time / 10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation()
limiter = setup_limiter_blend(blend_type = fo_blend(equation),
                              indicating_variables = conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false)
limiter = setup_limiter_none()
limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
