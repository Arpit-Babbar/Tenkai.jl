using StaticArrays
using Tenkai
Eq = Tenkai.EqBurg2D

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
initial_value = Eq.burger_sin_iv
exact_solution = Eq.burger_sin_exact
boundary_value = exact_solution # dummy function
boundary_condition = (periodic, periodic, periodic, periodic)
final_time = 0.1

degree = 1
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bound_limit = "no"
bflux = evaluate
numerical_flux = Eq.rusanov

nx, ny = 640, 640
bounds = ([-Inf],[Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation()
limiter = setup_limiter_blend(
                              blend_type = mh_blend(equation),
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              constant_node_factor = 1.0,
                              debug_blend = false,
                              pure_fv = true
                             )
# limiter = setup_limiter_none()
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

print(sol["errors"])

return sol;

