using StaticArrays
using Tenkai
Eq = Tenkai.EqBurg2D

#------------------------------------------------------------------------------
xmin, xmax = 0.0f0, 1.0f0
ymin, ymax = 0.0f0, 1.0f0
initial_value = Eq.burger_sin_iv
exact_solution = Eq.burger_sin_exact
boundary_value = exact_solution # dummy function
zero_source_terms = (u, x, t, eq) -> zero(u)

boundary_condition = (periodic, periodic, periodic, periodic)
final_time = 0.1f0

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bound_limit = "no"
bflux = evaluate
numerical_flux = Eq.rusanov

nx = 10
ny = 10
bounds = (Float32[-Inf], Float32[Inf])
cfl = 0.0f0
save_iter_interval = 0
save_time_interval = 0.0f0 # final_time / 10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = zero_source_terms)
equation = Eq.get_equation()
limiter = setup_limiter_blend(blend_type = mh_blend(equation),
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              constant_node_factor = 1.0f0,
                              debug_blend = false,
                              pure_fv = false)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, time_scheme = "RK4")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

print(sol["errors"])

return sol;
