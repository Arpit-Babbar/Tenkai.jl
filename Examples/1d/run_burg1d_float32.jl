using Tenkai
Eq = Tenkai.EqBurg1D

#------------------------------------------------------------------------------
# Float32 test - using Float32 for all domain and parameters
xmin, xmax = 0.0f0, 2.0f0 * Float32(pi)
initial_value = Eq.initial_value_burger_sin
boundary_value = Eq.zero_boundary_value # dummy function
boundary_condition = (periodic, periodic)
final_time = 0.1f0

exact_solution = Eq.exact_solution_burger_sin

source_terms(u, x, t, eq) = zero(u)

degree = 3
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bflux = evaluate
numerical_flux = Eq.rusanov
bound_limit = "no"

nx = 20  # Small grid for fast testing
cfl = 0.0f0
bounds = (Float32[-0.2], Float32[0.2])
tvbM = 0.0f0
save_iter_interval = 0
save_time_interval = 0.0f0
compute_error_interval = 0
animate = false
#------------------------------------------------------------------------------
grid_size = nx
domain = Float32[xmin, xmax]  # Explicit Float32 vector
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = source_terms)
equation = Eq.get_equation()
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, cfl_safety_factor = 0.95f0, eps = 1.0f0)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol
