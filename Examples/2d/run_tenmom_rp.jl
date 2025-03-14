using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqTenMoment2D
# Set backend

xmin, xmax = -0.5, 1.5
ymin, ymax = -0.5, 1.5
boundary_condition = (neumann, neumann, neumann, neumann)

dummy_bv(x, t) = 0.0

initial_value = Eq.two_rare_vacuum1d_iv
boundary_value = dummy_bv
exact_solution_rp(x, y, t) = initial_value(x, y)
exact_solution = exact_solution_rp
degree = 4

solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.05

nx, ny = 200, 200
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 10.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.8

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
eq = Eq.get_equation()
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = fo_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner")
# limiter = setup_limiter_none()
# limiter = setup_limiter_tvbβ(eq; tvbM = tvbM, beta = 1.0)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
