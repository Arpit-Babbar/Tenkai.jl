# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
Eq = Tenkai.EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0

boundary_condition = (dirichlet, dirichlet, dirichlet, dirichlet)
γ = 2.0
equation = Eq.get_equation(γ)

initial_value = Eq.hurricane_initial_solution

exact_solution = Eq.exact_solution_hurricane_critical
boundary_value = Eq.exact_solution_hurricane_critical
degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.045

nx = 200
ny = 200
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
mh = mh_blend(equation)
muscl = muscl_blend(equation)
limiter = setup_limiter_blend(blend_type = mh,
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "mdrk_results/output_hurricane_v0critical")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol
