# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
Eq = Tenkai.EqEuler2D
using Tenkai.EqEuler2D: hllc_bc
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 0.25
ymin, ymax = 0.0,  1.0

boundary_condition = (reflect, reflect, hllc_bc, dirichlet)
γ = 5.0/3.0
equation = Eq.get_equation(γ)


initial_value = Eq.initial_condition_rayleigh_taylor

exact_solution = Eq.boundary_condition_rayleigh_taylor
degree = 3
solver = "rkfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.5

nx = 20
ny = 4 * nx
cfl = 0.0 # 0.98 * 0.215
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 100.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, Eq.boundary_condition_rayleigh_taylor, boundary_condition,
                  final_time, exact_solution, source_terms = Eq.source_terms_rayleigh_taylor)
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
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol
