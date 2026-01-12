using Tenkai
Eq = Tenkai.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = -10.0, 10.0
ymin, ymax = -10.0, 10.0

boundary_value = Eq.zero_bv # dummy
boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

initial_value = Eq.isentropic_iv
exact_solution = Eq.isentropic_exact

degree = 4
solver = LWEnzymeTower()
# solver = LWTDEltWise()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.1 # 20 * sqrt(2.0) / 0.5

nx = 100
ny = 100
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 5.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
limiter = setup_limiter_blend(blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false,
                              pure_fv = true,
                              tvbM = Inf)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux, 2)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   time_scheme = "by degree")
#------------------------------------------------------------------------------
# problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
#                                           ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
