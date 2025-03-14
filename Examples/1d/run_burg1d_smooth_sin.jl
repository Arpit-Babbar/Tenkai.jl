using Tenkai
using StaticArrays
Eq = Tenkai.EqBurg1D
using Plots
gr()
#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
burg_smooth_ic = x -> sinpi(2.0 * x)
burg_smooth_exact = (x, t) -> sinpi(2.0 * (x - t))
burg_smooth_source_terms = (u, x, t, eq) -> SVector(pi * (sin(4.0 * pi * (x - t)) -
                                                     2.0 * cos(2.0 * pi * (x - t))))

boundary_value = Eq.zero_boundary_value # dummy function
boundary_condition = (periodic, periodic)
final_time = 2.0

degree = 3
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bflux = evaluate
numerical_flux = Eq.rusanov
bound_limit = "no"

nx = 200
cfl = 0.0
bounds = ([-0.2], [0.2])
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time/10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, burg_smooth_ic, boundary_value, boundary_condition,
                  final_time, burg_smooth_exact, source_terms = burg_smooth_source_terms)
equation = Eq.get_equation()
limiter = setup_limiter_none()
# limiter = setup_limiter_blend(
#                               blend_type = mh_blend(equation),
#                               # indicating_variables = Eq.rho_p_indicator!,
#                               indicating_variables = conservative_indicator!,
#                               reconstruction_variables = conservative_reconstruction,
#                               indicator_model = "gassner",
#                               debug_blend = false,
#                               pure_fv = true
#                              )
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol
