using StaticArrays
using Tenkai
using Plots
# Submodules
Eq = Tenkai.EqTenMoment1D
plotlyjs() # Set backend

xmin, xmax = -0.5, 0.5
boundary_condition = (neumann, neumann)

dummy_bv(x, t) = 0.0

initial_value = Eq.two_rare_vacuum_iv # choices - Eq.sod_iv, Eq.two_shock_iv, Eq.two_rare_iv, Eq.two_rare_vacuum_iv
boundary_value = dummy_bv
exact_solution_rp(x, t) = initial_value(x)
exact_solution = exact_solution_rp

degree = 2
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.05 # Choose 0.125 for sod, two_shock; 0.15 for two_rare_iv; 0.05 for two_rare_vacuum_iv

nx = ceil(Int64, 400)
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 10.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = nx
eq = Eq.get_equation()
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
# limiter = setup_limiter_none()
limiter = setup_limiter_blend(blend_type = mh_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner"
                              # pure_fv = true
                              )
limiter = setup_limiter_tvb(eq; tvbM = tvbM, beta = 1.4)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = 0.98)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;

sol["plot_data"].p_ua
