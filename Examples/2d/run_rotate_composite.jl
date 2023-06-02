using StaticArrays
using SSFR
Eq = SSFR.EqLinAdv2D

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
final_time = 2*pi
boundary_condition = (dirichlet, dirichlet, dirichlet, dirichlet)
velocity, initial_value, exact_solution = Eq.composite2d_data
boundary_value = exact_solution
#------------------------------------------------------------------------------
degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bound_limit = "no"
bflux = evaluate
numerical_flux = Eq.rusanov

nx, ny = 100, 100
bounds = ([-Inf],[Inf])
cfl = 0.0
tvbM = 100.0
save_iter_interval = 0
save_time_interval = 0.1*pi
compute_error_interval = 0
cfl_safety_factor = 0.5
pure_fv = false
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
eq = Eq.get_equation(velocity)
limiter = setup_limiter_blend(
                              blend_type = mh_blend(eq),
                              indicating_variables = conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false,
                              smooth_alpha = true,
                              pure_fv = pure_fv
                             )
# limiter = setup_limiter_tvb(eq; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, eq, ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
