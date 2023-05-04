using StaticArrays
using SSFR
Eq = SSFR.EqLinAdv2D

#------------------------------------------------------------------------------
(xmin, xmax, ymin, ymax, velocity, initial_value,
 exact_solution) = Eq.smooth_sin_vel1_data
boundary_value = exact_solution # dummy function
boundary_condition = (periodic, periodic, periodic, periodic)
final_time = 1.0

degree = 4
solver = "rkfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = extrapolate

nx, ny = 10, 10
bounds = ([-Inf],[Inf])
cfl = 0.0
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
compute_error_interval = 0
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation(velocity)
# limiter = setup_limiter_blend(
#                               blend_type = mh_blend(equation),
#                               indicating_variables = conservative_indicator!,
#                               reconstruction_variables = conservative_reconstruction,
#                               indicator_model = "gassner",
#                               debug_blend = false,
#                               pure_fv = true
#                              )
# limiter = setup_limiter_tvbβ(equation; tvbM = tvbM, beta = 1.0)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM, beta = 1.0)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux, 2)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval, time_scheme = "SSPRK54")
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
SSFR.solve(equation, problem, scheme, param);
