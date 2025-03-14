using StaticArrays
using Tenkai

Eq = Tenkai.EqLinAdv1D
 # Set backend

#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
velocity(x) = 1.0

boundary_condition = (dirichlet, neumann)
final_time = 1.0
initial_value, exact_solution = Eq.linear1d_data
boundary_value = exact_solution

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate

nx = 100
bounds = ([-Inf], [Inf])
cfl = 0.0
tvbM = 5.0
save_iter_interval = 0
save_time_interval = 0.0
compute_error_interval = 0
animate = true

cfl_safety_factor = 0.95

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation(velocity)
# limiter = Tenkai.setup_limiter_blend(
#                                  blend_type = Tenkai.mh_blend,
#                                  indicating_variables = Tenkai.conservative_indicator!,
#                                  reconstruction_variables = Tenkai.conservative_reconstruction,
#                                  indicator_model = "gassner",
#                                  debug_blend = false
#                                 )
limiter = setup_limiter_tvb(eq = equation, tvbM = tvbM)
# limiter = Tenkai.setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return errors, plot_data
