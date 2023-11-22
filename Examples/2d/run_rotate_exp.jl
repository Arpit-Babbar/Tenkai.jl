using StaticArrays
using Tenkai
Eq = Tenkai.EqLinAdv2D
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
final_time = 0.1 * pi # Kept small for unit tests. Correct in run_rotate_linadv2d.jl
boundary_condition = (neumann, dirichlet, dirichlet, neumann)
initial_value, velocity, exact_solution = Eq.rotate_exp_data
boundary_value = exact_solution
#------------------------------------------------------------------------------
degree = 4
solver = "rkfr"
solution_points = "gl"
correction_function = "radau"
bound_limit = "no"
bflux = evaluate
numerical_flux = Eq.rusanov

nx, ny = 10, 10
bounds = ([-Inf], [Inf])
cfl = 0.0
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation(velocity)
# limiter = setup_limiter_blend(
#                                  blend_type = fo_blend,
#                                  indicating_variables = conservative_indicator!,
#                                  reconstruction_variables = conservative_reconstruction,
#                                  indicator_model = "gassner",
#                                  debug_blend = false
#                                 )
# limiter = setup_limiter_tvb(eq; tvbM = tvbM)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
