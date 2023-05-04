using SSFR
Eq = SSFR.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

boundary_condition = (dirichlet, neumann, neumann, neumann)
γ = 1.4

initial_value, exact_solution = Eq.shock_vortex_data

boundary_value = exact_solution

degree = 2
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.hllc

bound_limit = "yes"
bflux = evaluate
final_time = 0.5 #  20 * sqrt(2.0) / 0.5

nx, ny = 100, 100
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 100.0
save_iter_interval = 0
save_time_interval = final_time / 20.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);

return errors, plot_data
