using StaticArrays
using Tenkai
Eq = Tenkai.EqLinAdv1D
# Submodules

plotlyjs() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.1, 1.0

boundary_condition = (dirichlet, neumann)
final_time = 1.0
velocity_, initial_value, exact_solution = Eq.or_data
boundary_value = exact_solution

degree = 1
solver = "rkfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.upwind

bound_limit = "no"
bflux = extrapolate

nx = 50
bounds = ([-1.0], [1.0])
cfl = 0.0
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.1
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
eq = Eq.get_equation(velocity_)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, eq,
                                          ARGS)
#------------------------------------------------------------------------------
errors, plot_data = Tenkai.solve(eq, problem, scheme, param)

return errors, plot_data
