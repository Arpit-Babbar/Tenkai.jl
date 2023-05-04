using StaticArrays
using SSFR
Eq = SSFR.EqLinAdv2D

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
velocity, initial_value, exact_solution = Eq.linear_vel1_data

boundary_value = exact_solution
boundary_condition = (dirichlet, neumann, dirichlet, neumann)
final_time = 1.0

degree = 3
solver = "rkfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux   = extrapolate

nx, ny = 50, 50
bounds = ([-Inf],[Inf])
cfl = 0.0
tvbM = 1.0
save_iter_interval = 0
save_time_interval = 0.05
compute_error_interval = 0
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                     final_time, exact_solution)
equation = Eq.get_equation(velocity)
limiter = setup_limiter_tvbβ(equation; tvbM = tvbM, beta = 1.0)
# limiter = FR.setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                   numerical_flux, bound_limit, limiter, bflux, 2)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval)
#------------------------------------------------------------------------------
problem, scheme, param, = ParseCommandLine(problem, param, scheme, equation,
                                           ARGS)
#------------------------------------------------------------------------------
SSFR.solve(equation, problem, scheme, param);
