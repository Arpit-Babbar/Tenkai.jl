import Roots.find_zero
using Tenkai
# Submodules

Eq = Tenkai.EqBuckleyLeverett1D
 # Set backend

#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
initial_value = Eq.hatbuck_iv
boundary_value = Eq.hatbuck_exact # dummy function
boundary_condition = (periodic, periodic)
exact_solution = Eq.hatbuck_exact_a025
final_time = 0.4

degree = 3
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bflux = evaluate
numerical_flux = Eq.upwind
bound_limit = "yes"

nx = 50
if degree == 3
    cfl = 0.079
else
    cfl = 0.0
end
bounds = ([0.0], [1.0])
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
eq = Eq.get_equation()
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_tvb(eq; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate)
#------------------------------------------------------------------------------
problem, scheme, param, = ParseCommandLine(problem, param, scheme, eq, ARGS)
#------------------------------------------------------------------------------
Tenkai.solve(eq, problem, scheme, param)
