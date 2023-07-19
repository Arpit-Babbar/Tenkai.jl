using StaticArrays
using Tenkai
using Plots
# Submodules
Eq = Tenkai.EqTenMoment1D
plotlyjs() # Set backend

xmin, xmax = 0.0, 1.0
boundary_condition = (periodic, periodic)

function dwave(x, equations::Eq.TenMoment1D)
   rho = 1.0 + 0.5*sinpi(2.0*x)
   v1 = v2 = P11 = P12 = P22 = 1.0
   return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

dummy_bv(x,t) = 0.0

eq = Eq.get_equation()

dwave(x) = dwave(x, eq)

exact_dwave(x,t) = dwave(x-t)

initial_value, exact_solution, boundary_value = dwave, exact_dwave, dummy_bv

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.hll
bound_limit = "no"
bflux = evaluate
final_time = 1.0

nx = ceil(Int64,100)
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
