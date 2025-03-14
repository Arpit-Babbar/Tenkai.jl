using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqTenMoment2D
plotlyjs() # Set backend

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
boundary_condition = (periodic, periodic, periodic, periodic)

function dwave(x, y, equations::Eq.TenMoment2D)
    rho = 2.0 + sinpi(2.0 * x) * sinpi(2.0 * y)
    v1 = 1.0
    v2 = 2.0
    P11 = 1.0
    P12 = 0.0
    P22 = 1.0
    return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

dummy_bv(x, t) = 0.0

eq = Eq.get_equation()

dwave(x, y) = dwave(x, y, eq)

exact_dwave(x, y, t) = dwave(x - t, y - 2.0 * t)

initial_value, exact_solution, boundary_value = dwave, exact_dwave, dummy_bv

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.hllc3
bound_limit = "no"
bflux = evaluate
final_time = 0.01

nx, ny = 20, 20
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = fo_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner")

# limiter = setup_limiter_tvbβ(eq; tvbM = tvbM, beta = 2.0)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
