# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
Eq = Tenkai.EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0

boundary_value = (x, t) -> 0.0 # dummy function
boundary_condition = (neumann, neumann, neumann, neumann)
γ = 2.0
equation = Eq.get_equation(γ)

function hurricane_initial_solution(x, y)
    A = 25.0
    gamma = 2.0
    gamma_minus_1 = gamma - 1.0
    theta = atan(y, x)
    r = sqrt(x^2 + y^2)
    rho = 1.0
    v0 = 12.5 # Choices - (10, 12.5, 7.5) which give M0 as (√2, >√2, <√2)
    p = A * rho^gamma

    v1 = v0 * sin(theta)
    v2 = -v0 * cos(theta)

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_e = p / gamma_minus_1 + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

initial_value = hurricane_initial_solution

exact_solution = (x, y, t) -> hurricane_initial_solution(x, y) # Dummy
degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.045

nx, ny = 400, 400 # 50, 50
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = final_time / 30.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.5

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
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
sol = Tenkai.solve(equation, problem, scheme, param);

return sol["errors"]
