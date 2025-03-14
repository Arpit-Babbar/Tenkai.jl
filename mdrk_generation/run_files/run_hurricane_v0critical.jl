# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
Eq = Tenkai.EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0

boundary_condition = (dirichlet, dirichlet, dirichlet, dirichlet)
γ = 2.0
equation = Eq.get_equation(γ)

function hurricane_initial_solution(x, y)
    A = 25.0
    gamma = 2.0
    gamma_minus_1 = gamma - 1.0
    theta = atan(y, x)
    r = sqrt(x^2 + y^2)
    rho = 1.0
    v0 = 10.0 # Choices - (10, 12.5, 7.5) which give M0 as (√2, >√2, <√2)
    p = A * rho^gamma

    v1 = v0 * sin(theta)
    v2 = -v0 * cos(theta)

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_e = p / gamma_minus_1 + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

function exact_solution_critical(x, y, t)
    A = 25.0
    gamma = 2.0
    gamma_minus_1 = gamma - 1.0
    theta = atan(y, x)
    r = sqrt(x^2 + y^2)
    rho0 = 1.0
    v0 = 10.0 # Choices - (10, 12.5, 7.5) which give M0 as (√2, >√2, <√2)
    p0 = A * rho0^gamma
    p0_prime = gamma * A * rho0^gamma_minus_1

    if r >= 2.0 * t * sqrt(p0_prime)
        rho = rho0

        v1 = 2.0 * t * p0_prime * cos(theta)
        v1 += sqrt(2.0 * p0_prime) * sqrt(r^2 - 2.0 * t^2 * p0_prime) * sin(theta)
        v1 /= r

        v2 = 2.0 * t * p0_prime * sin(theta)
        v2 -= sqrt(2.0 * p0_prime) * sqrt(r^2 - 2.0 * t^2 * p0_prime) * cos(theta)
        v2 /= r

        p = p0
    else
        rho = r^2 / (8.0 * A * t^2)

        v1 = (x + y) / (2.0 * t)
        v2 = (-x + y) / (2.0 * t)

        p = A * rho * gamma # Don't know if it is right,
        # but it does not matter in boundary value computation
    end

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_e = p / gamma_minus_1 + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

initial_value = hurricane_initial_solution

exact_solution = exact_solution_critical
boundary_value = exact_solution_critical
degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.045

nx, ny = 200, 200 # 50, 50
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = final_time / 30.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.98

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
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "mdrk_results/output_hurricane_v0critical")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol["errors"]
