# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
Eq = Tenkai.EqEuler2D
using Tenkai.EqEuler2D: hllc_bc
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 0.25
ymin, ymax = 0.0, 1.0

boundary_condition = (reflect, reflect, reflect, reflect)
γ = 1.4
equation = Eq.get_equation(γ)
function initial_condition_rayleigh_taylor(x, y)
    gamma = 1.4
    slope = 15.0
    if y < 0.5
        p = 2 * y + 1
    else
        p = y + 3 / 2
    end

    # smooth the discontinuity to avoid ambiguity at element interfaces
    smoothed_heaviside(x, left, right) = left + 0.5 * (1 + tanh(slope * x)) * (right - left)
    rho = smoothed_heaviside(y - 0.5, 2.0, 1.0)

    c = sqrt(gamma * p / rho)
    # the velocity is multiplied by sin(pi*y)^6 as in Remacle et al. 2003 to ensure that the
    # initial condition satisfies reflective boundary conditions at the top/bottom boundaries.
    v2 = -0.025 * c * cos(8 * pi * x[1]) * sin(pi * y)^6
    v1 = 0.0
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    gamma_minus_1 = gamma - 1.0
    rho_e = p / gamma_minus_1 + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

# Used to set the top and bottom boundary conditions
function boundary_condition_rayleigh_taylor(x, y, t)
    gamma = 1.4
    if y <= 0.5
        rho, v1, v2, p = (2.0, 0.0, 0.0, 1.0)
    else
        rho, v1, v2, p = (1.0, 0.0, 0.0, 2.5)
    end
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    gamma_minus_1 = gamma - 1.0
    rho_e = p / gamma_minus_1 + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

initial_value = initial_condition_rayleigh_taylor

source_terms_rayleigh_taylor(u, x, t, eq) = SVector(0.0, 0.0, u[1], u[3])

exact_solution = boundary_condition_rayleigh_taylor
degree = 3
solver = cRK44()
solution_points = "gll"
correction_function = "g2"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 2.5

nx = 256
ny = 4 * nx
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = final_time / 100.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_condition_rayleigh_taylor,
                  boundary_condition,
                  final_time, exact_solution, source_terms = source_terms_rayleigh_taylor)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "mdrk_results/output_rayleigh_taylor")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol["errors"]
