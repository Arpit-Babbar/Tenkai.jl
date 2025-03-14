# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
Eq = Tenkai.EqEuler2D
using Tenkai.EqEuler2D: hllc_bc
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 0.25
ymin, ymax = 0.0, 1.0

boundary_condition = (reflect, reflect, hllc_bc, dirichlet)
γ = 5.0 / 3.0
equation = Eq.get_equation(γ)
function initial_condition_rayleigh_taylor(x, y)
    gamma = 5.0 / 3.0
    if y <= 0.5
        rho = 2.0
        p = 2.0 * y + 1.0
        c = sqrt(gamma * p / rho)
        v1 = 0.0
        v2 = -0.025 * c * cospi(8.0 * x)
    else
        rho = 1.0
        p = 1.5 + y
        c = sqrt(gamma * p / rho)
        v1 = 0.0
        v2 = -0.025 * c * cospi(8.0 * x)
    end
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    gamma_minus_1 = gamma - 1.0
    rho_e = p / gamma_minus_1 + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

# Used to set the top and bottom boundary conditions
function boundary_condition_rayleigh_taylor(x, y, t)
    gamma = 5.0 / 3.0
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
solver = "mdrk"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 2.5

nx = 64
ny = 4 * nx
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = final_time / 100.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_condition_rayleigh_taylor,
                  boundary_condition,
                  final_time, exact_solution, source_terms = source_terms_rayleigh_taylor)
limiter = setup_limiter_blend(blend_type = fo_blend(equation),
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
                   saveto = "mdrk_results/output_rayleigh_taylor")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol["errors"]
