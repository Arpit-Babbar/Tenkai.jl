using Tenkai
Eq = Tenkai.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.1
ymin, ymax = 0.0, 1.1

boundary_condition = (reflect, neumann, reflect, neumann)
# boundary_condition = (neumann, neumann, neumann, neumann)
γ = 1.4
equation = Eq.get_equation(γ)

nx = ny = 160
function initial_value_sedov_zhang_shu(x, nx, ny)
    ρ = 1.0
    v1 = 0.0
    v2 = 0.0
    dx, dy = 1.1 / nx, 1.1 / ny
    d = dx
    # r = sqrt(x[1]^2 + x[2]^2)
    if x[1] > 1.5 * d || x[2] > 1.5 * d
        E = 10^(-12)
    else
        E = 0.244816 / (dx * dy)
    end
    return SVector(ρ, ρ * v1, ρ * v2, E)
end
initial_value = (x, y) -> initial_value_sedov_zhang_shu((x, y), nx, ny)
exact_solution = (x, y, t) -> initial_value(x, y)
boundary_value = exact_solution

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes"
bflux = evaluate
final_time = 1.0 #  20 * sqrt(2.0) / 0.5

cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time / 100.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.95

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
# equation = Eq.get_equation(γ)
problem = Problem(domain,
                  initial_value,
                  boundary_value, boundary_condition,
                  final_time, exact_solution)
blend = setup_limiter_blend(blend_type = mh_blend(equation),
                            indicating_variables = Eq.rho_p_indicator!,
                            reconstruction_variables = conservative_reconstruction,
                            indicator_model = "gassner",
                            debug_blend = false,
                            pure_fv = false)
limiter = blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor, eps = 1e-06)

#------------------------------------------------------------------------------
# @suppress
sol = Tenkai.solve(equation, problem, scheme, param);

return sol
