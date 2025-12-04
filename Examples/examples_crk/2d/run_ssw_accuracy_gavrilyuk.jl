using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.StaticArrays
using Tenkai.TenkaicRK.Tenkai

# Submodules
Eq = TenkaicRK.EqShearShallowWater2D

# Submodules
Eq = TenkaicRK.EqShearShallowWater2D
EqTenMom = Tenkai.EqTenMoment2D
using Tenkai: @unpack

xmin, xmax = 0.0, 10.0
ymin, ymax = 0.0, 10.0
boundary_condition = (dirichlet, dirichlet, dirichlet, dirichlet)

function exact_solution_ssw_convergence(x, y, t)

    # Constants
    h0 = 1.0
    beta = 0.001
    lambda = 0.1
    gamma = 0.01

    # Precompute some terms
    bt = beta * t
    bt2 = bt^2
    c1 = beta / (1.0 + bt2)
    c2 = 1.0 / (1.0 + bt2)^2

    # Calculate primitive variables
    h = h0 / (1.0 + bt2)
    v1 = c1 * (bt * x + y)
    v2 = c1 * (bt * y - x)

    # Calculate P values
    P11 = c2 * (lambda + gamma * bt2)
    P12 = c2 * (lambda - gamma) * bt
    P22 = c2 * (gamma + lambda * bt2)
    R11 = h * P11
    R12 = h * P12
    R22 = h * P22

    return EqTenMom.tenmom_prim2con((h, v1, v2, R11, R12, R22))
end

initial_value = (x, y) -> exact_solution_ssw_convergence(x, y, 0.0)

boundary_value = exact_solution_ssw_convergence
exact_solution = exact_solution_ssw_convergence

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 10.0

nx = ny = ceil(Int64, 80)
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.1

#------------------------------------------------------------------------------
grid_size = [nx, ny]
gravity = 9.81
eq = Eq.get_equation(; gravity)
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)

sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol
