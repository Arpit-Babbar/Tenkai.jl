using StaticArrays
using Tenkai

using LinearAlgebra: norm
# Submodules
Eq = Tenkai.EqTenMoment2D
plotlyjs() # Set backend

xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
boundary_condition = (neumann, neumann, neumann, neumann)

@inline function f_rare(a, b)
    if a < b
        return -2.0 * (a / b)^3 + 3.0 * (a / b)^2
    else
        return 1.0
    end
end

function near_vacuum_ic(x, y, Δx, equations::Eq.TenMoment2D)
    r = sqrt(x^2 + y^2)
    s = 0.06 * Δx
    local θ
    θ = atan(y, x) # The range of this function is [-π,π]
    rho = 1.0
    # The f smoothens the velocity near the origin
    v1 = 8.0 * cos(θ) * f_rare(r, s)
    v2 = 8.0 * sin(θ) * f_rare(r, s)
    P11 = 1.0
    P12 = 0.0
    P22 = 1.0
    return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

dummy_bv(x, t) = 0.0

eq = Eq.get_equation()

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.02

nx, ny = 100, 100

near_vacuum_ic(x, y) = near_vacuum_ic(x, y, (xmax - xmin) / nx, eq)

exact_near_vacuum(x, y, t) = near_vacuum_ic(x, y)

cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.25
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, near_vacuum_ic, dummy_bv, boundary_condition,
                  final_time, exact_near_vacuum)
limiter = setup_limiter_blend(blend_type = fo_blend(eq),
                              indicating_variables = Eq.conservative_indicator!,
                              #   indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner")

limiter = setup_limiter_tvbβ(eq; tvbM = tvbM, beta = 0.9)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
