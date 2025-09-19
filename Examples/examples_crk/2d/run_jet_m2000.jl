using Tenkai
# Run with dev branch with title 'Works best for M2000'
Eq = Tenkai.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = -0.5, 0.5

boundary_condition = (dirichlet, neumann, neumann, neumann)
γ = 5.0 / 3.0
equation = Eq.get_equation(γ)

function initial_value_astro_jet(eq, x, y)
    γ = eq.γ
    ρ = 0.5
    v1 = 0.0
    v2 = 0.0
    p = 0.4127
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

initial_value = (x, y) -> initial_value_astro_jet(equation, x, y)
exact_solution = (x, y, t) -> initial_value(x, y)

function boundary_value_astro_jet(eq, x, y)
    γ = eq.γ
    if y >= -0.05 && y <= 0.05
        ρ = 5.0
        v1 = 800.0
        v2 = 0.0
        p = 0.4127
    else
        ρ = 0.5
        v1 = 0.0
        v2 = 0.0
        p = 0.4127
    end
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

boundary_value = (x, y, t) -> boundary_value_astro_jet(equation, x, y)

degree = 3
solver = cRK44()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes"
bflux = evaluate
final_time = 0.001

nx = 400
ny = nx
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time / 100.0
compute_error_interval = 0

cfl_safety_factor = 0.5

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
MH = mh_blend(equation)
FO = fo_blend(equation)
problem = Problem(domain,
                  initial_value,
                  boundary_value, boundary_condition,
                  final_time, exact_solution)
blend = setup_limiter_blend(blend_type = MH,
                            indicating_variables = Eq.rho_p_indicator!,
                            reconstruction_variables = conservative_reconstruction,
                            indicator_model = "gassner",
                            bc_x = Eq.hllc_upwinding_normal_x)
limiter = blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = joinpath(@__DIR__, "output"))
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param)

return sol
