using Tenkai.StaticArrays
using Tenkai.SimpleUnPack
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
Eq = Tenkai.TenkaicRK.EqVarAdv1D
# Submodules

# Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.1, 1.0

boundary_condition = (dirichlet, neumann)
final_time = 1.0

adv = x -> 1.0 / x^2

struct ExactSolutionMyVar{InitialCondition}
    initial_condition::InitialCondition
end

function (exact_solution_myvar::ExactSolutionMyVar)(x, t)
    @unpack initial_condition = exact_solution_myvar
    return SVector(initial_condition(x^3 - 3.0 * t)[1], x)
end

function initial_condition_var_adv(x)
    return SVector(sinpi(x), x)
end

exact_solution = ExactSolutionMyVar(initial_condition_var_adv)

boundary_value = exact_solution

degree = 3
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate

nx = 40
bounds = ([-Inf], [Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.1
compute_error_interval = 1

diss = "2"
cfl_safety_factor = 0.5

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_condition_var_adv, boundary_value,
                  boundary_condition, final_time, exact_solution)
equation = Eq.get_equation(adv)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux,
                diss)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none", time_scheme = "SSPRK54")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

show(sol["errors"])

return sol;
