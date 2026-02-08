using Tenkai.TenkaicRK
using Tenkai
using Tenkai.StaticArrays
using Tenkai.TenkaicRK: newton_solver
Eq = TenkaicRK.EqSupBurg1D
# Submodules

#------------------------------------------------------------------------------
xmin, xmax = -5.0, 5.0

boundary_condition = (dirichlet, neumann)
final_time = 4.0

function initial_value_jump(x)
    if x >= 0.0
        return 0.0
    else
        return 1.0
    end
end

function exact_solution_burg1d(x, t)
    if x - 0.25 * t >= 0.0
        return 0.0
    else
        return 1.0
    end
end
initial_value, exact_solution = initial_value_jump, exact_solution_burg1d

boundary_value = exact_solution

degree = 3
solver = cRK44()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = extrapolate

nx = 10
bounds = ([-Inf], [Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
compute_error_interval = 1
animate_ = true
tvbM = 0.0

cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution)
equation = Eq.get_equation()
limiter = setup_limiter_blend(blend_type = fo_blend_imex(equation),
                              #   indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              amax = 1.0,
                              pure_fv = false,
                              smoothing_in_time = false)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval, animate = animate_,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

show(sol["errors"])

return sol;
p_ua = sol["plot_data"].p_ua
