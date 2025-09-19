using StaticArrays
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
Eq = Tenkai.EqBurg1D
# Submodules

# Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (dirichlet, dirichlet)
final_time = 10.0

function initial_value_jump(x)
    if x <= 0.1
        return 1.0
    else
        return -0.1
    end
end

function exact_solution_(x, t)
    if x <= 0.2
        return 1.0
    else
        return -0.1
    end
end

initial_value, exact_solution = initial_value_jump, exact_solution_

source_terms_mu(u, x, t, mu, eq) = SVector(mu * (6.0 * x - 3.0) * u[1])

source_terms_burg_stiff(u, x, t, eq) = source_terms_mu(u, x, t, 8.0, eq)

boundary_value = exact_solution

degree = 1
solver = cHT112()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate

nx = 1000
bounds = ([-Inf], [Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.1 * final_time
compute_error_interval = 1
animate_ = true
tvbM = 0.0

cfl_safety_factor = 0.98
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution,
                  source_terms = source_terms_burg_stiff)
equation = Eq.get_equation()
limiter = setup_limiter_none()
limiter = setup_limiter_tvb(equation; tvbM = tvbM)
# limiter = setup_limiter_blend(
#                               blend_type = fo_blend(equation),
#                               # indicating_variables = Eq.rho_p_indicator!,
#                               indicating_variables = conservative_indicator!,
#                               reconstruction_variables = conservative_reconstruction,
#                               indicator_model = "gassner",
#                               debug_blend = false,
#                               pure_fv = false
#                              )
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval, animate = animate_,
                   cfl_safety_factor = cfl_safety_factor, time_scheme = "Tsit5")
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

show(sol["errors"])

return sol;
p_ua = sol["plot_data"].p_ua
xlims!(p_ua, (0.0, 1.0))
ylims!(p_ua, (-6.5, 1.5))
