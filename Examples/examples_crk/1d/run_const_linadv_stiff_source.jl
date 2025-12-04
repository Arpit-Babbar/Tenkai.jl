using StaticArrays
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
Eq = Tenkai.EqLinAdv1D
# Submodules

# Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (dirichlet, dirichlet)
final_time = 0.3
velocity, initial_value, exact_solution = Eq.mult1d_data

function initial_value_jump(x)
    if x <= 0.3
        return 1.0
    else
        return 0.0
    end
end

exact_solution_jump(x, t) = initial_value_jump(x - velocity(x) * t)

initial_value, exact_solution = initial_value_jump, exact_solution_jump

function source_terms_linear_stiff(u, x, t, mu, eq)
    SVector(-mu * u[1] * (u[1] - 1.0) * (u[1] - 0.5))
end

source_terms_linear_stiff(u, x, t, eq) = source_terms_linear_stiff(u, x, t, 1000.0, eq)

boundary_value = exact_solution

degree = 3
# cSSP2IMEX222 works better without a limiter
solver = cSSP2IMEX433()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = extrapolate

nx = 70
bounds = ([-Inf], [Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # 0.1 * final_time
compute_error_interval = 1
animate_ = true
tvbM = 50.0

cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution,
                  source_terms = source_terms_linear_stiff)
equation = Eq.get_equation(velocity)
# limiter = setup_limiter_none()
limiter = setup_limiter_tvb(equation; tvbM = tvbM)
limiter_blend = setup_limiter_blend(blend_type = fo_blend_imex(equation),
                                    # indicating_variables = Eq.rho_p_indicator!,
                                    indicating_variables = conservative_indicator!,
                                    reconstruction_variables = conservative_reconstruction,
                                    indicator_model = "gassner",
                                    debug_blend = false,
                                    pure_fv = false)
limiter = limiter_blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval, animate = animate_,
                   cfl_safety_factor = cfl_safety_factor, time_scheme = "Tsit5")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

show(sol["errors"])

return sol;
p = sol["plot_data"].p_ua
