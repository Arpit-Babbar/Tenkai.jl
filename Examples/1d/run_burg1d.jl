using Tenkai
Eq = Tenkai.EqBurg1D

#------------------------------------------------------------------------------
# xmin, xmax = 0.0, 2.0 * pi
xmin, xmax = -1.0, 1.0

function initial_value_burg_marco(x)
    k = 1.0
    u = 2.0 + sinpi(k * (x[1] - 0.7))
    return u
    # 0.2 * sin(x)
end

function exact_solution_burger_marco(x, t_)
    t = min(0.3, t_)
    implicit_eqn(u) = u - initial_value_burg_marco(x - t * u)
    seed = initial_value_burg_marco(x)
    value = find_zero(implicit_eqn, seed)
    return value
end

initial_value = Eq.initial_value_burger_sin
initial_value = initial_value_burg_marco
boundary_value = Eq.zero_boundary_value # dummy function
boundary_condition = (periodic, periodic)
final_time = 0.5

exact_solution = exact_solution_burger_sin_marco

source_terms(u, x, t, eq) = zero(u)

degree = 3
solver = cRK33()
solution_points = "gl"
correction_function = "radau"
bflux = evaluate
numerical_flux = Eq.rusanov
bound_limit = "no"

nx = 100
cfl = 0.0
bounds = ([-0.2], [0.2])
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time/10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation()
limiter = setup_limiter_none()
limiter = setup_limiter_blend(blend_type = fo_blend(equation),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false,
                              pure_fv = false)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, saveto = "subcell_nx$nx")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol
sol["plot_data"].p_u1
