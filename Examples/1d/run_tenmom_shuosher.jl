using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqTenMoment1D
# Set backend

xmin, xmax = -5.0, 5.0
boundary_condition = (neumann, neumann)

dummy_bv(x, t) = 0.0

function initial_condition_shu_osher(x, equations::Eq.TenMoment1D)
    if x <= -4.0
        rho, v1, v2, P11, P12, P22 = 3.857143, 2.699369, 0.0, 10.33333, 0.0, 10.33333
    else
        rho, v1, v2, P11, P12, P22 = 1.0 + 0.2 * sin(5x), 0.0, 0.0, 1.0, 0.0, 1.0
    end
    return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

eq = Eq.get_equation()
initial_value = x -> initial_condition_shu_osher(x, eq) # choices - Eq.sod_iv, Eq.two_shock_iv, Eq.two_rare_iv, Eq.two_rare_vacuum_iv
boundary_value = dummy_bv
exact_solution = (x, t) -> initial_value(x)

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 1.8 # Choose 0.125 for sod, two_shock; 0.15 for two_rare_iv; 0.05 for two_rare_vacuum_iv

nx = ceil(Int64, 100)
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = fo_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner"
                              # pure_fv = true
                              )
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = 0.6)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;

sol["plot_data"].p_ua
