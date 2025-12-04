using StaticArrays
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
# Submodules
Eq = TenkaicRK.EqShearShallowWater1D
EqTenMom = Tenkai.EqTenMoment1D

xmin, xmax = -1.0, 1.0
boundary_condition = (neumann, neumann)

dummy_bv(x, t) = 0.0

function initial_value_rp(x)
    EqTenMom.rp(x,
                (0.02, 0.0, 0.0, 1e-4, 0.0, 1e-4),
                (0.01, 0.0, 0.0, 1e-4, 0.0, 1e-4),
                0.5)
end

initial_value = EqTenMom.sod_iv

boundary_value = dummy_bv
exact_solution_rp(x, t) = initial_value_rp(x)
exact_solution = exact_solution_rp

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.125 # Choose 0.125 for sod, two_shock; 0.15 for two_rare_iv; 0.05 for two_rare_vacuum_iv

nx = ceil(Int64, 1000)
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 10.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = nx
gravity = 0.0
eq = Eq.get_equation(gravity)
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
# limiter = setup_limiter_none()
limiter = setup_limiter_blend(blend_type = mh_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner"
                              # pure_fv = true
                              )
limiter = setup_limiter_tvb(eq; tvbM = tvbM, beta = 1.4)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = 0.98)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;

sol["plot_data"].p_ua
