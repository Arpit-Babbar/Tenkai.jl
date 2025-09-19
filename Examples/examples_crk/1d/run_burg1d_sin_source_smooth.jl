using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
Eq = Tenkai.EqBurg1D
using Tenkai.TenkaicRK.StaticArrays

#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
burg_smooth_ic = x -> sinpi(2.0 * x)
burg_smooth_exact = (x, t) -> sinpi(2.0 * (x - t))
burg_smooth_source_smooth = (u, x, t, eq) -> SVector(pi * (sin(4.0 * pi * (x - t)) -
                                                      2.0 * cos(2.0 * pi * (x - t))))

boundary_value = Eq.zero_boundary_value # dummy function
boundary_condition = (periodic, periodic)
final_time = 1.0

degree = 1
solver = cHT112()
solution_points = "gl"
correction_function = "radau"
bflux = evaluate
numerical_flux = Eq.rusanov
bound_limit = "no"

nx = 800
cfl = 0.0
bounds = ([-0.2], [0.2])
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.1 * final_time
compute_error_interval = 0
cfl_safety_factor = 0.5
animate = true
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, burg_smooth_ic, boundary_value, boundary_condition,
                  final_time, burg_smooth_exact, source_terms = burg_smooth_source_smooth)
equation = Eq.get_equation()
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol

sol["plot_data"].p_u1
