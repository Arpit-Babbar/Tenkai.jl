using StaticArrays
using Tenkai
Eq = Tenkai.EqBurg2D

#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
burg_smooth_ic = (x, y) -> SVector(sinpi(2.0 * x) + sinpi(2.0 * y))
burg_smooth_exact = (x, y, t) -> SVector(sinpi(2.0 * (x - t)) + sinpi(2.0 * (y - t)))
boundary_value = burg_smooth_exact # dummy function

function burg_smooth_source(u, x_, t, eq)
    x, y = x_
    xt, yt = x - t, y - t
    u_ex = sinpi(2.0 * xt) + sinpi(2.0 * yt)
    ut = -2.0 * pi * (cospi(2.0*xt) + cospi(2.0*yt))
    ux = 2.0 * pi * cospi(2.0*xt)
    uy = 2.0 * pi * cospi(2.0*yt)

    return SVector(ut + u_ex*(ux + uy))
end
source_terms = burg_smooth_source
boundary_condition = (periodic, periodic, periodic, periodic)
final_time = 0.1

degree = 1
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bound_limit = "no"
bflux = evaluate
numerical_flux = Eq.rusanov

nx, ny = 50, 50
bounds = ([-Inf], [Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, burg_smooth_ic, boundary_value, boundary_condition,
                  final_time, burg_smooth_exact, source_terms = burg_smooth_source)
equation = Eq.get_equation()
limiter = setup_limiter_blend(blend_type = fo_blend(equation),
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              constant_node_factor = 1.0,
                              debug_blend = false,
                              pure_fv = true)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, time_scheme = "RK4")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

print(sol["errors"])

return sol;
