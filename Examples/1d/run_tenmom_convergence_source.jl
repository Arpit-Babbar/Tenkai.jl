using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqTenMoment1D
# Set backend

xmin, xmax = -1.0, 1.0
boundary_condition = (periodic, periodic)

function initial_wave(x, equations::Eq.TenMoment1D)
    rho = 2.0 + sinpi(2.0 * x)
    v1 = 1.0
    v2 = 0.0
    P11 = 1.5 + 1.0 / 8.0 * (cospi(4.0 * x) - 8.0 * sinpi(2.0 * x))
    P12 = 0.0
    P22 = 1.0
    return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

dummy_bv(x, t) = 0.0

eq = Eq.get_equation()

initial_wave(x) = initial_wave(x, eq)

exact_wave(x, t) = initial_wave(x - t)

Wx(x, t) = 2.0 * pi * cospi(2.0 * (x - t))

source_terms = (u, x, t, equations) -> Eq.ten_moment_source_x(u, x, t, Wx, equations)

initial_value, exact_solution, boundary_value = initial_wave, exact_wave, dummy_bv

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.5

nx = 100
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = source_terms)
limiter = setup_limiter_blend(blend_type = fo_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              pure_fv = true)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
