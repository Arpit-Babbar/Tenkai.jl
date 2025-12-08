using Tenkai
using Tenkai.TenkaicRK
EqEuler2D = Tenkai.EqEuler2D
Eq = EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = -10.0, 10.0
ymin, ymax = -10.0, 10.0

boundary_value = EqEuler2D.zero_bv # dummy
boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

initial_value_ = EqEuler2D.isentropic_iv
initial_value = (x, y) -> SVector(initial_value_(x, y)..., 0.0)
exact_solution_ = EqEuler2D.isentropic_exact
exact_solution = (x, y, t) -> SVector(exact_solution_(x, y, t)..., 0.0)

degree = 2
solver = cRK33()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 1.0 # 20 * sqrt(2.0) / 0.5

nx = 100
ny = nx
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 5.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
diss = "2"
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
limiter = setup_limiter_blend(blend_type = fo_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false,
                              pure_fv = false,
                              tvbM = Inf)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux,
                diss)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
