using Tenkai
using Tenkai.TenkaicRK
# using Tenkai.TenkaicRK.Tenkai
EqBurg1D = Tenkai.EqBurg1D
Eq = TenkaicRK.EqJinXin1D
using Tenkai.TenkaicRK.StaticArrays
using Tenkai.EqBurg1D: find_zero

#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0

epsilon_relaxation = 1e-4

equation_burg = EqBurg1D.get_equation()
A = () -> 3.0

advection_jin_xin = (x, u, eq) -> A()^2 * u

advection_jin_xin_plus(ul, ur, F, eq) = 0.5 * (F[1] + A() * F[2]) * SVector(A(), 1.0)
advection_jin_xin_minus(ul, ur, F, eq) = 0.5 * (F[1] - A() * F[2]) * SVector(-A(), 1.0)

nx = 20

equation_jin_xin = Eq.get_equation(equation_burg, advection_jin_xin, advection_jin_xin_plus,
                                   advection_jin_xin_minus, epsilon_relaxation, nx;
                                   thresholds = (1.5e-12, 2e-3))

# initial_value_burg = EqBurg1D.initial_value_burger_sin

function initial_value_burg_marco(x)
    k = 1.0
    u = 2.0 + sinpi(k * (x[1] - 0.7))
    return u
end

initial_value_burg = initial_value_burg_marco

initial_value = Eq.JinXinICBC(initial_value_burg, equation_jin_xin)
initial_value_ = (x) -> initial_value(x)
boundary_value_burg = EqBurg1D.zero_boundary_value # dummy function
boundary_value = Eq.JinXinICBC(boundary_value_burg, equation_jin_xin)
boundary_value_ = (x, t) -> boundary_value(x, t)
boundary_condition = (periodic, periodic)
final_time = 0.5

exact_solution_burg = EqBurg1D.exact_solution_burger_sin

function exact_solution_burger_marco(x, t_)
    t = min(0.3, t_)
    implicit_eqn(u) = u - initial_value_burg_marco(x - t * u)
    seed = initial_value_burg_marco(x)
    value = find_zero(implicit_eqn, seed)
    return value
end

exact_solution = exact_solution_burger_marco
exact_solution_ = (x, t) -> exact_solution(x, t)

degree = 3
solver = cSSP2IMEX433()
solution_points = "gl"
correction_function = "radau"
bflux = evaluate
numerical_flux = Eq.rusanov
bound_limit = "no"

cfl = 0.0
bounds = ([-0.2], [0.2])
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time / 10.0
compute_error_interval = 0
animate = true
diss = "2"
cfl_safety_factor = 0.9
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value_, boundary_value_, boundary_condition,
                  final_time, exact_solution_,
                  source_terms = Eq.jin_xin_source)
# limiter = setup_limiter_none()
limiter = Tenkai.setup_limiter_blend(blend_type = Tenkai.fo_blend(equation_jin_xin),
                                     indicating_variables = Tenkai.conservative_indicator!,
                                     reconstruction_variables = Tenkai.conservative_reconstruction,
                                     indicator_model = "gassner",
                                     debug_blend = false)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux, diss)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor, saveto = "jinxin_nx$nx")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation_jin_xin, problem, scheme, param);

println(sol["errors"])

return sol

sol["plot_data"].p_u1
