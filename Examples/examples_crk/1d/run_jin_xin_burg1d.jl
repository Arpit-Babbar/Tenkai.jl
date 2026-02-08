using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
EqBurg1D = Tenkai.EqBurg1D
Eq = TenkaicRK.EqJinXin1D
using Tenkai.TenkaicRK.StaticArrays

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 2.0 * pi

equation_burg = EqBurg1D.get_equation()
A = () -> 0.2

advection_jin_xin = (x, u, eq) -> A()^2 * u

advection_jin_xin_plus(ul, ur, F, eq) = 0.5 * (F[1] + A() * F[2]) * SVector(A(), 1.0)
advection_jin_xin_minus(ul, ur, F, eq) = 0.5 * (F[1] - A() * F[2]) * SVector(-A(), 1.0)

equation_jin_xin = Eq.get_equation(equation_burg, advection_jin_xin, advection_jin_xin_plus,
                                   advection_jin_xin_minus, 0.001)

initial_value_burg = EqBurg1D.initial_value_burger_sin
initial_value = Eq.JinXinICBC(initial_value_burg, equation_jin_xin)
initial_value_ = initial_value
boundary_value_burg = EqBurg1D.zero_boundary_value # dummy function
boundary_value = Eq.JinXinICBC(boundary_value_burg, equation_jin_xin)
boundary_value_ = (x, t) -> boundary_value(x, t)
boundary_condition = (periodic, periodic)
final_time = 4.9

exact_solution_burg = EqBurg1D.exact_solution_burger_sin
exact_solution = Eq.JinXinICBC(exact_solution_burg, equation_jin_xin)
exact_solution_jin_xin = exact_solution

degree = 2
solver = cSSP2IMEX222()
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
save_time_interval = final_time / 10.0
compute_error_interval = 0
animate = true
diss = "2"
cfl_safety_factor = 0.9
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value_, boundary_value_, boundary_condition,
                  final_time, exact_solution_jin_xin,
                  source_terms = Eq.jin_xin_source)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux, diss)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation_jin_xin, problem, scheme, param);

println(sol["errors"])

return sol

sol["plot_data"].p_u1
