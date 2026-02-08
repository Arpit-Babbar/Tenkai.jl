import Roots.find_zero
using Tenkai
# Submodules

Eq = Tenkai.EqBuckleyLeverett1D
EqJinXin = Tenkai.TenkaicRK.EqJinXin1D
# Set backend

#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
initial_value = Eq.hatbuck_iv
boundary_condition = (periodic, periodic)
exact_solution = Eq.hatbuck_exact_a025
final_time = 0.15 # Larger epsilon is needed for time > 0.15

epsilon_relaxation = 1e-4

eq_bucklev = Eq.get_equation()
A = () -> 4.0

advection_jin_xin = (x, u, eq_bucklev) -> A()^2 * u

function advection_jin_xin_plus(ul, ur, F, eq_bucklev)
    0.5 * (F[1] + A() * F[2]) * SVector(A(), 1.0)
end
function advection_jin_xin_minus(ul, ur, F, eq_bucklev)
    0.5 * (F[1] - A() * F[2]) * SVector(-A(), 1.0)
end

equation_jin_xin = EqJinXin.get_equation(eq_bucklev, advection_jin_xin,
                                         advection_jin_xin_plus,
                                         advection_jin_xin_minus, epsilon_relaxation)

# Is it really a struct?
initial_value_struct = EqJinXin.JinXinICBC(Eq.hatbuck_iv, equation_jin_xin)
initial_value = (x) -> initial_value_struct(x)
boundary_value_struct = EqJinXin.JinXinICBC(Eq.hatbuck_exact, equation_jin_xin)
boundary_value = (x, t) -> boundary_value_struct(x, t)

degree = 3
solver = cSSP2IMEX433()
solution_points = "gl"
correction_function = "radau"
bflux = evaluate
numerical_flux = EqJinXin.rusanov
bound_limit = "no"

nx = 200
cfl = 0.0
bounds = ([0.0], [1.0])
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time * 0.1
compute_error_interval = 0
cfl_safety_factor = 0.98
animate = true
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = EqJinXin.jin_xin_source)
# limiter = setup_limiter_tvb(eq_bucklev; tvbM = tvbM)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   cfl_safety_factor = cfl_safety_factor,
                   animate = animate)
#------------------------------------------------------------------------------
problem, scheme, param, = ParseCommandLine(problem, param, scheme, eq_bucklev, ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation_jin_xin, problem, scheme, param)

println(sol["errors"])

sol["plot_data"].p_ua
