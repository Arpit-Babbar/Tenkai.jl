# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using Tenkai
using Tenkai.TenkaicRK

# Submodules
Eq = Tenkai.EqEuler1D
EqJinXin = Tenkai.TenkaicRK.EqJinXin1D
#  # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (periodic, periodic)
γ = 1.4

equation_euler = Eq.get_equation(γ)

A = () -> 4.0

advection_jin_xin = (x, u, equation_euler) -> A()^2 * u

function advection_jin_xin_plus(ul, ur, F, equation_euler)
    nothing
end
function advection_jin_xin_minus(ul, ur, F, equation_euler)
    nothing
end

epsilon_relaxation = 1e-10

equation_jin_xin = EqJinXin.get_equation(equation_euler, advection_jin_xin,
                                         advection_jin_xin_plus,
                                         advection_jin_xin_minus, epsilon_relaxation)

initial_value_dwave, exact_solution_dwave, final_time, ic_name = Eq.dwave_data
boundary_value_dwave = exact_solution_dwave # dummy function

initial_value_struct = EqJinXin.JinXinICBC(initial_value_dwave, equation_jin_xin)
# initial_value = (x) -> initial_value_struct(x)
initial_value = initial_value_struct
# initial_value = initial_value_dwave
boundary_value_struct = EqJinXin.JinXinICBC(exact_solution_dwave, equation_jin_xin)
boundary_value = (x, t) -> boundary_value_struct(x, t)
# boundary_value = boundary_value_dwave
exact_solution = boundary_value

degree = 2
solver = cSSP2IMEX433()
solution_points = "gl"
correction_function = "radau"
numerical_flux = EqJinXin.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 0.1

nx = 20
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = EqJinXin.jin_xin_source)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation_jin_xin, problem, scheme, param);

println(sol["errors"])

return sol;

sol["plot_data"].p_ua
