using Tenkai
using Tenkai.TenkaicRK
Eq = Tenkai.EqEuler2D
using StaticArrays
EqJinXin = Tenkai.TenkaicRK.EqJinXin2D
#------------------------------------------------------------------------------
xmin, xmax = -10.0, 10.0
ymin, ymax = -10.0, 10.0

boundary_value = Eq.zero_bv # dummy
boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

equation_euler = Eq.get_equation(γ)


epsilon_relaxation = 1e-12

nx = 120
ny = 120

equation_jin_xin = EqJinXin.get_equation(equation_euler, epsilon_relaxation, nx, ny)

initial_value = Eq.isentropic_iv
exact_solution = Eq.isentropic_exact

initial_value = EqJinXin.JinXinICBC(Eq.isentropic_iv, equation_jin_xin)
exact_solution = EqJinXin.JinXinICBC(Eq.isentropic_exact, equation_jin_xin)
boundary_value = exact_solution

degree = 3
solver = cBPR343()
solution_points = "gl"
correction_function = "radau"
numerical_flux = EqJinXin.rusanov
bound_limit = "no"
bflux = extrapolate
final_time = 1.0 # 20 * sqrt(2.0) / 0.5

cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 5.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = EqJinXin.jin_xin_source)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux,
                2)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   time_scheme = "by degree")
#------------------------------------------------------------------------------
# problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
#                                           ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation_jin_xin, problem, scheme, param);

println(sol["errors"])

return sol;
