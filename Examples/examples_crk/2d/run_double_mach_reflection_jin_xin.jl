using Tenkai
using Tenkai.TenkaicRK
Eq = Tenkai.EqEuler2D
EqJinXin = Tenkai.TenkaicRK.EqJinXin2D
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 4.0
ymin, ymax = 0.0, 1.0

bottom = Eq.double_mach_bottom
boundary_condition = (dirichlet, neumann, bottom, dirichlet)
γ = 1.4

equation_euler = Eq.get_equation(γ)

epsilon_relaxation = 1e-6

ny = 20
nx = 4 * ny

equation_jin_xin = EqJinXin.get_equation(equation_euler, epsilon_relaxation, nx, ny,
                                         thresholds = (1e-14, 1e-2),
                                         jin_xin_dt_scaling = 0.9)

initial_value = EqJinXin.JinXinICBC(Eq.double_mach_reflection_iv, equation_jin_xin)
exact_solution = EqJinXin.JinXinICBC(Eq.double_mach_reflection_bv, equation_jin_xin)
boundary_value = exact_solution

degree = 3
solver = cBPR343()
solution_points = "gl"
correction_function = "radau"
numerical_flux = EqJinXin.rusanov
bound_limit = "yes"
bflux = extrapolate
final_time = 0.2

cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 10.0
save_iter_interval = 30
save_time_interval = 0.0 # final_time / 40.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution,
                  source_terms = EqJinXin.jin_xin_source)
limiter = setup_limiter_blend(blend_type = fo_blend(equation_jin_xin),
                              indicating_variables = EqJinXin.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              constant_node_factor = 1.0,
                              debug_blend = false,
                              pure_fv = false)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux, 2)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation_jin_xin, problem, scheme, param);

return sol;
