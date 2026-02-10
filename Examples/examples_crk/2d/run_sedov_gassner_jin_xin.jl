using Tenkai
using Tenkai.TenkaicRK
Eq = Tenkai.EqEuler2D
using StaticArrays
EqJinXin = Tenkai.TenkaicRK.EqJinXin2D
#------------------------------------------------------------------------------
xmin, xmax = -1.5, 1.5
ymin, ymax = -1.5, 1.5

boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4
equation_euler = Eq.get_equation(γ)

nx = 64
ny = nx

epsilon_relaxation = 1e-12

equation_jin_xin = EqJinXin.get_equation(equation_euler, epsilon_relaxation, nx, ny,
                                         thresholds = (1e-14, 0.8e-3),
                                         jin_xin_dt_scaling = 0.8)

initial_value_sedov, exact_solution_sedov = Eq.sedov_data

initial_value = EqJinXin.JinXinICBC(initial_value_sedov, equation_jin_xin)
exact_solution = EqJinXin.JinXinICBC(exact_solution_sedov, equation_jin_xin)
boundary_value = exact_solution

degree = 3
solver = cBPR343()
solution_points = "gl"
correction_function = "radau"
numerical_flux = EqJinXin.rusanov

bound_limit = "yes"
bflux = extrapolate
final_time = 20.0

cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.1 * final_time
animate_time_factor = 1 # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = EqJinXin.jin_xin_source)
blend = setup_limiter_blend(blend_type = fo_blend(equation_jin_xin),
                            amax = 0.5,
                            indicating_variables = EqJinXin.rho_p_indicator!,
                            reconstruction_variables = conservative_reconstruction,
                            indicator_model = "gassner",
                            debug_blend = false,
                            pure_fv = false)
limiter = blend
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "output_jin_xin_nx$nx")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation_jin_xin, problem, scheme, param);

return sol
