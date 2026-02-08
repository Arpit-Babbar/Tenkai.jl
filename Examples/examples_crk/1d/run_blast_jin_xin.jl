using StaticArrays
using Tenkai
using Tenkai.TenkaicRK
# Submodules
Eq = Tenkai.EqEuler1D
EqJinXin = TenkaicRK.EqJinXin1D

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (reflect, reflect)
γ = 1.4
final_time = 0.038

equation_euler = Eq.get_equation(γ)

A = () -> 100.0

advection_jin_xin = (x, u, eq) -> A()^2 * u
nx = 400

epsilon_relaxation = 0.4e-7

advection_jin_xin_plus(ul, ur, F, eq) = nothing
advection_jin_xin_minus(ul, ur, F, eq) = nothing

equation = EqJinXin.get_equation(equation_euler, advection_jin_xin, advection_jin_xin_plus,
                                 advection_jin_xin_minus, epsilon_relaxation
                                 #   nx ; thresholds = (1.5e-1, 1e-1)
                                 )

initial_value = EqJinXin.JinXinICBC(Eq.blast, equation)
initial_value_ = initial_value
boundary_value = EqJinXin.JinXinICBC(Eq.exact_blast, equation)
boundary_value_ = (x, t) -> boundary_value(x, t)
exact_solution = boundary_value

degree = 3
solver = cSSP2IMEX433()
solution_points = "gl"
correction_function = "radau"
numerical_flux = EqJinXin.rusanov
bound_limit = "yes"
bflux = evaluate

cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

# blend parameters
indicator_model = "gassner"
debug_blend = false
cfl_safety_factor = 0.9
pure_fv = false
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution,
                  source_terms = EqJinXin.jin_xin_source)
# MH = mh_blend(equation)
# FO = fo_blend(equation)
# limiter_blend = setup_limiter_blend(blend_type = MH,
#                                     # indicating_variables = Eq.rho_p_indicator!,
#                                     indicating_variables = Eq.rho_p_indicator!,
#                                     reconstruction_variables = conservative_reconstruction,
#                                     indicator_model = indicator_model,
#                                     debug_blend = debug_blend,
#                                     pure_fv = pure_fv,
#                                     numflux = Eq.rusanov)
# limiter_tvb = setup_limiter_tvb(equation; tvbM = tvbM)
# limiter = limiter_blend
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate, cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK33", saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

print(sol["errors"])

return sol

sol["plot_data"].p_ua
