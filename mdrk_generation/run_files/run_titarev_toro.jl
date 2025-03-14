using StaticArrays
using Tenkai
using Plots
# Submodules
Eq = Tenkai.EqEuler1D
gr() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = -5.0, 5.0

boundary_value = Eq.dummy_zero_boundary_value # dummy function
boundary_condition = (neumann, neumann)
γ = 1.4
final_time = 5.0

initial_value = Eq.initial_value_titarev_toro
exact_solution = Eq.dummy_zero_boundary_value # Dummy function

degree = 3
solver = "mdrk"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 800
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

# blend parameters
indicator_model = "gassner"
debug_blend = false
cfl_safety_factor = 0.95
pure_fv = false

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution)
equation = Eq.get_equation(γ)
FO = fo_blend(equation)
MH = mh_blend(equation)
limiter = setup_limiter_blend(blend_type = MH,
                              indicating_variables = Eq.rho_p_indicator!,
                              #   indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = indicator_model,
                              constant_node_factor = 1.0,
                              amax = 1.0,
                              debug_blend = debug_blend,
                              pure_fv = pure_fv)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
# limiter = setup_limiter_hierarchical(alpha = 1.0,
#                                      reconstruction = characteristic_reconstruction)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK54",
                   saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
