# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqEuler1D
# plotlyjs() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (periodic, periodic)
γ = 1.4

initial_value, exact_solution, final_time, ic_name = Eq.dwave_data
boundary_value = exact_solution # dummy function

degree = 5
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 1.0

nx = ceil(Int64, 1600 / (degree + 1))
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = mh_blend(equation),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "model1",
                              debug_blend = false,
                              pure_fv = true)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
