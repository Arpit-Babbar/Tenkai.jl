# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using Tenkai
using Plots
# Submodules
Eq = Tenkai.EqEuler1D
plotlyjs() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (neumann, neumann)
γ = 1.4

# Get equation data from a file
initial_value_ref, exact_solution, final_time, ic_name = Eq.sod_data

boundary_value = exact_solution

degree = 3
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes" # positivity limiter
bflux = evaluate

nx = 50
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

# Keyword arguments
blend_type = "fo"
indicator_model = "model1"
positivity_fix = false
predictive_blend = false # Does nothing
pure_fv = false # Do a pure FO/MUSCL update
indicating_variables = "primitive"
reconstruction_variables = "primitive"
debug_blend = true
saveto = "./temp"
time_scheme = "ssprk54"
beta_muscl = 1.0

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value_ref, boundary_value, boundary_condition,
                     final_time, exact_solution)
equation = Eq.get_equation(γ)
# limiter = setup_limiter_blend(
#                                  blend_type = FR.fo_blend,
#                                  indicating_variables = FR.conservative_indicator!,
#                                  reconstruction_variables = FR.conservative_reconstruction,
#                                  indicator_model = "gassner",
#                                  debug_blend = false
#                                 )
limiter = setup_limiter_tvb(tvbM = tvbM, eq= equation)
scheme = Scheme(solver, degree, solution_points, correction_function,
                   numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                      save_time_interval, compute_error_interval;
                      animate = animate, saveto = saveto)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return errors, plot_data
