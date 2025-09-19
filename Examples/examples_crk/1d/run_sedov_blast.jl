using Tenkai
Eq = Tenkai.EqEuler1D

#------------------------------------------------------------------------------
xmin, xmax, nx, dx, initial_value, exact_solution = Eq.sedov_data

boundary_condition = (reflect, reflect)
γ = 1.4
final_time = 0.001

# nx = 201
# dx = (xmax - xmin) / nx

boundary_value = exact_solution # dummy function

degree = 3
solver = cRK44()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

# nx defined above
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 10.0
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
MH = mh_blend(equation)
FO = fo_blend(equation)
limiter_blend = setup_limiter_blend(blend_type = MH,
                                    # indicating_variables = Eq.rho_p_indicator!,
                                    indicating_variables = Eq.rho_p_indicator!,
                                    reconstruction_variables = conservative_reconstruction,
                                    indicator_model = indicator_model,
                                    debug_blend = debug_blend,
                                    pure_fv = pure_fv,
                                    numflux = Eq.rusanov)
limiter_tvb = setup_limiter_tvb(equation; tvbM = tvbM)
limiter = limiter_blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK54", saveto = "none")
#------------------------------------------------------------------------------
problem, scheme, param, = ParseCommandLine(problem, param, scheme, equation,
                                           ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol
