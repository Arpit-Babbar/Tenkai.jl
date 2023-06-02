using SSFR
Eq = SSFR.EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 4.0
ymin, ymax = 0.0, 1.0

bottom = Eq.double_mach_bottom
boundary_condition = (dirichlet, neumann, bottom, dirichlet)
γ = 1.4

boundary_value = Eq.double_mach_reflection_bv

initial_value  = Eq.double_mach_reflection_iv

exact_solution = boundary_value

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.2

ny = 150
nx = 4*ny
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 100.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
MH = mh_blend(equation)
FO = fo_blend(equation)
blend = setup_limiter_blend(
                              blend_type = MH,
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false
                             )
limiter = blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux, 2)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);

return sol;