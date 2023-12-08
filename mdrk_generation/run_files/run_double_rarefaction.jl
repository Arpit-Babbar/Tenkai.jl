using Tenkai
Eq = Tenkai.EqEuler1D
#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0

boundary_value = Eq.dummy_zero_boundary_value
boundary_condition = (neumann, neumann)
γ = 1.4

initial_value, exact_solution, final_time, ic_name = Eq.double_rarefaction_data

degree = 3
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes"
bflux = evaluate

nx = 200
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
diss = "2"
compute_error_interval = 1

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation(γ)
MH = mh_blend(equation)
FO = fo_blend(equation)
blend = setup_limiter_blend(blend_type = MH,
                            # indicating_variables = Eq.rho_p_indicator!,
                            indicating_variables = Eq.rho_p_indicator!,
                            reconstruction_variables = conservative_reconstruction,
                            indicator_model = "gassner",
                            numflux = Eq.rusanov)
tvb = setup_limiter_tvb(equation; tvbM = tvbM)
limiter = blend # To enable using trixi_include
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux, diss)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate, saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
