using Tenkai

Eq = Tenkai.EqEuler1D
#------------------------------------------------------------------------------
xmin, xmax, nx, dx, initial_value, exact_solution = Eq.sedov_data

boundary_value = Eq.dummy_zero_boundary_value
boundary_condition = (neumann, neumann)
γ = 1.4

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
final_time = 0.001
bound_limit = "yes"
bflux = evaluate

nx = 201
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

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
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval; saveto = "nothing")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
