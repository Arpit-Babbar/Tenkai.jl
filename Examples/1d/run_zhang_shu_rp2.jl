# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using SSFR
using Plots
# Submodules
Eq = SSFR.EqEuler1D
plotlyjs() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = -10.0, 10.0

zero_function(x,t) = 0.0
boundary_value = zero_function # dummy function
boundary_condition = (neumann, neumann)
γ = 1.4
# function riemann_problem(ul, ur, xs, x)
#    if x < xs
#       prim = ul
#    else
#       prim = ur
#    end
#    U = [prim[1], prim[1]*prim[2], prim[3]/(γ-1.0) + 0.5*prim[1]*prim[2]^2]
#    return U
# end

# initial_value =  x-> riemann_problem((2.0  , 0.0, 10.0^9),
#                                      (0.001, 0.0, 1.0),
#                                      0.0, x )
# exact_solution = (x,t) -> initial_value(x)
initial_value, exact_solution, final_time, ic_name = Eq.leblanc_data
degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 6400
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1


cfl_safety_factor = 0.98
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
equation = Eq.get_equation(γ)
limiter = setup_limiter_blend(
                              blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = true,
                              constant_node_factor = 0.1,
                              pure_fv = true
                             )
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(
                     grid_size, cfl, bounds, save_iter_interval,
                     save_time_interval, compute_error_interval;
                     animate = animate,
                     cfl_safety_factor = cfl_safety_factor,
                     time_scheme = "SSPRK54"
                  )
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);
plot_data = sol["plot_data"]

p_ua = plot_data.p_ua
plot!(p_ua, yscale = :log)
savefig(p_ua, "test.html")

return sol
