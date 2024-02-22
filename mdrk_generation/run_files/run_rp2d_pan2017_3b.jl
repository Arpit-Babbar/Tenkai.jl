# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
Eq = Tenkai.EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = -0.5, 1.5
ymin, ymax = -0.5, 1.5

boundary_value = (x, t) -> 0.0 # dummy function
boundary_condition = (neumann, neumann, neumann, neumann)
γ = 1.4
equation = Eq.get_equation(γ)
function riemann_problem_pan_1(x, y)
    Eq.riemann_problem(x, y, equation,
                       (1.0,  0.0312, 0.0312, 0.5),
                       (0.927,  -0.0312,  0.0312, 0.45),
                       (1.0, -0.0312,  -0.0312, 0.5),
                       (0.927, 0.0312, -0.0312, 0.45))
end

riemann_problem(x, y) = riemann_problem_pan_1(x, y)
rieman_problem_(x, y, t) = riemann_problem(x, y)
initial_value = riemann_problem

exact_solution = rieman_problem_
degree = 3
solver = "mdrk"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.35

nx, ny = 250, 250 # 50, 50
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = final_time / 30.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "mdrk_results/output_hurricane_rp2d_3b")

#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol["errors"]
