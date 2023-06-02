using SSFR
Eq = SSFR.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = -10.0, 10.0
ymin, ymax = -10.0, 10.0

boundary_value     = Eq.zero_bv # dummy
boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

initial_value  = Eq.isentropic_iv
exact_solution = Eq.isentropic_exact

degree              = 4
solver              = "lwfr"
solution_points     = "gl"
correction_function = "radau"
numerical_flux      = Eq.rusanov
bound_limit         = "no"
bflux               = evaluate
final_time          = 20 * sqrt(2.0) / 0.5

nx, ny = 160, 160
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time
compute_error_interval = 0
cfl_safety_factor = 0.95

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)

blend = setup_limiter_blend(
                              blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner"
                             )
no_limiter = setup_limiter_none()
limiter = no_limiter
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
ARGS = ["--degree", "4", "--solver", "lwfr", "--solution_points", "gl", "--correction_function",
        "radau", "--bflux", "evaluate", "--cfl_safety_factor", "0.95", "--bound_limit", "yes",
        "--cfl_style", "optimal", "--dissipation", "2",
        "--grid_size", "40", "40",
        "--final_time", "56.568542494923804",
 "--save_time_interval", "56.568542494923804", "--save_iter_interval", "0", "--animate", "true"]

problem2, scheme2, param2 = ParseCommandLine(problem, param, scheme, equation, ARGS)
sol = SSFR.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
