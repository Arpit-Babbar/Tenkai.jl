using StaticArrays
using Tenkai

Eq = Tenkai.EqMHD1D

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_value = Tenkai.EqEuler1D.dummy_zero_boundary_value # dummy function
boundary_condition = (periodic, periodic)
gamma = 5.0 / 3.0
final_time = 2.0

equation = Eq.get_equation(gamma)

exact_solution_alfven_wave = Eq.ExactSolutionAlfvenWave(equation)
initial_value_alfven_wave(x) = exact_solution_alfven_wave(x, 0.0)

degree = 3
solver = TrixiRKSolver(nothing) # Use TrixiRKSolver for MHD
solver = "rkfr"
solution_points = "gll"
correction_function = "g2"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 32
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in MHD
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
problem = Problem(domain, initial_value_alfven_wave, boundary_value,
                  boundary_condition, final_time, exact_solution_alfven_wave)
limiter = setup_limiter_none() # No limiter for this example
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK54")
#------------------------------------------------------------------------------

# cache for storing solution and other arrays
grid = Tenkai.make_cartesian_grid(problem, param.grid_size)
# fr operators like differentiation matrix, correction functions
op = Tenkai.fr_operators(scheme.degree, scheme.solution_points,
                    scheme.correction_function)
trixi_semi = Tenkai.tenkai2trixiode(equation, problem, scheme, param)
cache = (Tenkai.setup_arrays(grid, scheme, equation)..., ode = trixi_semi)
# auxiliary objects like plot data, blending limiter, etc.
aux = Tenkai.create_auxiliaries(equation, op, grid, problem, scheme, param,
                        cache)

sol = Tenkai.solve(equation, problem, scheme, param; grid, op, cache, aux);

println(sol["errors"])

return sol;
