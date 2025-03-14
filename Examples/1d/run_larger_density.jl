using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqEuler1D
 # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_value = Eq.dummy_zero_boundary_value # dummy function
boundary_condition = (neumann, neumann)
γ = 1.4
final_time = 0.15

function initial_value_high_density(x)
    γ = 1.4
    if x < 0.3
        rho, v, p = 1000.0, 0.0, 1000.0
    else
        rho, v, p = 1.0, 0.0, 1.0
    end
    return SVector(rho, rho * v, p / (γ - 1.0) + 0.5 * rho * v^2)
end
initial_value = initial_value_high_density
exact_solution = Eq.exact_solution_shuosher # Dummy function

degree = 3
solver = "mdrk"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 500
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
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
limiter = setup_limiter_blend(blend_type = mh_blend(equation),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = indicator_model,
                              constant_node_factor = 1.0,
                              amax = 1.0,
                              debug_blend = debug_blend,
                              pure_fv = pure_fv)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
# limiter = setup_limiter_hierarchical(alpha = 1.0,
#                                      reconstruction = characteristic_reconstruction)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK54")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
