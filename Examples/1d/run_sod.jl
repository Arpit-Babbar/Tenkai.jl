# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using Tenkai

# Submodules
Eq = Tenkai.EqEuler1D
# Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

boundary_condition = (neumann, neumann)
γ = 1.4

final_time = 0.2

function modified_sod(x)
    γ = 1.4
    if x < 0.3
        prim = (1, 0, 1, 0.0)
    else
        prim = (0.125, 0.0, 0.1, 0.0)
    end

    U = SVector(prim[1], prim[1] * prim[2],
                prim[3] / (γ - 1.0) + 0.5 * prim[1] * prim[2]^2)
    return U
end

initial_value = modified_sod

dummy_exact(x, t) = modified_sod(x)

exact_solution = dummy_exact

boundary_value = exact_solution # dummy function

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 1600
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 10.0
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
limiter = setup_limiter_blend(blend_type = Tenkai.fo_blend(equation),
                              indicating_variables = Eq.primitive_indicator!,
                              reconstruction_variables = Tenkai.conservative_reconstruction,
                              amax = 1.0,
                              indicator_model = "model1",
                              debug_blend = false)
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
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

sol["plot_data"].p_u1
sol["plot_data"].p_ua
