# For Riemann problems in domain [0.0,1.0]
using Tenkai
using StaticArrays
using Trixi
Eq = Tenkai.EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 40 / 3
ymin, ymax = 0.0, 40

boundary_condition = (reflect, reflect, reflect, reflect)
γ = 1.4
equation = Eq.get_equation(γ)
function initial_condition_richtmeyer_meshkov(x, y)
    gamma = 1.4
    slope = 2.0

    L = 40.0

    # smooth the discontinuity to avoid ambiguity at element interfaces
    smoothed_heaviside(x, left, right; slope = slope) = left +
                                                        0.5 * (1 + tanh(slope * x)) *
                                                        (right - left)

    L = 40 # domain size
    rho = smoothed_heaviside(y - (18 + 2 * cos(2 * pi * 3 / L * x)), 1.0, 0.25)
    rho = rho + smoothed_heaviside(abs(y - 4) - 2, 3.22, 0.0) # 2 < x < 6
    p = smoothed_heaviside(abs(y - 4) - 2, 4.9, 1.0)
    v1 = 0.0
    v2 = 0.0
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    gamma_minus_1 = gamma - 1.0
    rho_e = p / gamma_minus_1 + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

# Used to set the top and bottom boundary conditions
function boundary_condition_richtmeyer_meshkov(x, y, t)
    return SVector(0.0, 0.0, 0.0, 0.0) # Won't be used
end

initial_value = initial_condition_richtmeyer_meshkov

exact_solution = boundary_condition_richtmeyer_meshkov
degree = 3

volume_integral = Trixi.VolumeIntegralFluxDifferencing((Trixi.flux_ranocha,
                                                        nothing))

solver = cRK44(volume_integral)
solution_points = "gll"
correction_function = "g2"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 30.0

nx = 32
ny = 3 * nx
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = final_time / 100.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.1

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_condition_richtmeyer_meshkov,
                  boundary_condition, final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = fo_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              amax = 0.0002, # Crashes with amax = 0.0001
                              debug_blend = false)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "mdrk_results/output_richtmeyer_meshkov")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

return sol["errors"]
