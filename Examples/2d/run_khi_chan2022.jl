using Tenkai
Eq = Tenkai.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0

boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

volume_integral = Trixi.VolumeIntegralFluxDifferencing((Trixi.flux_ranocha,
                                                        nothing))

initial_value, exact_solution = Eq.kevin_helmholtz_schaal_data

function inital_data_khi_chan(x, y)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    RealT = typeof(x)
    slope = 15
    B = tanh(slope * y + 7.5f0) - tanh(slope * y - 7.5f0)
    rho = 0.5f0 + 0.75f0 * B
    v1 = 0.5f0 * (B - 1)
    v2 = convert(RealT, 0.1) * sinpi(2 * x)
    p = 1.0
    gamma = 1.4
    return SVector(rho, rho * v1, rho * v2,
                   p / (gamma - 1.0) + 0.5 * (rho * v1 * v1 + rho * v2 * v2))
end

initial_value = inital_data_khi_chan

boundary_value = exact_solution # dummy function

degree = 3
solver = cRK44(volume_integral)
solution_points = "gll"
correction_function = "g2"
numerical_flux = Eq.rusanov

# iter, dt, t =   249   3.5832e-04   9.0469e-02
# iter, dt, t =   738   2.9470e-04   2.2283e-01

bound_limit = "no"
bflux = evaluate
final_time = 3.0

nx, ny = 32, 32
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
save_iter_interval = 0
save_time_interval = final_time / 20.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.8

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
limiter = setup_limiter_blend(blend_type = fo_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              amax = 0.002,
                              debug_blend = false)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
