# For Riemann problems in domain [0.0, 1.0]
using StaticArrays
using Tenkai

using Tenkai.TenkaicRK
# Submodules
Eq = Tenkai.TenkaicRK.EqEulerReactive1D
# Set backend

#------------------------------------------------------------------------------
xmin, xmax = -5.0, 25.0

boundary_condition = (neumann, neumann)

function reactive_rp1(x)
    if x < 0.0
        rho = 1.6812
        v1 = 2.8867
        p = 21.5672
        z = 0.0
    else
        rho = 1.0
        v1 = 0.0
        p = 1.0
        z = 1.0
    end
    gamma = 1.4
    q0 = 25.0
    E = p / (gamma - 1.0) + 0.5 * rho * v1^2 + rho * z * q0
    return SVector(rho, rho * v1, E, rho * z)
end

A, TA = 16418.0, 25.0

source_terms_reactive_rp1 = Eq.SourceTermReactive(A, TA)

initial_value = reactive_rp1

exact_solution_reactive_rp1(x, t) = reactive_rp1(x)

exact_solution = exact_solution_reactive_rp1

boundary_value = exact_solution # dummy function

degree = 4
solver = cSSP2IMEX433()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = extrapolate
final_time = 1.0

nx = ceil(Int64, 4000 / (degree + 1))
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1
cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
gamma = 1.4
q0 = 25.0
equation = Eq.get_equation(gamma, q0)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution,
                  source_terms = source_terms_reactive_rp1)
MH = mh_blend(equation)
FO = fo_blend_imex(equation)
limiter_blend = setup_limiter_blend(blend_type = FO,
                                    indicating_variables = Eq.rho_p_indicator!,
                                    # indicating_variables = Eq.conservative_indicator!,
                                    reconstruction_variables = conservative_reconstruction,
                                    # indicator_model = "model1",
                                    indicator_model = "gassner",
                                    debug_blend = false,
                                    pure_fv = false,
                                    numflux = Eq.rusanov,
                                    smoothing_in_time = false)
# limiter_tvb = setup_limiter_tvb(equation; tvbM = tvbM)
# limiter = setup_limiter_none()
limiter = limiter_blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;

sol["plot_data"].p_ua
