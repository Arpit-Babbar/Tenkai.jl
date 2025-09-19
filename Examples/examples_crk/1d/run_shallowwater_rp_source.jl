# For Riemann problems in domain [0.0,1.0]
using StaticArrays
using Tenkai

using Tenkai.TenkaicRK
# Submodules
Eq = TenkaicRK.EqShallowWater1D
# Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 100.0

boundary_condition = (neumann, neumann)
gravity = 1.0

function dam_break(x)
    if x < 50.0
        height = 0.02
        h_v = 0.0
    else
        height = 0.01
        h_v = 0.04
    end
    return SVector(height, h_v)
end

function source_terms_dam_break(u, x, t, eq)
    gravity = eq.gravity
    h, hv = u
    Bx = -0.2
    n = 0.09
    source_y = -gravity * h * Bx
    source_y += -gravity * n^2 * abs(hv) * hv / h^(7 / 3)
    return SVector(0.0, source_y)
end

initial_value = dam_break

exact_solution_dambreak(x, t) = dam_break(x)

exact_solution = exact_solution_dambreak

boundary_value = exact_solution # dummy function

degree = 1
solver = cHT112()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 10.0

nx = 100
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.1 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1
cfl_safety_factor = 0.5

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
equation = Eq.get_equation(gravity)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution,
                  source_terms = source_terms_dam_break)
MH = mh_blend(equation)
FO = fo_blend(equation)
limiter_blend = setup_limiter_blend(blend_type = FO,
                                    # indicating_variables = Eq.rho_p_indicator!,
                                    indicating_variables = Eq.conservative_indicator!,
                                    reconstruction_variables = conservative_reconstruction,
                                    indicator_model = "model1",
                                    debug_blend = false,
                                    pure_fv = false,
                                    numflux = Eq.rusanov)
# limiter = setup_limiter_none()
limiter = limiter_blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;

sol["plot_data"].p_ua
