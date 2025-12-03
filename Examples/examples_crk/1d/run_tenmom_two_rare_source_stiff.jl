using Tenkai.TenkaicRK.StaticArrays
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai

# Submodules
Eq = Tenkai.EqTenMoment1D
# Set backend

xmin, xmax = 0.0, 4.0
boundary_condition = (neumann, neumann)

function initial_wave(x, equations::Eq.TenMoment1D)
    if x <= 2.0
        rho, v1, v2, P11, P12, P22 = 1.0, -4.0, 0.0, 9.0, 7.0, 9.0
    else
        rho, v1, v2, P11, P12, P22 = 1.0, 4.0, 0.0, 9.0, 7.0, 9.0
    end
    return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

dummy_bv(x, t) = 0.0

eq = Eq.get_equation()

initial_wave(x) = initial_wave(x, eq)

exact_wave(x, t) = initial_wave(x - t)

Wx(x, t) = -100000.0 * (x - 2.0) * exp(-20 * (x - 2.0)^2)

source_terms_tenmom_two_rare_stiff = (u, x, t, equations) -> Eq.ten_moment_source_x(u, x, t,
                                                                                    Wx,
                                                                                    equations)

initial_value, exact_solution, boundary_value = initial_wave, exact_wave, dummy_bv

degree = 3
solver = cHT112()
# solver = cRK44()
# solver = cIMEX111()
# solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.1

nx = 100
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.9 # Source term positivity condition has |∇W| making it 1e-4

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution,
                  source_terms = source_terms_tenmom_two_rare_stiff)
# FO Blending scheme gives good results with degree 4, amax = 0.5
limiter = setup_limiter_blend(blend_type = fo_blend(eq),
                              indicating_variables = Eq.rho_p_indicator!,
                              #   indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              amax = 0.5,
                              pure_fv = false)
# limiter = setup_limiter_tvb(eq; tvbM = tvbM, beta = 1.0)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = joinpath("paper_results", "output_N$(degree)_$nx"))
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
