using Tenkai.TenkaicRK.StaticArrays
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai

# Submodules
Eq = TenkaicRK.EqShearShallowWater1D
EqTenMom = Tenkai.EqTenMoment1D
using Tenkai.TenkaicRK.Tenkai: @unpack

xmin, xmax = -0.5, 0.5
boundary_condition = (periodic, periodic)

dummy_bv(x, t) = 0.0

function exact_solution_ssw_convergence(x, t)
    h = 2.0 + sinpi(2.0 * (x - t))
    v1 = 1.0
    v2 = 0.0
    P11 = P22 = 1.0
    P12 = 0.0
    R11, R22, R12 = h * P11, h * P22, h * P12
    return EqTenMom.tenmom_prim2con((h, v1, v2, R11, R12, R22))
end

initial_value = x -> exact_solution_ssw_convergence(x, 0.0)

function source_terms_ssw_convergence(u, x, t, eq)
    @unpack gravity = eq
    s1 = 0.0
    s2 = 2.0 * pi * cospi(2.0 * (x - t)) *
         (1.0 + 2.0 * gravity + gravity * sinpi(2.0 * (x - t)))
    s3 = 0.0
    s4 = s2
    s5 = 0.0
    s6 = 0.0
    return SVector(s1, s2, s3, s4, s5, s6)
end

boundary_value = dummy_bv
exact_solution = exact_solution_ssw_convergence

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = extrapolate
final_time = 0.1

nx = 20
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = nx
gravity = 9.81
eq = Eq.get_equation(gravity)
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution,
                  source_terms = source_terms_ssw_convergence)
limiter = setup_limiter_blend(blend_type = mh_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner"
                              # pure_fv = true
                              )
limiter = setup_limiter_tvb(eq; tvbM = tvbM, beta = 0.5)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
