using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
using Tenkai.TenkaicRK.StaticArrays
# Submodules
Eq = TenkaicRK.EqShearShallowWater1D
EqTenMom = Tenkai.EqTenMoment1D

xmin, xmax = 0.0, 1.3
boundary_condition = (periodic, periodic)

dummy_bv(x, t) = 0.0

function initial_condition_roll_wave(x)
    Cf = 0.0036
    Cr = 0.00035
    phi = 22.76
    rtheta = 0.05011
    g = 9.81
    h0 = 7.98e-3
    a = 0.05
    Lx = 1.3

    h = h0 * (1.0 + a * sinpi(2.0 * x / Lx))
    v1 = sqrt(g * h0 * tan(rtheta) / Cf)
    v2 = 0.0

    P11 = 0.5 * phi * h^2
    P12 = 0.0
    P22 = 0.5 * phi * h^2

    R11, R12, R22 = h * P11, h * P12, h * P22

    return EqTenMom.tenmom_prim2con((h, v1, v2, R11, R12, R22))
end

function bottom(x)
    rtheta = 0.05011
    return -tan(rtheta) * x
end

function bottom_dx(x)
    rtheta = 0.05011
    return -tan(rtheta)
end

function tenmom_con2prim(u)
    ρ = u[1]
    v1 = u[2] / ρ
    v2 = u[3] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    P12 = 2.0 * u[5] - ρ * v1 * v2
    P22 = 2.0 * u[6] - ρ * v2 * v2
    return SVector(ρ, v1, v2, P11, P12, P22)
end

function source_terms_ssw_roll_wave(u, x, t, eq)
    Cf = 0.0036
    Cr = 0.00035
    phi = 22.76
    rtheta = 0.05011
    g = 9.81
    h0 = 7.98e-3
    a = 0.05
    Lx = 1.3

    h, v1, v2, R11, R12, R22 = tenmom_con2prim(u)
    P11, P12, P22 = R11 / h, R12 / h, R22 / h
    v = sqrt(v1^2 + v2^2)
    T = P11 + P22
    alpha = max(0.0, Cr * (T - phi * h * h) / T^2)

    return SVector(0.0,
                   -g * h * bottom_dx(x) - Cf * v * v1,
                   -Cf * v * v2,
                   -alpha * v^3 * P11 - g * h * v1 * bottom_dx(x) - Cf * v * v1^2,
                   -alpha * v^3 * P12 - 0.5 * g * h * v2 * bottom_dx(x) - Cf * v * v1 * v2,
                   -alpha * v^3 * P22 - Cf * v * v2^2)
end

initial_value = initial_condition_roll_wave

boundary_value = dummy_bv
exact_solution_roll(x, t) = initial_condition_roll_wave(x)
exact_solution = exact_solution_roll

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 26.99

nx = ceil(Int64, 500)
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.1 * final_time
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
                  source_terms = source_terms_ssw_roll_wave)
# limiter = setup_limiter_none()
limiter = setup_limiter_blend(blend_type = mh_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner"
                              # pure_fv = true
                              )
limiter = setup_limiter_tvb(eq; tvbM = tvbM, beta = 1.0)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;

# using DelimitedFiles
# exact_brock = readdlm("exact_brock.txt", skipstart = 1)
# h0 = 7.98e-3

# using Tenkai.Plots
# p_u1 = sol["plot_data"].p_u1
# scatter!(p_u1[2], exact_brock[:, 1] * 1.3, exact_brock[:, 2] * h0, label = "Reference",
#          lw = 2)

# p = plot(p_u1.subplots[2])
# ylims!(p, (0.0, 0.013))
