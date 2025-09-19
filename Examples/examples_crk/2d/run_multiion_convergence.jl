using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.StaticArrays
using Tenkai.TenkaicRK.SimpleUnPack
Eq = Tenkai.TenkaicRK.EqMultiIonMHD2D
using Tenkai.TenkaicRK.EqMultiIonMHD2D: MultiIonMHD2D, IdealGlmMhdMultiIonEquations2D
import Tenkai.TenkaicRK.Trixi

###############################################################################
"""
  electron_pressure_alpha(u, equations::IdealMhdMultiIonEquations2D)
Returns a fraction (alpha) of the total ion pressure for the electron pressure.
"""
function electron_pressure_alpha(u, equations)
    alpha = 0.2
    prim = Trixi.cons2prim(u, equations)
    p_e = zero(u[1])
    for k in Trixi.eachcomponent(equations)
        _, _, _, _, p_k = Trixi.get_component(k, prim, equations)
        p_e += p_k
    end
    return alpha * p_e
end
# semidiscretization of the ideal multi-ion MHD equations

trixi_equations = IdealGlmMhdMultiIonEquations2D(gammas = (2.0, 4.0),
                                                 charge_to_mass = (2.0, 1.0),
                                                 electron_pressure = electron_pressure_alpha)

eq = Eq.get_equation(trixi_equations) # Deactivate GLM divergence cleaning

"""
Initial (and exact) solution for the the manufactured solution test. Runs with 
* gammas = (2.0, 4.0),å
* charge_to_mass = (2.0, 1.0)
* Domain size: [-1,1]²
"""
function initial_condition_manufactured_solution(x, t, equations)
    am = 0.1
    om = π
    h = am * sin(om * (x[1] + x[2] - t)) + 2
    hh1 = am * 0.4 * sin(om * (x[1] + x[2] - t)) + 1
    hh2 = h - hh1

    u1 = hh1
    u2 = hh1
    u3 = hh1
    u4 = 0.1 * hh1
    u5 = 2 * hh1^2 + hh1
    u6 = hh2
    u7 = hh2
    u8 = hh2
    u9 = 0.1 * hh2
    u10 = 2 * hh2^2 + hh2
    u11 = 0.25 * h
    u12 = -0.25 * h
    u13 = 0.1 * h

    return SVector{Trixi.nvariables(equations), real(equations)}([
                                                                     u11,
                                                                     u12,
                                                                     u13,
                                                                     u1,
                                                                     u2,
                                                                     u3,
                                                                     u4,
                                                                     u5,
                                                                     u6,
                                                                     u7,
                                                                     u8,
                                                                     u9,
                                                                     u10,
                                                                     0
                                                                 ])
end

"""
Source term that corresponds to the manufactured solution test. Runs with 
* gammas = (2.0, 4.0),
* charge_to_mass = (2.0, 1.0)
* Domain size: [-1,1]²
"""
function source_terms_manufactured_solution_pe(u, x, t, equations)
    am = 0.1
    om = pi
    h1 = am * sin(om * (x[1] + x[2] - t))
    hx = am * om * cos(om * (x[1] + x[2] - t))

    s1 = (2 * hx) / 5
    s2 = (38055 * hx * h1^2 + 185541 * hx * h1 + 220190 * hx) / (35000 * h1 + 75000)
    s3 = (38055 * hx * h1^2 + 185541 * hx * h1 + 220190 * hx) / (35000 * h1 + 75000)
    s4 = hx / 25
    s5 = (1835811702576186755 * hx * h1^2 + 8592627463681183181 * hx * h1 +
          9884050459977240490 * hx) / (652252660543767500 * h1 + 1397684272593787500)
    s6 = (3 * hx) / 5
    s7 = (76155 * hx * h1^2 + 295306 * hx * h1 + 284435 * hx) / (17500 * h1 + 37500)
    s8 = (76155 * hx * h1^2 + 295306 * hx * h1 + 284435 * hx) / (17500 * h1 + 37500)
    s9 = (3 * hx) / 50
    s10 = (88755 * hx * h1^2 + 338056 * hx * h1 + 318185 * hx) / (8750 * h1 + 18750)
    s11 = hx / 4
    s12 = -hx / 4
    s13 = hx / 10

    trixi_equation = equations.trixi_equations
    s = SVector{Trixi.nvariables(trixi_equation), real(trixi_equation)}(s11,
                                                                        s12,
                                                                        s13,
                                                                        s1,
                                                                        s2,
                                                                        s3,
                                                                        s4,
                                                                        s5,
                                                                        s6,
                                                                        s7,
                                                                        s8,
                                                                        s9,
                                                                        s10,
                                                                        0)
    S_std = Trixi.source_terms_lorentz(u, x, t, trixi_equation)

    return SVector{Trixi.nvariables(equations.trixi_equations),
                   real(equations.trixi_equations)}(S_std .+ s)
end

initial_condition = initial_condition_manufactured_solution

dummy_bv(x, y, t) = 0.0
boundary_value = dummy_bv
function exact_solution_manufactured_solution(x, y, t)
    initial_condition_manufactured_solution(SVector(x, y), t, eq.trixi_equations)
end

initial_value = (x, y) -> exact_solution_manufactured_solution(x, y, 0.0)

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 1.0

nx = 100
ny = 100
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.9

xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0

domain = [xmin, xmax, ymin, ymax]

boundary_condition = (periodic, periodic, periodic, periodic)

###############################################################################
# ODE solvers, callbacks etc.

grid_size = [nx, ny]

problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution_manufactured_solution,
                  source_terms = source_terms_manufactured_solution_pe)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)

sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol
