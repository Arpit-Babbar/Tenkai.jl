using Tenkai
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.StaticArrays
using Tenkai.TenkaicRK.SimpleUnPack
Eq = TenkaicRK.EqMultiIonMHD2D
using Tenkai.TenkaicRK.EqMultiIonMHD2D: MultiIonMHD2D, IdealGlmMhdMultiIonEquations2D
import Tenkai.TenkaicRK.Trixi

###############################################################################

# semidiscretization of the ideal multi-ion MHD equations
trixi_equations = IdealGlmMhdMultiIonEquations2D(gammas = (5 / 3, 1.4),
                                                 charge_to_mass = (1, 1 / 2),
                                                 initial_c_h = 0.0)

eq = Eq.get_equation(trixi_equations) # Deactivate GLM divergence cleaning

"""
Initial (and exact) solution for the the manufactured solution test. Runs with 
* gammas = (2.0, 4.0),
* charge_to_mass = (2.0, 1.0)
* Domain size: [-1,1]²
"""
function initial_condition_khi(x, t, equations::IdealGlmMhdMultiIonEquations2D)
    (; gammas) = equations
    ca = 0.1 # Alfvén speed
    theta = pi / 3 # Angle of initial magnetic field
    M = 1 # Mach number
    y0 = 1 / 20 #steepness of the shear
    v20 = 0.01 # Parameter of perturbation
    sigma = 0.1 # Parameter of perturbation

    rho1 = rho2 = 0.5
    p1 = 0.5 / gammas[1]
    p2 = 0.5 / gammas[2]
    B1 = ca * cos(theta)
    B2 = 0
    B3 = ca * sin(theta)
    v1 = 0.5 * M * tanh(x[2] / y0)
    v2 = v20 * sin(2 * pi * x[1]) * exp(-x[2]^2 / sigma^2)
    v3 = 0

    prim = SVector{Trixi.nvariables(equations), real(equations)}(B1,
                                                                 B2,
                                                                 B3,
                                                                 rho1,
                                                                 v1,
                                                                 v2,
                                                                 v3,
                                                                 p1,
                                                                 rho2,
                                                                 v1,
                                                                 v2,
                                                                 v3,
                                                                 p2,
                                                                 0)
    return Trixi.prim2cons(prim, equations)
end

initial_condition = initial_condition_khi

dummy_bv(x, y, t) = 0.0
boundary_value = dummy_bv
function exact_solution_khi(x, y, t)
    initial_condition_khi(SVector(x, y), t, eq.trixi_equations)
end

initial_value = (x, y) -> exact_solution_khi(x, y, 0.0)

degree = 3
volume_integral = Trixi.VolumeIntegralFluxDifferencing((Trixi.flux_ruedaramirez_etal,
                                                        Trixi.flux_nonconservative_ruedaramirez_etal))

solver = TrixiRKSolver(volume_integral)
# solver = RKFR(volume_integral)
solution_points = "gll"
correction_function = "g2"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 20.0

nx = 128
ny = 128
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 300
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.25

xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0

domain = [xmin, xmax, ymin, ymax]

boundary_condition = (periodic, periodic, reflect, reflect)

function source_terms_lorentz_khi(u, x, t, eq::MultiIonMHD2D)
    Trixi.source_terms_lorentz(u, x, t, eq.trixi_equations)
end

###############################################################################
# ODE solvers, callbacks etc.

grid_size = [nx, ny]

problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution_khi, source_terms = source_terms_lorentz_khi)
limiter_blend = setup_limiter_blend(blend_type = fo_blend(eq),
                                    # indicating_variables = Eq.rho_p_indicator!,
                                    indicating_variables = conservative_indicator!,
                                    reconstruction_variables = conservative_reconstruction,
                                    indicator_model = "gassner",
                                    debug_blend = false,
                                    amax = 0.01,
                                    pure_fv = false)
limiter = limiter_blend
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "by degree")

sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol
