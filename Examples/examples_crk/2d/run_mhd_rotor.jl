using Tenkai.TenkaicRK, TenkaicRK.Tenkai
using Tenkai.TenkaicRK.StaticArrays
using Tenkai.TenkaicRK.Trixi: False, True
using Tenkai.TenkaicRK.SimpleUnPack

Eq = TenkaicRK.EqMHD2D

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

boundary_value = Tenkai.EqEuler2D.zero_bv # dummy function
boundary_condition = (periodic, periodic, periodic, periodic)
gamma = 1.4
final_time = 0.15

equation = Eq.get_equation(gamma,
                           activate_nc = True())

struct ExactSolutionRotor{TrixiEquations}
    equations::Eq.MHD2D{TrixiEquations}
end

function exact_solution_rotor_trixi(x, t, equations::Eq.Trixi.IdealGlmMhdEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ = 1.4
    RealT = eltype(x)
    dx = x[1] - 0.5f0
    dy = x[2] - 0.5f0
    r = sqrt(dx^2 + dy^2)
    f = (convert(RealT, 0.115) - r) / convert(RealT, 0.015)
    if r <= RealT(0.1)
        rho = convert(RealT, 10)
        v1 = -20 * dy
        v2 = 20 * dx
    elseif r >= RealT(0.115)
        rho = one(RealT)
        v1 = zero(RealT)
        v2 = zero(RealT)
    else
        rho = 1 + 9 * f
        v1 = -20 * f * dy
        v2 = 20 * f * dx
    end
    v3 = 0
    p = 1
    B1 = 5 / sqrt(4 * convert(RealT, pi))
    B2 = 0
    B3 = 0
    psi = 0
    return Eq.Trixi.prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

function (exact_solution_rotor::ExactSolutionRotor)(x, y, t)
    @unpack equations = exact_solution_rotor
    @unpack trixi_equations = equations

    # Set up the Alfven wave initial condition
    return exact_solution_rotor_trixi(SVector(x, y), t, trixi_equations)
end

exact_solution_rotor = ExactSolutionRotor(equation)
initial_value_rotor(x, y) = exact_solution_rotor(x, y, 0.0)

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 100
ny = 100
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in MHD
tvbM = 300.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

# blend parameters
indicator_model = "gassner"
debug_blend = false
cfl_safety_factor = 0.95
pure_fv = false

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value_rotor, boundary_value,
                  boundary_condition, final_time, exact_solution_rotor)
limiter_blend = setup_limiter_blend(blend_type = fo_blend(equation),
                                    # indicating_variables = Eq.rho_p_indicator!,
                                    indicating_variables = conservative_indicator!,
                                    reconstruction_variables = conservative_reconstruction,
                                    indicator_model = "gassner",
                                    debug_blend = false,
                                    pure_fv = false)
limiter_none = setup_limiter_none() # No limiter for this example
limiter = limiter_blend
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval;
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK54")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
