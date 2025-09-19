using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.StaticArrays
using Tenkai.TenkaicRK.Trixi: False, True
using Tenkai.TenkaicRK.SimpleUnPack

Eq = Tenkai.TenkaicRK.EqMHD2D

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

boundary_value = Tenkai.EqEuler2D.zero_bv # dummy function
boundary_condition = (periodic, periodic, periodic, periodic)
gamma = 5.0 / 3.0
final_time = 0.5

equation = Eq.get_equation(gamma,
                           activate_nc = True())

struct ExactSolutionOrzagTang{TrixiEquations}
    equations::Eq.MHD2D{TrixiEquations}
end

function exact_solution_orszag_tang_trixi(x, t, equations::Eq.Trixi.IdealGlmMhdEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ = 5/3
    rho = 1
    v1 = -sinpi(2 * x[2])
    v2 = sinpi(2 * x[1])
    v3 = 0
    p = 1 / equations.gamma
    B1 = -sinpi(2 * x[2]) / equations.gamma
    B2 = sinpi(4 * x[1]) / equations.gamma
    B3 = 0
    psi = 0
    return Eq.Trixi.prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

function (exact_solution_orszag_tang::ExactSolutionOrzagTang)(x, y, t)
    @unpack equations = exact_solution_orszag_tang
    @unpack trixi_equations = equations

    # Set up the Alfven wave initial condition
    return exact_solution_orszag_tang_trixi(SVector(x, y), t, trixi_equations)
end

exact_solution_orszag_tang = ExactSolutionOrzagTang(equation)
initial_value_orszag_tang(x, y) = exact_solution_orszag_tang(x, y, 0.0)

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx = 512
ny = 512
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
problem = Problem(domain, initial_value_orszag_tang, boundary_value,
                  boundary_condition, final_time, exact_solution_orszag_tang)
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
