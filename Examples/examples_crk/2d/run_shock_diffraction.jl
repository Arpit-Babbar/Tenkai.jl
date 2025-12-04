using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
using Tenkai.TenkaicRK.StaticArrays
Eq = TenkaicRK.EqEulerReactive2D
EqEuler2D = Tenkai.EqEuler2D
Eq = EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 2.0
ymin, ymax = 0.0, 2.0

# More boundary condition objects will be there.
boundary_condition = (dirichlet, neumann,
                      reflect, # bottom includes three surfaces, 2 horizontal
                      # and one vertical
                      reflect)
γ = 1.4
# equation = Eq.get_equation(γ, 0.0, 0.0, 0.0)
equation = EqEuler2D.get_equation(γ)

function step_grid_sizes(ny)
    @assert ny % 4 == 0
    ny_tuple = (Int(ny / 2), ny)
    nx = ny
    nx_tuple = (Int(nx / 4), nx)
    return nx_tuple, ny_tuple
end

function initial_value_shock_diffraction(eq, x, y)
    γ = 1.4
    # Ma = 9.85914
    # ρ_r = 1.0
    # v1_l, v1_r = Ma,  0.0
    # v2_l, v2_r = 0.0, 0.
    # p_r  = 1.0
    # pl_by_pr = 1.0 + 2.0*γ/(γ+1.0) * (Ma^2 - 1.0)
    # ρl_by_ρr = (γ+1.0)*Ma^2 / (2.0 + (γ-1.0) * Ma^2)
    # p_l = pl_by_pr / p_r
    # ρ_l = ρl_by_ρr / ρ_r
    if x <= 0.5
        # ρ  = ρ_l
        # v1 = v1_l
        # v2 = v2_l
        # p  = p_l
        ρ = 5.9970
        v1 = 98.5914
        v2 = 0.0
        p = 11666.5
    else
        ρ = 1.0
        v1 = 0.0
        v2 = 0.0
        p = 1.0
    end

    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

initial_value = (x, y) -> initial_value_shock_diffraction(equation, x, y)
initial_value = (x, y) -> SVector(initial_value_shock_diffraction(equation, x, y)..., 0.0)
exact_solution = (x, y, t) -> initial_value(x, y)
exact_solution = (x, y, t) -> initial_value(x, y)

boundary_value = exact_solution

degree = 3
solver = cRK44()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes"
bflux = evaluate
final_time = 0.01

ny = 100
nx_tuple, ny_tuple = step_grid_sizes(ny)
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = final_time / 100.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = [nx_tuple, ny_tuple]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false,
                              pure_fv = false
                              # bc_x = Eq.hllc_upwinding_x
                              )
# limiter = setup_limiter_tvbβ(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK33",
                   saveto = "none")
#------------------------------------------------------------------------------
# problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
#                                           ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
