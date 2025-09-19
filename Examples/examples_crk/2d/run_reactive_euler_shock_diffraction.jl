using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai
Eq = TenkaicRK.EqEulerReactive2D
EqEuler2D = Tenkai.EqEuler2D
using Tenkai.TenkaicRK.StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 2.0
ymin, ymax = 0.0, 2.0

# More boundary condition objects will be there.
boundary_condition = (dirichlet, neumann,
                      reflect, # bottom includes three surfaces, 2 horizontal
                      # and one vertical
                      reflect)
gamma = 1.2
q = 50.0
tT = 50.0
tK = 2566.4
equation = Eq.get_equation(gamma, q, tK, tT)

function step_grid_sizes(ny)
    @assert ny % 4 == 0
    ny_tuple = (Int(ny / 2), ny)
    nx = ny
    nx_tuple = (Int(nx / 4), nx)
    return nx_tuple, ny_tuple
end

function initial_value_shock_diffraction(eq, x, y)
    gamma = eq.gamma
    if x <= 0.5
        rho = 11.0
        v1 = 6.18
        v2 = 0.0
        p = 970.0
        Y = 1.0
    else
        rho = 11.0
        v1 = 0.0
        v2 = 0.0
        p = 55.0
        Y = 1.0
    end

    return Eq.prim2con(eq, (rho, v1, v2, p, Y))
end

initial_value = (x, y) -> initial_value_shock_diffraction(equation, x, y)
exact_solution = (x, y, t) -> initial_value(x, y)

source_terms = (u, x, t, q) -> Eq.source_term_arrhenius(x, u, eq)

boundary_value = exact_solution

degree = 3
solver = cRK44()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes"
bflux = evaluate
final_time = 0.1

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
                  final_time, exact_solution,
                  source_terms = source_terms)
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
                   time_scheme = "SSPRK33")
#------------------------------------------------------------------------------
# problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
#                                           ARGS)
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
