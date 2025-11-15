using Tenkai.TenkaicRK
using Tenkai
using Tenkai.StaticArrays
using Tenkai.TenkaicRK: newton_solver
Eq = Tenkai.EqBurg1D
# Submodules

#------------------------------------------------------------------------------
xmin, xmax = -5.0, 5.0

boundary_condition = (dirichlet, neumann)
final_time = 5.0

function initial_value_jump(x)
    if x >= 0.0
        return 0.0
    else
        return 1.0
    end
end

function exact_solution_burg1d(x, t)
    if x - 0.5 * t >= 0.0
        return 0.0
    else
        return 1.0
    end
end
initial_value, exact_solution = initial_value_jump, exact_solution_burg1d

boundary_value = exact_solution

degree = 3
solver_single = cSSP2IMEX332()
solver = DoublecRKSourceSolver(solver_single)
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = extrapolate

nx_() = 10 # Kept small for CI
nx = nx_()
bounds = ([-Inf], [Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
compute_error_interval = 1
animate_ = true
tvbM = 0.0

cfl_safety_factor_() = 0.9
cfl_safety_factor = cfl_safety_factor_()

dt() = cfl_safety_factor_() / ((degree + 1) * nx_())
mu() = 151.0 / dt()
function source_terms_stiff_burg_non_linear(u, x, t, mu, eq)
    SVector(-mu * u[1] * (u[1] - 1.0) * (u[1] - 0.9))
end

function source_terms_stiff_burg_non_linear(u, x, t, eq)
    source_terms_stiff_burg_non_linear(u, x, t, mu(), eq)
end

function source_terms_stiff_burg_non_linear_single_crk(u, x, t, eq)
    source_terms_stiff_burg_non_linear(u, x, t, eq)
end

source_terms = source_terms_stiff_burg_non_linear

function TenkaicRK.get_cache_node_vars(aux, u1,
                                       problem::Problem{<:typeof(source_terms_stiff_burg_non_linear)},
                                       scheme,
                                       eq, i, cell)
    (; cache_homogeneous) = aux
    u1 = cache_homogeneous.u1
    # u1 = aux.cache_source.u1
    u_node = get_node_vars(u1, eq, i, cell)
    return u_node
end

function TenkaicRK.implicit_source_solve(lhs, eq, x, t, coefficient,
                                         source_terms::typeof(source_terms_stiff_burg_non_linear),
                                         u_node)
    if u_node[1] >= 0.9 # The paper used 0.9
        initial_guess = SVector(1.0)
    else
        initial_guess = SVector(0.0)
    end
    # return SVector(u_node)
    # initial_guess = u_node
    # return u_node
    # @show initial_guess
    # initial_guess = u_node
    # initial_guess = SVector(exact_solution(x,t+1.2616e-02))
    # TODO - Make sure that the final source computation is used after the implicit solve
    # @assert false
    implicit_F(u_new) = u_new - lhs -
                        coefficient * TenkaicRK.calc_source(u_new, x, t, source_terms, eq)

    tol = 1e-6
    maxiters = 1000
    # u_new = TenkaicRK.picard_solver(implicit_F, initial_guess, tol)
    u_new = newton_solver(implicit_F, initial_guess, tol, maxiters)
    source = TenkaicRK.calc_source(u_new, x, t, source_terms, eq)
    @assert maximum(implicit_F(u_new))<10 * tol implicit_F(u_new)
    return u_new, source
end

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution,
                  source_terms = source_terms)
equation = Eq.get_equation()
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
limiter = setup_limiter_blend(blend_type = fo_blend_imex(equation),
                              #   indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              amax = 1.0,
                              pure_fv = false,
                              smoothing_in_time = false)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval, animate = animate_,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(equation, problem, scheme, param);

show(sol["errors"])

return sol;
p_ua = sol["plot_data"].p_ua
