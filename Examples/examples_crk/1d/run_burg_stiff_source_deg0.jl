using Tenkai.TenkaicRK
using Tenkai
using Tenkai.StaticArrays
using Tenkai.TenkaicRK: newton_solver
using Tenkai.EqBurg1D
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

degree = 0
# solver_single = cSSP2IMEX433()
solver_single = cIMEX111()
solver = DoublecRKSourceSolver(solver_single)
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate

nx_() = 10 # Kept small for CI
nx = nx_()
bounds = ([-Inf], [Inf])
cfl = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
compute_error_interval = 1
animate_ = true
tvbM = 0.0

cfl_safety_factor_() = 0.5
cfl_safety_factor = cfl_safety_factor_()

dt() = cfl_safety_factor_() / ((degree + 1) * nx_())
# mu() = 151.0 / dt()
mu() = 10^3
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
                                       problem::Problem{<:Real,
                                                        <:typeof(source_terms_stiff_burg_non_linear)},
                                       scheme,
                                       eq::Burg1D, i, cell)
    (; cache_homogeneous) = aux
    u1 = cache_homogeneous.u1
    # u1 = aux.cache_source.u1
    u_node = get_node_vars(u1, eq, i, cell)
    return u_node
end

function TenkaicRK.implicit_source_solve(lhs, eq::Burg1D, x, t, coefficient,
                                         source_terms::typeof(source_terms_stiff_burg_non_linear),
                                         u_node)
    if u_node[1] > 0.5 # The paper used 0.9
        initial_guess = SVector(1.0)
    elseif u_node[1] < 0.5
        initial_guess = SVector(0.0)
    else
        initial_guess = u_node
    end

    implicit_F(u_new) = u_new - lhs -
                        coefficient * TenkaicRK.calc_source(u_new, x, t, source_terms, eq)

    implicit_F_no_lhs(u_new) = u_new -
                               coefficient *
                               TenkaicRK.calc_source(u_new, x, t, source_terms, eq)

    # tol = 10.0 / 110.0
    tol = 1.0 / nx_()
    maxiters = 1000000
    u_new = TenkaicRK.picard_solver(implicit_F, initial_guess, tol, maxiters, 0.0000001)
    # u_new = newton_solver(implicit_F, initial_guess, tol, maxiters)

    # implicit_F_scalar(u_new) = u_new[1] - lhs[1] -
    #                     coefficient * TenkaicRK.calc_source(u_new, x, t, source_terms, eq)[1]
    # u_new_ = Tenkai.newton_solver_scalar(implicit_F_scalar, initial_guess, tol, maxiters)
    # u_new = SVector(u_new_)

    # return initial_guess
    # if u_new[1] > 1.0
    #     return initial_guess
    # elseif u_new[1] < 0.0
    #     return initial_guess
    # end

    dt = 2.7778e-02

    exact = exact_solution_burg1d(x, t)
    if abs(u_new[1] - exact) > 0.1
        res_exact_1 = implicit_F(SVector(exact))
        res_exact_2 = implicit_F_no_lhs(SVector(exact))
        @show res_exact_1, res_exact_2
        # @show u_new, exact, u_node, lhs, x
        @show u_new, exact, lhs, coefficient, initial_guess, u_node
        # return SVector(exact)
        # @show x
        # @assert false
    end

    source = TenkaicRK.calc_source(u_new, x, t, source_terms, eq)

    # @assert maximum(implicit_F(u_new))<10 * tol implicit_F(u_new)
    return u_new, source
end

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution,
                  source_terms = source_terms)
equation = Eq.get_equation()
# limiter = setup_limiter_blend(blend_type = fo_blend_imex(equation),
#                               #   indicating_variables = Eq.rho_p_indicator!,
#                               indicating_variables = Eq.conservative_indicator!,
#                               reconstruction_variables = conservative_reconstruction,
#                               indicator_model = "gassner",
#                               amax = 1.0,
#                               pure_fv = false)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval, animate = animate_,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------

grid = Tenkai.make_cartesian_grid(problem, param.grid_size)
# fr operators like differentiation matrix, correction functions
op = Tenkai.fr_operators(scheme.degree, scheme.solution_points,
                         scheme.correction_function)
# cache for storing solution and other arrays
cache = (; Tenkai.setup_arrays(grid, scheme, equation)...,
         trixi_ode = Tenkai.tenkai2trixiode(scheme.solver, equation, problem,
                                            scheme, param))
# auxiliary objects like plot data, blending limiter, etc.
aux = Tenkai.create_auxiliaries(equation, op, grid, problem, scheme, param,
                                cache)

sol = Tenkai.solve(equation, problem, scheme, param;
                   grid = grid, op = op, aux = aux, cache = cache);

show(sol["errors"])

return sol;
p_ua = sol["plot_data"].p_ua

ua = sol["ua"]

u_homogeneous = sol["aux"].cache_homogeneous.ua
xc = sol["grid"].xc

using Plots
scatter(xc, u_homogeneous[1, 1:nx], label = "Homogeneous Solution", lw = 2)
scatter!(xc, ua[1, 1:nx], label = "CRK Solution", lw = 2)
exact_t = x -> exact_solution_burg1d(x, final_time)
plot!(xc, exact_t.(xc), label = "Exact Solution", lw = 2)
