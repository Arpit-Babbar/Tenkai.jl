using Tenkai.TenkaicRK
using Tenkai
using Tenkai.StaticArrays
using Tenkai.TenkaicRK: newton_solver, norm, newton_solver_scalar
Eq = Tenkai.EqBurg1D
# Submodules

#------------------------------------------------------------------------------
xmin, xmax = -5.0, 5.0

boundary_condition = (dirichlet, dirichlet)
final_time = 4.0

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
solver_single = cSSP2IMEX433()
# solver_single = cIMEX111()
# solver_single = cRK11()
solver = DoublecRKSourceSolver(solver_single)
# solver = cRK44()
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

cfl_safety_factor_() = 1.0
cfl_safety_factor = cfl_safety_factor_()

dt() = cfl_safety_factor_() / (nx_())
# dt() = 9.2694e-03
mu() = 151.0 / dt()
mu() = 10^0
function source_terms_stiff_burg_non_linear(u, x, t, mu, eq)
    SVector(mu * (1.0 - u[1]) * (u[1] - 0.9) * u[1])
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
                                       eq, i, cell)
    # u_node = get_node_vars(u1, eq, i, cell)
    # return u_node
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

    min_mu = 10^6
    # mu_ = max(min_mu, mu())
    mu_ = mu()

    implicit_F(u_new) = u_new - lhs -
                        # coefficient * TenkaicRK.calc_source(u_new, x, t, source_terms, eq)
                        coefficient *
                        source_terms_stiff_burg_non_linear(u_new, x, t, mu_, eq)

    implicit_F_scalar(u_new_scalar) = implicit_F(SVector(u_new_scalar))[1]
    tol = 1e-12
    maxiters = 1000
    # u_new = TenkaicRK.picard_solver(implicit_F, initial_guess, tol)
    u_new = newton_solver(implicit_F, initial_guess, tol, maxiters)
    # u_new = newton_solver_scalar(implicit_F_scalar, initial_guess, tol, maxiters); u_new = SVector(u_new)
    @assert maximum(implicit_F(u_new))<10 * tol implicit_F(u_new)
    # u_new = lhs + coefficient * source_terms_stiff_burg_non_linear(u_new, x, t, mu(), eq)
    # if implicit_F(initial_guess)[1] > 1e-4
    #     @show u_new, initial_guess, lhs, u_node
    # end

    # source = source_terms_stiff_burg_non_linear(u_new, x, t, mu_, eq)
    source = TenkaicRK.calc_source(u_new, x, t, source_terms, eq)

    return u_new, source
end

#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution,
                  source_terms = source_terms)
equation = Eq.get_equation()
limiter = setup_limiter_blend(blend_type = fo_blend_imex(equation),
                              #   indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              amax = 1.0,
                              pure_fv = false)
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

ua = sol["ua"]

u_homogeneous = sol["aux"].cache_homogeneous.ua
xc = sol["grid"].xc

using Plots
scatter(xc, u_homogeneous[1, 1:nx], label = "Homogeneous Solution", lw = 2)
scatter!(xc, ua[1, 1:nx], label = "CRK Solution", lw = 2)
exact_t = x -> exact_solution_burg1d(x, final_time)
plot!(xc, exact_t.(xc), label = "Exact Solution", lw = 2)
