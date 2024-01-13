using StaticArrays
using Tenkai
using Plots
using LinearAlgebra: norm
# Submodules
Eq = Tenkai.EqTenMoment2D
ten_moment_source = Eq.ten_moment_source
plotlyjs() # Set backend

xmin, xmax = 0.0, 4.0
ymin, ymax = 0.0, 4.0
boundary_condition = (neumann, neumann, neumann, neumann)

@inline function f_rare(a, b)
    if a < b
        return -2.0 * (a/b)^3 + 3.0 * (a/b)^2
    else
        return 1.0
    end
end

function uniform_plasma_ic(x, y, equations::Eq.TenMoment2D)
    rho = 0.1
    v1 = v2 = 0.0
    P11 = P22 = 9.0
    P12 = 7.0
    return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

# W(x,y) = c * exp(-200 * ((x-2)^2 + (y-2)^2) ) # c = 25
# Wx(x,y) = -400 * c * (x-2) * exp(-200 * ((x-2)^2 + (y-2)^2) )
# Wy(x,y) = -400 * c * (y-2) * exp(-200 * ((x-2)^2 + (y-2)^2) )

function gauss_source_x(x, y, t)
    a = - 200 * ( (x-2.0)^2 + (y-2.0)^2 )
    der_factor = - 400.0 * (x-2.0)
    return 25.0 * der_factor * exp(a)
end

function gauss_source_y(x, y, t)
    a = - 200 * ( (x-2.0)^2 + (y-2)^2 )
    der_factor = - 400.0 * (y-2.0)
    return 25.0 * der_factor * exp(a)
end

source_term = (u, x, t, equations::Eq.TenMoment2D) -> ten_moment_source(
    u, x[1], x[2], t, gauss_source_x, gauss_source_y, equations)

dummy_bv(x, t) = 0.0

eq = Eq.get_equation()

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate
final_time = 0.1

nx, ny = 100, 100

uniform_plasma_ic(x, y) = uniform_plasma_ic(x, y, eq)

exact_uniform_plasma(x, y, t) = uniform_plasma_ic(x, y)

cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.98
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, uniform_plasma_ic, dummy_bv, boundary_condition,
                  final_time, exact_uniform_plasma, source_terms = source_term)
limiter = setup_limiter_blend(blend_type = fo_blend(eq),
                              indicating_variables = Eq.conservative_indicator!,
                            #   indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner")

limiter = setup_limiter_tvbβ(eq; tvbM = tvbM, beta = 1.5)
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
