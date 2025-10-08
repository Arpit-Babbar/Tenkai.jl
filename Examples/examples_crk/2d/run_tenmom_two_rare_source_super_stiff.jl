using Tenkai.StaticArrays
using Tenkai.TenkaicRK
using Tenkai

# Submodules
Eq = Tenkai.EqTenMoment2D
# Set backend

xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0
boundary_condition = (neumann, neumann, neumann, neumann)

function initial_wave(x, y, equations::Eq.TenMoment2D)
    v1 = 4.0 * x / sqrt(x^2 + y^2 + 1e-16)
    v2 = 4.0 * y / sqrt(x^2 + y^2 + 1e-16)

    rho = 1.0
    P11, P12, P22 = 9.0, 7.0, 9.0
    return Eq.prim2con(equations, (rho, v1, v2, P11, P12, P22))
end

dummy_bv(x, y, t) = 0.0

eq = Eq.get_equation()

initial_wave(x, y) = initial_wave(x, y, eq)

exact_wave(x, y, t) = initial_wave(x, y, eq)

factor() = 1000000.0

function gauss_source_x(x, y, t)
    a = -20 * (x^2 + y^2)
    der_factor = -x
    return factor() * der_factor * exp(a)
end

function gauss_source_y(x, y, t)
    a = -20 * (x^2 + y^2)
    der_factor = -y
    return factor() * der_factor * exp(a)
end

struct MyTenMomentSource{WX, WY}
    Wx::WX
    Wy::WY
end

function (source::MyTenMomentSource)(u, x, t, equations::Eq.TenMoment2D)
    Wx = source.Wx(x[1], x[2], t)
    Wy = source.Wy(x[1], x[2], t)
    rho = u[1]
    rho_v1 = u[2]
    rho_v2 = u[3]
    term1 = SVector(0.0, -0.5 * rho * Wx, 0.0, -0.5 * rho_v1 * Wx, -0.25 * rho_v2 * Wx,
                    0.0)
    term2 = SVector(0.0, 0.0, -0.5 * rho * Wy, 0.0, -0.25 * rho_v1 * Wy,
                    -0.5 * rho_v2 * Wy)
    return term1 + term2
end

source_term1 = (u, x, t, equations::Eq.TenMoment2D) -> Eq.ten_moment_source(u, x[1], x[2],
                                                                            t,
                                                                            gauss_source_x,
                                                                            gauss_source_y,
                                                                            equations)

source_term = MyTenMomentSource(gauss_source_x, gauss_source_y)

initial_value, exact_solution, boundary_value = initial_wave, exact_wave, dummy_bv

degree = 0 # CHANGE TO 2 TO SEE THE CRASH
# solver = cRK44() # (ALSO UNCOMMENT TO SEE THE CRASH)
solver = cIMEX111()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no" # CHANGE TO "yes" TO FAIRLY SEE THE CRASH
bflux = extrapolate
final_time = 0.02

nx = 50
ny = 50
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.1 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.95 # Source term positivity condition has |∇W| making it 1e-4

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution, source_terms = source_term)
limiter = setup_limiter_none()

# # (ALSO UNCOMMENT THE LIMITER TO SEE THE CRASH)
# limiter = setup_limiter_blend(blend_type = fo_blend(eq),
#                             #   indicating_variables = Eq.rho_p_indicator!,
#                                 indicating_variables = Eq.conservative_indicator!,
#                               reconstruction_variables = conservative_reconstruction,
#                               indicator_model = "gassner",
#                               amax = 0.5,
#                               pure_fv = false)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   saveto = "none")
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
