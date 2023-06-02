using SSFR
Eq = SSFR.EqEuler2D

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

# initial_value, exact_solution, final_time, ic_name = Eq.dwave_data

v1_() = 1.0
v2_() = 1.0

function dwave(x, y)
   γ = 1.4 # RETHINK!
   ρ = 1.0 + 0.98*sinpi(2.0*(x+y))
   v1 = v1_()
   v2 = v2_()
   p   = 1.0
   ρ_v1, ρ_v2 = ρ*v1, ρ*v2
   ρ_e  = p/(γ-1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2)
   return SVector(ρ, ρ*v1, ρ*v2, ρ_e)
end

function dwave_exact(x,y,t)
   return dwave(x-v1_()*t,y-v2_()*t)
end
initial_value = dwave
exact_solution = dwave_exact



# initial_value = Eq.constant_state
# exact_solution = Eq.constant_state_exact
boundary_value = exact_solution

degree = 1
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 0.1

nx, ny = 80, 80
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution)
limiter = setup_limiter_blend(
                              blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false,
                              pure_fv = true,
                              tvbM = Inf
                             )
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;
