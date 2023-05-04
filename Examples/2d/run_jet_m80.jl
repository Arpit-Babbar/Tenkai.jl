using SSFR
Eq = SSFR.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax =  0.0, 2.0
ymin, ymax = -0.5, 0.5

boundary_condition = (dirichlet, neumann, neumann, neumann)
γ = 5.0/3.0
equation = Eq.get_equation(γ)
# initial_value, exact_solution = Eq.sedov_data

function initial_value_astro_jet(eq, x, y)
   γ = eq.γ
   ρ  = 0.5
   v1 = 0.0
   v2 = 0.0
   p = 0.4127
   ρ_v1 = ρ*v1
   ρ_v2 = ρ*v2
   return SVector(ρ, ρ*v1, ρ*v2, p/(γ-1.0) + 0.5*(ρ_v1*v1+ρ_v2*v2))
end

initial_value = (x,y) -> initial_value_astro_jet(equation, x,y)
exact_solution = (x,y,t) -> initial_value(x,y)

function boundary_value_astro_jet(eq, x, y, t)
   γ = eq.γ
   if t > 0.0 && y >= -0.05 && y <= 0.05 && x ≈ 0.0
      ρ  = 5.0
      v1 = 30.0
      v2 = 0.0
      p  = 0.4127
   else
      ρ  = 0.5
      v1 = 0.0
      v2 = 0.0
      p  = 0.4127
   end
   ρ_v1 = ρ*v1
   ρ_v2 = ρ*v2
   return SVector(ρ, ρ*v1, ρ*v2, p/(γ-1.0) + 0.5*(ρ_v1*v1+ρ_v2*v2))
end

boundary_value = (x,y,t) -> boundary_value_astro_jet(equation, x, y)

degree = 3
solver = "rkfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov

bound_limit = "yes"
bflux = evaluate
final_time = 0.07

nx, ny = 448, 224
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 10000.0
save_iter_interval = 0
save_time_interval = final_time / 100.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.98

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
# equation = Eq.get_equation(γ)
problem = Problem(domain,
                  initial_value,
                  boundary_value, boundary_condition,
                  final_time, exact_solution)
limiter = setup_limiter_blend(
                              blend_type = mh_blend(equation),
                              indicating_variables = Eq.rho_p_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner",
                              debug_blend = false,
                              pure_fv = false
                             )
# limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK54")
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol
