# For Riemann problems in domain [0.0,1.0]
using SSFR
Eq = SSFR.EqEuler2D
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

boundary_value = (x,t) -> 0.0 # dummy function
boundary_condition = (neumann, neumann, neumann, neumann)
γ = 1.4

# initial_value_ref, final_time, ic_name = Eq.dwave_data

function riemann_problem(x,y)
   if x >= 0.5 && y >= 0.5
      ρ  = 1.1
      v1 = 0.0
      v2 = 0.0
      p  = 1.1
   elseif x <= 0.5 && y >= 0.5
      ρ  = 0.5065
      v1 = 0.8939
      v2 = 0.0
      p  = 0.35
   elseif x <= 0.5 && y <= 0.5
      ρ  = 1.1
      v1 = 0.8939
      v2 = 0.8939
      p  = 1.1
   elseif x >= 0.5 && y <= 0.5
      ρ  = 0.5065
      v1 = 0.0
      v2 = 0.8939
      p  = 0.35
   end
   ρ_v1 = ρ*v1
   ρ_v2 = ρ*v2
   return SVector(ρ, ρ*v1, ρ*v2, p/(γ-1.0) + 0.5*(ρ_v1*v1+ρ_v2*v2))
end
rieman_problem_(x, y, t)= riemann_problem(x, y)
initial_value = riemann_problem

exact_solution = rieman_problem_
degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 0.25

nx, ny = 100, 100 # 50, 50
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 300.0
save_iter_interval = 0
save_time_interval = final_time / 30.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 1

cfl_safety_factor = 0.9

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                     final_time, exact_solution)
limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                   numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);

return errors, plot_data
