using StaticArrays
using SSFR
Eq = SSFR.EqLinAdv2D
#------------------------------------------------------------------------------
case = 3 # options: 1,2,3,4

if case == 1 || case == 2 # full domain
   xmin, xmax = -1.0, 1.0
   ymin, ymax = -1.0, 1.0
   final_time = 0.1
   initial_value__(x, y) = SVector(1.0 + exp(-50.0*((x-0.5)^2 + y^2)))
   if case == 1
      boundary_condition_ = (periodic, periodic, periodic, periodic)
   else
      boundary_condition_ = (dirichlet, dirichlet, dirichlet, dirichlet)
   end
elseif case == 3
   xmin, xmax = 0.0, 1.0
   ymin, ymax = 0.0, 1.0
   final_time = 0.5 * pi
   initial_value__(x, y) = SVector(1.0 + exp(-50.0*((x-0.5)^2 + y^2)))
   boundary_condition_ = (neumann, dirichlet, dirichlet, neumann)
else
   xmin, xmax = 0.0, 1.0
   ymin, ymax = 0.0, 1.0
   final_time = 2.0 * pi
   initial_value__(x, y) = SVector(sinpi(2 * x) * sinpi(2 * y))
   boundary_condition_ = (neumann, dirichlet, dirichlet, neumann)
end

velocity__(x, y) = SVector(-y, x)
exact_solution_(x,y,t) = initial_value__(x*cos(t) + y*sin(t), -x*sin(t) + y*cos(t))
boundary_value_(x,y,t) = exact_solution_(x, y, t)
#------------------------------------------------------------------------------
degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
bound_limit = "no"
bflux = extrapolate
numerical_flux = Eq.rusanov

nx, ny = 20, 20
bounds = ([-Inf],[Inf])
cfl = 0.0
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 10.0
compute_error_interval = 0
animate = true
#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
problem = Problem(domain, initial_value__, boundary_value_, boundary_condition_,
                  final_time, exact_solution_)
eq = Eq.get_equation(velocity__)
# limiter = setup_limiter_blend(
#                                  blend_type = fo_blend,
#                                  indicating_variables = conservative_indicator!,
#                                  reconstruction_variables = conservative_reconstruction,
#                                  indicator_model = "gassner",
#                                  debug_blend = false
#                                 )
limiter = setup_limiter_tvb(eq; tvbM = tvbM)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                      save_time_interval, compute_error_interval,
                      animate = animate)
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, eq, ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;
