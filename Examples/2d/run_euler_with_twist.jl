using Tenkai
Eq = Tenkai.EqEuler2D
using StaticArrays
using Tenkai.StructuredMeshes: save_mesh_file
import Trixi
using Trixi2Vtk
#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
boundary_value_(x,y,t) = 0.0
boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

function exact_solution_convergence_test(x_, y_, t, velocity)
   rho = 1.0
   v1 =  v2 = 1.0
   e = 1.0
   rho_v1 = rho*v1
   rho_v2 = rho*v2
   rho_e = rho*e

   return SVector(rho, rho_v1, rho_v2, rho_e)
end

velocity(x, y) = SVector(0.2, -0.7)
exact_solution_convergence_test(x, y, t) = exact_solution_convergence_test(x, y, t, velocity)
initial_value_convergence_test(x, y) = exact_solution_convergence_test(x, y, 0.0)

degree              = 6
solver              = "rkfr"
solution_points     = "gll"
correction_function = "g2"
numerical_flux      = Tenkai.flux_lax_friedrichs
bound_limit         = "no"
bflux               = extrapolate
final_time          = 1.0 # 20 * sqrt(2.0) / 0.5

nx, ny = 32, 32
cfl = 0.1
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 1
save_time_interval = 0.0 # final_time / 5.0
cfl_safety_factor = 0.1
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = (nx, ny)
domain_filler = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain_filler, initial_value_convergence_test,
                  boundary_value_, boundary_condition,
                  final_time, exact_solution_convergence_test)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux,
                2)
param = Parameters([grid_size[1], grid_size[2]], cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "SSPRK22")
default_mesh_file = joinpath(@__DIR__, "mesh_periodic_square_with_twist.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/12ce661d7c354c3d94c74b964b0f1c96/raw/8275b9a60c6e7ebbdea5fc4b4f091c47af3d5273/mesh_periodic_square_with_twist.mesh",
                                       default_mesh_file)
mesh_file = default_mesh_file

grid = UnstructuredMesh2D(mesh_file, periodicity=true)

#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          grid, ARGS)
#------------------------------------------------------------------------------
rm("output", force = true, recursive = true)
sol = Tenkai.solve(equation, grid, problem, scheme, param);
Trixi.save_mesh_file(grid, "output")
trixi2vtk("output/sol*.h5", output_directory="output")
println(sol["errors"])

return sol;
