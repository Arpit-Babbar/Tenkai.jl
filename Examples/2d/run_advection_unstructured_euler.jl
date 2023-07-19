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
boundary_condition = (periodic, periodic, periodic, periodic) # dummy
γ = 1.4

function exact_solution_eq(x, t, equations)
   c = 2
   A = 0.1
   L = 2
   f = 1/L
   ω = 2 * pi * f
   ini = c + A * sin(ω * (x[1] + x[2] - t))

   rho = ini
   rho_v1 = ini
   rho_v2 = ini
   rho_e = ini^2

   return SVector(rho, rho_v1, rho_v2, rho_e)
end

exact_solution(x,y,t)=exact_solution_eq((x,y),t,nothing)

bc = Trixi.BoundaryConditionDirichlet(exact_solution_eq)
boundary_conditions = Dict( :Slant  => bc,
                            :Bezier => bc,
                            :Right  => bc,
                            :Bottom => bc,
                            :Top    => bc )

initial_value_convergence_test(x,y)= exact_solution(x,y,0.0)

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
                  final_time, exact_solution,
                  boundary_conditions)
limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux,
                2)
param = Parameters([grid_size[1], grid_size[2]], cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate, cfl_safety_factor = cfl_safety_factor,
                   time_scheme = "RK11")
default_mesh_file = joinpath(@__DIR__, "mesh_trixi_unstructured_mesh_docs.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/52056f1487853fab63b7f4ed7f171c80/raw/9d573387dfdbb8bce2a55db7246f4207663ac07f/mesh_trixi_unstructured_mesh_docs.mesh",
                                       default_mesh_file)
mesh_file = default_mesh_file

grid = UnstructuredMesh2D(mesh_file)

#------------------------------------------------------------------------------
# problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
#                                           grid, ARGS)
#------------------------------------------------------------------------------
rm("output", force = true, recursive = true)
sol = Tenkai.solve(equation, grid, problem, scheme, param);
Trixi.save_mesh_file(grid, "output")
trixi2vtk("output/sol*.h5", output_directory="output")
println(sol["errors"])

return sol;
