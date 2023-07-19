using Tenkai
Eq = Tenkai.EqLinAdv2D
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

function exact_solution_eq1(x, t::Real, eq::Eq.LinAdv2D)
   # Store translated coordinate for easy use of exact solution
   return SVector(1.0)
   # x_trans = x - eq.velocity(x[1],x[2]) * t
   # c = 1.0
   # A = 0.5
   # L = 2
   # f = 1/L
   # omega = 2 * pi * f
   # scalar = c + A * sin(omega * sum(x_trans))
   # return SVector(scalar)
end

velocity(x, y) = SVector(0.2, -0.7)
equation = Eq.get_equation(velocity)
exact_solution1(x::Real, y::Float64, t::Float64) = exact_solution_eq1((x, y), t, equation)
initial_value(x::Real, y::Real) = exact_solution1(x, y, 0.0)

bc = Trixi.BoundaryConditionDirichlet(exact_solution_eq1)
boundary_conditions = Dict( :Body    => bc,
                            :Button1 => bc,
                            :Button2 => bc,
                            :Eye1    => bc,
                            :Eye2    => bc,
                            :Smile   => bc,
                            :Bowtie  => bc )

initial_value(x,y)= exact_solution(x,y,0.0)

degree              = 6
solver              = "rkfr"
solution_points     = "gll"
correction_function = "g2"
numerical_flux      = Tenkai.flux_lax_friedrichs
bound_limit         = "no"
bflux               = extrapolate
final_time          = 0.1 # 20 * sqrt(2.0) / 0.5

nx, ny = 32, 32
cfl = 0.1
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.01 # final_time / 5.0
cfl_safety_factor = 0.1
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

#------------------------------------------------------------------------------
grid_size = (nx, ny)
domain_filler = [xmin, xmax, ymin, ymax]
problem = Problem(domain_filler, initial_value,
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
default_mesh_file = joinpath(@__DIR__, "mesh_gingerbread_man.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/2c6440b5f8a57db131061ad7aa78ee2b/raw/1f89fdf2c874ff678c78afb6fe8dc784bdfd421f/mesh_gingerbread_man.mesh",
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
