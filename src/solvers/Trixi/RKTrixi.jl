import Trixi
abstract type AbstractTrixiSolver <: AbstractRKSolver end

struct TrixiRKSolver{RKSolver} <: AbstractTrixiSolver
    RKSolver::RKSolver
end

solver2enum(solver::TrixiRKSolver) = rktrixi # solver type enum

include(joinpath(@__DIR__, "RKTrixi1D.jl"))
include(joinpath(@__DIR__, "RKTrixi2D.jl"))

function tenkai2trixiequation(equation::EqEuler1D.Euler1D)
    Trixi.CompressibleEulerEquations1D(equation.gamma)
end

function tenkai2trixiequation(equations::EqMHD1D.MHD1D)
    equations.trixi_equations
end

function tenkai2trixisemi(solver::TrixiRKSolver, equation, problem, scheme, param)
    @unpack grid_size = param
    @assert ispow2(grid_size) "Grid size must be a power of 2 for TreeMesh."
    @assert scheme.solution_points == "gll" "Only GLL solution points are supported for Trixi."
    @assert scheme.correction_function == "g2" "Only G2 correction function is supported for Trixi."
    trixi_equations = tenkai2trixiequation(equation)
    initial_condition(x, t, equations) = problem.exact_solution(x..., t)
    dg_solver = Trixi.DGSEM(polydeg = scheme.degree, surface_flux = Trixi.flux_lax_friedrichs,
                             volume_integral = Trixi.VolumeIntegralWeakForm())
    mesh = Trixi.TreeMesh(problem.domain[1], problem.domain[2],
                          initial_refinement_level = Int(log2(grid_size)),
                          n_cells_max = 10000000)
    semi = Trixi.SemidiscretizationHyperbolic(mesh, trixi_equations, initial_condition, dg_solver)
    tspan = (0.0, problem.final_time)
    ode = Trixi.semidiscretize(semi, tspan)
end
