using Trixi: Trixi, TreeMesh, StructuredMesh, UnstructuredMesh2D, P4estMesh, T8codeMesh,
             True,
             False, DGSEM, eachnode, nnodes
abstract type AbstractTrixiSolver <: AbstractRKSolver end

struct TrixiRKSolver{RKSolver} <: AbstractTrixiSolver
    RKSolver::RKSolver
end

solver2enum(solver::TrixiRKSolver) = rktrixi # solver type enum

include(joinpath(@__DIR__, "RKTrixi1D.jl"))
include(joinpath(@__DIR__, "RKTrixi2D.jl"))
