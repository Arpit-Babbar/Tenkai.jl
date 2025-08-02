abstract type AbstractTrixiSolver <: AbstractRKSolver end

struct TrixiRKSolver{RKSolver} <: AbstractTrixiSolver
    RKSolver::RKSolver
end

solver2enum(solver::TrixiRKSolver) = rktrixi # solver type enum
