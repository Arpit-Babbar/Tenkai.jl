using Tenkai: set_initial_condition!,
              compute_cell_average!,
              compute_face_residual!,
              write_soln!,
              compute_error,
              post_process_soln,
              modal_smoothness_indicator, # KLUDGE - This shouldn't be here
              AbstractEquations, ssfr, BFluxType,
              cRKSolver, AbstractDissipation
using Tenkai: apply_limiter!
using Tenkai
using Tenkai: VolumeIntegralWeak
using SimpleNonlinearSolve
using StaticArrays
using SimpleUnPack
using Printf
using LinearAlgebra: axpy!, dot
using Accessors: @set
using TimerOutputs
using MuladdMacro

import Tenkai: apply_limiter!, compute_time_step, adjust_time_step,
               pre_process_limiter!, get_cfl, save_solution, calc_source,
               update_solution_cRK!, multiply_add_to_node_vars!, get_node_vars

import Tenkai: initialize_solution!, evolve_solution!, Scheme

struct EmptyEquations <: AbstractEquations{0, 0} end

const EMPTY_EQUATIONS = EmptyEquations()

# cRK stuff in Tenkai. jl
using Tenkai: cRKSolver, AbstractDissipation, cRK64, cRK44, cRK33, cRK22, cRK11

# Functions used by implicit solvers

function picard_iteration(func, y)
    F = func(y) # Update F
    stepsize = 0.1 # TODO - Adaptively choose the stepsize
    y = y - stepsize * F
    return y, norm(F)
end

function picard_solver(func, y0, tol = 1e-12)
    y = y0
    n_iters = 0
    y, norm_F = picard_iteration(func, y)
    while norm_F > tol && n_iters < 10000
        y, norm_F = picard_iteration(func, y)
        n_iters += 1
    end
    if norm_F > 1e-8
        println("Picard solver did not converge and res = $norm_F")
    end

    return y
end

function newton_solver(func, y0, tol = 1e-14, maxiters = 1e3)
    p = nothing # The func doesn't have any parameters
    f = (x, p) -> func(x)
    prob = NonlinearProblem{false}(f, y0, p)
    sol = solve(prob, SimpleNewtonRaphson(), abstol = tol, reltol = tol, verbose = true,
                maxiters = maxiters)
    return sol.u
end

struct cIMEX111 <: cRKSolver end

# Two-stage second order IMEX scheme with one implicit solve
struct cHT112{ImplicitSolver, VolumeIntegral} <: cRKSolver
    implicit_solver::ImplicitSolver
    volume_integral::VolumeIntegral
end

# Constructor for cHT112 with a default implicit solver
function cHT112(; implicit_solver = newton_solver,
                volume_integral = VolumeIntegralWeak())
    return cHT112{typeof(implicit_solver), typeof(volume_integral)}(implicit_solver,
                                                                    volume_integral)
end

struct cHT112Explicit <: cRKSolver end # Above scheme treating source terms as explicit
struct cSSP2IMEX222{ImplicitSolver, VolumeIntegral} <: cRKSolver
    implicit_solver::ImplicitSolver
    volume_integral::VolumeIntegral
end # Two-stage second order IMEX scheme with two implicit solves

# Constructor for cSSP2IMEX222 with a default implicit solver
function cSSP2IMEX222(; implicit_solver = newton_solver,
                      volume_integral = VolumeIntegralWeak())
    return cSSP2IMEX222{typeof(implicit_solver), typeof(volume_integral)}(implicit_solver,
                                                                          volume_integral)
end

struct cARS222{ImplicitSolver, VolumeIntegral} <: cRKSolver
    implicit_solver::ImplicitSolver
    volume_integral::VolumeIntegral
end # Two-stage second order ARS scheme with two implicit solves

# Constructor for cARS222 with a default implicit solver
function cARS222(; implicit_solver = newton_solver,
                 volume_integral = VolumeIntegralWeak())
    return cARS222{typeof(implicit_solver), typeof(volume_integral)}(implicit_solver,
                                                                     volume_integral)
end

struct cBPR343{ImplicitSolver, VolumeIntegral} <: cRKSolver
    implicit_solver::ImplicitSolver
    volume_integral::VolumeIntegral
end # Three-stage third order BRB scheme with three implicit solves

# Constructor for cBPR343 with a default implicit solver
function cBPR343(; implicit_solver = newton_solver,
                 volume_integral = VolumeIntegralWeak())
    return cBPR343{typeof(implicit_solver), typeof(volume_integral)}(implicit_solver,
                                                                     volume_integral)
end

struct cAGSA343{ImplicitSolver, VolumeIntegral} <: cRKSolver
    implicit_solver::ImplicitSolver
    volume_integral::VolumeIntegral
end # Three-stage third order AGSA scheme with three implicit solves

# Constructor for cAGSA343 with a default implicit solver
function cAGSA343(; implicit_solver = newton_solver,
                  volume_integral = VolumeIntegralWeak())
    return cAGSA343{typeof(implicit_solver), typeof(volume_integral)}(implicit_solver,
                                                                      volume_integral)
end

struct cSSP2IMEX332{ImplicitSolver, VolumeIntegral} <: cRKSolver
    implicit_solver::ImplicitSolver
    volume_integral::VolumeIntegral
end # Three-stage second order IMEX scheme with three implicit solves

function cSSP2IMEX332(; implicit_solver = newton_solver,
                      volume_integral = VolumeIntegralWeak())
    return cSSP2IMEX332{typeof(implicit_solver), typeof(volume_integral)}(implicit_solver,
                                                                          volume_integral)
end

struct cSSP2IMEX433{ImplicitSolver, VolumeIntegral} <: cRKSolver
    implicit_solver::ImplicitSolver
    volume_integral::VolumeIntegral
end # Four-stage third order IMEX scheme with four implicit solves

function cSSP2IMEX433(; implicit_solver = newton_solver,
                      volume_integral = VolumeIntegralWeak())
    return cSSP2IMEX433{typeof(implicit_solver), typeof(volume_integral)}(implicit_solver,
                                                                          volume_integral)
end

function get_cache_node_vars(aux, u1, problem, scheme, eq, i, cell)
    u_node = get_node_vars(u1, eq, i, cell)
    return u_node
end

# SolverSource will be the actual solver with source terms, while
# SolverHomogeneous will be the solver without source terms.
struct DoublecRKSourceSolver{SolverSource <: cRKSolver} <: cRKSolver
    single_crk_solver::SolverSource
end

# TODO - Bad practice

non_conservative_equation(eq::AbstractEquations) = EMPTY_EQUATIONS

# This will compute the term to be differentiated. Its size should be known
# at the compile time by using the `nvariables` function.
calc_non_cons_gradient(u_node, x, t, eq::AbstractEquations{1}) = zero(u_node)

calc_non_cons_gradient(u_node, x, y, t, eq::AbstractEquations{2}) = zero(u_node)

# This will compute the action of B on u_non_cons. The u_non_cons may
# be the derivative or it may not. Both quantities need to be computed.
calc_non_cons_Bu(u_node, u_non_cons, x, t, eq::AbstractEquations{1}) = zero(u_node)

calc_non_cons_Bu(u_node, u_non_cons, x, y, t, eq::AbstractEquations{2}) = zero(u_node)

function calc_non_cons_B(u, x_, t, eq::AbstractEquations{1})
    @assert false "Only for non-conservative equations"
end
#--------------------------------------------------------------------------
# Double cRK solver methods
#--------------------------------------------------------------------------

# A scheme function with DoublecRKSourceSolver will just contain the standard
# scheme object within it in the cache subfield
function Scheme(solver::DoublecRKSourceSolver{<:cRKSolver},
                degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux::BFluxType,
                dissipation = "2"; cache = (;))
    # KLUDGE - The dissipation should have been a key word argument, but
    # I don't want to change all the run files now.
    scheme_single_solver = Scheme(solver.single_crk_solver, degree, solution_points,
                                  correction_function, numerical_flux, bound_limit,
                                  limiter, bflux, dissipation; cache = cache)

    cache = (; cache..., scheme_single_solver = scheme_single_solver)

    # scheme = @set scheme_single_solver.solver = solver
    # scheme = @set scheme.cache = cache

    # The @set macro does not work because the Scheme type has (degree + 1)^2 which cannot
    # be set automatically. This is bad design.
    scheme = Scheme{typeof(solver), typeof(scheme_single_solver.dissipation),
                    typeof(scheme_single_solver.numerical_flux),
                    typeof(scheme_single_solver.limiter),
                    typeof(scheme_single_solver.bflux), typeof(cache),
                    scheme_single_solver.degree + 1,
                    (scheme_single_solver.degree + 1)^2}(solver,
                                                         scheme_single_solver.solver_enum,
                                                         scheme_single_solver.degree,
                                                         scheme_single_solver.solution_points,
                                                         scheme_single_solver.correction_function,
                                                         scheme_single_solver.numerical_flux,
                                                         scheme_single_solver.bound_limit,
                                                         scheme_single_solver.limiter,
                                                         scheme_single_solver.bflux,
                                                         scheme_single_solver.dissipation,
                                                         cache)

    return scheme
end

import Tenkai: create_auxiliaries
# TODO - Remove "Tenkai." from here
function Tenkai.create_auxiliaries(eq, op, grid, problem,
                                   scheme::Scheme{<:DoublecRKSourceSolver{<:cRKSolver}},
                                   param, cache_source)
    @unpack scheme_single_solver = scheme.cache

    aux = create_auxiliaries(eq, op, grid, problem, scheme_single_solver, param,
                             cache_source)

    cache_homogeneous = deepcopy(cache_source)
    problem_homogeneous = @set problem.source_terms = nothing

    aux = (; aux..., cache_homogeneous, problem_homogeneous, cache_source)

    return aux
end

function initialize_solution!(eq, grid, op, problem,
                              scheme::Scheme{<:DoublecRKSourceSolver{<:cRKSolver}}, param,
                              aux,
                              cache)
    @unpack cache_homogeneous, problem_homogeneous = aux

    @unpack scheme_single_solver = scheme.cache

    # Do the homogeneous part first as it is used by non-homogeneous
    initialize_solution!(eq, grid, op, problem_homogeneous, scheme_single_solver, param,
                         aux,
                         cache_homogeneous)

    initialize_solution!(eq, grid, op, problem, scheme_single_solver, param, aux, cache)

    return nothing
end

function evolve_solution!(eq, grid, op, problem,
                          scheme::Scheme{<:DoublecRKSourceSolver{<:cRKSolver}},
                          param, aux, iter, t, dt, fcount, cache)
    @unpack cache_homogeneous, problem_homogeneous = aux

    @unpack scheme_single_solver = scheme.cache

    # Svard's paper suggests using old homogeneous solution as the initial guess
    evolve_solution!(eq, grid, op, problem, scheme_single_solver, param, aux, iter, t, dt,
                     fcount,
                     cache)

    evolve_solution!(eq, grid, op, problem_homogeneous, scheme_single_solver, param, aux,
                     iter, t,
                     dt, fcount, cache_homogeneous)

    return nothing
end

@inline function multiply_add_to_node_vars!(B::AbstractArray,
                                            factor::Real, B_node::SMatrix{<:Any},
                                            equations::AbstractEquations,
                                            equations_nc::AbstractNonConservativeEquations,
                                            indices...)
    for v_nc in eachvariable(equations_nc), v in eachvariable(equations)
        B[v, v_nc, indices...] = B[v, v_nc, indices...] + factor * B_node[v, v_nc]
    end
    return nothing
end

@inline function get_node_vars(u, eq::AbstractEquations,
                               eq_nc::AbstractNonConservativeEquations,
                               indices...)
    u_ = @view u[:, :, indices...]
    nvar = nvariables(eq)
    nvar_nc = nvariables(eq_nc)
    SMatrix{nvar, nvar_nc}(ntuple(@inline(v->u_[v]), Val(nvar * nvar_nc)))
end
