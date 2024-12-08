using ..Basis
using ..CartesianGrids
using ..Equations: AbstractEquations, nvariables, eachvariable

import Tenkai

using OffsetArrays # OffsetArray, OffsetMatrix, OffsetVector
using LinearAlgebra
using Printf
using WriteVTK
using FLoops
using Polyester
using TimerOutputs
using StaticArrays
using UnPack
using Plots
using DelimitedFiles
using LoopVectorization
using MuladdMacro
using JSON3

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Temporary hack
using FastGaussQuadrature

@enum BCType periodic dirichlet neumann reflect hllc_bc
@enum BFluxType extrapolate evaluate
@enum SolverType rkfr lwfr mdrk ssfr # TODO - Is this only needed for D2 / D1?

#-------------------------------------------------------------------------------
# Create a struct of problem description
#-------------------------------------------------------------------------------
struct Problem{F1, F2, F3 <: Function, BoundaryCondition <: Tuple, SourceTerms}
    domain::Vector{Float64}
    initial_value::F1
    boundary_value::F2
    boundary_condition::BoundaryCondition
    source_terms::SourceTerms
    periodic_x::Bool
    periodic_y::Bool
    final_time::Float64
    exact_solution::F3
end

# Constructor
function Problem(domain::Vector{Float64},
                 initial_value::Function,
                 boundary_value::Function,
                 boundary_condition::Tuple,
                 final_time::Float64,
                 exact_solution::Function;
                 source_terms = nothing)
    if length(domain) == 2
        @assert length(boundary_condition)==2 "Invalid Problem"
        left, right = boundary_condition
        if (left == periodic && right != periodic)
            println("Incorrect use of periodic bc")
            @assert false
        elseif left == periodic && right == periodic
            periodic_x = true
        else
            periodic_x = false
        end
        return Problem(domain, initial_value, boundary_value, boundary_condition, source_terms,
                       periodic_x,
                       false, # Put dummy place holder for periodic_y
                       final_time, exact_solution)
    elseif length(domain) == 4
        @assert length(boundary_condition)==4 "Invalid Problem"
        left, right, bottom, top = boundary_condition

        if ((left == periodic && right != periodic) ||
            (left != periodic && right == periodic))
            println("Incorrect use of periodic bc")
            @assert false
        elseif left == periodic && right == periodic
            periodic_x = true
        else
            periodic_x = false
        end

        if ((bottom == periodic && top != periodic) ||
            (bottom != periodic && top == periodic))
            println("Incorrect use of periodic bc")
            @assert false
        elseif bottom == periodic && top == periodic
            periodic_y = true
        else
            periodic_y = false
        end
        return Problem(domain, initial_value, boundary_value, boundary_condition, source_terms,
                       periodic_x, periodic_y, final_time, exact_solution)
    else
        @assert false, "Invalid domain"
    end
end

#-------------------------------------------------------------------------------
# Create a struct of scheme description
#-------------------------------------------------------------------------------
struct Scheme{Solver, NumericalFlux, Limiter, BFlux <: NamedTuple{<:Any, <:Any},
              Dissipation}
    solver::Solver
    solver_enum::SolverType
    degree::Int64
    solution_points::String
    correction_function::String
    numerical_flux::NumericalFlux
    bound_limit::String
    limiter::Limiter
    bflux::BFlux
    dissipation::Dissipation
end

function solver2enum(solver)
    if solver == "lwfr"
        solver_enum = lwfr
    elseif solver == "rkfr"
        solver_enum = rkfr
    elseif solver == "mdrk"
        solver_enum = mdrk
    end
end

function diss_arg2method(dissipation)
    @assert dissipation in ("1", "2", 1, 2,
                            get_first_node_vars, # kludge
                            get_second_node_vars) "dissipation = $dissipation"
    if dissipation in ("1", 1)
        get_dissipation_node_vars = get_first_node_vars
    else
        @assert dissipation in ("2", 2)
        get_dissipation_node_vars = get_second_node_vars
    end
    return get_dissipation_node_vars
end

# Constructor
function Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux::BFluxType,
                dissipation = "2")
    bflux_data = (; bflux_ind = bflux,
                  compute_bflux! = get_bflux_function(solver, degree, bflux))
    solver_enum = solver2enum(solver)
    get_dissipation_node_vars = diss_arg2method(dissipation)

    Scheme(solver, solver_enum, degree, solution_points, correction_function,
           numerical_flux, bound_limit, limiter, bflux_data,
           get_dissipation_node_vars)
end

function Scheme(solver, solver_enum::SolverType, degree, solution_points,
                correction_function,
                numerical_flux, bound_limit, limiter, bflux::BFluxType,
                dissipation = "2")
    bflux_data = (; bflux_ind = bflux,
                  compute_bflux! = get_bflux_function(solver, degree, bflux))
    get_dissipation_node_vars = diss_arg2method(dissipation)

    Scheme(solver, solver_enum, degree, solution_points, correction_function,
           numerical_flux, bound_limit, limiter, bflux_data,
           get_dissipation_node_vars)
end

trivial_function(x) = nothing

function get_bflux_function(solver, degree, bflux)
    if solver == "rkfr"
        return trivial_function
    elseif solver == "mdrk"
        if bflux == extrapolate
            return extrap_bflux!
        else
            eval_bflux! = Tenkai.eval_bflux_mdrk!
            return eval_bflux!
        end
    else
        if bflux == extrapolate
            return extrap_bflux!
        else
            if degree == 0
                return extrap_bflux!
            elseif degree == 1
                return eval_bflux1!
            elseif degree == 2
                return eval_bflux2!
            elseif degree == 3
                return eval_bflux3!
            elseif degree == 4
                return eval_bflux4!
            elseif degree == 5
                return eval_bflux5!
            else
                @assert false "Incorrect degree"
            end
        end
    end
end

#------------------------------------------------------------------------------
# Create a struct of parameters
#------------------------------------------------------------------------------
struct Parameters{T1 <: Union{Int64, Vector{Int64}}}
    grid_size::T1
    cfl::Float64
    bounds::Tuple{Vector{Float64}, Vector{Float64}}
    save_iter_interval::Int64
    save_time_interval::Float64
    compute_error_interval::Int64
    animate::Bool
    saveto::String      # Directory where a copy of output will be sent
    time_scheme::String # Time integration used by Runge-Kutta
    cfl_safety_factor::Float64
    cfl_style::String
    eps::Float64
end

# Constructor
function Parameters(grid_size, cfl, bounds, save_iter_interval,
                    save_time_interval, compute_error_interval;
                    animate = false, cfl_safety_factor = 0.98,
                    time_scheme = "by degree",
                    saveto = "none",
                    cfl_style = "optimal",
                    eps = 1e-12)
    @assert (cfl>=0.0) "cfl must be >= 0.0"
    @assert (save_iter_interval>=0) "save_iter_interval must be >= 0"
    @assert (save_time_interval>=0.0) "save_time_interval must be >= 0.0"
    @assert (!(save_iter_interval > 0 &&
               save_time_interval > 0.0)) "Both save_(iter,time)_interval > 0"
    @assert cfl_style in ["lw", "optimal"]

    Parameters(grid_size, cfl, bounds, save_iter_interval,
               save_time_interval, compute_error_interval, animate,
               saveto, time_scheme, cfl_safety_factor, cfl_style, eps)
end


#------------------------------------------------------------------------------
# A struct which gives zero whenever you try to index it as a zero
#------------------------------------------------------------------------------
struct EmptyZeros{RealT <: Real} end
@inline Base.getindex(::EmptyZeros{RealT}, i...) where RealT = zero(RealT)
EmptyZeros(RealT) = EmptyZeros{RealT}()
EmptyZeros() = EmptyZeros{Float64}()

#------------------------------------------------------------------------------
# Methods which need to be defined in Equation modules
#------------------------------------------------------------------------------
flux(x, u, eq) = @assert false "method not defined for equation"
prim2con(eq, prim) = @assert false "method not defined for equation"
prim2con!(eq, prim) = @assert false "method not defined for equation"
prim2con!(eq, prim, U) = @assert false "method not defined for equation"
con2prim(eq, U) = @assert false "method not defined for equation"
con2prim!(eq, U) = @assert false "method not defined for equation"
con2prim!(eq, U, prim) = @assert false "method not defined for equation"
eigmatrix(eq, u) = @assert false "method not defined for equation"
fo_blend(eq) = @assert false "method not defined for equation"
mh_blend(eq) = @assert false "method not defined for equation"
no_upwinding_x() = @assert false "method not defined"

#------------------------------------------------------------------------------
# First order source term functions needed by blending and (later) RKFR
#------------------------------------------------------------------------------

# If there is no source_term, there is a nothing object in its place.
# With that information, we can use multiple dispatch to create a source term function which
# are zero functions when there is no source terms (i.e., it is a Nothing object)

function calc_source(u, x, t, source_terms, eq::AbstractEquations)
    return source_terms(u, x, t, eq)
end

function calc_source(u, x, t, source_terms::Nothing, eq::AbstractEquations)
    return zero(u)
end

#-------------------------------------------------------------------------------
# Apply given command line arguments
#-------------------------------------------------------------------------------
# TODO - Remove this function
function ParseCommandLine(problem, param, scheme, eq, args;
                          limiter_ = nothing, numerical_flux_ = nothing,
                          initial_value_ = nothing)
    return problem, scheme, param
end

#-------------------------------------------------------------------------------
# Defining threads, as done by Trixi.jl, see
# https://github.com/trixi-framework/Trixi.jl/blob/main/src/auxiliary/auxiliary.jl#L177
# for things to keep in mind
#-------------------------------------------------------------------------------
macro threaded(expr)
    return esc(quote
                   let
                       if Threads.nthreads() == 1
                           $(expr)
                       else
                           Threads.@threads $(expr)
                       end
                   end
               end)

    # Use this for a single threaded code without restarting REPL
    # return esc(quote
    #   let
    #      $(expr)
    #   end
    # end)

    # Use this for Polyester threads
    #  return esc(quote Polyester.@batch $(expr) end) #  < - Polyester threads, to be tested
end

#-------------------------------------------------------------------------------
# C = a1 * A1 * B1 + a2 * A2 * B2
# Not used. Remove?
#-------------------------------------------------------------------------------
@inline function gemm!(a1, A1, B1, a2, A2, B2, C)
    mul!(C, A1, B1)         # C = A1 * B1
    mul!(C, A2, B2, a2, a1) # C = a1 * C + a2 * A2 * B2
    return nothing
end

#------------------------------------------------------------------------------
# Static array operations
#------------------------------------------------------------------------------

@inline function get_node_vars(u, eq, indices...)
    SVector(ntuple(@inline(v->u[v, indices...]), Val(nvariables(eq))))
end

@inline function get_first_node_vars(u1, u2, eq, indices...)
    SVector(ntuple(@inline(v->u1[v, indices...]), Val(nvariables(eq))))
end

@inline function get_second_node_vars(u1, u2, eq, indices...)
    SVector(ntuple(@inline(v->u2[v, indices...]), Val(nvariables(eq))))
end

@inline function set_node_vars!(u, u_node::SVector{<:Any}, eq, indices...)
    for v in eachvariable(eq)
        u[v, indices...] = u_node[v]
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor::Real, u_node::SVector{<:Any},
                                            equations::AbstractEquations,
                                            indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + factor * u_node[v]
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             factor::Real, u_node::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = factor * u_node[v]
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             factor1::Real, u_node1::SVector{<:Any},
                                             factor2::Real, u_node2::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = factor1 * u_node1[v] + factor2 * u_node2[v]
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             factor::Real,
                                             factor1::Real, u_node1::SVector{<:Any},
                                             factor2::Real, u_node2::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = factor * (factor1 * u_node1[v] + factor2 * u_node2[v])
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             factor::Real,
                                             factor1::Real, u_node1::SVector{<:Any},
                                             factor2::Real, u_node2::SVector{<:Any},
                                             factor3::Real, u_node3::SVector{<:Any},
                                             factor4::Real, u_node4::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = factor * (factor1 * u_node1[v] + factor2 * u_node2[v]
                            + factor3 * u_node3[v] + factor4 * u_node4[v])
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             factor1::Real, u_node1::SVector{<:Any},
                                             factor2::Real, u_node2::SVector{<:Any},
                                             factor3::Real, u_node3::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = (factor1 * u_node1[v]
                            + factor2 * u_node2[v]
                            + factor3 * u_node3[v])
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             u_node_factor_less1::SVector{<:Any},
                                             factor::Real, u_node::SVector{<:Any},
                                             u_node_factor_less2::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u_node_factor_less1[v] + factor * u_node[v] +
                           u_node_factor_less2[v]
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             u_node_factor_less::SVector{<:Any},
                                             factor1::Real,
                                             u_node1::SVector{<:Any},
                                             factor2::Real,
                                             u_node2::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u_node_factor_less[v] + (factor1 * u_node1[v]
                            +
                            factor2 * u_node2[v])
    end
    return nothing
end

@inline function multiply_add_set_node_vars!(u::AbstractArray,
                                             factor::Real,
                                             factor1::Real, u_node1::SVector{<:Any},
                                             factor2::Real, u_node2::SVector{<:Any},
                                             factor3::Real, u_node3::SVector{<:Any},
                                             factor4::Real, u_node4::SVector{<:Any},
                                             factor5::Real, u_node5::SVector{<:Any},
                                             equations::AbstractEquations,
                                             indices...)
    for v in eachvariable(equations)
        u[v, indices...] = factor * (factor1 * u_node1[v] + factor2 * u_node2[v]
                            + factor3 * u_node3[v] + factor4 * u_node4[v]
                            + factor5 * u_node5[v])
    end
    return nothing
end

@inline function add_to_node_vars!(u::AbstractArray, u_node::SVector{<:Any},
                                   equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + u_node[v]
    end
    return nothing
end

@inline function subtract_from_node_vars!(u::AbstractArray,
                                          u_node::SVector{<:Any},
                                          equations::AbstractEquations,
                                          indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] - u_node[v]
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            factor2::Real, u_node2::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + factor1 * u_node1[v] +
                           factor2 * u_node2[v]
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor::Real,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            factor2::Real, u_node2::SVector{<:Any},
                                            factor3::Real, u_node3::SVector{<:Any},
                                            factor4::Real, u_node4::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] +
                           factor * (factor1 * u_node1[v]
                            + factor2 * u_node2[v]
                            + factor3 * u_node3[v]
                            + factor4 * u_node4[v])
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor::Real,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            factor2::Real, u_node2::SVector{<:Any},
                                            factor3::Real, u_node3::SVector{<:Any},
                                            factor4::Real, u_node4::SVector{<:Any},
                                            factor5::Real, u_node5::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] +
                           factor * (factor1 * u_node1[v]
                            + factor2 * u_node2[v]
                            + factor3 * u_node3[v]
                            + factor4 * u_node4[v]
                            + factor5 * u_node5[v])
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            factor2::Real, u_node2::SVector{<:Any},
                                            factor3::Real, u_node3::SVector{<:Any},
                                            factor4::Real, u_node4::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + (factor1 * u_node1[v]
                            + factor2 * u_node2[v]
                            + factor3 * u_node3[v]
                            + factor4 * u_node4[v])
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            factor2::Real, u_node2::SVector{<:Any},
                                            factor3::Real, u_node3::SVector{<:Any},
                                            factor4::Real, u_node4::SVector{<:Any},
                                            factor5::Real, u_node5::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + (factor1 * u_node1[v]
                            + factor2 * u_node2[v]
                            + factor3 * u_node3[v]
                            + factor4 * u_node4[v]
                            + factor5 * u_node5[v])
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            factor2::Real, u_node2::SVector{<:Any},
                                            factor3::Real, u_node3::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + (factor1 * u_node1[v]
                            + factor2 * u_node2[v]
                            + factor3 * u_node3[v])
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            u_node_factor_less::SVector{<:Any},
                                            factor1::Real, u_node1::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + u_node_factor_less[v] +
                           factor1 * u_node1[v]
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            u_node2::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + factor1 * (u_node1[v] + u_node2[v])
    end
    return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor1::Real, u_node1::SVector{<:Any},
                                            factor2::Real, u_node2::SVector{<:Any},
                                            u_node3::SVector{<:Any},
                                            equations::AbstractEquations, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = (u[v, indices...] + factor1 * u_node1[v]
                            + factor2 * (u_node2[v]
                                         +
                                         u_node3[v]))
    end
    return nothing
end

@inline function comp_wise_mutiply_node_vars!(u::AbstractArray,
                                              u_node::SVector{<:Any},
                                              equations::AbstractEquations,
                                              indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] * u_node[v]
    end
    return nothing
end

function zhang_shu_flux_fix(eq::AbstractEquations,
                            uprev,    # Solution at previous time level
                            ulow,     # low order update
                            Fn,       # Blended flux candidate
                            fn_inner, # Inner part of flux
                            fn,       # low order flux
                            c)
    # This method is to be defined for each equation
    return Fn
end

function admissibility_tolerance(eq::AbstractEquations)
    return 0.0
end

#-------------------------------------------------------------------------------
# Set up arrays
#------------------------------------------------------------------------------
function setup_arrays(grid, scheme, equation)
    @unpack solver = scheme
    if solver == "lwfr"
        return setup_arrays_lwfr(grid, scheme, equation)
    elseif solver == "rkfr"
        return setup_arrays_rkfr(grid, scheme, equation)
    elseif solver == "mdrk"
        return setup_arrays_mdrk(grid, scheme, equation)
    else
        @assert false "Incorrect solver"
    end
end

# Construct `cache_size` number of objects with `constructor`
# and store them in an SVector
function alloc(constructor, cache_size)
    SVector{cache_size}(constructor(undef) for _ in Base.OneTo(cache_size))
end

# Create the result of `alloc` for each thread. Basically,
# for each thread, construct `cache_size` number of objects with
# `constructor` and store them in an SVector
function alloc_for_threads(constructor, cache_size)
    nt = Threads.nthreads()
    SVector{nt}([alloc(constructor, cache_size) for _ in Base.OneTo(nt)])
end

#-------------------------------------------------------------------------------
# Create tuple of auxiliary objects created from user supplied data which are
# not a core part of the LWFR/RKFR scheme
#-------------------------------------------------------------------------------
function create_auxiliaries(eq, op, grid, problem, scheme, param, cache)
    # Setup plotting
    @unpack u1, ua = cache
    timer = TimerOutput()
    plot_data = initialize_plot(eq, op, grid, problem, scheme, timer, u1, ua)
    # Setup blending limiter
    blend = Tenkai.Blend(eq, op, grid, problem, scheme, param, plot_data)
    hierarchical = Tenkai.Hierarchical(eq, op, grid, problem, scheme, param,
                                       plot_data)
    aux_cache = create_aux_cache(eq, op)
    error_file = open("error.txt", "w")
    aux = (; plot_data, blend,
           hierarchical,
           error_file, timer,
           aux_cache) # named tuple;
    return aux
end

#-------------------------------------------------------------------------------
# Limiter functions
#-------------------------------------------------------------------------------
function setup_limiter_none()
    limiter = (; name = "none")
    return limiter
end

function setup_limiter_blend(; blend_type, indicating_variables,
                             reconstruction_variables, indicator_model,
                             smooth_alpha = true, smooth_factor = 0.5,
                             amax = 1.0, constant_node_factor = 1.0,
                             constant_node_factor2 = 1.0,
                             a = 0.5, c = 1.8, amin = 0.001,
                             debug_blend = false, super_debug = false,
                             pure_fv = false,
                             bc_x = no_upwinding_x, tvbM = 0.0,
                             numflux = nothing)
    limiter = (; name = "blend", blend_type, indicating_variables,
               reconstruction_variables, indicator_model,
               amax, smooth_alpha, smooth_factor,
               constant_node_factor, constant_node_factor2,
               a, c, amin,
               super_debug, debug_blend, pure_fv, bc_x, tvbM, numflux)
    return limiter
end

function setup_limiter_hierarchical(; alpha,
                                    reconstruction = conservative_reconstruction)
    limiter = (; name = "hierarchical", alpha, reconstruction)
    return limiter
end

function apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    @timeit aux.timer "Limiter" begin
    #! format: noindent
    # TOTHINK - (very small) allocations happening outside limiter funcs?
    # Are they just because of the assertion in tvbβ or is there a worrisome cause?
    @unpack limiter = scheme
    if limiter.name == "tvb"
        apply_tvb_limiter!(eq, problem, scheme, grid, param, op, ua, u1, aux)
    elseif limiter.name == "tvbβ"
        apply_tvb_limiterβ!(eq, problem, scheme, grid, param, op, ua, u1, aux)
    elseif limiter.name == "hierarchical"
        apply_hierarchical_limiter!(eq, problem, scheme, grid, param, op, ua, u1,
                                    aux)
    end
    apply_bound_limiter!(eq, grid, scheme, param, op, ua, u1, aux)
    return nothing
    end # timer
end

function minmod(a, b, c, Mdx2)
    slope = min(abs(a), abs(b), abs(c))
    if abs(a) < Mdx2
        return a
    end
    s1, s2, s3 = sign(a), sign(b), sign(c)
    if (s1 != s2) || (s2 != s3)
        return zero(a)
    else
        slope = s1 * slope
        return slope
        # slope = s1 * min(abs(a),abs(b),abs(c))
        # return slope
    end
end

function minmod(a, b, c, beta, Mdx2)
    # beta = 1.0
    slope = min(abs(a), beta * abs(b), beta * abs(c))
    if abs(a) < Mdx2
        return a
    end
    s1, s2, s3 = sign(a), sign(b), sign(c)
    if (s1 != s2) || (s2 != s3)
        return zero(a)
    else
        slope = s1 * slope
        return slope
        # slope = s1 * min(abs(a),abs(b),abs(c))
        # return slope
    end
end

function finite_differences(h1, h2, ul, u, ur)
    # TODO - This is only be needed for GLL points. Use multiple dispatch maybe
    fwd_diff = (ur - u) / h2
    back_diff = (u - ul) / h1
    a, b, c = -(h2 / (h1 * (h1 + h2))), (h2 - h1) / (h1 * h2), (h1 / (h2 * (h1 + h2)))
    cent_diff = a * ul + b * u + c * ur
    if abs(h1) < 1e-12
        back_diff = cent_diff = zero(u)
    end
    if abs(h2) < 1e-12
        fwd_diff = cent_diff = zero(u)
    end
    return back_diff, cent_diff, fwd_diff
end

function limit_variable_slope(eq, variable, slope, u_star_ll, u_star_rr, ue, xl, xr)
    # By Jensen's inequality, we can find theta's directly for the primitives
    var_star_ll, var_star_rr = variable(eq, u_star_ll), variable(eq, u_star_rr)
    var_low = variable(eq, ue)
    threshold = 0.1 * var_low
    eps = 1e-10
    if var_star_ll < eps || var_star_rr < eps
        ratio_ll = abs(threshold - var_low) / (abs(var_star_ll - var_low) + 1e-13)
        ratio_rr = abs(threshold - var_low) / (abs(var_star_rr - var_low) + 1e-13)
        theta = min(ratio_ll, ratio_rr, 1.0)
        slope *= theta
        u_star_ll = ue + 2.0 * xl * slope
        u_star_rr = ue + 2.0 * xr * slope
    end
    return slope, u_star_ll, u_star_rr
end

function implicit_source_update(eq, u, x, t, dt, source_terms) # u, s are SVectors
    tolE = 1e-8
    unp1 = u
    unp1 = u + dt * source_terms(unp1, x, t, eq)
    local L = unp1 - u - dt * source_terms(unp1, x, t, eq)
    while norm(L) > tolE
        unp1 = u + dt * source_terms(unp1, x, t, eq)
        L = unp1 - u - dt * source_terms(unp1, x, t, eq)
    end

    return source_terms(unp1, x, t, eq)
end


function pre_process_limiter!(eq, t, iter, fcount, dt, grid, problem, scheme,
                              param, aux, op, u1, ua)
    @timeit aux.timer "Limiter" begin
    #! format: noindent
    @timeit aux.timer "Pre process limiter" begin
    #! format: noindent
    # TOTHINK - (very small) allocations happening outside limiter funcs?
    # Are they just because of the assertion in tvbβ or is there a worrisome cause?
    @unpack limiter = scheme
    if limiter.name == "blend"
        update_ghost_values_u1!(eq, problem, grid, op, u1, aux, t)
        modal_smoothness_indicator(eq, t, iter, fcount, dt, grid, scheme,
                                   problem, param, aux, op, u1, ua)
        return nothing
    elseif limiter.name == "tvb"
        update_ghost_values_u1!(eq, problem, grid, op, u1, aux, t)
        return nothing
    end
    end # timer
    end # timer
end

@inline function conservative_indicator!(un, eq::AbstractEquations)
    nvar = nvariables(eq)
    n_ind_var = nvar
    return n_ind_var
end

@inbounds @inline function conservative2conservative_reconstruction!(ue, ua,
                                                                     eq::AbstractEquations)
    return ue
end

@inbounds @inline function prim2con!(eq::AbstractEquations{<:Any, 1}, # For scalar equations
                                     ue)
    return nothing
end

@inbounds @inline function con2prim!(eq::AbstractEquations{<:Any, 1}, # For scalar equations
                                     ue)
end

@inbounds @inline function conservative2primitive_reconstruction!(ue, ua,
                                                                  eq::AbstractEquations)
    Tenkai.con2prim(eq, ue)
end

@inbounds @inline function primitive2conservative_reconstruction!(ue, ua,
                                                                  eq::AbstractEquations)
    Tenkai.prim2con(eq, ue)
end

@inbounds @inline function conservative2characteristic_reconstruction!(ue, ua,
                                                                       ::AbstractEquations{
                                                                                           <:Any,
                                                                                           1
                                                                                           })
    return nothing
end

@inbounds @inline function characteristic2conservative_reconstruction!(ue, ua,
                                                                       ::AbstractEquations{
                                                                                           <:Any,
                                                                                           1
                                                                                           })
    return nothing
end

@inline @inbounds function refresh!(u)
    @turbo u .= zero(eltype(u))
end

#-------------------------------------------------------------------------------
# Return string of the form base_name00c with total number of digits = ndigits
#-------------------------------------------------------------------------------
function get_filename(base_name, ndigits, c)
    if c > 10^ndigits - 1
        println("get_filename: Not enough digits !!!")
        println("   ndigits =", ndigits)
        println("   c       =", c)
        @assert false
    end
    number = lpad(c, ndigits, "0")
    return string(base_name, number)
end

#-------------------------------------------------------------------------------
# Struct storing plot information
#-------------------------------------------------------------------------------
struct PlotData{T} # Parametrized to support all backends
    p_ua::Plots.Plot{T}
    anim_ua::Animation
    p_u1::Plots.Plot{T}
    anim_u1::Animation
end

#-------------------------------------------------------------------------------
# Adjust dt to reach final time or the next time when solution has to be saved
#-------------------------------------------------------------------------------
function adjust_time_step(problem, param, t, dt, aux)
    @timeit aux.timer "Time step computation" begin
    #! format: noindent
    # Adjust to reach final time exactly
    @unpack final_time = problem
    if t + dt > final_time
        dt = final_time - t
        return dt
    end

    # Adjust to reach next solution saving time
    @unpack save_time_interval = param
    if save_time_interval > 0.0
        next_save_time = ceil(t / save_time_interval) * save_time_interval
        # If t is not a plotting time, we check if the next time
        # would step over the plotting time to adjust dt
        if abs(t - next_save_time) > 1e-10 && t + dt - next_save_time > -1e-10
            dt = next_save_time - t
            return dt
        end
    end

    return dt
    end # timer
end

#-------------------------------------------------------------------------------
# Check if we have to save solution
#-------------------------------------------------------------------------------
function save_solution(problem, param, t, iter)
    # Save if we have reached final time
    @unpack final_time = problem
    if abs(t - final_time) < 1.0e-10
        return true
    end

    # Save after specified time interval
    @unpack save_time_interval = param
    if save_time_interval > 0.0
        k1, k2 = ceil(t / save_time_interval), floor(t / save_time_interval)
        if (abs(t - k1 * save_time_interval) < 1e-10 ||
            abs(t - k2 * save_time_interval) < 1e-10)
            return true
        end
    end

    # Save after specified number of iterations
    @unpack save_iter_interval = param
    if save_iter_interval > 0
        if mod(iter, save_iter_interval) == 0
            return true
        end
    end

    return false
end

#------------------------------------------------------------------------------
# Methods that are extended/defined in FR1D, FR2D
#------------------------------------------------------------------------------
set_initial_condition!() = nothing
compute_cell_average!() = nothing
get_cfl() = nothing
compute_time_step() = nothing
compute_face_residual!() = nothing
apply_bound_limiter!() = nothing
setup_limiter_tvb() = nothing
setup_limiter_tvbβ() = nothing
apply_tvb_limiter!() = nothing
apply_tvb_limiterβ!() = nothing
Hierarchical() = nothing
apply_hierarchical_limiter!() = nothing
set_blend_dt!() = nothing
compute_error() = nothing
initialize_plot() = nothing
write_soln!() = nothing
create_aux_cache() = nothing
write_poly() = nothing
post_process_soln() = nothing
update_ghost_values_periodic!() = nothing
update_ghost_values_u1!() = nothing
update_ghost_values_fn_blend!() = nothing
limit_slope() = nothing
is_admissible() = nothing
modal_smoothness_indicator() = nothing
modal_smoothness_indicator_gassner() = nothing
Blend() = nothing
setup_arrays_lwfr() = nothing
setup_arrays_rkfr() = nothing
setup_arrays_mdrk() = nothing

# These methods are primarily for LWFR.jl, but are also needed here for
# get_bflux_function()
eval_bflux1!() = nothing
eval_bflux2!() = nothing
eval_bflux3!() = nothing
eval_bflux4!() = nothing
eval_bflux5!() = nothing
extrap_bflux!() = nothing

#-------------------------------------------------------------------------------
# Solve the problem
#-------------------------------------------------------------------------------
function solve(equation, problem, scheme, param)
    println("Number of julia threads = ", Threads.nthreads())
    @unpack grid_size, cfl, compute_error_interval = param

    # Make 1D/2D grid
    grid = make_cartesian_grid(problem, grid_size)

    # Make fr operators
    @unpack degree, solution_points, correction_function = scheme

    op = fr_operators(degree, solution_points, correction_function)

    cache = setup_arrays(grid, scheme, equation)
    aux = create_auxiliaries(equation, op, grid, problem, scheme, param, cache)

    @unpack solver = scheme
    if solver == "lwfr"
        # SSFR = Single Stage Flux Reconstruction. It defaults to LWFR
        out = solve_ssfr(equation, problem, scheme, param, grid, op, aux,
                         cache)
    elseif solver == "rkfr"
        out = solve_rkfr(equation, problem, scheme, param, grid, op, aux,
                         cache)
    elseif solver == "mdrk"
        out = solve_mdrk(equation, problem, scheme, param, grid, op, aux,
                         cache)
    else
        @assert !(solver isa String) "Solver not implemented"
        println("About to use a solver written by user...")
        out = solve_ssfr(equation, problem, scheme, param, grid, op, aux,
                         cache)
    end
    return out
end

solve(equation, grid, problem, scheme, param) = solve(equation, problem, scheme, param)
end # @muladd
