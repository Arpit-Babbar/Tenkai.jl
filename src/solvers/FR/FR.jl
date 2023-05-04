module FR

using ..Basis
using ..Grid
using ..Equations: AbstractEquations, nvariables, eachvariable

import SSFR

using ArgParse
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

# Temporary hack
using FastGaussQuadrature

@enum BCType periodic dirichlet neumann reflect
@enum BFluxType extrapolate evaluate
@enum SolverType rkfr lwfr # TEMPORARY HACK, AVOID REPETETION

#-------------------------------------------------------------------------------
# Create a struct of problem description
#-------------------------------------------------------------------------------
struct Problem{F1,F2,F3 <: Function, BoundaryCondition <: Tuple}
   domain::Vector{Float64}
   initial_value::F1
   boundary_value::F2
   boundary_condition::BoundaryCondition
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
                 exact_solution::Function)
   if length(domain) == 2
      @assert length(boundary_condition) == 2 "Invalid Problem"
      left, right = boundary_condition
      if (left == periodic && right != periodic)
         println("Incorrect use of periodic bc")
         @assert false
      elseif left == periodic && right == periodic
         periodic_x = true
      else
         periodic_x = false
      end
      return Problem(domain, initial_value, boundary_value, boundary_condition,
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
      return Problem(domain, initial_value, boundary_value, boundary_condition,
                     periodic_x, periodic_y, final_time, exact_solution)
   else
      @assert false,"Invalid domain"
   end
end

#-------------------------------------------------------------------------------
# Create a struct of scheme description
#-------------------------------------------------------------------------------
struct Scheme{NumericalFlux, Limiter, BFlux <: NamedTuple{<:Any, <:Any}, Dissipation}
   solver::String
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
   end
end

function diss_arg2method(dissipation)
   @assert dissipation in ("1","2",1,2,
                           get_first_node_vars, # TODO - This is bad design
                           get_second_node_vars) "dissipation = $dissipation"
   if dissipation in ("1",1)
      get_dissipation_node_vars = get_first_node_vars
   else
      @assert dissipation in ("2",2)
      get_dissipation_node_vars = get_second_node_vars
   end
   return get_dissipation_node_vars
end

# Constructor
function Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux::BFluxType,
                dissipation = "2")
   bflux_data = (; bflux_ind = bflux,
                   compute_bflux! = get_bflux_function(solver,degree,bflux) )
   solver_enum = solver2enum(solver)
   get_dissipation_node_vars = diss_arg2method(dissipation)

   Scheme(solver, solver_enum, degree, solution_points, correction_function,
          numerical_flux, bound_limit, limiter, bflux_data,
          get_dissipation_node_vars)
end

function Scheme(solver, solver_enum::SolverType, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux::BFluxType,
                dissipation = "2")
   bflux_data = (; bflux_ind = bflux,
                   compute_bflux! = get_bflux_function(solver,degree,bflux) )
   get_dissipation_node_vars = diss_arg2method(dissipation)

   Scheme(solver, solver_enum, degree, solution_points, correction_function,
          numerical_flux, bound_limit, limiter, bflux_data,
          get_dissipation_node_vars)
end

trivial_function(x) = nothing

function get_bflux_function(solver, degree, bflux)
   if solver == "rkfr"
      return trivial_function
   else
      if bflux == extrapolate
         return extrap_bflux!
      else
         if degree == 1
            return eval_bflux1!
         elseif degree == 2
            return eval_bflux2!
         elseif degree == 3
            return eval_bflux3!
         elseif degree == 4
            return eval_bflux4!
         else
            @assert false "Incorrect degree"
         end
      end
   end
end

#------------------------------------------------------------------------------
# Create a struct of parameters
#------------------------------------------------------------------------------
struct Parameters{T1 <: Union{Int64,Vector{Int64}}}
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
end

# Constructor
function Parameters(grid_size, cfl, bounds, save_iter_interval,
                    save_time_interval, compute_error_interval;
                    animate = false, cfl_safety_factor = 0.98,
                    time_scheme="by degree",
                    saveto = "none",
                    cfl_style = "optimal")
   @assert (cfl >= 0.0) "cfl must be >= 0.0"
   @assert (save_iter_interval >= 0) "save_iter_interval must be >= 0"
   @assert (save_time_interval >= 0.0) "save_time_interval must be >= 0.0"
   @assert (!(save_iter_interval > 0 &&
            save_time_interval > 0.0)) "Both save_(iter,time)_interval > 0"
   @assert cfl_style in ["lw", "optimal"]

   Parameters(grid_size, cfl, bounds, save_iter_interval,
              save_time_interval, compute_error_interval, animate,
              saveto, time_scheme, cfl_safety_factor, cfl_style)
end

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

#-------------------------------------------------------------------------------
# Apply given command line arguments
#-------------------------------------------------------------------------------
function ParseCommandLine(problem, param, scheme, eq, args;
                          limiter_ = nothing, numerical_flux_ = nothing,
                          initial_value_ = nothing)
   @unpack numfluxes = eq
   @unpack ( domain, initial_value, boundary_value, boundary_condition,
             final_time, exact_solution ) = problem
   @unpack ( grid_size, cfl, bounds, cfl_safety_factor,
             save_time_interval, save_iter_interval,
             compute_error_interval, saveto, time_scheme,
             animate, cfl_style ) = param
   @unpack ( solver, degree, solution_points, correction_function,
             numerical_flux, bound_limit, limiter, bflux,
             dissipation ) = scheme
   size = length(grid_size)
   s = ArgParseSettings()
   @add_arg_table s begin
      "--initial_value"
         help = "Choose Initial Condition from Equation file"
         arg_type = Union{String,Nothing}
         default = nothing
      "--final_time"
         help = "Final Time solution is computed at"
         arg_type = Float64
         default = final_time
      "--grid_size"
         help = "Grid size along x-axis"
         arg_type = Int64
         nargs = size
         default = vcat(grid_size) # convert to vector, needed in 1D
      "--cfl"
         help = "Choose CFL number"
         arg_type = Float64
         default = cfl
      "--cfl_safety_factor"
         help = "Choose CFL safety factor"
         arg_type = Float64
         default = cfl_safety_factor
      "--cfl_style"
         help = "Choose CFL safety factor"
         arg_type = String
         default = cfl_style
      "--save_iter_interval"
         help = "Solution saving iteration interval size"
         arg_type = Int64
         default = save_iter_interval
      "--save_time_interval"
         help = "Solution saving time interval size"
         arg_type = Float64
         default = save_time_interval
      "--compute_error_interval"
         help = "Error computing iteration interval size"
         arg_type = Int64
         default = compute_error_interval
      "--animate"
         help = "Factor multiplied by save_time_interval or save_iter_interval to determine number of plots made"
         arg_type = Bool
         default = animate
      "--saveto"
         help = "Directory where a copy of output data is sent"
         arg_type = String
         default = saveto
      "--solver"
         help = "Flux Reconstruction solver"
         arg_type = String
         default = solver
      "--degree"
         help = "Degree of approximating polynomials"
         arg_type = Int64
         default = degree
      "--solution_points"
         help = "Solution Points"
         arg_type = String
         default = solution_points
      "--correction_function"
         help = "Flux correction function"
         arg_type = String
         default = correction_function
      "--tvbM"
         help = "TVB parameter M"
      "--bound_limit"
         help = "Enable or disable the bounds limiter with yes/no"
         arg_type = String
         default = bound_limit
      "--bflux"
         help = "Boundary Flux"
         default = bflux
         # TODO - Improve this by overloading ArgParse.parse_item, see
         # https://argparsejl.readthedocs.io/en/latest/argparse.html#available-actions-and-nargs-values
      "--dissipation"
         help = "Dissipation type"
      "--time_scheme"
         help = "Time integration scheme for RKFR"
         arg_type = String
         default = time_scheme
   end

   # TODO - Add animate, bounds limiter, upper and lower bounds

   args_dict = parse_args(args, s)

   if args_dict["bflux"] in ["extrapolate", extrapolate]
      args_dict["bflux"] = extrapolate
   elseif args_dict["bflux"] in ["evaluate", evaluate]
      args_dict["bflux"] = evaluate
   end

   if length(args_dict["grid_size"]) == 1 # 1-D has scalar grid_size
      args_dict["grid_size"], = [a for a in args_dict["grid_size"]]
   end

   @unpack ( final_time, grid_size, cfl, tvbM,
             save_time_interval, save_iter_interval,
             compute_error_interval, animate, solver, degree,
             solution_points, correction_function,
             bflux, saveto, cfl_safety_factor,
             bound_limit, time_scheme, cfl_style
             ) = args_dict

   args_diss = args_dict["dissipation"]
   if args_diss !== nothing
      dissipation = args_diss
   end

   if solver == "lwfr"
      solver_enum = lwfr
   else
      @assert solver == "rkfr"
      solver_enum = rkfr
   end

   problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                     final_time, exact_solution)
   scheme = Scheme(solver, solver_enum, degree, solution_points,
                   correction_function, numerical_flux, bound_limit,
                   limiter, bflux, dissipation)
   param = Parameters(grid_size, cfl, bounds,
                      save_iter_interval, save_time_interval,
                      compute_error_interval, animate, saveto,
                      time_scheme, cfl_safety_factor, cfl_style)
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

# TODO : Is Base.OneTo good for performance

@inline function get_node_vars(u, eq, indices...)
   SVector(ntuple(@inline(v -> u[v, indices...]), Val(nvariables(eq))))
end

@inline function get_first_node_vars(u1, u2, eq, indices...)
   SVector(ntuple(@inline(v -> u1[v, indices...]), Val(nvariables(eq))))
end

@inline function get_second_node_vars(u1, u2, eq, indices...)
   SVector(ntuple(@inline(v -> u2[v, indices...]), Val(nvariables(eq))))
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

# TODO - Rename to multiply_add_set_node_vars! ?

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
      u[v, indices...] = factor * (   factor1 * u_node1[v] + factor2 * u_node2[v]
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
      u[v, indices...] = (  factor1 * u_node1[v]
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
      u[v, indices...] = u_node_factor_less1[v] + factor * u_node[v] + u_node_factor_less2[v]
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
                                                  + factor2 * u_node2[v])
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
      u[v, indices...] = factor * (  factor1*u_node1[v] + factor2*u_node2[v]
                                   + factor3*u_node3[v] + factor4*u_node4[v]
                                   + factor5*u_node5[v])
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
      u[v, indices...] = u[v, indices...] + factor1 * u_node1[v] + factor2 * u_node2[v]
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
      u[v, indices...] = u[v, indices...] + factor * ( factor1 * u_node1[v]
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
      u[v, indices...] = u[v, indices...] + factor * ( factor1 * u_node1[v]
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
      u[v, indices...] = u[v, indices...] + ( factor1 * u_node1[v]
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
      u[v, indices...] = u[v, indices...] + ( factor1 * u_node1[v]
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
      u[v, indices...] = u[v, indices...] + (  factor1 * u_node1[v]
                                             + factor2 * u_node2[v]
                                             + factor3 * u_node3[v] )
   end
   return nothing
end

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                             u_node_factor_less::SVector{<:Any},
                                             factor1::Real, u_node1::SVector{<:Any},
                                             equations::AbstractEquations, indices...)
   for v in eachvariable(equations)
      u[v, indices...] = u[v, indices...] + u_node_factor_less[v] + factor1 * u_node1[v]
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
                                                        + u_node3[v]) )
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

#-------------------------------------------------------------------------------
# Set up arrays
#------------------------------------------------------------------------------
function setup_arrays(grid, scheme, equation)
   @unpack solver = scheme
   if solver == "lwfr"
      return setup_arrays_lwfr(grid, scheme, equation)
   elseif solver == "rkfr"
      return setup_arrays_rkfr(grid, scheme, equation)
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
   plot_data = initialize_plot(eq, op, grid, problem, scheme, timer, u1, ua);
   # Setup blending limiter
   blend = SSFR.Blend(eq, op, grid, problem, scheme, param, plot_data)
   hierarchical = SSFR.Hierarchical(eq, op, grid, problem, scheme, param,
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
   limiter = (;name = "none")
   return limiter
end

function setup_limiter_blend(; blend_type, indicating_variables,
                               reconstruction_variables, indicator_model,
                               smooth_alpha = true, smooth_factor = 0.5,
                               amax = 1.0, constant_node_factor = 1.0,
                               constant_node_factor2 = 1.0,
                               a = 0.5, c = 1.8, amin = 0.001,
                               debug_blend = false , pure_fv = false)
   limiter = (; name = "blend", blend_type, indicating_variables,
                reconstruction_variables, indicator_model,
                amax, smooth_alpha, smooth_factor,
                constant_node_factor, constant_node_factor2,
                a, c, amin,
                debug_blend, pure_fv)
   return limiter
end

function setup_limiter_hierarchical(; alpha,
                                      reconstruction = conservative_reconstruction)
   limiter = (; name = "hierarchical", alpha, reconstruction)
   return limiter
end

function apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
   @timeit aux.timer "Limiter" begin
   # TODO - Why are allocations happening here outside of TVB limiter?
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
   slope = min(abs(a),abs(b),abs(c))
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
   slope = min(beta*abs(a),abs(b),beta*abs(c))
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

function pre_process_limiter!(eq, t, iter, fcount, dt, grid, problem, scheme,
                              param, aux, op, u1, ua)
   @timeit aux.timer "Limiter" begin
   @timeit aux.timer "Pre process limiter" begin
   # TODO - There are allocations in this function, can they be avoided?
   @unpack limiter = scheme
   if limiter.name == "blend"
      update_ghost_values_u1!(eq, problem, grid, op, u1, t)
      modal_smoothness_indicator(eq, t, iter, fcount, dt, grid, scheme,
                                 problem, param, aux, op, u1, ua)
   elseif limiter.name == "tvb"
      nothing
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
   return nothing
end

@inbounds @inline function prim2con!(eq::AbstractEquations{<:Any,1}, # For scalar equations
                                     ue)
   return nothing
end

@inbounds @inline function con2prim!(eq::AbstractEquations{<:Any,1}, # For scalar equations
                                     ue)
end

@inbounds @inline function conservative2primitive_reconstruction!(ue, ua,
                                                                  eq::AbstractEquations)
   con2prim!(eq, ue)
end

@inbounds @inline function primitive2conservative_reconstruction!(ue, ua,
                                                                  eq::AbstractEquations)
   prim2con!(eq, ue)
end

@inbounds @inline function conservative2characteristic_reconstruction!(ue, ua,
                                                                       ::AbstractEquations{<:Any,1})
   return nothing
end

@inbounds @inline function characteristic2conservative_reconstruction!(ue, ua,
                                                                       ::AbstractEquations{<:Any, 1})
   return nothing
end

refresh!(arr) = fill!(arr, eltype(arr))

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
   # Adjust to reach final time exactly
   @unpack final_time = problem
   if t + dt > final_time
      dt = final_time - t
      return dt
   end

   # Adjust to reach next solution saving time
   @unpack save_time_interval = param
   if save_time_interval > 0.0
      next_save_time = ceil(t/save_time_interval) * save_time_interval
      # If t is not a plotting time, we check if the next time
      # would step over the plotting time to adjust dt
      if abs(t-next_save_time) > 1e-10 && t + dt - next_save_time > -1e-10
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
      k1, k2 = ceil(t/save_time_interval), floor(t/save_time_interval)
      if (abs(t-k1*save_time_interval) < 1e-10 ||
          abs(t-k2*save_time_interval) < 1e-10)
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
limit_slope!() = nothing
is_admissible() = nothing
modal_smoothness_indicator() = nothing
modal_smoothness_indicator_gassner() = nothing
Blend() = nothing
setup_arrays_lwfr() = nothing
setup_arrays_rkfr() = nothing
solve_lwfr() = nothing
solve_rkfr() = nothing

# These methods are primarily for LWFR.jl, but are also needed here for
# get_bflux_function()
eval_bflux1!() = nothing
eval_bflux2!() = nothing
eval_bflux3!() = nothing
eval_bflux4!() = nothing
extrap_bflux!() = nothing

#-------------------------------------------------------------------------------
# Solve the problem
#-------------------------------------------------------------------------------
function solve(equation, problem, scheme, param);
   println("Number of julia threads = ", Threads.nthreads())
   println("Number of BLAS  threads = ", BLAS.get_num_threads())
   @unpack grid_size, cfl, compute_error_interval = param

   # Make 1D/2D grid
   grid = make_grid(problem, grid_size)

   # Make fr operators
   @unpack degree, solution_points, correction_function = scheme

   op = fr_operators(degree, solution_points, correction_function)

   cache = setup_arrays(grid, scheme, equation)
   aux = create_auxiliaries(equation, op, grid, problem, scheme, param, cache)

   @unpack solver = scheme
   if solver == "lwfr"
      out = solve_lwfr(equation, problem, scheme, param, grid, op, aux,
                       cache)
   elseif solver == "rkfr"
      out = solve_rkfr(equation, problem, scheme, param, grid, op, aux,
                       cache)
   else
      println("Solver not implemented")
      @assert false
   end
   return out
end

end # @muladd

end # module