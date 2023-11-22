module MDRK

( # FR Methods unified as 1D and 2D in Tenkai
using Tenkai: set_initial_condition!,
            compute_cell_average!,
            compute_face_residual!,
            write_soln!,
            compute_error,
            post_process_soln
)

using Tenkai.LWFR: update_ghost_values_lwfr!, update_solution_lwfr!
using LoopVectorization
using Tenkai

include("$(Tenkai.mdrk_dir)/MDRK1D.jl")
include("$(Tenkai.mdrk_dir)/MDRK2D.jl")

#------------------------------------------------------------------------------
# Extending methods needed in FR.jl which are defined here
#------------------------------------------------------------------------------
# Dimension independent methods in FR
(
using ..FR: apply_limiter!, compute_time_step, adjust_time_step,
            pre_process_limiter!, get_cfl, save_solution
)

import Tenkai.FR: solve_mdrk

using Printf
using LinearAlgebra: axpy!, dot
using UnPack
using TimerOutputs
using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#-------------------------------------------------------------------------------
# Perform a full stage of MDRK
#-------------------------------------------------------------------------------
function perform_mdrk_step!(eq, t, iter, fcount, dt, grid, problem, scheme,
                            param, aux, op, uprev, ua, res, Fb, Ub, cache,
                            unew, cell_residual!, boundary_scaling_factor)
   @timeit aux.timer "MDRK Stages" begin
   pre_process_limiter!(eq, t, iter, fcount, dt, grid, problem, scheme,
                        param, aux, op, uprev, ua)
   @timeit aux.timer "Cell Residual" cell_residual!(eq, grid, op, scheme, aux,
                                                    t, dt, uprev, res, Fb, Ub,
                                                    cache)

   update_ghost_values_lwfr!(problem, scheme, eq, grid, aux, op, cache, t,
                             dt, boundary_scaling_factor)
   compute_face_residual!(eq, grid, op, scheme, param, aux, t, dt, uprev,
                          Fb, Ub, ua, res)
   @turbo unew .= uprev # Does nothing in the second stage. TODO - Fix this for performance
   update_solution_lwfr!(unew, res, aux) # s1: us = u1 - res, s2: u1 = u1 - res
   compute_cell_average!(ua, unew, t, eq, grid, problem, scheme, aux, op)
   apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, unew)
   return nothing
   end # timer
end

#-------------------------------------------------------------------------------
# Apply LWFR scheme and solve the problem
# N = degree of solution space
#-------------------------------------------------------------------------------
function solve_mdrk(eq, problem, scheme, param, grid, op, aux, cache)
   println("Solving ",eq.name," using LWFR")

   @unpack final_time = problem
   @unpack grid_size, cfl, compute_error_interval = param

   # Allocate memory
   @unpack u1, ua, us, res, Fb, Ub = cache

   # Set initial condition
   set_initial_condition!(u1, eq, grid, op, problem)

   # Compute cell average for initial condition
   compute_cell_average!(ua, u1, 0.0, eq, grid, problem, scheme, aux, op)

   # Apply limiter to handle discontinuities of the initial solution
   apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)

   # Initialize counters
   iter, t, fcount = 0, 0.0, 0

   # Save initial solution to file
   fcount = write_soln!("sol", fcount, iter, t, 0.0, eq, grid, problem, param, op,
                         ua, u1, aux)

   # Choose CFL number
   if cfl > 0.0
      @printf("CFL: specified value = %f\n", cfl)
   else
      cfl = get_cfl(eq, scheme, param)
      @printf("CFL: based on stability = %f\n", cfl)
   end

   # Compute initial error norm
   error_norm = compute_error(problem, grid, eq, aux, op, u1, t)

   println("Starting time stepping")
   while t < final_time
      dt = compute_time_step(eq, grid, aux, op, cfl, u1, ua)
      dt = adjust_time_step(problem, param, t, dt, aux)

      # First stage
      perform_mdrk_step!(eq, t, iter, fcount, dt, grid, problem, scheme,
                         param, aux, op, u1, ua, res, Fb, Ub, cache, us,
                         compute_cell_residual_mdrk_1!, 0.5)

      # Second stage
      perform_mdrk_step!(eq, t, iter, fcount, dt, grid, problem, scheme,
                         param, aux, op, u1, ua, res, Fb, Ub, cache, u1,
                         compute_cell_residual_mdrk_2!, 1.0)

      t += dt; iter += 1
      @printf("iter,dt,t = %5d %12.4e %12.4e\n", iter, dt, t)
      if save_solution(problem, param, t, iter)
         fcount = write_soln!("sol", fcount, iter, t, dt, eq, grid, problem, param,
                              op, ua, u1, aux)
      end
      if (compute_error_interval > 0 && mod(iter, compute_error_interval) == 0)
         error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
      end
   end
   error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
   post_process_soln(eq, aux, problem, param)

   return Dict("u" => u1, "ua" => ua, "errors" => error_norm,
               "plot_data" => aux.plot_data, "grid" => grid,
               "op" => op)
end

end # @muladd

end # module
