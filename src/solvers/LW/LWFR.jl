module LWFR


( # FR Methods unified as 1D and 2D in SSFR
using SSFR: set_initial_condition!,
            compute_cell_average!,
            compute_face_residual!,
            write_soln!,
            compute_error,
            post_process_soln
)

#------------------------------------------------------------------------------
# Extending methods needed in FR.jl which are defined here
#------------------------------------------------------------------------------
import ..SSFR: solve_lwfr

# Dimension independent methods in FR
(
using ..FR: apply_limiter!, compute_time_step, adjust_time_step,
            pre_process_limiter!, get_cfl, save_solution
)

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

#------------------------------------------------------------------------------
# Methods to be defined in LWFR1D, LWFR2D
#------------------------------------------------------------------------------
compute_cell_residual_1!() = nothing
compute_cell_residual_2!() = nothing
compute_cell_residual_3!() = nothing
compute_cell_residual_4!() = nothing
update_ghost_values_lwfr!() = nothing
( # These LWFR methods were needed in FR.jl, so they were defined there
import ..FR: eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
             extrap_bflux!, setup_arrays_lwfr
)
# eval_bflux1!() = nothing
# eval_bflux2!() = nothing
# eval_bflux3!() = nothing
# eval_bflux4!() = nothing
# extrap_bflux!() = nothing

#-------------------------------------------------------------------------------
# Update solution
#-------------------------------------------------------------------------------
function update_solution_lwfr!(u1, res, aux)
   @timeit aux.timer "Update solution" begin
   axpy!(-1.0, res, u1) # u1 = (-1.0)*res + u1
   return nothing
   end # timer
end

#-------------------------------------------------------------------------------
# Compute cell residual for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual!(eq, grid, op, scheme, aux, t, dt, u1, res, Fb,
                                Ub, cache)
   @timeit aux.timer "Cell Residual" begin

   N = op.degree
   if N == 1
      compute_cell_residual_1!(eq, grid, op, scheme, aux, t, dt, u1, res, Fb, Ub,
                               cache)
   elseif N == 2
      compute_cell_residual_2!(eq, grid, op, scheme, aux, t, dt, u1, res, Fb, Ub,
                               cache)
   elseif N == 3
      compute_cell_residual_3!(eq, grid, op, scheme, aux, t, dt, u1, res, Fb, Ub,
                               cache)
   elseif N == 4
      compute_cell_residual_4!(eq, grid, op, scheme, aux, t, dt, u1, res, Fb, Ub,
                               cache)
   else
      println("compute_cell_residual: Not implemented for degree > 1")
      @assert false
   end

   return nothing
   end # timer
end

#-------------------------------------------------------------------------------
# Apply LWFR scheme and solve the problem
# N = degree of solution space
#-------------------------------------------------------------------------------
function solve_lwfr(eq, problem, scheme, param, grid, op, aux, cache)
   println("Solving ",eq.name," using LWFR")

   @unpack final_time = problem
   @unpack grid_size, cfl, compute_error_interval = param

   # Allocate memory
   @unpack u1, ua, res, Fb, Ub = cache

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

      pre_process_limiter!(eq, t, iter, fcount, dt, grid, problem, scheme,
                           param, aux, op, u1, ua)
      compute_cell_residual!(eq, grid, op, scheme, aux, t, dt, u1, res, Fb, Ub,
                             cache)
      update_ghost_values_lwfr!(problem, scheme, eq, grid, aux, op, cache, t,
                                dt)
      compute_face_residual!(eq, grid, op, scheme, param, aux, t, dt, u1,
                              Fb, Ub, ua, res)
      update_solution_lwfr!(u1, res, aux) # u1 = u1 - res
      compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
      apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
      t += dt; iter += 1
      @printf("iter,dt,t = %5d %12.4e %12.4e\n", iter, dt, t)
      if save_solution(problem, param, t, iter)
         fcount = write_soln!("sol", fcount, iter, t, dt, eq, grid, problem, param,
                              op, ua, u1, aux)
      end
      if (compute_error_interval > 0 && mod(iter,compute_error_interval) == 0)
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