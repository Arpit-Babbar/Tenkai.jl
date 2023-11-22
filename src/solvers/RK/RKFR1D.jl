module RKFR1D

(
import ..Tenkai: setup_arrays_rkfr,
               compute_cell_residual_rkfr!,
               update_ghost_values_rkfr!,
               update_ghost_values_fn_blend!,
               flux
)

using UnPack
using TimerOutputs
using Polyester
using MuladdMacro
using OffsetArrays

using ..FR: @threaded
using ..Equations: AbstractEquations, nvariables, eachvariable

(
using ..Tenkai: update_ghost_values_periodic! # KLUDGE - Should this be taken from FR?
)

(
using ..FR: periodic, dirichlet, neumann, reflect, extrapolate, evaluate,
            get_node_vars, set_node_vars!,
            add_to_node_vars!, subtract_from_node_vars!,
            multiply_add_to_node_vars!, multiply_add_set_node_vars!,
            comp_wise_mutiply_node_vars!
)

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#------------------------------------------------------------------------------
function setup_arrays_rkfr(grid, scheme, eq::AbstractEquations{1})
   gArray(nvar,nx) = OffsetArray(zeros(Float64, nvar,nx+2),
                                 OffsetArrays.Origin(1,0))
   gArray(nvar,n1,nx) = OffsetArray(zeros(Float64, nvar,n1,nx+2),
                                    OffsetArrays.Origin(1,1,0))
   # Allocate memory
   @unpack degree = scheme
   nd = degree + 1
   nx = grid.size
   nvar = eq.nvar
   u0  = gArray(nvar, nd, nx) # ghost indices not needed, only for copyto!
   u1  = gArray(nvar, nd, nx) # ghost indices needed for blending limiter
   ua  = gArray(nvar,nx)
   res = gArray(nvar,nd,nx)
   Fb  = gArray(nvar,2,nx)
   ub  = gArray(nvar,2,nx)

   cache = (; u0, u1, ua, res, Fb, ub)
   return cache
end

#------------------------------------------------------------------------------
function update_ghost_values_rkfr!(problem, scheme, eq::AbstractEquations{1},
                                   grid, aux, op, cache, t)
   @timeit aux.timer "Update ghost values" begin
   @unpack Fb = cache
   Ub = cache.ub
   update_ghost_values_periodic!(eq, problem, Fb, Ub)

   if problem.periodic_x
      return nothing
   end

   nx = grid.size
   xf = grid.xf
   nvar = eq.nvar
   left, right = problem.boundary_condition
   @unpack boundary_value = problem

   # ub = zeros(nvar)
   # fb = zeros(nvar)
   # For Dirichlet bc, use upwind flux at faces by assigning both physical
   # and ghost cells through the bc.
   if left == dirichlet
      x = xf[1]
      ub = boundary_value(x,t)
      fb = flux(x, ub, eq)
      for n=1:nvar
         Ub[n, 1, 1] = Ub[n, 2, 0] = ub[n]    # upwind
         Fb[n, 1, 1] = Fb[n, 2, 0] = fb[n]    # upwind
      end
   elseif left == neumann
      for n=1:nvar
         Ub[n, 2, 0] = Ub[n, 1, 1]
         Fb[n, 2, 0] = Fb[n, 1, 1]
      end
   elseif left == reflect
      # velocity reflected back in opposite direction and density is same
      for n=1:nvar
         Ub[n, 2, 0] = Ub[n, 1, 1]
         Fb[n, 2, 0] = Fb[n, 1, 1]
      end
      Ub[2, 2, 0] = -Ub[2, 2, 0] # velocity reflected back
      Fb[1, 2, 0], Fb[3, 2, 0] = -Fb[1, 2, 0], -Fb[3, 2, 0] # vel multiple term
   else
      println("Incorrect bc specified at left.")
      @assert false
   end

   if right == dirichlet
      x = xf[nx+1]
      ub = boundary_value(x,t)
      fb = flux(x, ub, eq)
      for n=1:nvar
         Ub[n, 2, nx] = Ub[n, 1, nx+1] = ub[n] # upwind
         Fb[n, 2, nx] = Fb[n, 1, nx+1] = fb[n] # upwind
      end
   elseif right == neumann
      for n=1:nvar
         Ub[n, 1, nx+1] = Ub[n, 2, nx]
         Fb[n, 1, nx+1] = Fb[n, 2, nx]
      end
   elseif right == reflect
      # velocity reflected back in opposite direction and density is same
      for n=1:nvar
         Ub[n, 1, nx+1] = Ub[n, 2, nx]
         Fb[n, 1, nx+1] = Fb[n, 2, nx]
      end
      Ub[2, 1, nx+1] = -Ub[2, 1, nx+1] # velocity reflected back
      Fb[1, 1, nx+1], Fb[3, 1, nx+1] = (-Fb[1, 1, nx+1],
                                        -Fb[3, 1, nx+1]) # vel multiple term
   else
      println("Incorrect bc specified at right.")
      @assert false
   end

   if scheme.limiter.name == "blend"
      update_ghost_values_fn_blend!(eq, problem, grid, aux)
   end

   return nothing
   end # timer
end

#------------------------------------------------------------------------------
function compute_cell_residual_rkfr!(eq::AbstractEquations{1}, grid, op,
                                     scheme, aux, t, dt, u1, res, Fb, ub, cache)
   @timeit aux.timer "Cell residual" begin
   @unpack xg, D1, Vl, Vr = op
   @unpack blend = aux
   nx  = grid.size
   nd  = length(xg)
   @unpack bflux_ind = scheme.bflux
   refresh!(u) = fill!(u,0.0)

   refresh!.((ub, Fb, res))
   nvar = nvariables(eq)
   f = zeros(nvar, nd)
   @timeit aux.timer "Cell loop" begin
      @inbounds for cell=1:nx
         dx     = grid.dx[cell]
         xc     = grid.xc[cell]
         lamx   = dt / dx
         xl, xr = grid.xf[cell], grid.xf[cell+1]
         for ix in Base.OneTo(nd)
            # Solution points
            x = xc - 0.5 * dx + xg[ix] * dx
            u_node = get_node_vars(u1, eq, ix, cell)
            # Compute flux at all solution points
            flux1 = flux(x, u_node, eq)
            set_node_vars!(f, flux1, eq, ix)
            # KLUDGE - Remove dx, xf arguments. just pass grid and i
            for iix in 1:nd
               multiply_add_to_node_vars!(res, lamx * D1[iix,ix], flux1, eq,
                                          iix, cell)
            end
            multiply_add_to_node_vars!(ub, Vl[ix], u_node, eq, 1, cell)
            multiply_add_to_node_vars!(ub, Vr[ix], u_node, eq, 2, cell)
            if bflux_ind == extrapolate
               multiply_add_to_node_vars!(Fb, Vl[ix], flux1, eq, 1, cell)
               multiply_add_to_node_vars!(Fb, Vr[ix], flux1, eq, 2, cell)
            else
               ubl, ubr = get_node_vars(ub, eq, 1, cell), get_node_vars(ub, eq, 2, cell)
               fbl, fbr = flux(xl, ubl, eq), flux(xr, ubr, eq)
               set_node_vars!(Fb, fbl, eq, 1, cell)
               set_node_vars!(Fb, fbr, eq, 2, cell)
            end
         end
         u = @view u1[:,:,cell]
         r = @view res[:,:,cell]
         blend.blend_cell_residual!(cell, eq, scheme, aux, lamx, dt, dx,
                                    grid.xf[cell], op, u1 , u, cache.ua, f, r)
      end
   end # timer
   return nothing
   end # timer
end

end # muladd

end # module