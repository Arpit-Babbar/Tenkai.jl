module FR2D

using UnPack
using MuladdMacro
using TimerOutputs
using Printf
using Polyester
using OffsetArrays # OffsetArray, OffsetMatrix, OffsetVector
using ElasticArrays
using StaticArrays
using WriteVTK
using FLoops
using LoopVectorization
using LinearAlgebra: lmul!, mul!

using ..FR: refresh!

using ..SSFR: fr_dir, lwfr_dir, rkfr_dir, eq_dir

using ..Basis

(
using ..FR: periodic, dirichlet, neumann, reflect,
            lwfr, rkfr,
            minmod,
            @threaded, Problem, Scheme, Parameters,
            get_filename,
            alloc_for_threads,
            get_node_vars, set_node_vars!,
            get_first_node_vars, get_second_node_vars,
            add_to_node_vars!, subtract_from_node_vars!,
            multiply_add_to_node_vars!, multiply_add_set_node_vars!,
            comp_wise_mutiply_node_vars!
)

using ..Equations: AbstractEquations, nvariables, eachvariable

( # Methods to be extended from SSFR
import ..SSFR: update_ghost_values_periodic!,
               update_ghost_values_u1!,
               update_ghost_values_fn_blend!,
               modal_smoothness_indicator,
               modal_smoothness_indicator_gassner,
               set_initial_condition!,
               compute_cell_average!,
               get_cfl,
               compute_time_step,
               compute_face_residual!,
               setup_limiter_tvb,
               setup_limiter_tvbβ,
               apply_bound_limiter!,
               apply_tvb_limiter!,
               apply_tvb_limiterβ!,
               Blend,
               fo_blend,
               limit_slope!,
               is_admissible,
               mh_blend,
               set_blend_dt!,
               Hierarchical,
               apply_hierarchical_limiter!,
               compute_error,
               initialize_plot,
               write_soln!,
               create_aux_cache,
               write_poly,
               write_soln!,
               post_process_soln
)

using TimerOutputs
using OffsetArrays

using JSON3

using SSFR: flux, con2prim, con2prim!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#-------------------------------------------------------------------------------
# Set initial condition by interpolation in all real cells
#-------------------------------------------------------------------------------
function set_initial_condition!(u, eq::AbstractEquations{2}, grid, op, problem)
   println("Setting initial condition")
   @unpack initial_value = problem
   nx, ny = grid.size
   xg = op.xg
   nd = length(xg)
   for el_y=1:ny, el_x=1:nx
      dx, dy = grid.dx[el_x], grid.dy[el_y] # cell size
      xc, yc = grid.xc[el_x], grid.yc[el_y] # cell center
      for j=1:nd, i=1:nd
         x = xc - 0.5 * dx + xg[i] * dx
         y = yc - 0.5 * dy + xg[j] * dy
         iv = initial_value(x, y)
         set_node_vars!(u, iv, eq, i, j, el_x, el_y)
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell average in all real cells
#-------------------------------------------------------------------------------
function compute_cell_average!(ua, u1, t, eq::AbstractEquations{2}, grid,
                               problem, scheme, aux, op)
   @timeit aux.timer "Cell averaging" begin
   nx, ny = grid.size
   @unpack limiter = scheme
   @unpack xc = grid
   @unpack xg, wg, Vl, Vr = op
   @unpack periodic_x, periodic_y = problem
   @unpack boundary_condition, boundary_value = problem
   left, right, bottom, top = boundary_condition
   nd = length(wg)
   fill!(ua, zero(eltype(ua)))
   # Compute cell averages
   @threaded for element in CartesianIndices((1:nx, 1:ny))
      el_x, el_y = element[1], element[2]
      u1_ = @view u1[:,:,:,el_x,el_y]
      for j in Base.OneTo(nd), i in Base.OneTo(nd)
         u_node = get_node_vars(u1_, eq, i, j)
         multiply_add_to_node_vars!(ua, wg[i]*wg[j], u_node, eq, el_x, el_y)
         # Maybe put w2d in op and use it here
      end
   end

   # Update ghost values of ua by periodicity or with neighbours
   if periodic_x
      for j=1:ny
         ua_node = get_node_vars(ua, eq, nx  , j)
         set_node_vars!(ua, ua_node, eq, 0   , j)
         ua_node = get_node_vars(ua, eq, 1   , j)
         set_node_vars!(ua, ua_node, eq, nx+1, j)
      end
   else
      if left == dirichlet
         @threaded for el_y=1:ny
            x = grid.xf[1]
            for j in Base.OneTo(nd)
               y = grid.yf[el_y] + grid.dy[el_y]*xg[j]
               uval = boundary_value(x,y,t)
               for i in Base.OneTo(nd)
                  multiply_add_to_node_vars!(ua, wg[i]*wg[j], uval, eq, 0,
                                             el_y)
               end
            end
         end
      else
         @threaded for el_y=1:ny
            ua_ = get_node_vars(ua, eq, 1, el_y)
            set_node_vars!(ua, ua_, eq, 0, el_y)
         end
      end

      if right == dirichlet
         @threaded for el_y=1:ny
            x = grid.xf[nx+1]
            for j in Base.OneTo(nd)
               y = grid.yf[el_y] + grid.dy[el_y]*xg[j]
               uval = boundary_value(x,y,t)
               for i in Base.OneTo(nd)
                  multiply_add_to_node_vars!(ua, wg[i]*wg[j], uval, eq, nx+1,
                                             el_y)
               end
            end
         end
      else
         @threaded for el_y=1:ny
            ua_ = get_node_vars(ua, eq, nx  , el_y)
            set_node_vars!(ua, ua_, eq, nx+1, el_y)
         end
      end

      if left == reflect
         @turbo for el_y=1:ny
            ua[2,0,el_y] = -ua[2,1,el_y]
         end
      end

      if right == reflect
         @turbo for el_y=1:ny
            ua[2,nx+1,el_y] = -ua[2,nx,el_y]
         end
      end
   end

   if periodic_y
      # Bottom ghost cells
      for el_x in Base.OneTo(nx)
         ua_node = get_node_vars(ua, eq, el_x, ny)
         set_node_vars!(ua, ua_node, eq, el_x, 0)
         ua_node = get_node_vars(ua, eq, el_x, 1)
         set_node_vars!(ua, ua_node, eq, el_x, ny+1)
      end
   else
      if bottom in (reflect, neumann, dirichlet)
         if bottom == dirichlet
            @threaded for el_x=1:nx
               y = grid.yf[1]
               for i in Base.OneTo(nd)
                  x = grid.xf[el_x] + grid.dx[el_x]*xg[i]
                  uval = boundary_value(x,y,t)
                  for j in Base.OneTo(nd)
                     multiply_add_to_node_vars!(ua, wg[j]*wg[i], uval, eq,
                                                el_x, 0)
                  end
               end
            end
         else
            for el_x in Base.OneTo(nx)
               ua_ = get_node_vars(ua, eq, el_x, 1)
               set_node_vars!(ua, ua_, eq, el_x, 0)
            end
            if bottom == reflect
               @turbo for i=1:nx
                  ua[3,i,0] = -ua[3,i,1]
               end
            end
         end
      else
         @assert typeof(bottom)<: Tuple{Any, Any, Any}
         ua_bc! = bottom[2]
         ua_bc!(grid, eq, ua)
      end

      if top in (reflect, neumann, dirichlet)
         if top == dirichlet
            @threaded for el_x=1:nx
               y = grid.yf[ny+1]
               for i in Base.OneTo(nd)
                  x = grid.xf[el_x] + grid.dx[el_x]*xg[i]
                  uval = boundary_value(x,y,t)
                  for j in Base.OneTo(nd)
                     multiply_add_to_node_vars!(ua, wg[j]*wg[i], uval, eq,
                                                el_x, ny+1)
                  end
               end
            end
         else
            for el_x in Base.OneTo(nx)
               ua_ = get_node_vars(ua, eq, el_x, ny)
               set_node_vars!(ua, ua_, eq, el_x, ny+1)
            end
            if top == reflect
               for i=1:nx
                  ua[3,i,ny+1] = -ua[3,i,ny]
               end
            end
         end
      else
         @assert typeof(top) <: Tuple{Any, Any, Any}
         bc_ua! = top[2]
         bc_ua!(grid, eq, ua)
      end
   end
   end # timer
   return nothing
end

#------------------------------------------------------------------------------
# Choose cfl based on degree and correction function
#------------------------------------------------------------------------------
function get_cfl(eq::AbstractEquations{2}, scheme, param)
   os_vector(v)  = OffsetArray(v, OffsetArrays.Origin(0))
   @unpack solver, degree, correction_function = scheme
   @unpack cfl_safety_factor, cfl_style = param
   diss = scheme.dissipation
   @assert (degree >= 0 && degree < 5) "Invalid degree"
   if solver == "lwfr" || cfl_style == "lw"
      if diss == get_second_node_vars # Diss 2
         cfl_radau = os_vector([1.0, 0.259, 0.170, 0.103, 0.069]) # TODO: check
         cfl_g2    = os_vector([1.0, 0.511, 0.333, 0.170, 0.103])
         if solver == "rkfr"
            println("Using LW-D2 CFL with RKFR")
         else
            println("Using LW-D2 CFL with LW-D2")
         end
      elseif diss == get_first_node_vars # Diss 1
         # @assert degree > 1 "CFL of D1 not known for degree 1"
         cfl_radau = os_vector([1.0, 0.226, 0.117, 0.072, 0.049]) # TODO: check
         cfl_g2    = os_vector([1.0, 0.465, 0.204, 0.116, 0.060])
         if solver == "rkfr"
            println("Using LW-D1 CFL with RKFR")
         else
            println("Using LW-D1 CFL with LW-D1")
         end
      end
      if solver == "rkfr"
         println("Using LW-D2 CFL with RKFR")
      end
   elseif solver == "rkfr"
      cfl_radau = os_vector([1.0, 0.333, 0.209, 0.145, 0.110])
      cfl_g2    = os_vector([1.0, 1.0, 0.45, 0.2875, 0.212])
      # Source - Gassner,Dumbser,Hindenlang,Munz(2010) & Gassner,Kopriva(2011)
   end
   # Reduce this cfl by a small amount
   if correction_function == "radau"
      return cfl_safety_factor * cfl_radau[degree]
   elseif correction_function == "g2"
      return cfl_safety_factor * cfl_g2[degree]
   else
      println("get_cfl: unknown correction function")
      @assert false
   end
end

#-------------------------------------------------------------------------------
# Compute dt using cell average
#-------------------------------------------------------------------------------
function compute_time_step(eq::AbstractEquations{2,1}, grid, aux, op, cfl, u1,
                           ua)
   @timeit aux.timer "Time Step computation" begin
   nx, ny = grid.size
   xc, yc = grid.xc, grid.yc
   dx, dy = grid.dx, grid.dy
   den    = 0.0
   den2   = 0.0
   for j=1:ny, i=1:nx
      ua_    = get_node_vars(ua, eq, i, j)
      sx, sy = eq.speed((xc[i],yc[j]), ua_, eq)
      den    = max(den, abs(sx)/dx[i] + abs(sy)/dy[j] + 1.0e-12)

   end
   dt = cfl / den
   return dt
   end # timer
end

#-------------------------------------------------------------------------------
# Compute cartesian fluxes (f,g) at all solution points in one cell
# Not in use, not supported
#-------------------------------------------------------------------------------
@inline function compute_flux!(eq::AbstractEquations{2}, flux!, x, y, u, f, g)
   nd = length(x)
   for j=1:nd, i=1:nd
      @views f[:,i,j], g[:,i,j] .= flux([x[i],y[j]], u[:,i,j], eq)
   end
   return nothing
end

@inline function compute_flux!(eq::AbstractEquations{2}, x, y, u)
   @views f[:,i,j], g[:,i,j] .= flux([x[i],y[j]], u[:,i,j], eq)
   return nothing
end

#-------------------------------------------------------------------------------
# Currently not being used, not supported
# Interpolate average flux and solution to the four faces of cell
#  Fb[:,1] = F' * Vl
#  Fb[:,2] = F' * Vr
#  Fb[:,3] = G  * Vl
#  Fb[:,4] = G  * Vr
#
#  Ub[:,1] = U' * Vl
#  Ub[:,2] = U' * Vr
#  Ub[:,3] = U  * Vl
#  Ub[:,4] = U  * Vr
#-------------------------------------------------------------------------------
@inline function interpolate_to_face2D!(Vl, Vr, F, G, U, Fb, Ub)
   @views mult!(Ub[:,:,1], Vl, U)
   @views mult!(Ub[:,:,2], Vr, U)
   @views mult!(Ub[:,:,3], U, Vl)
   @views mult!(Ub[:,:,4], U, Vr)

   @views mult!(Fb[:,:,1], Vl, F)
   @views mult!(Fb[:,:,2], Vr, F)
   @views mult!(Fb[:,:,3], G, Vl)
   @views mult!(Fb[:,:,4], G, Vr)
   return nothing
end

@inline function interpolate_to_face2D!(Vl, Vr, U, Ub)
   @views mult!(Ub[:,:,1], Vl, U)
   @views mult!(Ub[:,:,2], Vr, U)
   @views mult!(Ub[:,:,3], U, Vl)
   @views mult!(Ub[:,:,4], U, Vr)
   return nothing
end

#-------------------------------------------------------------------------------
# Add numerical flux to residual
#-------------------------------------------------------------------------------
# res = ∂_x F_h + ∂_y G_h where F_h, G_h are continuous fluxes. We write it as
# res = D1*F_δ + gL'*Fn_L + gR'*Fn_R + G_δ*D1T + gL'*Fn_D + gR'*Fn_U.
# The gL,gR part is what we include in the face residual.
# We pre-allocate allocate bl,br=gL', gR' at soln points. Then,
# In reference coordinates, at nodes, the face residual is written as
# res[i,j,el_x,el_y] +=  dt/dx*(bL[i]*Fn[j,el_x-1/2,el_y] + bR[i]*Fn[j,el_x+1/2,el_y])
#                      + dt/dy*(bL[j]*Fn[i,el_x,el_y-1/2] + bR[j]*Fn[i,el_x,el_y+1/2])

function compute_face_residual!(eq::AbstractEquations{2}, grid, op, scheme,
                                param, aux, t, dt, u1,
                                Fb, Ub, ua, res)
   @timeit aux.timer "Face residual" begin
   @unpack bl, br, xg, degree = op
   nd = degree + 1
   nx, ny = grid.size
   @unpack dx, dy, xf, yf = grid
   @unpack numerical_flux = scheme
   @unpack blend = aux
   @unpack blend_face_residual_x!, blend_face_residual_y! = blend.subroutines

   # Vertical faces, x flux
   for i=1:nx+1
      @threaded for j=1:ny # thread here to avoid race condition
         # Face between (i-1,j) and (i,j)
         x = xf[i]
         ual, uar = get_node_vars(ua, eq, i-1, j), get_node_vars(ua, eq, i, j)
         for jy in Base.OneTo(nd)
            y = yf[j] + xg[jy] * dy[j]
            Fl, Fr = (get_node_vars(Fb, eq, jy, 2, i-1, j),
                      get_node_vars(Fb, eq, jy, 1, i  , j))
            Ul, Ur = (get_node_vars(Ub, eq, jy, 2, i-1, j),
                      get_node_vars(Ub, eq, jy, 1, i  , j))
            X = SVector{2}(x,y)
            Fn = numerical_flux(X, ual, uar, Fl, Fr, Ul, Ur, eq, 1)
            Fn, blend_factors = blend_face_residual_x!(i, j, jy, x, y, u1, ua,
                                                       eq, dt, grid, op,
                                                       scheme, param,Fn, aux,
                                                       res)
            for ix in Base.OneTo(nd)
               multiply_add_to_node_vars!(res,
                                          blend_factors[1] * dt/dx[i-1] * br[ix], Fn,
                                          eq,
                                          ix, jy, i-1, j )

               multiply_add_to_node_vars!(res,
                                          blend_factors[2] * dt/dx[i]   * bl[ix], Fn,
                                          eq,
                                          ix, jy, i, j )
            end
         end
      end
   end

   # Horizontal faces, y flux
   for j=1:ny+1
      @threaded for i=1:nx # thread here to avoid race condition
         # Face between (i,j-1) and (i,j)
         y = yf[j]
         ual, uar = get_node_vars(ua, eq, i, j-1), get_node_vars(ua, eq, i, j)
         for ix in Base.OneTo(nd)
            x = xf[i] + xg[ix] * dx[i]
            Fl, Fr = get_node_vars(Fb, eq, ix, 4, i, j-1), get_node_vars(Fb, eq, ix, 3, i, j)
            Ul, Ur = get_node_vars(Ub, eq, ix, 4, i, j-1), get_node_vars(Ub, eq, ix, 3, i, j)
            X  = SVector{2}(x,y)
            Fn = numerical_flux(X, ual, uar, Fl, Fr, Ul, Ur, eq, 2)
            Fn, blend_factors = blend_face_residual_y!(i, j, ix, x, y,
                                                       u1, ua, eq, dt, grid, op,
                                                       scheme, param, Fn, aux,
                                                       res)
            for jy in Base.OneTo(nd)
               multiply_add_to_node_vars!(res,
                                          blend_factors[1] * dt/dy[j-1] * br[jy], Fn,
                                          eq,
                                          ix, jy, i, j-1 )
               multiply_add_to_node_vars!(res,
                                          blend_factors[2] * dt/dy[j]   * bl[jy], Fn,
                                          eq,
                                          ix, jy, i, j   )
            end
         end
      end
   end
   return nothing
   end # timer
end

#-------------------------------------------------------------------------------
# Fill some data in ghost cells using periodicity
#-------------------------------------------------------------------------------
function update_ghost_values_periodic!(eq::AbstractEquations{2}, problem, Fb,
                                       Ub)
   nvar, nd, nx, ny = size(Fb,1), size(Fb,2), size(Fb,4)-2, size(Fb,5)-2
   @unpack periodic_x, periodic_y = problem
   if periodic_x
      # Left ghost cells
      for j=1:ny, ix=1:nd
         Ub_node = get_node_vars(Ub, eq, ix, 2, nx, j)
         set_node_vars!(Ub, Ub_node, eq, ix, 2, 0, j)

         Fb_node = get_node_vars(Fb, eq, ix, 2, nx, j)
         set_node_vars!(Fb, Fb_node, eq, ix, 2, 0, j)
      end

      # Right ghost cells
      for j=1:ny, ix=1:nd
         Ub_node = get_node_vars(Ub, eq, ix, 1, 1, j)
         set_node_vars!(Ub, Ub_node, eq, ix, 1, nx+1, j)

         Fb_node = get_node_vars(Fb, eq, ix, 1, 1, j)
         set_node_vars!(Fb, Fb_node, eq, ix, 1, nx+1, j)
      end
   end

   if periodic_y
      # Bottom ghost cells
      for iy=1:nd, i=1:nx
         Ub_node = get_node_vars(Ub, eq, iy, 4, i, ny)
         set_node_vars!(Ub, Ub_node, eq, iy, 4, i, 0)

         Fb_node = get_node_vars(Fb, eq, iy, 4, i, ny)
         set_node_vars!(Fb, Fb_node, eq, iy, 4, i, 0)
      end

      # Top ghost cells
      for iy=1:nd, i=1:nx
         Ub_node = get_node_vars(Ub, eq, iy, 3, i, 1)
         set_node_vars!(Ub, Ub_node, eq, iy, 3, i, ny+1)

         Fb_node = get_node_vars(Fb, eq, iy, 3, i, 1)
         set_node_vars!(Fb, Fb_node, eq, iy, 3, i, ny+1)
      end
   end

   return nothing
end

function update_ghost_values_fn_blend!(eq::AbstractEquations{2}, problem, grid,
                                       aux)
   # This looks unnecessary. The places where these values are used are redundant.
   @unpack blend = aux
   @unpack fn_low = blend.cache
   nx, ny = grid.size
   nd = size(fn_low, 3)
   nvar = nvariables(eq)
   # if problem.periodic_x || problem.periodic_y
   #    # Create a function that does it like this
   #    # fn_ghost = @view fn_low[:,:,2,0,:]
   #    # fn_physical = @view fn_low[:,2,nx,:]
   #    # @turbo fn_ghost .= fn_physical

   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 2:2, 0:0, 1:ny)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 2:2, nx:nx, 1:ny)))
   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 1:1, nx+1:nx+1, 1:ny)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 1:1, 1:1 , 1:ny)))

   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 4:4, 1:nx, 0:0)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 4:4, 1:nx, ny:ny)))
   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 3:3, 1:nx, ny+1:ny+1)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 3:3, 1:nx, 1:1)))
   # else
   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 2:2, 0:0, 1:ny)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 2:2, 1:1, 1:ny)))
   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 1:1, nx+1:nx+1, 1:ny)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 1:1, nx:nx, 1:ny)))

   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 4:4, 1:nx, 0:0)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 4:4, 1:nx, 1:1)))
   #    copyto!(fn_low, CartesianIndices((1:nvar, 1:nd, 3:3, 1:nx, ny+1:ny+1)),
   #            fn_low, CartesianIndices((1:nvar, 1:nd, 3:3, 1:nx, ny:ny)))
   # end
end

#-------------------------------------------------------------------------------
# Limiter function
#-------------------------------------------------------------------------------
function apply_bound_limiter!(eq::AbstractEquations{2,1}, grid, scheme, param, op,
                              ua, u1, aux)
   nx, ny = grid.size
   @unpack nvar = eq
   return nothing
end

function setup_limiter_tvb(eq::AbstractEquations{2}; tvbM = 0.0, beta = 1.0)
   cache_size = 28
   # Make the particular initializers into functions
   MArr = MArray{Tuple{nvariables(eq),1}, Float64}
   cache = alloc_for_threads(MArr, cache_size)
   limiter = (;name = "tvb", tvbM = tvbM, cache, beta = beta)
   return limiter
end

function setup_limiter_tvbβ( eq::AbstractEquations{2}; tvbM = 0.0, beta = 1.0)
   cache_size = 24
   # Make the particular initializers into functions
   MArr = MArray{Tuple{nvariables(eq),1}, Float64}
   cache = alloc_for_threads(MArr, cache_size)
   limiter = (;name = "tvbβ", tvbM = tvbM, cache, beta = beta)
   return limiter
end

function apply_tvb_limiter!(eq::AbstractEquations{2,1}, problem, scheme, grid,
                            param, op, ua, u1, aux)
   @timeit aux.timer "TVB Limiter" begin
   nx, ny = grid.size
   @unpack xg, wg, Vl, Vr = op
   @unpack tvbM, cache, beta = scheme.limiter
   nd = length(wg)
   # Loop over cells

   u1_ = @view u1[1,:,:,:,:]
   ua_ = @view ua[1,:,:]

   @threaded for element in CartesianIndices((1:nx, 1:ny))
      el_x, el_y = element[1], element[2]
      # face averages
      ul, ur, ub, ut = 0.0, 0.0, 0.0, 0.0
      for jj=1:nd, ii=1:nd
         ul += u1_[jj, ii, el_x, el_y] * Vl[jj] * wg[ii] # transpose(u1) * Vl . wg
         ur += u1_[jj, ii, el_x, el_y] * Vr[jj] * wg[ii] # transpose(u1) * Vr . wg
         ub += u1_[ii, jj, el_x, el_y] * Vl[jj] * wg[ii] # u1 * Vl * wg
         ut += u1_[ii, jj, el_x, el_y] * Vr[jj] * wg[ii] # u1 * Vr * wg
      end

      # slopes b/w centres and faces
      dul, dur = ua_[el_x,el_y] - ul, ur - ua_[el_x,el_y]
      dub, dut = ua_[el_x,el_y] - ub, ut - ua_[el_x,el_y]
      # minmod to detect jumps
      Mdx2, Mdy2 = tvbM * grid.dx[el_x]^2, tvbM * grid.dy[el_y]^2
      dulm = minmod(dul, ua_[el_x,el_y] - ua_[el_x-1,el_y], ua_[el_x+1,el_y] - ua_[el_x,el_y], Mdx2)
      durm = minmod(dur, ua_[el_x,el_y] - ua_[el_x-1,el_y], ua_[el_x+1,el_y] - ua_[el_x,el_y], Mdx2)
      dubm = minmod(dub, ua_[el_x,el_y] - ua_[el_x,el_y-1], ua_[el_x,el_y+1] - ua_[el_x,el_y], Mdy2)
      dutm = minmod(dut, ua_[el_x,el_y] - ua_[el_x,el_y-1], ua_[el_x,el_y+1] - ua_[el_x,el_y], Mdy2)
      # limit if jumps are detected
      if ( (abs(dul-dulm)>1e-06 || abs(dur-durm)>1e-06) ||
            (abs(dub-dubm)>1e-06 || abs(dut-dutm)>1e-06) )
         dux, duy = 0.5 * (dulm+durm), 0.5 * (dutm+dubm)
          for jj=1:nd, ii=1:nd # Adding @turbo here was giving bugs. WHY!?
            u1_[ii,jj,el_x,el_y] = ( ua_[el_x,el_y] + 2.0 * (xg[ii]-0.5) * dux
                                                    + 2.0 * (xg[jj]-0.5) * duy )
         end
      end
   end
   return nothing
   end # timer
end

# dflo version
function apply_tvb_limiterβ!(eq::AbstractEquations{2,1}, problem, scheme, grid,
                             param, op, ua, u1, aux)
   @timeit aux.timer "TVB Limiter" begin
   nx, ny = grid.size
   @unpack xg, wg, Vl, Vr = op
   @unpack dx, dy = grid
   @unpack tvbM, cache, beta = scheme.limiter
   @unpack nvar = eq
   nd = length(wg)

   refresh!(u) = fill!(u, zero(eltype(u)))
   # Pre-allocate for each thread

   # Loop over cells
   @threaded for ij in CartesianIndices((1:nx, 1:ny))
      id = Threads.threadid()
      el_x, el_y = ij[1], ij[2]
      # face averages
      (ul, ur, ud, uu,
       dux, duy,
       dual, duar, duad, duau,
       duxm, duym, dux, duy) = cache[id]
      u1_ = @view u1[:, :, :, el_x, el_y]
      ua_, ual, uar, uad, uau = (get_node_vars(ua, eq, el_x  , el_y),
                                 get_node_vars(ua, eq, el_x-1, el_y),
                                 get_node_vars(ua, eq, el_x+1, el_y),
                                 get_node_vars(ua, eq, el_x  , el_y-1),
                                 get_node_vars(ua, eq, el_x  , el_y+1))
      refresh!.((ul, ur, ud, uu))
      for j in Base.OneTo(nd), i in Base.OneTo(nd)
         u_ = get_node_vars(u1_, eq, i, j)
         multiply_add_to_node_vars!(ul, Vl[i]*wg[j], u_, eq, 1)
         multiply_add_to_node_vars!(ur, Vr[i]*wg[j], u_, eq, 1)
         multiply_add_to_node_vars!(ud, Vl[j]*wg[i], u_, eq, 1)
         multiply_add_to_node_vars!(uu, Vr[j]*wg[i], u_, eq, 1)
      end
      # TODO - Give better names to these quantities
      # slopes b/w centres and faces
      ul_ , ur_  = get_node_vars(ul, eq, 1), get_node_vars(ur, eq, 1)
      ud_ , uu_  = get_node_vars(ud, eq, 1), get_node_vars(uu, eq, 1)
      ual_, uar_ = get_node_vars(ual, eq, 1), get_node_vars(uar, eq, 1)
      uad_, uau_ = get_node_vars(uad, eq, 1), get_node_vars(uau, eq, 1)

      multiply_add_set_node_vars!(dux, 1.0, ur_, -1.0, ul_, eq, 1)
      multiply_add_set_node_vars!(duy, 1.0, uu_, -1.0, ud_, eq, 1)

      multiply_add_set_node_vars!(dual, 1.0, ua_ , -1.0, ual_, eq, 1)
      multiply_add_set_node_vars!(duar, 1.0, uar_, -1.0, ua_ , eq, 1)
      multiply_add_set_node_vars!(duad, 1.0, ua_ , -1.0, uad_, eq, 1)
      multiply_add_set_node_vars!(duau, 1.0, uau_, -1.0, ua_ , eq, 1)

      dux_ = get_node_vars(dux, eq, 1)
      dual_, duar_ = get_node_vars(dual, eq, 1), get_node_vars(duar, eq, 1)
      duy_ = get_node_vars(duy, eq, 1)
      duad_, duau_ = get_node_vars(duad, eq, 1,), get_node_vars(duau, eq, 1)

      Mdx2, Mdy2 = tvbM * dx[el_x]^2, tvbM * dy[el_y]^2
      for n in Base.OneTo(nvar)
         duxm[n] = minmod(dux_[n], beta*dual_[n], beta*duar_[n], Mdx2)
         duym[n] = minmod(duy_[n], beta*duad_[n], beta*duau_[n], Mdy2)
      end

      jump_x = jump_y = 0.0
      duxm_ = get_node_vars(duxm, eq, 1)
      duym_ = get_node_vars(duym, eq, 1)
      for n=1:nvar
         jump_x += abs(dux_[n] - duxm_[n])
         jump_y += abs(duy_[n] - duym_[n])
      end
      jump_x /= nvar
      jump_y /= nvar
      if jump_x + jump_y > 1e-10
         for j in Base.OneTo(nd), i in Base.OneTo(nd)
            multiply_add_set_node_vars!(u1_,
                                        1.0, ua_,
                                       #  2.0 * (xg[i] - 0.5),
                                        xg[i] - 0.5,
                                        duxm_,
                                       #  2.0 * (xg[j] - 0.5),
                                        xg[j] - 0.5,
                                        duym_,
                                        eq, i, j)
         end
      end
   end
   # @assert false
   return nothing
   end # timer
end

struct Blend2D{Cache, Parameters, Subroutines}
   cache::Cache
   parameters::Parameters
   subroutines::Subroutines
end

function modal_smoothness_indicator(eq::AbstractEquations{2}, t, iter, fcount,
                                    dt, grid, scheme,
                                    problem, param, aux, op, u1, ua)
   modal_smoothness_indicator_gassner(eq, t, iter, fcount, dt, grid,
                                      scheme, problem, param, aux, op,
                                      u1, ua)
end

function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3}, matrix::AbstractMatrix,
                                 data_in:: AbstractArray{<:Any, 3},
                                 tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix, 1), size(matrix, 2)))

  # Interpolate in x-direction
  # @tullio threads=false tmp1[v, i, j]     = matrix[i, ii] * data_in[v, ii, j]
  @turbo for j in axes(tmp1, 3), i in axes(tmp1, 2), v in axes(tmp1, 1)
    res = zero(eltype(tmp1))
    for ii in axes(matrix, 2)
      res += matrix[i, ii] * data_in[v, ii, j]
    end
    tmp1[v, i, j] = res
  end

  # Interpolate in y-direction
  # @tullio threads=false data_out[v, i, j] = matrix[j, jj] * tmp1[v, i, jj]
  @turbo for j in axes(data_out, 3), i in axes(data_out, 2), v in axes(data_out, 1)
    res = zero(eltype(data_out))
    for jj in axes(matrix, 2)
      res += matrix[j, jj] * tmp1[v, i, jj]
    end
    data_out[v, i, j] = res
  end

  return nothing
end

function modal_smoothness_indicator_gassner(eq::AbstractEquations{2}, t, iter,
                                            fcount, dt, grid,
                                            scheme, problem, param, aux, op,
                                            u1, ua)
   @timeit aux.timer "Blending limiter" begin
   @unpack dx, dy = grid
   nx, ny = grid.size
   @unpack nvar = eq
   @unpack xg = op
   nd = length(xg)
   @unpack limiter = scheme
   @unpack blend = aux
   @unpack constant_node_factor = blend.parameters
   @unpack amax = blend.parameters      # maximum factor of the lower order term
   @unpack tolE = blend.parameters      # tolerance for denominator
   @unpack E = blend.cache            # content in high frequency nodes
   @unpack alpha, alpha_temp = blend.cache    # vector containing smoothness indicator values
   @unpack (c, a, amin, a0, a1, smooth_alpha, smooth_factor
            ) = blend.parameters # smoothing coefficients
   @unpack get_indicating_variables! = blend.subroutines
   @unpack cache = blend
   @unpack Pn2m = cache

   @threaded for element in CartesianIndices((1:nx, 1:ny))
      el_x, el_y = element[1], element[2]
      un, um, tmp = cache.nodal_modal[Threads.threadid()]
      # Continuous extension to faces
      u = @view u1[:,:,:,el_x,el_y]
      @turbo un .= u

      # Copying is needed because we replace these with variables actually
      # used for indicators like primitives or rho*p, etc.

      # Convert un to ind var, get no. of variables used for indicator
      n_ind_nvar = get_indicating_variables!(un, eq)

      multiply_dimensionwise!(um, Pn2m, un, tmp)

      # ind = zeros(n_ind_nvar)
      ind = 0.0
      # TODO - URGENTLY FIX IT, YOU ARE ASSUMING n_ind_var = 1 !!

      for n=1:n_ind_nvar
         # um[n,1,1] *= constant_node_factor
         # TODO - avoid redundant calculations in total_energy_clip1, 2, etc.?
         total_energy = total_energy_clip1 = total_energy_clip2 = 0.0
         for j in Base.OneTo(nd), i in Base.OneTo(nd) # TODO - Why is @turbo bad here?
            total_energy += um[n,i,j]^2
         end
         total_energy += -um[n,1,1]^2 + (constant_node_factor*um[n,1,1])^2
         for j in Base.OneTo(nd-1), i in Base.OneTo(nd-1)
            total_energy_clip1 += um[n,i,j]^2
         end
         for j in Base.OneTo(nd-2), i in Base.OneTo(nd-2)
            total_energy_clip2 += um[n,i,j]^2
         end

         if total_energy > tolE
            ind1 = (total_energy - total_energy_clip1) / total_energy
         else
            ind1 = 0.0
         end

         if total_energy_clip1 > tolE
            ind2 = (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
         else
            ind2 = 0.0
         end

         ind = max(ind1, ind2)
      end
      E[el_x,el_y] = maximum(ind) # maximum content among all indicating variables

      T = a * 10^( -c * nd^(0.25) ) # TODO - Should we have something other than N+1 here?
      # alpha(E=0) = 0.0001
      s = log( (1.0 - 0.0001)/0.0001 )  # chosen to ensure so that E = 0 => alpha = amin
      alpha[el_x,el_y] = 1.0 / (1.0 + exp( (-s/T) * (E[el_x,el_y] - T) ))

      if alpha[el_x,el_y] < amin # amin = 0.0001. TODO - Will this ever even happen?
         alpha[el_x,el_y] = 0.0
      elseif alpha[el_x,el_y] > 1.0 - amin
         alpha[el_x,el_y] = 1.0
      end

      alpha[el_x,el_y] = min(alpha[el_x,el_y], amax)
   end

   if problem.periodic_x
      @turbo for j=1:ny
         alpha[0,j] = alpha[nx,j]
         alpha[nx+1,j] = alpha[1,j]
      end
   else
      @turbo for j=1:ny
         alpha[0,j] = alpha[1,j]
         alpha[nx+1,j] = alpha[nx,j]
      end
   end

   if problem.periodic_y
      @turbo for i=1:nx
         alpha[i,0] = alpha[i,ny]
         alpha[i,ny+1] = alpha[i,1]
      end
   else
      @turbo for i=1:nx
         alpha[i,0] = alpha[i,1]
         alpha[i,ny+1] = alpha[i,ny]
      end
   end

   # Smoothening of alpha
   if smooth_alpha == true
      @turbo alpha_temp .= alpha
      for j=1:ny, i=1:nx
         alpha[i,j] = max(smooth_factor*alpha_temp[i-1,j],
                          smooth_factor*alpha_temp[i,j-1],
                        #   smooth_factor*alpha_temp[i-1,j-1],
                          alpha[i,j],
                          smooth_factor*alpha_temp[i+1,j],
                          smooth_factor*alpha_temp[i,j+1],
                        #   smooth_factor*alpha_temp[i+1,j+1],
                        )
      end
   end

   if dt > 0.0
      blend.cache.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
   end

   if limiter.pure_fv == true
      @assert scheme.limiter.name == "blend"
      @turbo alpha .= one(eltype(alpha))
   end

   end # timer
end

# TODO - Wouldn't it better to just convert u1 in place?
function Hierarchical(eq::AbstractEquations{2}, op, grid, problem,
                                   scheme, param,
                                   plot_data)
   return nothing
end

#------------------------------------------------------------------------------
# Blending Limiter
#------------------------------------------------------------------------------
function set_blend_dt!(eq::AbstractEquations{2}, aux, dt)
   @unpack blend = aux
   blend.cache.dt[1] = dt
end

# TODO - This is repeated in FR1D
@inline function conservative_indicator!(un, eq::AbstractEquations{2})
   @unpack nvar = eq
   n_ind_var = nvar
   return n_ind_var
end

@inbounds @inline function conservative_reconstruction!(ue, ua,
                                                        eq::AbstractEquations{2})
   return nothing
end

@inbounds @inline function conservative2conservative!(ue, ua,
                                                      eq::AbstractEquations{2})
   return nothing
end

conservative_reconstruction = (conservative_reconstruction!,
                               conservative2conservative!)

@inbounds @inline function primitive_reconstruction!(ue, ua,
                                                     eq::AbstractEquations{2})
   con2prim!(eq, ue)
end

@inbounds @inline function primitive2conservative!(ue, ua,
                                                   eq::AbstractEquations{2})
   prim2con(eq, ue)
end

primitive_reconstruction = (primitive_reconstruction!,
                            primitive2conservative!)

# Create Blend2D struct
function Blend(eq::AbstractEquations{2}, op, grid,
               problem::Problem,
               scheme::Scheme,
               param::Parameters,
               plot_data)
   @unpack limiter = scheme

   if limiter.name != "blend"
      subroutines = (;blend_cell_residual! = trivial_cell_residual,
                      blend_face_residual_x! = trivial_face_residual,
                      blend_face_residual_y! = trivial_face_residual)
      cache = (;
                dt = MVector(1.0e20) # filler
               )
      # If limiter is not blend, replace blending with 'do nothing functions'
      return (; subroutines,cache
               )
   end

   println("Setting up blending limiter...")

   # TODO - Add strings with names to these parameters like
   # indicating_variables, reconstruction_variables, etc
   @unpack ( blend_type, indicating_variables, reconstruction_variables,
             indicator_model, amax, constant_node_factor,
             smooth_alpha, smooth_factor,
             c, a, amin,
             debug_blend, pure_fv ) = limiter


   @unpack xc, yc, xf, yf, dx, dy = grid
   nx, ny = grid.size
   @unpack degree, xg = op
   # @assert Threads.nthreads() == 1
   @assert indicator_model == "gassner" "Other models not implemented"
   @assert degree > 2 || pure_fv == true
   nd = degree + 1
   @unpack nvar = eq

   E1 = a*10^(-c*(degree+3)^0.25)
   E0 = E1 * 1e-2 # E < E0 implies smoothness
   tolE = 1.0e-6  # If denominator < tolE, do purely high order
   a0 = 1.0/3.0; a1 = 1.0 - 2.0*a0              # smoothing coefficients
   parameters = (; E1, E0, tolE, amax, a0, a1, constant_node_factor,
                   smooth_alpha, smooth_factor,
                   c, a, amin,
                   pure_fv, debug=debug_blend)

   # Big arrays
   E = zeros(nx, ny)
   alpha = OffsetArray(zeros(nx+2, ny+2), OffsetArrays.Origin(0,0))
   alpha_temp = similar(alpha)
   fn_low = OffsetArray(zeros(nvar,
                               nd, # Dofs on each face
                               4,  # 4 faces
                               nx+2, ny+2),
                        OffsetArrays.Origin(1,1,1,0,0))

   # Small cache of many MMatrix with one copy per thread
   abstract_constructor(tuple_,x , origin) = [OffsetArray(MArray{tuple_,Float64}(x),
                                                         OffsetArrays.Origin(origin))]
   # These square brackets are needed when cache_size = 1. Also, when
   # Cache size is > 1, a [1] is needed in constructor

   constructor = x -> abstract_constructor(Tuple{nvar,nd+2,nd+2}, x, (1,0,0))
   ue = alloc_for_threads(constructor, 1) # u extended by face extrapolation
   constructor = x -> abstract_constructor(Tuple{nd+1},x, (0))[1]
   subcell_faces = alloc_for_threads(constructor, 2) # faces of subcells

   constructor = x -> abstract_constructor(Tuple{nd+2}, x, (0))[1]
   solution_points = alloc_for_threads(constructor, 2)

   constructor = x -> abstract_constructor(Tuple{nvar,4,nd+2,nd+2}, x,
                                           (1,1,0,0))
   unph = alloc_for_threads(constructor, 1)

   constructor = MArray{Tuple{nvar,nd,nd}, Float64}
   nodal_modal = alloc_for_threads(constructor, 3) # stores un, um and a temp

   constructor = MArray{Tuple{nvar}, Float64}
   slopes = alloc_for_threads(constructor, 2)

   Pn2m = nodal2modal(xg)

   cache = (;alpha, alpha_temp, E, ue,
             subcell_faces,
             solution_points,
             fn_low, dt = MVector(1.0),
             nodal_modal, unph, Pn2m, slopes)

   println("Setting up $blend_type blending limiter with $indicator_model "
            * "with $indicating_variables indicating variables")

   @show blend_type
   @unpack cell_residual!, face_residual_x!, face_residual_y! = blend_type
   conservative2recon!, recon2conservative! = reconstruction_variables

   subroutines = (; blend_cell_residual! = cell_residual!,
                    blend_face_residual_x! = face_residual_x!,
                    blend_face_residual_y! = face_residual_y!,
                    conservative2recon!, recon2conservative!,
                    get_indicating_variables! = indicating_variables )

   Blend2D(cache, parameters, subroutines)
end

function update_ghost_values_u1!(eq::AbstractEquations{2}, problem, grid, op, u1, t)
   nx, ny = grid.size
   nd = op.degree + 1
   @unpack xg = op
   nvar = size(u1,1)
   if problem.periodic_x
      copyto!(u1, CartesianIndices((1:nvar, 1:nd, 1:nd, 0:0, 1:ny)),
              u1, CartesianIndices((1:nvar, 1:nd, 1:nd, nx:nx, 1:ny)))
      copyto!(u1, CartesianIndices((1:nvar, 1:nd, 1:nd, nx+1:nx+1, 1:ny)),
              u1, CartesianIndices((1:nvar, 1:nd, 1:nd, 1:1, 1:ny)))
   end

   if problem.periodic_y
      copyto!(u1, CartesianIndices((1:nvar, 1:nd, 1:nd, 1:nx, 0:0)),
              u1, CartesianIndices((1:nvar, 1:nd, 1:nd, 1:nx, ny:ny)))
      copyto!(u1, CartesianIndices((1:nvar, 1:nd, 1:nd, 1:nx, ny+1:ny+1)),
              u1, CartesianIndices((1:nvar, 1:nd, 1:nd, 1:nx, 1:1)))
   end

   if problem.periodic_x && problem.periodic_y
      return nothing
   end
   left, right, bottom, top = problem.boundary_condition
   boundary_value = problem.boundary_value
   @unpack dx, dy, xf, yf = grid
   ub = zeros(nvar)
   if left == dirichlet
      x  = xf[1]
      for j=1:ny
         for k=1:nd
            y  = yf[j] + xg[k] * dy[j]
            ub .= boundary_value(x,y, t) # TODO - Don't allocate so much
            for n=1:nvar
               u1[n,1:nd,k,0,j] .= ub[n]
            end
            # Purely upwind. # TODO - Remove it soon!!
            # if abs(y) < 0.05
            #    for n=1:nvar
            #       u1[n,1:nd,k,1,j] .= ub[n]
            #    end
            # end
         end
      end
   elseif left == neumann
      for j=1:ny
         for n=1:nvar
            u1[n,1:nd,1:nd,0,j] .= @view u1[n,1:nd,1:nd,1,j]
         end
      end
   elseif left == reflect
      for j=1:ny
         for n=1:nvar
            u1[n,1:nd,1:nd,0,j] .= @view u1[n,1:nd,1:nd,1,j]
         end
         @views u1[2,1:nd,1:nd,0,j] .= -u1[2,1:nd,1:nd,0,j]
      end
   else
      println("Incorrect bc specified at left.")
      @assert false
   end

   if right == dirichlet
      for j=1:ny
         for k=1:nd
            x   = xf[nx+1]
            y   = yf[j] + xg[k] * dy[j]
            ub .= boundary_value(x,y, t)
            for n=1:nvar
               u1[n,1:nd,k,nx+1,j] .= ub[n]
            end
         end
      end
   elseif right == neumann
      for j=1:ny
         u1[1:nvar,1:nd,1:nd,nx+1,j] .= @view u1[1:nvar, 1:nd, 1:nd, nx, j]
      end
   elseif right == reflect
      for j=1:ny
         for n=1:nvar
            u1[n,1:nd,1:nd,nx+1,j] .= @view u1[n, 1:nd, 1:nd, nx, j]
         end
         @views u1[2,1:nd,1:nd,nx+1,j] .= -u1[2,1:nd,1:nd,nx+1,j]
      end
   else
      println("Incorrect bc specified at right.")
      @assert false
   end

   if bottom == dirichlet
      y = yf[1]
      for i=1:nx
         for k=1:nd
            x = xf[i] + xg[k]*dx[i]
            ub = boundary_value(x,y, t)
            u1[1:nvar, k, 1:nd, i, 0] .= ub
         end
      end
   elseif bottom == neumann
      u1[1:nvar,1:nd,1:nd,1:nx,0] .= @view u1[1:nvar,1:nd,1:nd,1:nx,1]
   elseif bottom == reflect
      u1[1:nvar,1:nd,1:nd,1:nx,0] .= @view u1[1:nvar,1:nd,1:nd,1:nx,1]
      @views u1[3,1:nd,1:nd,1:nx,0] .= -u1[3,1:nd,1:nd,1:nx,0]
   else
      bc! = bottom[3]
      bc!(op, grid, u1)
   end

   if top == dirichlet
      y = yf[ny+1]
      for i=1:nx
         for k=1:nd
            x = xf[i] + xg[k]*dx[i]
            ub = boundary_value(x,y, t)
            u1[1:nvar, k, 1:nd, i, ny+1] .= ub
         end
      end
   elseif top == neumann
      u1[1:nvar, 1:nd, 1:nd, 1:nx, ny+1] .= @view u1[1:nvar, 1:nd, 1:nd, 1:nx, ny]
   elseif top == reflect
      u1[1:nvar, 1:nd, 1:nd, 1:nx, ny+1] .= @view u1[1:nvar, 1:nd, 1:nd, 1:nx, ny]
      @views u1[3, 1:nd, 1:nd, 1:nx, ny+1] .= -u1[3, 1:nd, 1:nd, 1:nx, ny+1]
   else
      top[3](op, grid, u1)
   end

end

function zhang_shu_flux_fix(eq::AbstractEquations{2},
                            uprev,    # Solution at previous time level
                            ulow,     # low order update
                            Fn,       # Blended flux candidate
                            fn_inner, # Inner part of flux
                            fn,       # low order flux
                            c         # c is such that unew = u - c(fr-fl)
                            )
   # This method is to be defined for each equation
   return Fn
end

function blend_cell_residual_fo!(el_x, el_y, eq::AbstractEquations{2}, scheme,
                                 aux, dt, dx, dy, xf, yf, op, u1, u_, f, res)
   @timeit_debug aux.timer "Blending limiter" begin
   @unpack blend = aux
   @unpack Vl, Vr, xg, wg = op
   num_flux = scheme.numerical_flux
   nd = length(xg)

   id = Threads.threadid()
   xxf, yyf = blend.cache.subcell_faces[id]
   @unpack fn_low = blend.cache
   alpha = blend.cache.alpha[el_x,el_y]

   u = @view u1[:,:,:,el_x,el_y]
   r = @view res[:,:,:,el_x,el_y]

   # limit the higher order part
   lmul!(1.0 - alpha, r)

   # compute subcell faces
   xxf[0], yyf[0] = xf, yf
   for ii in Base.OneTo(nd)
      xxf[ii] = xxf[ii-1] + dx*wg[ii]
      yyf[ii] = yyf[ii-1] + dy*wg[ii]
   end

   # loop over vertical inner faces between (ii-1,jj) and (ii,jj)
   for ii in 2:nd # skipping the supercell face for blend_face_residual
      xx = xxf[ii-1] # Face x coordinate, offset because index starts from 0
      for jj in Base.OneTo(nd)
         yy = yf + dy*xg[jj] # Face y coordinates picked same as soln points
         X  = SVector(xx, yy)
         ul, ur = get_node_vars(u, eq, ii-1, jj), get_node_vars(u, eq, ii, jj)
         fl, fr = flux(xx, yy, ul, eq, 1), flux(xx, yy, ur, eq, 1)
         fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)
         multiply_add_to_node_vars!(r, # r[ii-1,jj]+=alpha*dt/(dx*wg[ii-1])*fn
                                    alpha * dt / (dx*wg[ii-1]),
                                    fn, eq, ii-1, jj)
         multiply_add_to_node_vars!(r, # r[ii,jj]+=alpha*dt/(dx*wg[ii])*fn
                                    - alpha * dt / (dx*wg[ii]),
                                    fn, eq, ii, jj)
         # TODO - Can checking this in every step of the loop be avoided
         if ii == 2
            set_node_vars!(fn_low, fn, eq, jj, 1, el_x, el_y)
         elseif ii == nd
            set_node_vars!(fn_low, fn, eq, jj, 2, el_x, el_y)
         end
      end
   end

   # loop over horizontal inner faces between (ii,jj-1) and (ii,jj)
   for jj in 2:nd
      yy = yyf[jj-1] # face y coordinate, offset because index starts from 0
      for ii in Base.OneTo(nd)
         xx = xf + dx*xg[ii] # face x coordinate picked same as soln pt
         X  = SVector(xx, yy)
         ul, ur = get_node_vars(u, eq, ii, jj-1), get_node_vars(u, eq, ii, jj)
         fl, fr = flux(xx, yy, ul, eq, 2), flux(xx, yy, ur, eq, 2)
         fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 2)
         multiply_add_to_node_vars!(r, # r[ii,jj-1]+=alpha*dt/(dy*wg[jj-1])*fn
                                    alpha * dt/(dy*wg[jj-1]),
                                    fn,
                                    eq, ii, jj-1
                                   )
         multiply_add_to_node_vars!(r, # r[ii,jj]+=alpha*dt/(dy*wg[jj])*fn
                                    - alpha * dt/(dy*wg[jj]),
                                    fn,
                                    eq, ii, jj
                                    )
         # TODO - Can checking this in every step of the loop be avoided
         if jj == 2
            set_node_vars!(fn_low, fn, eq, ii, 3, el_x, el_y)
         elseif jj == nd
            set_node_vars!(fn_low, fn, eq, ii, 4, el_x, el_y)
         end
      end
   end

   end # timer
end

function finite_differences(h1, h2, ul, u, ur)
   back_diff = (u - ul)/h1
   fwd_diff = (ur - u)/h2
   a, b, c = -( h2/(h1*(h1+h2)) ), (h2-h1)/(h1*h2), ( h1/(h2*(h1+h2)) )
   cent_diff = a*ul + b*u + c*ur
   return back_diff, cent_diff, fwd_diff
end

function blend_cell_residual_muscl!(el_x, el_y, eq::AbstractEquations{2}, scheme,
                                    aux, dt, dx, dy, xf, yf, op, u1, ::Any, f, res)
   @timeit_debug aux.timer "Blending limiter" begin
   @unpack blend = aux
   @unpack xg, wg = op
   num_flux = scheme.numerical_flux
   nd = length(xg)
   nvar = nvariables(eq)

   id = Threads.threadid()
   xxf, yyf = blend.cache.subcell_faces[id]
   xe, ye   = blend.cache.solution_points[id] # cell points & 2 from neighbours
   ue = blend.cache.ue[id][1] # solution values in cell + 2 from neighbours
   unph = blend.cache.unph[id][1] # face values evolved to to time level n+1/2
   dt = blend.cache.dt[1] # For support with DiffEq
   @unpack fn_low = blend.cache
   alpha = blend.cache.alpha[el_x,el_y]
   # beta = 2.0 - alpha # TODO - Move beta to blend.parameters

   @unpack conservative2recon!, recon2conservative! = blend.subroutines

   u = @view u1[:,:,:,el_x,el_y]
   r = @view res[:,:,:,el_x,el_y]

   # limit the higher order part
   lmul!(1.0 - alpha, r)

   # compute subcell faces
   xxf[0], yyf[0] = xf, yf
   for ii in Base.OneTo(nd)
      xxf[ii] = xxf[ii-1] + dx*wg[ii]
      yyf[ii] = yyf[ii-1] + dy*wg[ii]
   end

   # Get solution points
   # TODO - You are incorrectly assuming uniform grid here, you should have
   # xe[0] = xf - dx[el_x-1]*(1.0-xg[nd])
   xe[0] = xf - dx*(1.0 - xg[nd])   # Last solution point of left cell
   xe[1:nd] .= xf .+ dx*xg          # Solution points inside the cell
   xe[nd+1] = xf + dx*(1.0 + xg[1]) # First point of right cell

   ye[0] = yf - dy*(1.0 - xg[nd])   # Last solution point of lower cell
   ye[1:nd] .= yf .+ dy*xg           # solution points inside the cell
   ye[nd+1] = yf + dy*(1.0 + xg[1]) # First point of upper cell

   # get solution point values # TODO - Add @turbo here
   # ue[:,1:nd,1:nd] .= u # values from current cell

   for j in Base.OneTo(nd), i in Base.OneTo(nd), n in eachvariable(eq)
      ue[n,i,j] = u[n,i,j] # values from current cell
   end

   for k in Base.OneTo(nd), n in eachvariable(eq)
      ue[n,0   ,k] = u1[n,nd,k,el_x-1,el_y] # values from left neighbour
      ue[n,nd+1,k] = u1[n,1 ,k,el_x+1,el_y] # values from right neighbour

      ue[n,k,0   ] = u1[n,k,nd,el_x,el_y-1] # values from lower neighbour
      ue[n,k,nd+1] = u1[n,k,1 ,el_x,el_y+1] # values from upper neighbour
   end

   # @views ue[:,0   ,1:nd] .= u1[:,nd,1:nd,el_x-1,el_y] # values from left neighbour
   # @views ue[:,nd+1,1:nd] .= u1[:,1 ,1:nd,el_x+1,el_y] # values from right neighbour

   # @views ue[:,1:nd,0   ] .= u1[:,1:nd,nd,el_x,el_y-1] # values from lower neighbour
   # @views ue[:,1:nd,nd+1] .= u1[:,1:nd,1 ,el_x,el_y+1] # values from upper neighbour

   # Loop over subcells
   for jj in Base.OneTo(nd), ii in Base.OneTo(nd)
      u_ = get_node_vars(ue, eq, ii  , jj)
      ul = get_node_vars(ue, eq, ii-1, jj)
      ur = get_node_vars(ue, eq, ii+1, jj)
      ud = get_node_vars(ue, eq, ii  , jj-1)
      uu = get_node_vars(ue, eq, ii  , jj+1)

      # TODO - Add this feature
      # u_, ul, ur, ud, uu = conservative2recon.((u_,ul,ur,ud,uu))

      # Compute finite differences
      Δx1, Δx2 = xe[ii]-xe[ii-1], xe[ii+1]-xe[ii]
      back_x, cent_x, fwd_x = finite_differences(Δx1, Δx2, ul, u_, ur)
      Δy1, Δy2 = ye[jj]-ye[jj-1], ye[jj+1]-ye[jj]
      back_y, cent_y, fwd_y = finite_differences(Δy1, Δy2, ud, u_, uu)

      # Slopes of linear approximation in cell
      # Ideally, I'd name both the tuples as slope_tuple, but that
      # was giving a type instability
      # beta1 = 2.0
      # if blend.parameters.pure_fv == true # TODO - Do this in blend.paramaeters
      beta1, beta2 = 2.0-alpha, 2.0-alpha # Unfortunate way to fix type instability
      # else
      #    beta1, beta2 = 2.0 - alpha, 2.0 - alpha
      # end
      slope_tuple_x = (minmod(back_x[n], cent_x[n], fwd_x[n], beta1, 0.0)
                       for n in eachvariable(eq))

      # slope_tuple_x = (minmod(back_x[n], cent_x[n], fwd_x[n], 0.0)
      #                for n in eachvariable(eq))
      slope_x = SVector{nvar}(slope_tuple_x)
      # beta2 = 2.0
      slope_tuple_y = (minmod(back_y[n], cent_y[n], fwd_y[n], beta2, 0.0)
                       for n in eachvariable(eq))
      # slope_tuple_y = (minmod(back_y[n], cent_y[n], fwd_y[n], 0.0)
      #                for n in eachvariable(eq))
      slope_y = SVector{nvar}(slope_tuple_y)
      ufl = u_ + slope_x*(xxf[ii-1] - xe[ii]) # left  face value u_{i-1/2,j}
      ufr = u_ + slope_x*(xxf[ii]   - xe[ii]) # right face value u_{i+1/2,j}

      ufd = u_ + slope_y*(yyf[jj-1] - ye[jj]) # lower face value u_{i,j-1/2}
      ufu = u_ + slope_y*(yyf[jj]   - ye[jj]) # upper face value u_{i,j+1/2}

      # TODO - u_star's are not needed in this function, just create and use them
      # in limit_slope!

      u_star_l = u_ + 2.0*slope_x*(xxf[ii-1] - xe[ii])
      u_star_r = u_ + 2.0*slope_x*(xxf[ii]   - xe[ii])

      u_star_d = u_ + 2.0*slope_y*(yyf[jj-1] - ye[jj])
      u_star_u = u_ + 2.0*slope_y*(yyf[jj]   - ye[jj])

      ufl, ufr = limit_slope!(eq, slope_x, ufl, u_star_l, ufr, u_star_r, u_,
                              xxf[ii-1]-xe[ii], xxf[ii]-xe[ii])
      ufd, ufu = limit_slope!(eq, slope_y, ufd, u_star_d, ufu, u_star_u, u_,
                              yyf[jj-1]-ye[jj], yyf[jj]-ye[jj])

      # TODO - Add this feature
      # Convert back to conservative variables for update
      # ufl, ufr, ufd, ufu = recon2conservative.((ufl,ufr,ufd,ufu))

      fl = flux(xxf[ii-1], ye[jj], ufl, eq, 1)
      fr = flux(xxf[ii]  , ye[jj], ufr, eq, 1)

      gd = flux(xe[ii], yyf[jj-1], ufd, eq, 2)
      gu = flux(xe[ii], yyf[jj]  , ufu, eq, 2)

      # Use finite difference method to evolve face values to time level n+1/2

      multiply_add_set_node_vars!(unph, # u_{i-1/2+,j}=u_{i-1/2,j}-0.5*dt*(fr-fl)/(xfr-xfl)
                                  ufl,  # u_{i-1/2,j}
                                  - 0.5*dt/(xxf[ii]-xxf[ii-1]),
                                  fr,
                                  - 0.5*dt/(xxf[ii]-xxf[ii-1]),
                                  -fl,
                                  eq,
                                  1, # Left face
                                  ii, jj)
      multiply_add_set_node_vars!(unph, # u_{i+1/2-,j}=u_{i+1/2,j}-0.5*dt*(fr-fl)/(xfr-xfl)
                                  ufr,  # u_{i+1/2,j}
                                  - 0.5*dt/(xxf[ii]-xxf[ii-1]),
                                  fr,
                                  - 0.5*dt/(xxf[ii]-xxf[ii-1]),
                                  - fl,
                                  eq,
                                  2, # Right face
                                  ii, jj)

      multiply_add_set_node_vars!(unph, # u_{i,j-1/2+}=u_{i,j-1/2}-0.5*dt*(gu-gd)/(yfu-yfd)
                                  ufd,  # u_{i,j-1/2}
                                  - 0.5*dt/(yyf[jj]-yyf[jj-1]),
                                  gu,
                                  - 0.5*dt/(yyf[jj]-yyf[jj-1]),
                                  -gd,
                                  eq,
                                  3, # Bottom face
                                  ii, jj)
      multiply_add_set_node_vars!(unph, # u_{i,j+1/2-}=u_{i,j+1/2}-0.5*dt*(gu-gd)/(yfu-yfd)
                                  ufu,  # u_{i,j+1/2}
                                  -0.5*dt/(yyf[jj]-yyf[jj-1]),
                                  gu,
                                  -0.5*dt/(yyf[jj]-yyf[jj-1]),
                                  -gd,
                                  eq,
                                  4, # Top face
                                  ii, jj)
   end

   # Now we loop over faces and perform the update

   # loop over vertical inner faces between (ii-1,jj) and (ii,jj)
   for ii in 2:nd # Supercell faces will be done in blend_face_residual
      xx = xxf[ii-1] # face x coordinate, offset because index starts from 0
      for jj in Base.OneTo(nd)
         yy = yf + dy*xg[jj] # y coordinate same as solution point
         X  = SVector(xx, yy)
         ul = get_node_vars(unph, eq, 2, ii-1, jj)
         ur = get_node_vars(unph, eq, 1, ii  , jj)
         fl, fr = flux(xx, yy, ul, eq, 1), flux(xx, yy, ur, eq, 1)
         fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)
         multiply_add_to_node_vars!(r, # r[ii-1,jj] += dt/(dx*wg[ii-1])*fn
                                    alpha * dt/(dx*wg[ii-1]),
                                    fn,
                                    eq, ii-1, jj)
         multiply_add_to_node_vars!(r, # r[ii,jj] += dt/(dx*wg[ii])*fn
                                    -alpha * dt/(dx*wg[ii]),
                                    fn,
                                    eq, ii, jj)
         # TODO - Can checking this at every iteration be avoided?
         if ii == 2
            set_node_vars!(fn_low, fn, eq, jj, 1, el_x, el_y)
         elseif ii == nd
            set_node_vars!(fn_low, fn, eq, jj, 2, el_x, el_y)
         end
      end
   end

   # loop over horizontal inner faces between (ii,jj-1) and (ii,jj)
   for jj in 2:nd
      yy = yyf[jj-1] # face y coordinate, offset because index starts from 0
      for ii in Base.OneTo(nd)
         xx = xf + dx*xg[ii] # face x coordinate picked same as the soln point
         X  = SVector(xx,yy)
         ud = get_node_vars(unph, eq, 4, ii, jj-1)
         uu = get_node_vars(unph, eq, 3, ii, jj)
         gd, gu = flux(xx, yy, ud, eq, 2), flux(xx, yy, uu, eq, 2)
         gn = num_flux(X, ud, uu, gd, gu, ud, uu, eq, 2)
         multiply_add_to_node_vars!(r, # r[ii,jj-1]+=alpha*dt/(dy*wg[jj-1])*gn
                                    alpha*dt/(dy*wg[jj-1]),
                                    gn,
                                    eq, ii, jj-1)
         multiply_add_to_node_vars!(r, # r[ii,jj]-=alpha*dt/(dy*wg[jj])*gn
                                    - alpha*dt/(dy*wg[jj]),
                                    gn,
                                    eq, ii, jj)
         # TODO - Can checking this at every iteration be avoided
         if jj == 2
            set_node_vars!(fn_low, gn, eq, ii, 3, el_x, el_y)
         elseif jj == nd
            set_node_vars!(fn_low, gn, eq, ii, 4, el_x, el_y)
         end
      end
   end

   end # timer
end

function blending_flux_factors(::AbstractEquations{2}, ::Any, ::Any,
                               ::Any)
   # This method is different for different equations
   λx = λy = 0.5
   return λx, λy
end

# We blend the lower order flux with flux at interfaces
function get_blended_flux_x(el_x, el_y, jy, eq::AbstractEquations{2}, dt, grid,
                            blend, scheme, xf, y, u1, ua, fn, Fn, op)
   if scheme.solver_enum == rkfr
      return Fn
   end
   @unpack alpha, fn_low = blend.cache
   @unpack dx, dy = grid
   @unpack wg = op
   nd = length(wg)
   nx, ny = grid.size
   # Initial trial blended flux
   # TODO (URGENT) - Fix this hacky fix
   # if abs(y) < 0.06
   #    # u_node = get_node_vars(u1, eq, nd, jy, el_x-1, el_y)
   #    # return flux(xf, y, u_node, eq, 1)
   #    alp = 1.0
   # else
   #    alp = 0.5*(alpha[el_x-1,el_y] + alpha[el_x,el_y])
   # end

   alp = 0.5*(alpha[el_x-1,el_y] + alpha[el_x,el_y])
   Fn  = (1.0-alp)*Fn + alp*fn

   ua_node = get_node_vars(ua, eq, el_x-1, el_y)
   λx, λy = blending_flux_factors(eq, ua_node, dx[el_x-1], dy[el_y])

   u_node = get_node_vars(u1, eq, nd, jy, el_x-1, el_y)

   # lower order flux on neighbouring subcell face
   fn_inner = get_node_vars(fn_low, eq, jy, 2, el_x-1, el_y)

   # Test whether lower order update is even admissible
   lower_order_update = u_node - (dt/dx[el_x-1]) / (wg[nd]*λx) * (fn - fn_inner)

   if is_admissible(eq, lower_order_update) == false && el_x > 1
      @warn "Low x-flux not admissible at " (el_x-1),el_y,xf,y
   end

   test_update = u_node - (dt/dx[el_x-1]) / (wg[nd]*λx) * (Fn - fn_inner)

   if is_admissible(eq, test_update) == false
      @debug "Using first order x-flux at " (el_x-1),el_y,xf,y
      Fn = zhang_shu_flux_fix(eq, u_node, lower_order_update, Fn, fn_inner,
                              fn, (dt/dx[el_x-1]) / (wg[nd]*λx))
   end

   # Now we ensure candidate Fn is admissible for (el_x,el_y)
   ua_node = get_node_vars(ua, eq, el_x, el_y)
   λx, λy = blending_flux_factors(eq, ua_node, dx[el_x], dy[el_y])

   u_node = get_node_vars(u1, eq, 1, jy, el_x, el_y)

   # lower order flux on neighbouring subcell face
   fn_inner = get_node_vars(fn_low, eq, jy, 1, el_x, el_y)

   # Test whether lower order update is even admissible
   lower_order_update = u_node - (dt/dx[el_x]) / (wg[1]*λx) * (fn_inner-fn)

   if is_admissible(eq, lower_order_update) == false && el_x < nx+1
      @warn "Lower x-flux not admissible at " el_x,el_y,xf,y
   end

   test_update = u_node - (dt/dx[el_x]) / (wg[1]*λx) * (fn_inner - Fn)

   if is_admissible(eq, test_update) == false
      @debug "Using first order x-flux at " el_x, el_y, xf, y
      Fn = zhang_shu_flux_fix(eq, u_node, lower_order_update, Fn, fn_inner,
                              fn, -(dt/dx[el_x]) / (wg[1]*λx))
   end

   return Fn
end

# TODO - Do a unified face residual for x,y. The direction should be a parameter.
function blend_face_residual_fo_x!(el_x, el_y, jy, xf, y, u1, ua,
                                   eq::AbstractEquations{2}, dt, grid, op,
                                   scheme, param,
                                   Fn, aux, res)
   @timeit_debug aux.timer "Blending limiter" begin # TODO - Check the overhead,
   #                                  # it's supposed to be 0.25 microseconds
   @unpack blend = aux
   @unpack alpha = blend.cache
   @unpack dx, dy = grid
   num_flux = scheme.numerical_flux

   # Add if alpha_l, alpha_r < 1e-12
   # Fn = get_blended_flux_x
   # Also, in this case, lower order flux wouldn't be available in fn_low and
   # have to be recomputed

   @unpack xg, wg = op
   nd = length(xg)

   ul = get_node_vars(u1, eq, nd, jy, el_x-1, el_y)
   ur = get_node_vars(u1, eq, 1 , jy, el_x  , el_y)
   fl, fr = flux(xf, y, ul, eq, 1), flux(xf, y, ur, eq, 1)

   X  = SVector(xf, y)
   fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)

   if el_x == 1
      if abs(y) < 0.05
         fn = fl
      end
   end

   Fn = get_blended_flux_x(el_x, el_y, jy, eq, dt, grid,
                           blend, scheme, xf, y, u1, ua, fn, Fn, op)

   r = @view res[:, :, jy, el_x-1, el_y]
   multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                              alpha[el_x-1,el_y]*dt/(dx[el_x-1]*wg[nd]), Fn,
                              eq, nd)

   r = @view res[:, :, jy, el_x, el_y]
   multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                              - alpha[el_x,el_y]*dt/(dx[el_x]*wg[1]), Fn,
                              eq, 1)

   return Fn, (1.0-alpha[el_x-1, el_y], 1.0-alpha[el_x, el_y])
   end # timer
end

function limit_slope!(::AbstractEquations{2, 1}, slope,
                        ufl, u_star_l, ufr, u_star_r,
                        ue, xl, xr)
   return ufl, ufr
end

function blend_face_residual_muscl_x!(el_x, el_y, jy, xf, y, u1, ua,
                                      eq::AbstractEquations{2,<:Any}, dt, grid,
                                      op, scheme, param, Fn, aux, res)
   @timeit_debug aux.timer "Blending limiter" begin
   @unpack blend = aux
   @unpack alpha = blend.cache
   @unpack dx, dy = grid
   nvar = nvariables(eq)
   num_flux = scheme.numerical_flux

   id = Threads.threadid()

   unph_ = blend.cache.unph[id][1]
   unph = @view unph_[:,1:2,1,1] # Load nvar x 2 array

   dt = blend.cache.dt[1] # For support with DiffEq

   @unpack xg, wg = op
   nd = length(xg)

   # We first find the u^{n+1/2} at the face. For that, there are two
   # relevant cells. To do everything in one loop, we stack all
   # quantitites relevant to the computation at the very beginning

   # Stack all relevant arrays (ul, u, ur)
   arrays1  = (get_node_vars(u1, eq, nd-1, jy, el_x-1, el_y),
               get_node_vars(u1, eq, nd, jy, el_x-1, el_y),
               get_node_vars(u1, eq, 1 , jy, el_x, el_y))
   arrays2 = (get_node_vars(u1, eq, nd, jy, el_x-1, el_y),
              get_node_vars(u1, eq, 1 , jy, el_x, el_y),
              get_node_vars(u1, eq, 2 , jy, el_x, el_y))
   solns = (arrays1, arrays2)

   # Stack x coordinates of solution points (xl, x, xr)
   sol_coords = ( (xf - dx[el_x-1] + xg[nd-1]*dx[el_x-1], xf - dx[el_x-1] + xg[nd]*dx[el_x-1],
                   xf + xg[1]*dx[el_x]),
                  (xf - dx[el_x-1] + xg[nd]*dx[el_x-1], xf + xg[1]*dx[el_x],
                   xf + xg[2]*dx[el_x]) )

   # Stack x coordinates of faces (xfl, xf, xfr)
   face_coords = ( (xf-wg[nd]*dx[el_x-1],xf), (xf, xf+wg[1]*dx[el_x]) )

   betas = (2.0 - alpha[el_x-1,el_y], 2.0 - alpha[el_x,el_y])
   if blend.parameters.pure_fv == true
      betas = (2.0, 2.0)
   end
   for i in 1:2 # Loop over cells
      ul, u_, ur = solns[i]

      # TODO - Add this feature
      # u_, ul, ur = conservative2recon.((u_,ul,ur))

      xl, x, xr = sol_coords[i]
      xfl, xfr = face_coords[i]
      Δx1, Δx2 = x-xl, xr-x
      back, cent, fwd = finite_differences(Δx1, Δx2, ul, u_, ur)
      beta = betas[i]
      slope_tuple = (minmod(beta*back[n], cent[n], beta*fwd[n], 0.0)
                     for n in eachvariable(eq) )
      slope = SVector{nvar}(slope_tuple)

      ufl = u_ + slope*(xfl-x)
      ufr = u_ + slope*(xfr-x)

      u_star_l = u_ + 2.0*slope*(xfl - x)
      u_star_r = u_ + 2.0*slope*(xfr - x)

      # TODO - Remove the !, it is not mutating!
      ufl, ufr = limit_slope!(eq, slope, ufl, u_star_l, ufr, u_star_r, u_,
                              xfl - x, xfr - x)

      # TODO - Add this feature
      # Convert back to conservative variables for update
      # ufl, ufr = recon2conservative.((ufl,ufr))

      fl = flux(xfl, y, ufl, eq, 1)
      fr = flux(xfr, y, ufr, eq, 1)

      if i == 1
         uf = ufr # The relevant face is on the right
      elseif i == 2
         uf = ufl # The relevant face is on the left
      end

      # Use finite difference method to evolve face values to time level n+1/2
      multiply_add_set_node_vars!(unph, # unph = uf - 0.5*dt*(fr-fl)/(xfr-xfl)
                                  uf,
                                  - 0.5*dt/(xfr-xfl),
                                  fr,
                                  - 0.5*dt/(xfr-xfl),
                                  -fl,
                                  eq,
                                  i)
   end
   ul = get_node_vars(unph, eq, 1)
   ur = get_node_vars(unph, eq, 2)
   fl, fr = flux(xf, y, ul, eq, 1), flux(xf, y, ur, eq, 1)
   X = SVector(xf,y)
   fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)

   # TODO (URGENT) - Remove this hacky fix!
   # if el_x == 1
   #    if abs(y) < 0.0535
   #       # u_node = get_node_vars(u1, eq, nd, jy, el_x-1, el_y)
   #       # fn = flux(xf, y, u_node, eq, 1)
   #       fn = fl
   #    end
   # end

   Fn = get_blended_flux_x(el_x, el_y, jy, eq, dt, grid,
                           blend, scheme, xf, y, u1, ua, fn, Fn, op)

   r = @view res[:, :, jy, el_x-1, el_y]
   multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                              alpha[el_x-1,el_y]*dt/(dx[el_x-1]*wg[nd]), Fn,
                              eq, nd)

   r = @view res[:, :, jy, el_x, el_y]
   multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                              - alpha[el_x,el_y]*dt/(dx[el_x]*wg[1]), Fn,
                              eq, 1)

   return Fn, (1.0-alpha[el_x-1, el_y], 1.0-alpha[el_x, el_y])

   end # timer
end

function get_blended_flux_y(el_x, el_y, ix, eq::AbstractEquations{2}, dt, grid,
                            blend, scheme, x, yf, u1, ua, fn, Fn, op)
   if scheme.solver_enum == rkfr
      return Fn
   end
   @unpack alpha, fn_low = blend.cache
   @unpack dx, dy = grid
   @unpack wg = op
   nd = length(wg)
   nx, ny = grid.size
   # Initial trial blended flux
   alp = 0.5 * (alpha[el_x,el_y-1] + alpha[el_x,el_y])
   Fn = (1.0-alp)*Fn + alp*fn

   # First we ensure that the candidate Fn is admissible for (el_x, el_y-1)
   ua_node = get_node_vars(ua, eq, el_x, el_y-1)
   λx, λy = blending_flux_factors(eq, ua_node, dx[el_x], dy[el_y-1])

   u_node = get_node_vars(u1, eq, ix, nd, el_x, el_y-1)

   # lower order flux on neighbouring subcell face
   fn_inner = get_node_vars(fn_low, eq, ix, 4, el_x, el_y-1)

   # test whether lower order update is even admissible
   lower_order_update = u_node - (dt/dy[el_y-1]) / (wg[nd]*λy) * (fn-fn_inner)

   if is_admissible(eq, lower_order_update) == false && el_y > 1
      @warn "Low y-flux not admissible at " el_x,(el_y-1),x,yf
   end

   test_update = u_node - (dt/dy[el_y-1]) / (wg[nd]*λy) * (Fn - fn_inner)

   if is_admissible(eq, test_update) == false
      @debug "Using first order y-flux at " el_x,(el_y-1),x,yf
      Fn = zhang_shu_flux_fix(eq, u_node, lower_order_update, Fn, fn_inner,
                              fn, (dt/dy[el_y-1]) / (wg[nd]*λy))
   end

   # Now we ensure candidate Fn is admissible in (el_x, el_y)
   ua_node = get_node_vars(ua, eq, el_x, el_y)
   λx, λy = blending_flux_factors(eq, ua_node, dx[el_x], dy[el_y])

   u_node = get_node_vars(u1, eq, ix, 1, el_x, el_y)

   # lower order flux on neighbouring subcell face
   fn_inner = get_node_vars(fn_low, eq, ix, 3, el_x, el_y)

   # Test whether lower order update is even admissible
   lower_order_update = u_node - (dt/dy[el_y]) / (wg[1]*λy) * (fn_inner - fn)

   if is_admissible(eq, lower_order_update) == false && el_y < ny+1
      @warn "Lower y-flux not admissible at " el_x,el_y,x,yf
   end

   test_update = u_node - (dt/dy[el_y]) / (wg[1]*λy) * (fn_inner - Fn)

   if is_admissible(eq, test_update) == false
      @debug "Using first order y-flux at " el_x,el_y,x,yf
      Fn = zhang_shu_flux_fix(eq, u_node, lower_order_update, Fn, fn_inner,
                              fn, -(dt/dy[el_y]) / (wg[1]*λy))
   end

   return Fn
end

# TODO - Do a unified face residual for x,y. The direction should be a parameter.
function blend_face_residual_fo_y!(el_x, el_y, ix, x, yf, u1, ua,
                                   eq::AbstractEquations{2}, dt, grid, op,
                                   scheme, param, Fn, aux, res)
   @timeit_debug aux.timer "Blending limiter" begin # TODO - Check the overhead,
   #                                  # it's supposed to be 0.25 microseconds
   @unpack blend = aux
   @unpack alpha = blend.cache
   num_flux = scheme.numerical_flux
   @unpack dx, dy = grid

   @unpack xg, wg = op
   nd = length(xg)

   ul = get_node_vars(u1, eq, ix, nd, el_x, el_y-1)
   fl = flux(x, yf, ul, eq, 2)
   ur = get_node_vars(u1, eq, ix, 1 , el_x, el_y)
   fr = flux(x, yf, ur, eq, 2)
   X  = SVector(x, yf)
   fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 2)

   Fn = get_blended_flux_y(el_x, el_y, ix, eq, dt, grid, blend,
                           scheme, x, yf, u1, ua, fn, Fn, op)

   r = @view res[:,ix,:,el_x,el_y-1]

   multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                              alpha[el_x,el_y-1] * dt/(dy[el_y-1]*wg[nd]),
                              Fn,
                              eq, nd
                              )

   r = @view res[:,ix,:,el_x,el_y]

   multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                              - alpha[el_x,el_y] * dt/(dy[el_y]*wg[1]),
                              Fn,
                              eq, 1)

   return Fn, (1.0-alpha[el_x,el_y-1], 1.0-alpha[el_x,el_y])
   end # timer
end

function blend_face_residual_muscl_y!(el_x, el_y, ix, x, yf, u1, ua,
                                      eq::AbstractEquations{2,<:Any}, dt, grid, op,
                                      scheme, param, Fn, aux, res)
   @timeit_debug aux.timer "Blending limiter" begin
   @unpack blend = aux
   @unpack alpha = blend.cache
   @unpack dx, dy = grid
   num_flux = scheme.numerical_flux
   nvar = nvariables(eq)

   id = Threads.threadid()

   dt = blend.cache.dt[1] # For support with DiffEq

   unph_ = blend.cache.unph[id][1]
   unph = @view unph_[:,1:2,1,1] # Load nvar x 2 array

   @unpack xg, wg = op
   nd = length(xg)

   # We first find u^{n+1/2}_{±} at the face. For that, there are two
   # relevant cells. To do everything in one loop, we stack all
   # quantities relevant to the computation at the very beginning

   # Stack all relevant arrays (ul, u, ur)

   arrays1 = (get_node_vars(u1, eq, ix, nd-1, el_x, el_y-1),
              get_node_vars(u1, eq, ix, nd  , el_x, el_y-1),
              get_node_vars(u1, eq, ix, 1   , el_x, el_y))
   arrays2 = (get_node_vars(u1, eq, ix, nd, el_x, el_y-1),
              get_node_vars(u1, eq, ix, 1 , el_x, el_y),
              get_node_vars(u1, eq, ix, 2 , el_x, el_y))

   solns = (arrays1, arrays2)

   # stack x coordinates of solution points (yd, y, yu)
   sol_coords = ((yf - dy[el_y-1] + xg[nd-1]*dy[el_y-1], yf - dy[el_y-1] + xg[nd]*dy[el_y-1],
                  yf + xg[1]*dy[el_y] ),
                 (yf - dy[el_y-1] + xg[nd]*dy[el_y-1], yf + xg[1]*dy[el_y],
                   yf + xg[2]*dy[el_y]))

   face_coords = ( (yf-wg[nd]*dy[el_y-1], yf), (yf, yf+wg[1]*dy[el_y]) )

   betas = (2.0 - alpha[el_x,el_y-1], 2.0 - alpha[el_x,el_y])
   if blend.parameters.pure_fv == true
      betas = (2.0, 2.0)
   end

   for i in 1:2 # Loop over the two relevant cells
      ud, u_, uu = solns[i]

      # TODO - Add this feature
      # ud, u_, uu = conservative2recon.((ud,u_,uu))

      yd, y, yu = sol_coords[i]
      yfd, yfu = face_coords[i]
      Δy1, Δy2 = y-yd, yu-y
      back, cent, fwd = finite_differences(Δy1, Δy2, ud, u_, uu)
      beta = betas[i]
      slope_tuple = (minmod(beta*back[n], cent[n], beta*fwd[n], 0.0)
                     for n in eachvariable(eq))
      slope = SVector{nvar}(slope_tuple)

      ufd = u_ + slope*(yfd-y)
      ufu = u_ + slope*(yfu-y)

      u_star_d = u_ + 2.0*slope*(yfd - y)
      u_star_u = u_ + 2.0*slope*(yfu - y)

      ufd, ufu = limit_slope!(eq, slope, ufd, u_star_d, ufu, u_star_u, u_,
                              yfd - y, yfu - y)

      # TODO - add this features
      # Convert back to conservative variables for update
      # ufl, ufr = recon2conservative.((ufl, ufr))

      gd = flux(x, yfd, ufd, eq, 2)
      gu = flux(x, yfu, ufu, eq, 2)

      if i == 1
         uf = ufu # relevant face is the one above
      elseif i == 2
         uf = ufd # relevant face is the one below
      end

      # use finite difference method to evolve face values to time n+1/2
      multiply_add_set_node_vars!(unph, # unph = uf - 0.5*dt*(gu-gd)/(yfu-yfd)
                                  uf,
                                  -0.5*dt/(yfu-yfd),
                                  gu,
                                  -0.5*dt/(yfu-yfd),
                                  -gd,
                                  eq,
                                  i)
   end

   ud = get_node_vars(unph, eq, 1)
   uu = get_node_vars(unph, eq, 2)
   gd, gu = flux(x, yf, ud, eq, 2), flux(x, yf, uu, eq, 2)
   X = SVector(x, yf)
   fn = num_flux(X, ud, uu, gd, gu, ud, uu, eq, 2)

   Fn = get_blended_flux_y(el_x, el_y, ix, eq, dt, grid, blend,
                           scheme, x, yf, u1, ua, fn, Fn, op)

   r = @view res[:,ix,:,el_x,el_y-1]

   multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                              alpha[el_x,el_y-1] * dt/(dy[el_y-1]*wg[nd]),
                              Fn,
                              eq, nd
                              )

   r = @view res[:,ix,:,el_x,el_y]

   multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                              - alpha[el_x,el_y] * dt/(dy[el_y]*wg[1]),
                              Fn,
                              eq, 1)

   return Fn, (1.0-alpha[el_x,el_y-1], 1.0-alpha[el_x,el_y])

   end # timer
end

fo_blend(::AbstractEquations{2,<:Any}) = (;
                                            cell_residual! = blend_cell_residual_fo!,
                                            face_residual_x! = blend_face_residual_fo_x!,
                                            face_residual_y! = blend_face_residual_fo_y!,
                                            name = "fo")

mh_blend(::AbstractEquations{2,<:Any}) = (;
                                            cell_residual! = blend_cell_residual_muscl!,
                                            face_residual_x! = blend_face_residual_muscl_x!,
                                            face_residual_y! = blend_face_residual_muscl_y!,
                                            name = "muscl")

function trivial_cell_residual(i, j, eq::AbstractEquations{2}, scheme, aux,
                                 dt, dx, dy, xf, yf,
                                 op, u1, u, f, r)
   return nothing
end

function trivial_face_residual(i, j, k, x, yf, u1, ua, eq::AbstractEquations{2},
                               dt, grid, op,
                               scheme, param, Fn, aux, res)
   return Fn, (1.0, 1.0)
end

function is_admissible(::AbstractEquations{2,<:Any}, ::AbstractVector)
   # Check if the invariant domain in preserved. This has to be
   # extended in Equation module
   return true
end

#-------------------------------------------------------------------------------
# Compute error norm
#-------------------------------------------------------------------------------
function compute_error(problem, grid, eq::AbstractEquations{2}, aux, op, u1, t)
   @timeit aux.timer "Compute error" begin
   @unpack error_file, aux_cache = aux
   @unpack error_cache = aux_cache
   xmin, xmax, ymin, ymax = grid.domain
   @unpack xg = op
   nd = length(xg)

   refresh!(u) = fill!(u, zero(eltype(u)))

   # TODO: Assuming periodicity
   @unpack exact_solution = problem

   @unpack xq, w2d, V, arr_cache = error_cache

   nq = length(xq)

   nx, ny = grid.size
   @unpack xc, yc, dx, dy = grid

   l1_error, l2_error, energy = 0.0, 0.0, 0.0
   @inbounds @floop for element in CartesianIndices((1:nx, 1:ny))
   # for element in CartesianIndices((1:nx, 1:ny))
      el_x, el_y = element[1], element[2]
      ue, un = arr_cache[Threads.threadid()]
      for j=1:nq, i=1:nq
         x = xc[el_x] - 0.5 * dx[el_x] + dx[el_x] * xq[i]
         y = yc[el_y] - 0.5 * dy[el_y] + dy[el_y] * xq[j]
         ue_node = exact_solution(x,y, t)
         set_node_vars!(ue, ue_node, eq, i, j)
      end
      u1_ = @view u1[:, :, :, el_x, el_y]
      refresh!(un)
      for j=1:nd, i=1:nd
         u_node = get_node_vars(u1_, eq, i, j)
         for jj=1:nq, ii=1:nq
            # un = V*u*V', so that
            # un[ii,jj] = ∑_ij V[ii,i]*u[i,j]*V[jj,j]
            multiply_add_to_node_vars!(un, V[ii,i]*V[jj,j], u_node, eq, ii, jj)
         end
      end
      l1 = l2 = e = 0.0
      for j=1:nq, i=1:nq
         un_node = get_node_vars(un, eq, i, j)
         ue_node = get_node_vars(ue, eq, i, j) # TODO - allocated ue is not needed
         for n in 1:1 # Only computing error in first conservative variable
            du  = abs(un_node[n] - ue_node[n])
            l1 += du           * w2d[i,j]
            l2 += du*du         * w2d[i,j]
            e  += un_node[n]^2 * w2d[i,j]
         end
      end
      l1 *= dx[el_x]*dy[el_y]
      l2 *= dx[el_x]*dy[el_y]
      e  *= dx[el_x]*dy[el_y]
      @reduce(l1_error += l1, l2_error += l2, energy += e)
      # l1_error += l1; l2_error += l2; energy += e
   end
   domain_size = (xmax - xmin) * (ymax - ymin)
   l1_error = l1_error/domain_size
   l2_error = sqrt(l2_error/domain_size)
   energy   = energy/domain_size
   @printf(error_file, "%.16e %.16e %.16e %.16e\n", t, l1_error[1], l2_error[1],
           energy[1])

   return Dict("l1_error" => l1_error, "l2_error" => l2_error,
               "energy" => energy)
   end # timer
end

#-------------------------------------------------------------------------------
# Write solution to a vtk file
#-------------------------------------------------------------------------------
function initialize_plot(::AbstractEquations{2,1}, op, grid, problem, scheme,
                         aux, u1, ua)
   rm("output", force=true, recursive=true)
   mkdir("output")
   return nothing
end

function create_aux_cache(eq, op)
   @unpack xg = op
   nd = length(xg)
   nvar = nvariables(eq)
   nq     = nd + 3    # number of quadrature points in each direction
   xq, wq = weights_and_points(nq, "gl")

   V = Vandermonde_lag(xg, xq) # matrix evaluating at `xq`
   # using values at solution points `xg`

   MArr = MArray{Tuple{nvar, nq, nq}, Float64}

   # for each thread, construct `cache_size` number of objects with
   # `constructor` and store them in an SVector
   arr_cache = alloc_for_threads(MArr, 2)

   w2d = SMatrix{nq, nq}(wq*wq')

   error_cache = (; xq, w2d, V, arr_cache)

   MArr = MArray{Tuple{nd}, Float64}

   bound_limiter_cache = alloc_for_threads(MArr, 4) # ul, ur, ud, uu

   aux_cache = (; error_cache, bound_limiter_cache)
   return aux_cache
end

function write_poly(::AbstractEquations{2,1}, grid, op, u1, fcount)
   filename = get_filename("output/sol", 3, fcount)
   @unpack xf, yf, dx, dy = grid
   nx, ny = grid.size
   @unpack degree, xg = op
   nd = degree + 1
   # Clear and re-create output directory

   nu = max(nd, 2)
   xu = LinRange(0.0, 1.0, nu)
   Vu = Vandermonde_lag(xg, xu)
   Mx, My = nx*nu, ny*nu
   grid_x = zeros(Mx)
   grid_y = zeros(My)
   for i = 1:nx
      i_min = (i-1)*nu + 1
      i_max = i_min + nu-1
      # grid_x[i_min:i_max] .= LinRange(xf[i], xf[i+1], nu)
      grid_x[i_min:i_max] .= xf[i] .+ dx[i]*xg
   end

   for j = 1:ny
      j_min = (j-1)*nu + 1
      j_max = j_min + nu-1
      # grid_y[j_min:j_max] .= LinRange(yf[j], yf[j+1], nu)
      grid_y[j_min:j_max] .= yf[j] .+ dy[j]*xg
   end

   vtk_sol = vtk_grid(filename, grid_x, grid_y)

   u_equi = zeros(Mx, My)
   u = zeros(nu)
   for j=1:ny
      for i=1:nx
         # TODO - Don't do this, use all values in the cell
         # to get values in the equispaced thing
         for jy=1:nd
            i_min = (i-1)*nu + 1
            i_max = i_min + nu-1
            u_ = @view u1[1, :, jy, i, j]
            mul!(u, Vu, u_)
            j_index = (j-1)*nu + jy
            u_equi[i_min:i_max, j_index] .= @view u1[1,:,jy,i,j]
         end
      end
   end
   vtk_sol["sol"] = u_equi

   out = vtk_save(vtk_sol)
   return out
end

function write_soln!(base_name, fcount, iter, time, eq::AbstractEquations{2,1},
                     grid, problem::Problem, ::Parameters, op,
                     z, u1, aux, ndigits=3)
   @timeit aux.timer "Write solution" begin
   @unpack final_time = problem

   # Output cell-averages
   filename = get_filename("output/avg", ndigits, fcount)
   # filename = string("output/", filename)
   vtk_avg = vtk_grid(filename, grid.xc, grid.yc)
   nx, ny = grid.size
   vtk_avg["Cell Averages"] = @view z[1,1:nx,1:ny]
   vtk_avg["CYCLE"] = iter
   vtk_avg["TIME"] = time
   out = vtk_save(vtk_avg)
   println("Wrote file ", out[1])
   write_poly(eq, grid, op, u1, fcount)
   if final_time - time < 1e-10
      cp("$filename.vtr","./output/avg.vtr")
      println("Wrote final solution to avg.vtr.")
   end

   # Output uniformly spaced solution point values only
   # at the last solution point.

   fcount += 1
   return fcount
   end # timer
end

function post_process_soln(::AbstractEquations{2}, aux, problem, param)
   @unpack timer, error_file = aux
   # Print timer data on screen
   print_timer(timer, sortby = :firstexec); print("\n")
   show(timer); print("\n")
   timer_file = open("./output/timer.json", "w")
   JSON3.write(timer_file, TimerOutputs.todict(timer))
   close(timer_file)
   close(error_file)

   @unpack saveto = param

   if saveto != "none"
      if saveto[end] == "/"
         saveto = saveto[1:end-1]
      end
      mkpath(saveto)
      cp("./error.txt", "$saveto/error.txt", force=true)
      for file in readdir("./output")
         cp("./output/$file", "$saveto/$file", force=true)
      end
      println("Saved output files to $saveto")
   end

   return nothing
end

export update_ghost_values_periodic!

end # @muladd

end
