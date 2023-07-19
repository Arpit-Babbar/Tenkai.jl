module EqLinAdv2D

using StaticArrays
using MuladdMacro

using Tenkai
using TimerOutputs
using Tenkai.FR2D: correct_variable!
using UnPack

# methods to be extended in this module
import Tenkai: flux
import Tenkai.FR: admissibility_tolerance

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct LinAdv2D{Speed, Velocity <: Function} <: AbstractEquations{2,1}
   speed::Speed
   velocity::Velocity
   name::String
   nvar::Int64
   numfluxes::Dict{String, Function}
end

# Upwind flux
function upwind(x, ual, uar, Fl, Fr, Ul, Ur, eq::LinAdv2D, dir)
   v = eq.velocity(x)[dir]
   Fn = (v > 0.0) ? Fl[1] : Fr[1]
   return SVector(Fn)
end

# Rusanov flux
function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::LinAdv2D, dir)
   v = eq.velocity(x[1], x[2])[dir]
   Fn = 0.5*(Fl[1] + Fr[1] - abs(v)*(Ur[1] - Ul[1]))
   return SVector(Fn)
end

function Tenkai.flux(x,y,u, eq::LinAdv2D)
   v1, v2 = eq.velocity(x, y)
   # return SVector(v1*u, v2*u)
   f1, f2 = v1*u[1], v2*u[1]
   return SVector(f1), SVector(f2)
end

function Tenkai.flux(x,y,u, eq::LinAdv2D, orientation::Integer)
   v1, v2 = eq.velocity(x, y)
   if orientation == 1
      f1 = v1*u[1]
      return SVector(f1)
   else
      f2 = v2*u[1]
      return SVector(f2)
   end
end

function linear_vel1_iv(x, y)
   return SVector(1.0 + 0.5*x + 0.5*y)
end

function linear_vel1_exact(x, y, t)
   return linear_vel1_iv(x-t, y-t)
end

linear_vel1_data = ((x,y) -> SVector(1.0, 1.0), linear_vel1_iv, linear_vel1_exact)

function smooth_sin_vel1_iv(x, y, xmin, xmax, ymin, ymax)
   return SVector(1.0 + (0.5 * sinpi(2.0*xmin + 2.0*(xmax-xmin)*x)
                             * sinpi(2.0*ymin + 2.0*(ymax-ymin)*y)))
end

function smooth_sin_vel1_exact(x, y, t, xmin, xmax, ymin, ymax)
   return smooth_sin_vel1_iv(x-t, y-t, xmin, xmax, ymin, ymax)
end

## ss = smooth_sine
xmin_ss, xmax_ss, ymin_ss, ymax_ss = 0.0, 1.0, 0.0, 1.0

smooth_sin_vel1_data = (xmin_ss, xmax_ss, ymin_ss, ymax_ss,
                        (x,y) -> SVector(1.0, 1.0),
                        (x,y) -> smooth_sin_vel1_iv(x, y,
                                                    xmin_ss, xmax_ss,
                                                    ymin_ss, ymax_ss),
                        (x,y,t) -> smooth_sin_vel1_exact(x, y, t,
                                                         xmin_ss, xmax_ss,
                                                         ymin_ss, ymax_ss))

rotate_exp_iv(x,y) = SVector(1.0 + exp(-50.0*((x-0.5)^2 + y^2)))
velocity_exp_iv(x,y) = SVector(-y, x)
exact_exp_iv(x,y,t) = rotate_exp_iv(x*cos(t) + y*sin(t), -x*sin(t) + y*cos(t))

rotate_exp_data = (rotate_exp_iv, velocity_exp_iv, exact_exp_iv)

function smooth_hump2d(x, y)
   r0 = 0.15
   d = sqrt( (x-0.25)^2+(y-0.5)^2 )
   q = min(d, r0)/r0
   return 0.25*(1.0+cospi(q))
end

function cone2d(x, y)
   r0 = 0.15
   d  = sqrt( (x-0.5)^2+(y-0.25)^2 )
   if d < r0
      return 1.0-d/r0
   else
      return 0.0
   end
end

function slotted_disc2d(x, y)
   r0 = 0.15
   d = sqrt( (x-0.5)^2+(y-0.75)^2 )
   if d < r0
      if (x> 0.5-r0*0.25 && x < 0.5+r0*0.25) && y < 0.75+r0*0.7
         return 0.0
      else
         return  1.0
      end
   else
      return 0.0
   end
end

function composite2d(x, y)
   return SVector(smooth_hump2d(x, y)+cone2d(x, y)+slotted_disc2d(x, y))
end

function Tenkai.apply_bound_limiter!(eq::LinAdv2D, grid, scheme, param, op, ua,
                                   u1, aux)
   if scheme.bound_limit == "no"
      return nothing
   end
   @timeit aux.timer "Bound limiter" begin
   # variables = (get_density, get_pressure)
   # for variable in variables
   #    correct_variable!(eq, variable, op, aux, grid, u1, ua)
   # end # KLUDGE Fix the type instability and do it with a loop
   # https://github.com/trixi-framework/Trixi.jl/blob/0fd86e4bd856d894de6a7514edcb9758bf6f8e1e/src/callbacks_stage/positivity_zhang_shu.jl#L39   correct_variable!(eq, get_density,  op, aux, grid, u1, ua)

   # TODO - Allow user to pass these variables
   correct_variable!(eq, (eq, u) -> first(u), op, aux, grid, u1, ua)
   correct_variable!(eq, (eq, u) -> 1.0 - first(u), op, aux, grid, u1, ua)
   return nothing
   end # timer
end

function Tenkai.limit_slope(eq::LinAdv2D, s, ufl, u_s_l, ufr, u_s_r, ue, xl, xr)
   eps = 1e-10

   variables = ((eq, u) -> first(u), (eq,u) -> 1.0 - first(u))

   for variable in variables
      var_star_tuple = (variable(eq, u_s_l), variable(eq, u_s_r))
      var_low = variable(eq, ue)

      theta = 1.0
      for var_star in var_star_tuple
         if var_star < eps
            # TOTHINK - Replace eps here by 0.1*var_low
            ratio = abs(0.1*var_low - var_low) / (abs(var_star - var_low) + 1e-13 )
            theta = min(ratio, theta)
         end
      end
      s *= theta
      u_s_l = ue + 2.0*theta*xl*s
      u_s_r = ue + 2.0*theta*xr*s
   end

   ufl = ue + xl*s
   ufr = ue + xr*s

   return ufl, ufr
end

function Tenkai.zhang_shu_flux_fix(eq::LinAdv2D,
                            Fn, fn,                   # high order flux, low order flux
                            u_prev_ll, u_prev_rr,     # solution at previous time level
                            u_low_ll, u_low_rr,       # low order updates
                            fn_inner_ll, fn_inner_rr, # flux from inner solution points
                            c_ll, c_rr                # c such that unew = u - c*(Fn-f_inner)
                           )
   variables = ((eq, u) -> first(u), (eq, u) -> 1.0 - first(u))
   # TODO - Allow general variables!
   for variable in variables
      u_high_ll, u_high_rr = (u_prev_ll - c_ll * (Fn-fn_inner_ll), # high order candidates
                              u_prev_rr - c_rr * (Fn-fn_inner_rr))
      var_high_ll, var_high_rr = (variable(eq, u_high_ll), variable(eq, u_high_rr))
      var_low_ll,  var_low_rr  = (variable(eq, u_low_ll ), variable(eq, u_low_rr ))
      eps_ll, eps_rr = 0.1*var_low_ll, 0.1*var_low_rr
      # eps_ll = eps_rr = 0.0
      # Maybe changing this to 1e-10 will get symmetry?
      ratio_ll = abs(eps_ll-var_low_ll)/(abs(var_high_ll-var_low_ll)+1e-13)
      ratio_rr = abs(eps_rr-var_low_rr)/(abs(var_high_rr-var_low_rr)+1e-13)
      theta    = min(ratio_ll, ratio_rr, 1.0)
      if theta < 1.0
         Fn = theta*Fn + (1.0-theta)*fn
      end
   end
   return Fn
end

function Tenkai.is_admissible(eq::LinAdv2D, u::AbstractVector)
   # Check if the invariant domain in preserved. This has to be
   # extended in Equation module
   return u[1] >= 0.0 && u[1] <= 1.0
end

function admissibility_tolerance(eq::LinAdv2D)
   return -1e-16
end

exact_solution_composite2d(x,y,t) = composite2d((x-0.5)*cos(t)+(y-0.5)*sin(t)+0.5,
                                                -(x-0.5)*sin(t)+(y-0.5)*cos(t)+0.5)

composite2d_data = ( (x,y) -> SVector(0.5 - y[1], x[1] - 0.5),
                     composite2d,
                     exact_solution_composite2d)

function get_equation(velocity)
   speed(x,u, eq::LinAdv2D) = velocity(x[1], x[2])
   name = "2d Linear Advection Equation"
   numfluxes = Dict("upwind"  => upwind,
                    "rusanov" => rusanov)
   nvar = 1
   return LinAdv2D(speed, velocity, name, nvar, numfluxes)
   # return (;flux, speed, velocity, name, nvar, numfluxes)
end

export flux

end # @muladd

end