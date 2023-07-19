# f(u) = u*u/(u*u+1(1-u)^2)
module EqBuckleyLeverett1D

using Tenkai

# flux function will be extended to EqBuckleyLeverett1D
import Tenkai: flux

import Roots: find_zero

using StaticArrays
using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct BuckleyLeverret1D <: AbstractEquations{1,1}
   speed::Function
   fprime::Function
   a_buck::Float64 # Constant in Buckley-Leverret model
   u_buck::Float64 # Convex in [0,u_bucklev], concave in [u_bucklev,1]
   nvar::Int64
   name::String
   numfluxes::Dict{String, Function}
end

# Extending the flux function
function Tenkai.flux(x, u, eq::BuckleyLeverret1D)
   u_, a = u[1], eq.a_buck
   return SVector(u_*u_/( u_*u_ + a*(1.0-u_)^2 ))
   return nothing
end

function fprime(u_, eq::BuckleyLeverret1D)
   a = eq.a_buck
   u = u_[1]
   L = u*u + a*(1.0-u)^2
   value = (2.0*a*u*(1.0-u)) / L^2
   return value
end

function fprime_a025(u_)
   a = 0.25
   u = u_[1]
   L = u*u + a*(1.0-u)^2
   value = (2.0*a*u*(1.0-u)) / L^2
   return value
end

function max_speed_bucklev(Ul, Ur, eq::BuckleyLeverret1D)
   u_buck = eq.u_buck
   ul, ur = Ul[1], Ur[1]
   umin = max(min(ul,ur), 0.0)
   umax = min(max(ul,ur), 1.0)
   if umin > u_buck || umax < u_buck
      value = max( fprime(umin, eq), fprime(umax, eq) )
   else
      value = fprime(u_buck, eq)
   end
   return value
end

function max_speed(x, u, eq::BuckleyLeverret1D)
   smax = fprime(eq.u_buck, eq)
   smax = max(smax, abs(fprime(u, eq)))
   return smax
end

# Rusanov flux
function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::BuckleyLeverret1D, dir)
   lam = max_speed_bucklev(ual, uar, eq)
   Fn  = 0.5*(Fl[1] + Fr[1] - lam*(Ur[1] - Ul[1]))
   return SVector(Fn)
end

function upwind(x, ual, uar, Fl, Fr, Ul, Ur, eq::BuckleyLeverret1D, dir)
   Fn = Fl[1]
   return SVector(Fn)
end

#------------------------------------------------------------------------------
# Initial values and boundary values
#------------------------------------------------------------------------------
function hatbuck_iv(x)
   if x < -0.5 || x > 0.0
      value = 0.0
   else
      value = 1.0
   end
   return value
end

function hatbuck_exact(x, t, eq)
   a_buck = eq.a_buck
   u_buck = eq.u_buck
   fprime = eq.fprime
   u_s  = sqrt(a_buck/(1.0+a_buck)) # (f(u_s)-f(0))/(u_s-0)=f'(u_s)
   u_ss = 1.0-1.0/sqrt(1.0+a_buck)  # (f(u_ss)-f(1))/(u_ss-1)=f'(u_ss)

   # Define inverse of f' for computing rarefaction

   # Inverse of f' restricted to [u_buck,1.0], an interval containing [u_s,1.0]
   function inv_f_s(v)
      # Inverse of f' at v equals root of this polynomial in [0.5,1]
      function p(u)
         value = v*(u*u+a_buck*(1.0-u)^2)^2-2.0*a_buck*u*(1.0-u)
         return value
      end
      output = find_zero(p, 0.5*(u_s+1.0))
      return output
   end

   # Inverse of f' restricted to [0,u_buck], an interval that contains [0,u_ss]
   function inv_f_ss(v)
      # Inverse of f' at v equals root of this polynomial in [0.5,1]
      function p(u)
         value = v*(u*u+a_buck*(1.0-u)^2)^2-2.0*a_buck*u*(1.0-u)
         return value
      end
      output = find_zero(p, 0.5*u_buck)
      return output
   end

   if x <= -0.5
      y =  0.0
   elseif -0.5 < x <= -0.5+fprime(u_ss,eq)*t
      y = inv_f_ss((x+0.5)/t)
   elseif -0.5+fprime(u_ss,eq)*t < x <= 0.0
      y = 1.0
   elseif 0.0 < x <= fprime(u_s,eq)*t
      y = inv_f_s(x/t)
   else
      y = 0.0
   end
   return y
end

function hatbuck_exact_a025(x, t)
   a_buck = 0.25
   u_buck = 0.287141
   u_s  = sqrt(a_buck/(1.0+a_buck)) # (f(u_s)-f(0))/(u_s-0)=f'(u_s)
   u_ss = 1.0-1.0/sqrt(1.0+a_buck)  # (f(u_ss)-f(1))/(u_ss-1)=f'(u_ss)

   # Define inverse of f' for computing rarefaction

   # Inverse of f' restricted to [u_buck,1.0], an interval containing [u_s,1.0]
   function inv_f_s(v)
      # Inverse of f' at v equals root of this polynomial in [0.5,1]
      function p(u)
         value = v*(u*u+a_buck*(1.0-u)^2)^2-2.0*a_buck*u*(1.0-u)
         return value
      end
      output = find_zero(p, 0.5*(u_s+1.0))
      return output
   end

   # Inverse of f' restricted to [0,u_buck], an interval that contains [0,u_ss]
   function inv_f_ss(v)
      # Inverse of f' at v equals root of this polynomial in [0.5,1]
      function p(u)
         value = v*(u*u+a_buck*(1.0-u)^2)^2-2.0*a_buck*u*(1.0-u)
         return value
      end
      output = find_zero(p, 0.5*u_buck)
      return output
   end

   if x <= -0.5
      y =  0.0
   elseif -0.5 < x <= -0.5+fprime_a025(u_ss)*t
      y = inv_f_ss((x+0.5)/t)
   elseif -0.5+fprime_a025(u_ss)*t < x <= 0.0
      y = 1.0
   elseif 0.0 < x <= fprime_a025(u_s)*t
      y = inv_f_s(x/t)
   else
      y = 0.0
   end
   return y
end

# only implemented for a_buck = 0.25
function get_equation()
   a_buck = 0.25
   u_buck = 0.287141
   name = "Buckley-Leverett equation"
   numfluxes = Dict("rusanov" => rusanov,
                    "upwind"  => upwind)
   nvar = 1
   BuckleyLeverret1D(max_speed, fprime, a_buck, u_buck, nvar, name, numfluxes)
end

export flux

end # @muladd

end