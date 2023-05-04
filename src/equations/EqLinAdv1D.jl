module EqLinAdv1D

using StaticArrays
using MuladdMacro

using SSFR

# flux function will be extended to LinAdv1D
import SSFR: flux

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct LinAdv1D{F2,F3 <: Function} <: AbstractEquations{1,1}
   speed::F2
   velocity::F3
   nvar::Int64
   name::String
   initial_values::Dict{String, Tuple{Function,Function}}
   numfluxes::Dict{String, Function}
end

function upwind(x, ual, uar, Fl, Fr, Ul, Ur, eq::LinAdv1D, dir)
   v = eq.velocity(x)
   (v > 0) ? Fn = Fl[1] : Fn = Fr[1]
   return SVector(Fn)
end

# Rusanov flux
@inbounds @inline function rusanov(x,ual,uar,Fl,Fr,Ul,Ur,eq::LinAdv1D,dir)
   v = eq.velocity(x)
   # v = 1.0
   Fn = 0.5*(Fl[1] + Fr[1] - abs(v)*(Ur[1] - Ul[1]))
   return SVector(Fn)
end


#------------------------------------------------------------------------------
# Initial Values
#------------------------------------------------------------------------------
function flat_hat1d(x)
   xmin, xmax = -1.0, 1.0
   dx = xmax - xmin
   if x > xmax
      y = x - dx*floor((x+xmin)/dx)
   elseif x < xmin
      y = x + dx*floor((xmax-x)/dx)
   else
      y = x
   end
   if y > xmin + 0.25*dx && y < xmax - 0.25*dx
      return 1.0
   else
      return 0.0
   end
end

flat_hat_data = (x -> 1.0, flat_hat1d, (x,t) -> flat_hat1d(x-t))

function mult1d(x) # xmin = -1.0, xmax = 1.0
   xmin, xmax = -1.0, 1.0
   dx = xmax - xmin
   if x > xmax
      y = x - dx*floor((x+xmin)/dx)
   elseif x < xmin
      y = x + dx*floor((xmax-x)/dx)
   else
      y = x
   end

   if y > -0.8 && y < -0.6
      value = exp(-log(2.0)*(y+0.7)^2/0.0009)
   elseif y > -0.4 && y < -0.2
      value = 1.0;
   elseif y > 0.0 && y < 0.2
      value =  1.0 - abs(10.0*(y-0.1))
   elseif y > 0.4 && y < 0.6
      value = sqrt(1.0 - 100.0*(y-0.5)^2)
   else
      value = 0.0
   end
   return value
end

mult1d_data = ( x -> 1.0,mult1d, (x,t) -> mult1d(x-t))

function cts_sin1d(x)
   xmin, xmax = -1.0, 1.0
   dx = xmax - xmin
   if x > xmax
      y = x - dx*floor((x+xmin)/dx)
   elseif x < xmin
      y = x + dx*floor((xmax-x)/dx)
   else
      y = x
   end

   if y > xmin + 0.25*dx && y < xmax - 0.25*dx
      value = sinpi(4.0*y)
   else
      value = 0.0
   end
   return value
end

cts_sin1d_data = ( x -> 1.0,cts_sin1d, (x,t) -> cts_sin1d(x-t))

function smooth_sin1d(x)
   value = sinpi(2.0*x)
   return value
end

function smooth_sin1d_exact(x,t)
   value = sinpi(2.0(x-t))
end

smooth_sin1d_data = ( x -> 1.0,smooth_sin1d, (x,t) -> smooth_sin1d(x-t))

function step1d(x)
   xmin, xmax = -1.0, 1.0
   dx = xmax - xmin
   if x > xmax
      y = x - dx*floor((x+xmin)/dx)
   elseif x < xmin
      y = x + dx*floor((xmax-x)/dx)
   else
      y = x
   end
   if x < xmin
      value = 0.0
   else
      value = 1.0
   end
   return value
end

step1d_data = ( x -> 1.0,step1d, (x,t) -> step1d(x-t))

function wpack1d(x)
   xmin, xmax = -1.0, 1.0
   dx = xmax - xmin
   if x > xmax
      y = x - dx*floor((x+xmax)/(xmax-xmin))
   elseif x < xmin
      y = x + dx*floor((xmax-x)/(xmax-xmin))
   else
      y = x
   end
   value = sin(10.0*pi*y)*exp(-10*y^2)
   return value
end

wpack1d_data = ( x -> 1.0,wpack1d, (x,t) -> wpack1d(x-t))

function linear_iv(x)
   return 1.0 + 0.5*x
end

# Offner Ranocha test case
or_velocity(x) = x^2
or_ic(x) = cospi(x / 2.0)
or_exact(x,t) = or_ic(x / (1.0 + t*x)) / (1.0 + t*x)^2

or_data = (or_velocity, or_ic, or_exact)

linear1d_data = ( linear_iv, (x,t) -> linear_iv(x-t))

initial_values_la = Dict{String, Tuple{Function,Function}}()
(
   initial_values_la["flat_hat1d"], initial_values_la["mult1d"],
   initial_values_la["cts_sin1d"], initial_values_la["smooth_sin1d"],
   initial_values_la["step1d"], initial_values_la["wpack1d"]
) = ((flat_hat1d, (x,t) -> flat_hat1d(x-t)), (mult1d, (x,t) -> mult1d(x-t)),
      (cts_sin1d, (x,t) -> cts_sin1d(x-t)),
      (smooth_sin1d, (x,t) -> smooth_sin1d(x-t)),
      (step1d, (x,t) -> step1d(x-t)), (wpack1d, (x,t) -> wpack1d(x-t)))

#------------------------------------------------------------------------------
# Initializing Function
#------------------------------------------------------------------------------

# Extending the flux function
@inbounds @inline function SSFR.flux(x, u, eq::LinAdv1D)
   f = eq.velocity(x) * u[1]
   return SVector(f)
   return nothing
end

function get_equation(velocity)
   speed(x,u, eq::LinAdv1D) = velocity(x)
   name = "1d Linear Advection Equation"
   numfluxes = Dict("upwind"  => upwind,
                    "rusanov" => rusanov)
   nvar = 1
   return LinAdv1D(
                   speed, velocity, nvar, name, initial_values_la,
                   numfluxes)
end

export LinAdv1D, upwind, rusanov, flux

end # @muladd

end # module