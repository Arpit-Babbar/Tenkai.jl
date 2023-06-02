module Basis

using FastGaussQuadrature
using LinearAlgebra
using StaticArrays
using Printf
using UnPack
using TimerOutputs
using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#-------------------------------------------------------------------------------
# Legendre polynomials on [-1,+1]
#-------------------------------------------------------------------------------
function Legendre(n, x)
   if n == 0
      value = 1.0
   elseif n == 1
      value = x
   else
      value = ((2.0*n-1.0)/n * x * Legendre(n-1, x)
               - (n-1.0)/n * Legendre(n-2, x))
   end

   return value
end

#-------------------------------------------------------------------------------
# Derivative of Legendre
#-------------------------------------------------------------------------------
function dLegendre(n, x)
   if n == 0
      value = 0.0
   elseif n == 1
      value = 1.0
   else
      value = n * Legendre(n-1, x) + x * dLegendre(n-1, x)
   end

   return value
end
#-------------------------------------------------------------------------------
# Normalize Legendre polynomials to unit L2 norm in [0,1]
#-------------------------------------------------------------------------------
function nLegendre(n, x)
   value = sqrt(2.0*n+1.0) * Legendre(n,x)
   return value
end

#-------------------------------------------------------------------------------
# Return n points and weights for the interval [0,1]
#-------------------------------------------------------------------------------
function weights_and_points(n, type)
   if type == "gl"
      x, w = gausslegendre(n)
   elseif type == "gll"
      x, w = gausslobatto(n)
   else
      println("Unknown solution points")
      @assert false
   end
   w *= 0.5
   x  = 0.5*(x .+ 1.0)
   return SVector{n}(x), SVector{n}(w)
end

#-------------------------------------------------------------------------------
# xp = set of grid points
# Returns i'th Lagrange polynomial value at x
#-------------------------------------------------------------------------------
function Lagrange(i, xp, x)
   value = 1.0
   n     = length(xp)
   for j=1:n
      if j != i
         value *= (x - xp[j]) / (xp[i] - xp[j])
      end
   end
   return value
end

#-------------------------------------------------------------------------------
# Vandermonde Matrix for Lagrange polynomials
# xp: grid points
# x:  evaluation points
#-------------------------------------------------------------------------------
function Vandermonde_lag(xp, x)
   n = length(xp)
   m = length(x)
   V = zeros(Float64, m, n)
   for j=1:n
      for i=1:m
         V[i,j] = Lagrange(j, xp, x[i])
      end
   end
   return SMatrix{m,n}(V)
end

#-------------------------------------------------------------------------------
# Vandermonde matrix for Legendre polynomials
# k : degree
# x : evaluation points in [0,1]
#-------------------------------------------------------------------------------
function Vandermonde_leg(k,x)
   n = k + 1
   m = length(x)
   V = zeros(Float64, m, n)
   for j=1:n
      for i=1:m
         V[i, j] = nLegendre(j-1, 2.0*x[i]-1.0)
      end
   end
   return V
end

#-------------------------------------------------------------------------------
# krivodonova
# Every thing is in [-1,1] for this
# Legendre polynomials are normalized so that $P_n(1) = 1$.
#-------------------------------------------------------------------------------
# Pass x in [-1,1] and get Vandermonde matrix
function Vandermonde_leg_krivodonova(k,x)
   n = k + 1
   m = length(x)
   V = zeros(Float64, m, n)
   for j=1:n
      for i=1:m
         # krivodonova's normalization, the division is redundant
         V[i, j] = Legendre(j-1, x[i])/Legendre(j-1,1.0)
      end
   end
   return V
end

# Pass xg for [-1,1] and get the nodal2modal map.
function nodal2modal_krivodonova(xg)
   nd  = length(xg)
   k   = nd - 1  # highest degree Legendre polynomial

   nq   = k + 1             # quadrature points for projection
   x, w = gausslegendre(nq) # x,w correspond to [-1,1]

   Vleg = Vandermonde_leg_krivodonova(k, x)

   # Legendre polynomials evaluated at quadrature points
   Vlag = Vandermonde_lag(xg, x)

   M = zeros(Float64, nd)
   for i=1:nd
      M[i] = @views sum(Vleg[:,i] .* Vleg[:,i] .* w)
   end

   A = zeros(Float64, nd, nd) # projection matrix
   for j=1:nd
      for i=1:nd
         A[i,j] = @views sum(Vleg[:,i] .* Vlag[:,j] .* w)
      end
   end

   for i=1:nd
      A[i,:] .= @views A[i,:] ./ M[i]
   end

   return A
end

#-------------------------------------------------------------------------------
# Projection matrix: nodal --> modal
# xg must be in [0,1]
#-------------------------------------------------------------------------------
function nodal2modal(xg)
   nd = length(xg)
   k  = nd - 1  # highest degree Legendre polynomial

   nq = k + 1 # quadrature points for projection
   x, w = weights_and_points(nq, "gl")

   Vleg = Vandermonde_leg(k,  x)
   Vlag = Vandermonde_lag(xg, x)

   M = zeros(Float64, nd)
   for i=1:nd
      M[i] = @views sum(Vleg[:,i] .* Vleg[:,i] .* w)
   end
   err = maximum(abs.(M - ones(nd)))
   if err > 1e-10
      println("Legendre mass matrix = ", M)
      @assert false
   end

   A = zeros(Float64, nd, nd) # projection matrix
   for j=1:nd
      for i=1:nd
         A[i,j] = @views sum(Vleg[:,i] .* Vlag[:,j] .* w)
      end
   end

   for i=1:nd
      A[i,:] .= @views A[i,:] ./ M[i]
   end

   return A
end
#-------------------------------------------------------------------------------
function barycentric_weights(x)
   n = length(x)
   w = ones(Float64, n)

   for j=2:n
      for k in 1:j-1
         w[k] *= x[k] - x[j] # all i > j cases
         w[j] *= x[j] - x[k] # all i < j cases
      end
   end

   value = 1.0 ./ w
   return value
end

#-------------------------------------------------------------------------------
# Differentiation matrix
# D[i,j] = l_j'(x_i)
#-------------------------------------------------------------------------------
function diff_mat(x)
   w = barycentric_weights(x)
   n = length(x)
   D = zeros(Float64, n, n)

   for j=1:n
      for i=1:n
         if j != i
            D[i,j] = (w[j]/w[i]) * 1.0/(x[i]-x[j])
            D[i,i]-= D[i,j]
         end
      end
   end
   return SMatrix{n,n}(D)
end

#-------------------------------------------------------------------------------
# FR Radau correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function gl_radau(k, x)
    value = 0.5 * (-1)^k * (Legendre(k,x) - Legendre(k+1,x))
    return value
end

function gr_radau(k,x)
    value = 0.5 * (Legendre(k,x) + Legendre(k+1,x))
    return value
end

#-------------------------------------------------------------------------------
# Derivatives of FR Radau correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function dgl_radau(k, x)
    value = 0.5 * (-1)^k * (dLegendre(k,x) - dLegendre(k+1,x))
    return value
end

function dgr_radau(k, x)
   value = 0.5 * (dLegendre(k,x) + dLegendre(k+1,x))
   return value
end

#-------------------------------------------------------------------------------
# FR g2 correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function gl_g2(k, x)
   value = 0.5 * (-1)^k * (Legendre(k,x) - ((k+1.0)*Legendre(k-1,x) +
                                             k*Legendre(k+1,x))/(2.0*k+1.0))
   return value
end

function gr_g2(k,x)
   value = gl_g2(k,-x)
   return value
end

#-------------------------------------------------------------------------------
# Derivatives of FR g2 correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function dgl_g2(k, x)
   value = 0.5 * (-1)^k * (1.0 - x) * dLegendre(k,x)
   return value
end

function dgr_g2(k, x)
   value = -dgl_g2(k,-x)
   return value
end

#-------------------------------------------------------------------------------
# sol_pts = gl, gll
# N       = degree
#-------------------------------------------------------------------------------
# Struct storing FR Operators
struct OP{T1,T2}
   degree::Int64
   xg::T1
   wg::T1
   Vl::T1
   Vr::T1
   bl::T1
   br::T1
   Dm::T2
   DmT::T2
   D1::T2
   D1T::T2
   Vgll::T2
end

function fr_operators(N, sol_pts, cor_fun)
   println("Setting up differentiation operators")
   @printf("   Degree     = %d\n", N)
   @printf("   Sol points = %s\n", sol_pts)
   @printf("   Cor fun    = %s\n", cor_fun)

   nd = N + 1 # number of dofs
   xg, wg = weights_and_points(nd, sol_pts)

   # Required to evaluate solution at face
   Vl, Vr = zeros(Float64, nd), zeros(Float64, nd)
   for i=1:nd
      Vl[i] = Lagrange(i, xg, 0.0)
      Vr[i] = Lagrange(i, xg, 1.0)
   end

   # Correction terms
   if cor_fun == "radau"
      dgl, dgr = dgl_radau, dgr_radau
   elseif cor_fun == "g2"
      dgl, dgr = dgl_g2, dgr_g2
   else
      prinln("Unknown cor_fun = ",cor_fun)
      @assert false
   end

   bl, br = zeros(nd), zeros(nd)
   for i=1:nd
      bl[i] = 2.0 * dgl(N, 2.0*xg[i]-1.0)
      br[i] = 2.0 * dgr(N, 2.0*xg[i]-1.0)
   end

   # Convert vectors to SVector for optimized operations
   Vl, Vr, bl, br = ( SVector{nd}(Vl), SVector{nd}(Vr), SVector{nd}(bl),
                      SVector{nd}(br) )

   # Differentiation matrix
   Dm = diff_mat(xg)
   D1 = Dm - bl * Vl' - br * Vr'

   DmT = SMatrix{nd,nd}(Dm')
   D1T = SMatrix{nd,nd}(D1')

   # Vandermonde matrix to convert to gll points, used by bounds limiter
   if nd > 1
      xgll, wgll = weights_and_points(nd, "gll")
      Vgll = Vandermonde_lag(xg, xgll)
   else # GLL points not defined for nd=1, so we put identity matrix then
      Vgll = Matrix(1.0*I, nd, nd)
      Vgll = SMatrix{nd, nd}(Vgll)
   end

   op = OP(N, xg, wg, Vl, Vr, bl, br, Dm, DmT, D1, D1T, Vgll)
   return op
end



export weights_and_points
export fr_operators
export Vandermonde_lag
(
export nodal2modal, nodal2modal_krivodonova, Vandermonde_leg,
       Vandermonde_leg_krivodonova
)

end

end # @muladd