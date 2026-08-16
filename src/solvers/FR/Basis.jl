module Basis

using FastGaussQuadrature
using LinearAlgebra
using StaticArrays
using Printf
using SimpleUnPack
using TimerOutputs
using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#-------------------------------------------------------------------------------
# Legendre polynomials on [-1,+1]
#-------------------------------------------------------------------------------
function Legendre(n, x)
    if n == 0
        value = one(x)
    elseif n == 1
        value = x
    else
        value = (oftype(x, 2 * n - 1) / n * x * Legendre(n - 1, x)
                 -
                 oftype(x, n - 1) / n * Legendre(n - 2, x))
    end

    return value
end

#-------------------------------------------------------------------------------
# Derivative of Legendre
#-------------------------------------------------------------------------------
function dLegendre(n, x)
    if n == 0
        value = zero(x)
    elseif n == 1
        value = one(x)
    else
        value = n * Legendre(n - 1, x) + x * dLegendre(n - 1, x)
    end

    return value
end
#-------------------------------------------------------------------------------
# Normalize Legendre polynomials to unit L2 norm in [0,1]
#-------------------------------------------------------------------------------
function nLegendre(n, x)
    value = sqrt(oftype(x, 2 * n + 1)) * Legendre(n, x)
    return value
end

#-------------------------------------------------------------------------------
# Legendre polynomial P_n and its first two derivatives at x, evaluated with a
# non-recursive three term recurrence in the arithmetic of `x`.
# Used to compute Gauss quadrature nodes in arbitrary precision.
#-------------------------------------------------------------------------------
function legendre_derivatives(n, x::RealT) where {RealT <: Real}
    if n == 0
        return one(RealT), zero(RealT), zero(RealT)
    end
    p_prev, p = one(RealT), x
    for k in 2:n
        p_prev, p = p, ((2 * k - 1) * x * p - (k - 1) * p_prev) / k
    end
    # (1 - x^2) P_n' = n * (P_{n-1} - x * P_n)
    one_minus_x2 = (one(RealT) - x) * (one(RealT) + x)
    dp = n * (p_prev - x * p) / one_minus_x2
    # Legendre's differential equation, (1 - x^2) P'' - 2 x P' + n (n+1) P = 0
    ddp = (2 * x * dp - n * (n + 1) * p) / one_minus_x2
    return p, dp, ddp
end

#-------------------------------------------------------------------------------
# Newton refinement of a quadrature node, starting from a `Float64` guess.
# Each iteration doubles the number of correct digits, so a handful of steps
# suffice for any double-double or quad-double type. The iteration is stopped
# once the correction no longer decreases, which is the best that can be done
# in the working precision.
#-------------------------------------------------------------------------------
function refine_node(f_and_df, x0::RealT) where {RealT <: Real}
    x = x0
    # Any upper bound on the first correction works here; the `Float64` initial
    # guess is already accurate to about 1e-16.
    dx_prev = one(RealT)
    for _ in 1:100
        f, df = f_and_df(x)
        dx = f / df
        x -= dx
        abs_dx = abs(dx)
        # Stop as soon as we stop making progress; `iszero` catches exact
        # convergence (e.g. the node x = 0 of odd rules).
        if iszero(abs_dx) || abs_dx >= dx_prev
            break
        end
        dx_prev = abs_dx
    end
    return x
end

#-------------------------------------------------------------------------------
# Is `RealT` wider than the `Float64` rules of FastGaussQuadrature? If not, the
# `Float64` nodes are already correct to the last bit of `RealT` and refining
# them in the narrower arithmetic could only make them worse.
#-------------------------------------------------------------------------------
needs_refinement(RealT::Type{<:Real}) = eps(RealT) < eps(Float64)

#-------------------------------------------------------------------------------
# Gauss-Legendre nodes and weights on [-1,1] in arbitrary precision.
# The `Float64` nodes of FastGaussQuadrature are used as initial guesses.
#-------------------------------------------------------------------------------
function gauss_legendre_nodes(n, RealT::Type{<:Real})
    x64, w64 = gausslegendre(n)
    needs_refinement(RealT) || return Vector{RealT}(x64), Vector{RealT}(w64)
    x = Vector{RealT}(x64)
    w = Vector{RealT}(w64)
    for i in 1:n
        x[i] = refine_node(x0 -> begin
                               p, dp, _ = legendre_derivatives(n, x0)
                               (p, dp)
                           end, x[i])
        _, dp, _ = legendre_derivatives(n, x[i])
        # w_i = 2 / ((1 - x_i^2) * P_n'(x_i)^2)
        w[i] = 2 / ((one(RealT) - x[i]) * (one(RealT) + x[i]) * dp * dp)
    end
    return x, w
end

#-------------------------------------------------------------------------------
# Gauss-Lobatto nodes and weights on [-1,1] in arbitrary precision.
# The interior nodes are the roots of P_{n-1}', the end points are +-1.
#-------------------------------------------------------------------------------
function gauss_lobatto_nodes(n, RealT::Type{<:Real})
    @assert n>=2 "Gauss-Lobatto quadrature needs at least 2 points"
    x64, w64 = gausslobatto(n)
    needs_refinement(RealT) || return Vector{RealT}(x64), Vector{RealT}(w64)
    x = Vector{RealT}(x64)
    w = Vector{RealT}(w64)
    m = n - 1
    x[1], x[n] = -one(RealT), one(RealT)
    for i in 2:(n - 1)
        x[i] = refine_node(x0 -> begin
                               _, dp, ddp = legendre_derivatives(m, x0)
                               (dp, ddp)
                           end, x[i])
    end
    for i in 1:n
        p, _, _ = legendre_derivatives(m, x[i])
        # w_i = 2 / (n * (n-1) * P_{n-1}(x_i)^2)
        w[i] = 2 / (n * m * p * p)
    end
    return x, w
end

#-------------------------------------------------------------------------------
# Return n points and weights for the interval [0,1]
#-------------------------------------------------------------------------------
function weights_and_points(n, type, RealT::Type{<:Real} = Float64)
    if type == "gl"
        x, w = gauss_legendre_nodes(n, RealT)
    elseif type == "gll"
        x, w = gauss_lobatto_nodes(n, RealT)
    else
        println("Unknown solution points")
        @assert false
    end
    w *= 0.5f0
    x = 0.5f0 * (x .+ 1)
    return SVector{n}(x), SVector{n}(w)
end

#-------------------------------------------------------------------------------
# xp = set of grid points
# Returns i'th Lagrange polynomial value at x
#-------------------------------------------------------------------------------
function Lagrange(i, xp, x)
    T = promote_type(eltype(xp), typeof(x))
    value = one(T)
    n = length(xp)
    for j in 1:n
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
    T = promote_type(eltype(xp), eltype(x))
    V = zeros(T, m, n)
    for j in 1:n
        for i in 1:m
            V[i, j] = Lagrange(j, xp, x[i])
        end
    end
    return SMatrix{m, n}(V)
end

#-------------------------------------------------------------------------------
# Vandermonde matrix for Legendre polynomials
# k : degree
# x : evaluation points in [0,1]
#-------------------------------------------------------------------------------
function Vandermonde_leg(k, x)
    n = k + 1
    m = length(x)
    T = eltype(x)
    V = zeros(T, m, n)
    for j in 1:n
        for i in 1:m
            V[i, j] = nLegendre(j - 1, 2 * x[i] - one(x[i]))
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
function Vandermonde_leg_krivodonova(k, x)
    n = k + 1
    m = length(x)
    T = eltype(x)
    V = zeros(T, m, n)
    for j in 1:n
        for i in 1:m
            # krivodonova's normalization, the division is redundant
            V[i, j] = Legendre(j - 1, x[i]) / Legendre(j - 1, one(x[i]))
        end
    end
    return V
end

# Pass xg for [-1,1] and get the nodal2modal map.
function nodal2modal_krivodonova(xg)
    nd = length(xg)
    k = nd - 1  # highest degree Legendre polynomial

    nq = k + 1             # quadrature points for projection
    # x,w correspond to [-1,1]
    x, w = gauss_legendre_nodes(nq, eltype(xg))

    Vleg = Vandermonde_leg_krivodonova(k, x)

    # Legendre polynomials evaluated at quadrature points
    Vlag = Vandermonde_lag(xg, x)

    T = eltype(xg)
    M = zeros(T, nd)
    for i in 1:nd
        M[i] = @views sum(Vleg[:, i] .* Vleg[:, i] .* w)
    end

    A = zeros(T, nd, nd) # projection matrix
    for j in 1:nd
        for i in 1:nd
            A[i, j] = @views sum(Vleg[:, i] .* Vlag[:, j] .* w)
        end
    end

    for i in 1:nd
        A[i, :] .= @views A[i, :] ./ M[i]
    end

    return A
end

#-------------------------------------------------------------------------------
# Projection matrix: nodal --> modal
# xg must be in [0,1]
#-------------------------------------------------------------------------------
function nodal2modal(xg)
    nd = length(xg)
    k = nd - 1  # highest degree Legendre polynomial

    nq = k + 1 # quadrature points for projection
    x, w = weights_and_points(nq, "gl", eltype(xg))

    Vleg = Vandermonde_leg(k, x)
    Vlag = Vandermonde_lag(xg, x)

    T = eltype(xg)
    M = zeros(T, nd)
    for i in 1:nd
        M[i] = @views sum(Vleg[:, i] .* Vleg[:, i] .* w)
    end
    err = maximum(abs.(M - ones(nd)))
    if err > 1e-10
        println("Legendre mass matrix = ", M)
        @assert false
    end

    A = zeros(T, nd, nd) # projection matrix
    for j in 1:nd
        for i in 1:nd
            A[i, j] = @views sum(Vleg[:, i] .* Vlag[:, j] .* w)
        end
    end

    for i in 1:nd
        A[i, :] .= @views A[i, :] ./ M[i]
    end

    return A
end
#-------------------------------------------------------------------------------
function barycentric_weights(x)
    n = length(x)
    T = eltype(x)
    w = ones(T, n)

    for j in 2:n
        for k in 1:(j - 1)
            w[k] *= x[k] - x[j] # all i > j cases
            w[j] *= x[j] - x[k] # all i < j cases
        end
    end

    T = eltype(x)
    value = one(T) ./ w
    return value
end

#-------------------------------------------------------------------------------
# Differentiation matrix
# D[i,j] = l_j'(x_i)
#-------------------------------------------------------------------------------
function diff_mat(x)
    w = barycentric_weights(x)
    n = length(x)
    T = eltype(x)
    D = zeros(T, n, n)

    for j in 1:n
        for i in 1:n
            if j != i
                D[i, j] = (w[j] / w[i]) * one(T) / (x[i] - x[j])
                D[i, i] -= D[i, j]
            end
        end
    end
    return SMatrix{n, n}(D)
end

#-------------------------------------------------------------------------------
# FR Radau correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function gl_radau(k, x)
    value = 0.5f0 * (-1)^k * (Legendre(k, x) - Legendre(k + 1, x))
    return value
end

function gr_radau(k, x)
    value = 0.5f0 * (Legendre(k, x) + Legendre(k + 1, x))
    return value
end

#-------------------------------------------------------------------------------
# Derivatives of FR Radau correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function dgl_radau(k, x)
    value = 0.5f0 * (-1)^k * (dLegendre(k, x) - dLegendre(k + 1, x))
    return value
end

function dgr_radau(k, x)
    value = 0.5f0 * (dLegendre(k, x) + dLegendre(k + 1, x))
    return value
end

#-------------------------------------------------------------------------------
# FR g2 correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function gl_g2(k, x)
    value = 0.5f0 * (-1)^k *
            (Legendre(k, x) -
             ((k + one(x)) * Legendre(k - 1, x) +
              k * Legendre(k + 1, x)) / (2 * k + one(x)))
    return value
end

function gr_g2(k, x)
    value = gl_g2(k, -x)
    return value
end

#-------------------------------------------------------------------------------
# Derivatives of FR g2 correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function dgl_g2(k, x)
    value = 0.5f0 * (-1)^k * (one(x) - x) * dLegendre(k, x)
    return value
end

function dgr_g2(k, x)
    value = -dgl_g2(k, -x)
    return value
end

#-------------------------------------------------------------------------------
# sol_pts = gl, gll
# N       = degree
#-------------------------------------------------------------------------------
function fr_operators(N, sol_pts, cor_fun, RealT::Type{<:Real} = Float64)
    println("Setting up differentiation operators")
    @printf("   Degree     = %d\n", N)
    @printf("   Sol points = %s\n", sol_pts)
    @printf("   Cor fun    = %s\n", cor_fun)
    println("   Real type  = $RealT")

    nd = N + 1 # number of dofs
    xg, wg = weights_and_points(nd, sol_pts, RealT)

    # Required to evaluate solution at face
    T = eltype(xg)
    Vl, Vr = zeros(T, nd), zeros(T, nd)
    for i in 1:nd
        Vl[i] = Lagrange(i, xg, zero(T))
        Vr[i] = Lagrange(i, xg, one(T))
    end

    # Correction terms
    if cor_fun == "radau"
        dgl, dgr = dgl_radau, dgr_radau
    elseif cor_fun == "g2"
        dgl, dgr = dgl_g2, dgr_g2
    else
        prinln("Unknown cor_fun = ", cor_fun)
        @assert false
    end

    T = eltype(xg)
    bl, br = zeros(T, nd), zeros(T, nd)
    for i in 1:nd
        bl[i] = 2 * dgl(N, 2 * xg[i] - one(xg[i]))
        br[i] = 2 * dgr(N, 2 * xg[i] - one(xg[i]))
    end

    # Convert vectors to SVector for optimized operations
    Vl, Vr, bl, br = (SVector{nd}(Vl), SVector{nd}(Vr), SVector{nd}(bl),
                      SVector{nd}(br))

    # Differentiation matrix
    Dm = diff_mat(xg)
    bV = -bl * Vl' - br * Vr'
    D1 = Dm + bV
    Dsplit = 2 * Dm + bV

    DmT = SMatrix{nd, nd}(Dm')
    D1T = SMatrix{nd, nd}(D1')

    # Vandermonde matrix to convert to gll points, used by bounds limiter
    if nd > 1
        xgll, wgll = weights_and_points(nd, "gll", RealT)
        Vgll = Vandermonde_lag(xg, xgll)
    else # GLL points not defined for nd=1, so we put identity matrix then
        Vgll = Matrix(one(T) * I, nd, nd)
        Vgll = SMatrix{nd, nd}(Vgll)
    end

    wg_inv = one(T) ./ wg

    op = (; degree = N, xg, wg, wg_inv, Vl, Vr, bl, br, Dm, DmT, bV, D1, D1T, Dsplit,
          Vgll)
    return op
end

export weights_and_points
export fr_operators
export Vandermonde_lag
(export nodal2modal, nodal2modal_krivodonova, Vandermonde_leg,
        Vandermonde_leg_krivodonova)
end

end # @muladd
