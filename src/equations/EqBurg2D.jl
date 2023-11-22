module EqBurg2D

using MuladdMacro
using StaticArrays
using Roots: find_zero

using Tenkai

# methods to be extended in this module
import Tenkai: flux

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct Burg2D{Speed <: Function} <: AbstractEquations{2, 1}
    speed::Speed
    name::String
    nvar::Int64
    numfluxes::Dict{String, Function}
end

@inbounds @inline function Tenkai.flux(x, y, u, eq::Burg2D)
    f1 = f2 = 0.5 * u[1]^2
    return SVector(f1), SVector(f2)
end

@inbounds @inline function Tenkai.flux(x, y, u, eq::Burg2D, orientation::Integer)
    f = 0.5 * u[1]^2
    return SVector(f)
end

speed(x, u, eq::Burg2D) = SVector(u[1], u[1])

# Rusanov flux
@inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::Burg2D, dir)
    # lam = max |f'(u)| for u b/w ual and uar
    # laml, lamr = speed(x, ual, eq)[dir], speed(x, uar, eq)[dir]
    laml, lamr = ual[1], uar[1]
    lam = max(abs(laml), abs(lamr))
    Fn = 0.5 * (Fl[1] + Fr[1] - lam * (Ur[1] - Ul[1]))
    return SVector(Fn)
end

@inline function roe(x, ual, uar, Fl, Fr, Ul, Ur, eq::Burg2D, dir)
    if abs(ual[1] - uar[1]) < 1e-10
        a = speed(x, ual, eq)
    else
        fl, fr = flux(x[1], x[2], ual, eq, dir), flux(x[1], x[2], uar, eq, dir)
        a = (fl - fr) / (ual[1] - uar[1])
    end
    Fn = 0.5 * (Fl[1] + Fr[1] - abs(a[1]) * (Ur[1] - Ul[1]))
    return SVector(Fn)
end

function burger_sin_iv(x, y)
    return SVector(0.25 + 0.5 * sinpi(2.0 * (x + y)))
end

function burger_sin_exact(x, y, t)
    implicit_eqn(u) = u - burger_sin_iv(x - t * u, y - t * u)[1]
    avg_iv = 0.5 # avg of initial_value to seed find_zero
    value = find_zero(implicit_eqn, avg_iv)
    return SVector(value)
end

function get_equation()
    name = "2d Burger's equation"
    numfluxes = Dict("rusanov" => rusanov,
                     "roe" => roe)
    nvar = 1
    Burg2D(speed, name, nvar, numfluxes)
end

export flux
end # @muladd

end # module
