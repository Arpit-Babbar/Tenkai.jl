module EqSupBurg1D

using StaticArrays
using MuladdMacro

using Tenkai

# flux function will be extended to SupBurg1D
import Tenkai: flux

import Roots.find_zero

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# struct SupBurg1D{F1,F2 <: Function} <: AbstractScalarEq1D
struct SupBurg1D{F2 <: Function} <: AbstractEquations{1, 1}
    speed::F2
    nvar::Int64
    name::String
end

# Extending the flux function
function Tenkai.flux(x, u, eq::SupBurg1D)
    return SVector(0.25 * u[1]^4)
end

speed(x, u, eq::SupBurg1D) = u[1]^3

# Rusanov flux
function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::SupBurg1D, dir)
    # lam = max |f'(u)| for u b/w ual and uar
    laml, lamr = speed(x, ual, eq), speed(x, uar, eq)
    lam = max(abs(laml), abs(lamr))
    Fn = 0.5 * (Fl[1] + Fr[1] - lam * (Ur[1] - Ul[1]))
    return SVector(Fn)
end

# Function initializing a particular form of the equation struct
function get_equation()
    name = "1d Burger's equation"
    nvar = 1
    SupBurg1D(speed, nvar, name)
end

export flux, speed
end # @muladd

end
