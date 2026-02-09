module EqBurg1D

using StaticArrays
using MuladdMacro

using Tenkai

# flux function will be extended to Burg1D
import Tenkai: flux

import Tenkai.EqEuler1D: max_abs_eigen_value

import Roots.find_zero

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# struct Burg1D{F1,F2 <: Function} <: AbstractScalarEq1D
struct Burg1D{F2 <: Function} <: AbstractEquations{1, 1}
    speed::F2
    nvar::Int64
    name::String
    initial_values::Dict{String, Tuple{Function, Function}}
    numfluxes::Dict{String, Function}
end

# Extending the flux function
function Tenkai.flux(x, u, eq::Burg1D)
    return SVector(0.5 * u[1]^2)
    return nothing
end

speed(x, u, eq::Burg1D) = u[1]

function max_abs_eigen_value(eq::Burg1D, u)
    return abs(u[1])
end

# Rusanov flux
function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::Burg1D, dir)
    # lam = max |f'(u)| for u b/w ual and uar
    laml, lamr = speed(x, ual, eq), speed(x, uar, eq)
    lam = max(abs(laml), abs(lamr))
    Fn = 0.5 * (Fl[1] + Fr[1] - lam * (Ur[1] - Ul[1]))
    return SVector(Fn)
    return nothing
end

function roe(x, ual, uar, Fl, Fr, Ul, Ur, eq::Burg1D, dir)
    if abs(ual[1] - uar[1]) < 1e-10
        a = speed(x, ual, eq)
    else
        a = 0.5 * (ual[1] + uar[1])
    end
    Fn = 0.5 * (Fl[1] + Fr[1] - abs(a) * (Ur[1] - Ul[1]))
    return SVector(Fn)
end

function initial_value_burger_sin(x)
    return 0.2 * sin(x)
end

function exact_solution_burger_sin(x, t)
    t = min(t, 5.0)
    implicit_eqn(u) = u - initial_value_burger_sin(x - t * u)
    seed = initial_value_burger_sin(x)
    value = find_zero(implicit_eqn, seed)
    return value
end

function zero_boundary_value(x, t)
    return zero(eltype(x))
end

function initial_value_hat(x)
    if x <= 0.0
        value = 0.0
    elseif 0.0 < x <= 0.5
        value = 1.0
    else
        value = 0.0
    end
    return value
end

function exact_solution_hat(x, t)
    if t <= 1.0
        if x <= 0.0
            value = 0.0
        elseif 0.0 < x <= t
            value = x / t
        elseif t < x <= 0.5 + 0.5 * t
            value = 1.0
        else
            value = 0.0
        end
    else
        if x <= 0.0
            value = 0.0
        elseif x <= sqrt(t)
            value = x / t
        else
            value = 0.0
        end
    end
    return value
end

initial_values_burg = Dict{String,
                           Tuple{Function, Function}}("hat"
                                                      => (initial_value_hat,
                                                          exact_solution_hat))

# Function initializing a particular form of the equation struct
function get_equation()
    name = "1d Burger's equation"
    numfluxes = Dict("rusanov" => rusanov,
                     "roe" => roe)
    nvar = 1
    Burg1D(speed, nvar, name, initial_values_burg, numfluxes)
end

export flux, speed, Burg1D
end # @muladd

end
