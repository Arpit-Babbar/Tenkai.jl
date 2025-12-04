module EqShearShallowWater1D
#! format: noindent

import GZip
using Tenkai.DelimitedFiles
using Plots
using LinearAlgebra
using Tenkai.SimpleUnPack
using Printf
using TimerOutputs
using StaticArrays
using Tenkai.Polyester
using Tenkai.LoopVectorization
using Tenkai.JSON3

using Tenkai
import Tenkai.EqTenMoment1D
using Tenkai.Basis

import Tenkai.TenkaicRK: test_var

import Tenkai.EqTenMoment1D: rho_p_indicator!, density_constraint, trace_constraint,
                             det_constraint

import Tenkai: admissibility_tolerance

(import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
                eigmatrix,
                limit_slope, zhang_shu_flux_fix,
                apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln,
                correct_variable_bound_limiter!)

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!, update_ghost_values_lwfr!, refresh!,
               calc_source, cRKSolver)

import ..TenkaicRK: calc_non_cons_gradient, calc_non_cons_Bu, non_conservative_equation,
                    update_ghost_values_ub_N!, AbstractNonConservativeEquations,
                    calc_non_cons_B

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This is used my multiply_add_to_node_vars! to find out how many variables have to be
# differentiated, which is just 1 (density) in the case of the shear shallow water equations.
struct ShearShallowWaterNonConservative1D <: AbstractNonConservativeEquations{1, 1}
end

# Shear shallow water equations in 1D
struct ShearShallowWater1D <: AbstractNonConservativeEquations{1, 6}
    gravity::Float64 # Gravity
    nvar::Int64
    name::String
    initial_values::Dict{String, Function}
    numfluxes::Dict{String, Function}
    non_conservative_part::ShearShallowWaterNonConservative1D
    ten_moment_problem::EqTenMoment1D.TenMoment1D
end

#-------------------------------------------------------------------------------
# PDE Information
#-------------------------------------------------------------------------------
@inbounds @inline function flux(x, u, eq::ShearShallowWater1D)
    @unpack gravity = eq
    h = u[1]
    flux_ = flux(x, u, eq.ten_moment_problem)
    return SVector(flux_[1],
                   flux_[2],
                   flux_[3], flux_[4], flux_[5], flux_[6])
end

@inbounds @inline flux(U, eq::ShearShallowWater1D) = flux(1.0, U, eq)

@inline function waterheight(::ShearShallowWater1D, u::AbstractArray)
    ρ = u[1]
    return ρ
end

@inline function trace_constraint(eq::ShearShallowWater1D, u::AbstractArray)
    return trace_constraint(eq.ten_moment_problem, u)
end

@inline function det_constraint(eq::ShearShallowWater1D, u::AbstractArray)
    return det_constraint(eq.ten_moment_problem, u)
end

# function converting primitive variables to PDE variables
function prim2con(eq::ShearShallowWater1D, prim) # primitive, gas constant
    return prim2con(eq.ten_moment_problem, prim)
end

# function converting pde variables to primitive variables
@inbounds @inline function con2prim(eq::ShearShallowWater1D, U)
    return con2prim(eq.ten_moment_problem, prim)
end

function con2prim!(eq::ShearShallowWater1D, cons, prim)
    prim .= con2prim(eq, cons)
end

non_conservative_equation(eq::ShearShallowWater1D) = eq.non_conservative_part

# This will compute the term to be differentiated.
calc_non_cons_gradient(u_node, x_, t, eq::ShearShallowWater1D) = SVector(u_node[1])

function calc_non_cons_B(u, x_, t, eq::ShearShallowWater1D)
    @unpack gravity = eq
    h, h_v1, h_v2 = u[1], u[2], u[3]
    B = SMatrix{6, 1}(0.0, gravity * h, 0.0, gravity * h_v1, 0.5 * gravity * h_v2, 0.0)
    # By_u = SVector(0.0, 0.0, 0.0, 0.0, 0.5 * g * h * v1 * h_nc, g * h * v2 * h_nc)
    return B # + By_u
end

# This will compute the action of B on u_non_cons. The u_non_cons may
# be the derivative or it may not. Both quantities need to be computed.
function calc_non_cons_Bu(u, u_non_cons, x_, t, eq::ShearShallowWater1D)
    @unpack gravity = eq
    h_nc = u_non_cons[1]
    h = u[1]
    v1 = u[2] / h
    v2 = u[3] / h
    Bx_u = SVector(0.0,
                   gravity * h * h_nc,
                   0.0,
                   gravity * h * v1 * h_nc, 0.5 * gravity * h * v2 * h_nc, 0.0)
    # By_u = SVector(0.0, 0.0, 0.0, 0.0, 0.5 * g * h * v1 * h_nc, g * h * v2 * h_nc)
    return Bx_u # + By_u
end

function compute_time_step(eq::ShearShallowWater1D, problem, grid, aux, op, cfl, u1, ua)
    @unpack source_terms = problem
    nx = grid.size
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        u = get_node_vars(ua, eq, i)
        smax = max_abs_eigen_value(eq, u)
        den = max(den, smax / dx[i])
    end
    dt = cfl / den
    return dt
end

# TODO - Move to shear shallow water equations file
function test_var(val_min, eq::Union{ShearShallowWater1D},
                  variable::typeof(det_constraint), el_x, el_y)
    return nothing # Shear shallow water equations can have negative determinant in low order solution
end

function max_abs_eigen_value(eq::ShearShallowWater1D, u)
    @unpack gravity = eq
    h = u[1]
    v1 = u[2] / h
    R11 = 2.0 * u[4] - h * v1 * v1
    P11 = R11 / h
    return abs(v1) + sqrt(3.0 * P11 + gravity * h)
end

function max_abs_eigen_value(eq::ShearShallowWater1D, ul, ur)
    @unpack gravity = eq
    hl, hr = ul[1], ur[1]
    v1l, v1r = ul[2] / hl, ur[2] / hr
    R11l, R11r = 2.0 * ul[4] - hl * v1l * v1l, 2.0 * ur[4] - hr * v1r * v1r
    P11l, P11r = R11l / hl, R11r / hr
    λ = max(abs(v1l), abs(v1r)) +
        max(sqrt(3.0 * P11l + gravity * hl), sqrt(3.0 * P11r + gravity * hr))
    return λ
end

@inbounds @inline function rho_p_indicator!(un, eq::ShearShallowWater1D)
    rho_p_indicator!(un, eq.ten_moment_problem)
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::ShearShallowWater1D,
                                   dir)
    λ = max(max_abs_eigen_value(eq, ual), max_abs_eigen_value(eq, uar)) # local wave speed

    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function Tenkai.eigmatrix(eq::ShearShallowWater1D, u)
    @unpack gravity = eq

    @assert false "To be implemented!"

    # Inverse eigenvector-matrix
    L11 = 1.0 - g2 * v^2 / c^2
    L21 = (g2 * v^2 - v * c) / (d * c)

    L12 = g1 * v / c^2
    L22 = (c - g1 * v) / (d * c)

    L = SMatrix{nvariables(eq), nvariables(eq)}(L11, L21,
                                                L12, L22)

    # Eigenvector matrix
    R11 = 1.0
    R21 = v

    R12 = f
    R22 = (v + c) * f

    R = SMatrix{nvariables(eq), nvariables(eq)}(R11, R21,
                                                R12, R22)

    return R, L
end

#-------------------------------------------------------------------------------
# Limiters
#-------------------------------------------------------------------------------
function Tenkai.apply_bound_limiter!(eq::ShearShallowWater1D, grid, scheme, param, op,
                                     ua,
                                     u1, aux)
    apply_bound_limiter!(eq.ten_moment_problem, grid, scheme, param, op, ua, u1, aux)
end

function Tenkai.apply_tvb_limiter!(eq::ShearShallowWater1D, problem, scheme, grid,
                                   param, op, ua,
                                   u1, aux)
    apply_tvb_limiter!(eq.ten_moment_problem, problem, scheme, grid, param, op, ua, u1,
                       aux)
end

admissibility_tolerance(eq::ShearShallowWater1D) = 1e-10

function Tenkai.limit_slope(eq::ShearShallowWater1D, s, ufl, u_s_l, ufr, u_s_r, ue, xl,
                            xr)
    limit_slope(eq.ten_moment_problem, s, ufl, u_s_l, ufr, u_s_r, ue, xl, xr)
end

function Tenkai.zhang_shu_flux_fix(eq::ShearShallowWater1D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    zhang_shu_flux_fix(eq.ten_moment_problem, uprev, ulow, Fn, fn_inner, fn, c)
end

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

function Tenkai.initialize_plot(eq::ShearShallowWater1D, op, grid, problem, scheme,
                                timer, u1,
                                ua)
    initialize_plot(eq.ten_moment_problem, op, grid, problem, scheme, timer, u1, ua)
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::ShearShallowWater1D,
                            grid,
                            problem, param, op, ua, u1, aux, ndigits = 3)
    write_soln!(base_name, fcount, iter, time, dt, eq.ten_moment_problem, grid,
                problem, param, op, ua, u1, aux, ndigits)
end

function Tenkai.post_process_soln(eq::ShearShallowWater1D, aux, problem, param, scheme)
    post_process_soln(eq.ten_moment_problem, aux, problem, param, scheme)
end

function get_equation(gravity)
    name = "1D shear shallow water equations"
    numfluxes = Dict("rusanov" => rusanov)
    nvar = 6
    initial_values = Dict()
    non_conservative_part = ShearShallowWaterNonConservative1D()

    return ShearShallowWater1D(gravity, nvar, name, initial_values, numfluxes,
                               non_conservative_part,
                               EqTenMoment1D.get_equation())
end
end # muladd
end # module
