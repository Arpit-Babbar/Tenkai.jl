module EqShearShallowWater2D
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
using MuladdMacro

import Tenkai.EqTenMoment2D

import Tenkai.EqTenMoment2D: varnames

import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
               eigmatrix,
               limit_slope, zhang_shu_flux_fix,
               apply_tvb_limiter!, apply_tvb_limiterβ!,
               apply_bound_limiter!, initialize_plot,
               write_soln!, compute_time_step, post_process_soln,
               correct_variable_bound_limiter!,
               update_ghost_values_rkfr!

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

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct ShearShallowWaterNonConservative2D <: AbstractNonConservativeEquations{2, 1}
end

# Shear shallow water equations in 1D
struct ShearShallowWater2D{RealT <: Real} <: AbstractNonConservativeEquations{2, 6}
    gravity::RealT # Gravity
    nvar::Int64
    name::String
    initial_values::Dict{String, Function}
    numfluxes::Dict{String, Function}
    non_conservative_part::ShearShallowWaterNonConservative2D
    ten_moment_problem::EqTenMoment2D.TenMoment2D
end

@inbounds @inline function flux(x, y, u, eq::ShearShallowWater2D, orientation)
    return flux(x, y, u, eq.ten_moment_problem, orientation)
end

# Non-variable specific flux
@inbounds @inline flux(u, eq::ShearShallowWater2D) = flux(one(eltype(u)),
                                                          one(eltype(u)), u, eq)

@inbounds @inline function flux(x, y, u, eq::ShearShallowWater2D)
    return flux(x, y, u, eq, 1), flux(x, y, u, eq, 2)
end

@inline function waterheight(::ShearShallowWater2D, u::AbstractArray)
    ρ = u[1]
    return ρ
end

# function converting primitive variables to PDE variables
function prim2con(eq::ShearShallowWater2D, prim) # primitive, gas constant
    return prim2con(eq.ten_moment_problem, prim)
end

# function converting pde variables to primitive variables
@inbounds @inline function con2prim(eq::ShearShallowWater2D, U)
    return con2prim(eq.ten_moment_problem, prim)
end

function con2prim!(eq::ShearShallowWater2D, cons, prim)
    prim .= con2prim(eq, cons)
end

non_conservative_equation(eq::ShearShallowWater2D) = eq.non_conservative_part

# This will compute the term to be differentiated.
function calc_non_cons_gradient(u_node, x, y, t, eq::ShearShallowWater2D)
    SVector(u_node[1])
end

function calc_non_cons_B(u, x, y, t, orientation::Int, eq::ShearShallowWater2D)
    @unpack gravity = eq
    h = u[1]
    h_v1 = u[2]
    h_v2 = u[3]
    if orientation == 1
        B_x = SMatrix{6, 1}(0.0,
                            gravity * h,
                            0.0,
                            gravity * h_v1,
                            0.5 * gravity * h_v2,
                            0.0)
        return B_x
    else
        B_y = SMatrix{6, 1}(0.0,
                            0.0,
                            gravity * h,
                            0.0,
                            0.5 * gravity * h_v1,
                            gravity * h_v2)
        return B_y
    end
end

function calc_non_cons_B(u, x, y, t, eq::ShearShallowWater2D)
    @unpack gravity = eq
    return calc_non_cons_B(u, x, y, t, 1, eq), calc_non_cons_B(u, x, y, t, 2, eq)
end

# This will compute the action of B on u_non_cons. The u_non_cons may
# be the derivative or it may not. Both quantities need to be computed.
function calc_non_cons_Bu(u, u_non_cons, x, y, t, orientation::Int,
                          eq::ShearShallowWater2D)
    @unpack gravity = eq
    h_nc = u_non_cons[1]
    h = u[1]
    h_v1 = u[2]
    h_v2 = u[3]
    if orientation == 1
        B_x = SVector(0.0,
                      gravity * h * h_nc,
                      0.0,
                      gravity * h_v1 * h_nc,
                      0.5 * gravity * h_v2 * h_nc,
                      0.0)
        return B_x
    else
        B_y = SVector(0.0,
                      0.0,
                      gravity * h * h_nc,
                      0.0,
                      0.5 * gravity * h_v1 * h_nc,
                      gravity * h_v2 * h_nc)
        return B_y
    end
end

function max_abs_eigen_value(eq::ShearShallowWater2D, u, dir)
    @unpack gravity = eq
    h = u[1]
    if dir == 1
        v1 = u[2] / h
        R11 = 2.0 * u[4] - h * v1 * v1
        P11 = R11 / h
        return abs(v1) + sqrt(3.0 * P11 + gravity * h)
    else
        v2 = u[3] / h
        R22 = 2.0 * u[6] - h * v2 * v2
        P22 = R22 / h
        return abs(v2) + sqrt(3.0 * P22 + gravity * h)
    end
end

function compute_time_step(eq::ShearShallowWater2D, problem, grid, aux, op, cfl, u1, ua)
    @timeit aux.timer "Time Step computation" begin
    #! format: noindent
    @unpack dx, dy = grid
    nx, ny = grid.size
    @unpack wg = op
    den = 0.0
    corners = ((0, 0), (nx + 1, 0), (0, ny + 1), (nx + 1, ny + 1))
    for element in CartesianIndices((0:(nx + 1), 0:(ny + 1)))
        el_x, el_y = element[1], element[2]
        if (el_x, el_y) ∈ corners # KLUDGE - Temporary hack
            continue
        end
        u_node = get_node_vars(ua, eq, el_x, el_y)
        sx, sy = max_abs_eigen_value(eq, u_node, 1),
                 max_abs_eigen_value(eq, u_node, 2)
        den = max(den, abs(sx) / dx[el_x] + abs(sy) / dy[el_y] + 1e-12)
        # den = max(den, abs(sx) / dx[el_x] + 1e-12)
    end

    dt = cfl / den

    return dt
    end # timer
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::ShearShallowWater2D,
                                   dir)
    λ = max(max_abs_eigen_value(eq, ual, dir), max_abs_eigen_value(eq, uar, dir)) # local wave speed
    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function Tenkai.eigmatrix(eq::ShearShallowWater2D, u)
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

function Tenkai.apply_bound_limiter!(eq::ShearShallowWater2D, grid, scheme, param, op,
                                     ua, u1, aux)
    apply_bound_limiter!(eq.ten_moment_problem, grid, scheme, param, op, ua, u1, aux)
end

function Tenkai.apply_tvb_limiterβ!(eq::ShearShallowWater2D, problem, scheme, grid,
                                    param,
                                    op, ua, u1, aux)
    apply_tvb_limiterβ!(eq.ten_moment_problem, problem, scheme, grid, param, op, ua, u1,
                        aux)
end

function Tenkai.apply_tvb_limiter!(eq::ShearShallowWater2D, problem, scheme, grid,
                                   param,
                                   op, ua, u1, aux)
    apply_tvb_limiter!(eq.ten_moment_problem, problem, scheme, grid, param, op, ua, u1,
                       aux)
end

admissibility_tolerance(eq::ShearShallowWater2D) = 1e-10

function Tenkai.limit_slope(eq::ShearShallowWater2D, s, ufl, u_s_l, ufr, u_s_r, ue, xl,
                            xr)
    limit_slope(eq.ten_moment_problem, s, ufl, u_s_l, ufr, u_s_r, ue, xl, xr)
end

function Tenkai.zhang_shu_flux_fix(eq::ShearShallowWater2D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    zhang_shu_flux_fix(eq.ten_moment_problem, uprev, ulow, Fn, fn_inner, fn, c)
end

function Tenkai.update_ghost_values_lwfr!(problem, scheme, eq::ShearShallowWater2D,
                                          grid, aux, op, cache, t, dt,
                                          scaling_factor = 1.0)
    update_ghost_values_lwfr!(problem, scheme, eq.ten_moment_problem, grid, aux, op,
                              cache, t, dt, scaling_factor)
end

function update_ghost_values_rkfr!(problem, scheme,
                                   eq::ShearShallowWater2D,
                                   grid, aux, op, cache, t)
    update_ghost_values_rkfr!(problem, scheme, eq.ten_moment_problem,
                              grid, aux, op, cache, t)
end

varnames(eq::ShearShallowWater2D) = varnames(eq.ten_moment_problem)
varnames(eq::ShearShallowWater2D, i::Int) = varnames(eq.ten_moment_problem, i)

function Tenkai.initialize_plot(eq::ShearShallowWater2D, op, grid, problem, scheme,
                                timer, u1,
                                ua)
    initialize_plot(eq.ten_moment_problem, op, grid, problem, scheme, timer, u1, ua)
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::ShearShallowWater2D,
                            grid,
                            problem, param, op, ua, u1, aux, ndigits = 3)
    write_soln!(base_name, fcount, iter, time, dt, eq.ten_moment_problem, grid,
                problem, param, op, ua, u1, aux, ndigits)
end

function Tenkai.post_process_soln(eq::ShearShallowWater2D, aux, problem, param, scheme)
    post_process_soln(eq.ten_moment_problem, aux, problem, param, scheme)
end

function get_equation(; gravity = 9.81)
    name = "2D shear shallow water equations"
    numfluxes = Dict{String, Function}("rusanov" => rusanov)
    nvar = 6
    initial_values = Dict{String, Function}()
    non_conservative_part = ShearShallowWaterNonConservative2D()

    return ShearShallowWater2D(gravity, nvar, name, initial_values, numfluxes,
                               non_conservative_part, EqTenMoment2D.get_equation())
end
end # muladd
end # module
