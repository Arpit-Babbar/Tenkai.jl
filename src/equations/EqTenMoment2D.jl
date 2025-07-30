module EqTenMoment2D

using TimerOutputs
using StaticArrays
using LinearAlgebra
using SimpleUnPack
using Plots
using Printf
using JSON3
using GZip
using DelimitedFiles
using WriteVTK

using Tenkai
using Tenkai.Basis
using Tenkai: correct_variable!
using Tenkai: limit_variable_slope

using Tenkai.CartesianGrids: CartesianGrid2D, save_mesh_file

import Tenkai: flux, prim2con, con2prim, limit_slope, zhang_shu_flux_fix,
               apply_bound_limiter!, initialize_plot,
               write_soln!, compute_time_step, post_process_soln,
               eigmatrix

using Tenkai: eachvariable, PlotData, get_filename

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The conservative variables are
# ρ, ρv1, ρv2, E11, E12, E22

struct TenMoment2D <: AbstractEquations{2, 6}
    varnames::Vector{String}
    name::String
end

# The primitive variables are
# ρ, v1, v2, P11, P12, P22.
# Tensor P is defined so that tensor E = 0.5 * (P + ρ*v⊗v)

@inline density(::TenMoment2D, u) = u[1]

@inline energy11(::TenMoment2D, u) = u[4]
@inline energy12(::TenMoment2D, u) = u[5]
@inline energy22(::TenMoment2D, u) = u[6]

@inline energy_components(::TenMoment2D, u) = u[4], u[5], u[6]

@inline velocity(::TenMoment2D, u) = u[2] / u[1], u[3] / u[1]

@inline function pressure_components(eq::TenMoment2D, u)
    ρ = density(eq, u)
    v1, v2 = velocity(eq, u)
    E11, E12, E22 = energy_components(eq, u)
    P11 = 2.0 * E11 - ρ * v1 * v2
    P12 = 2.0 * E12 - ρ * v1 * v2
    P22 = 2.0 * E22 - ρ * v2 * v2
    return P11, P12, P22
end

@inline @inbounds function con2prim(eq::TenMoment2D, u)
    ρ = u[1]
    v1 = u[2] / ρ
    v2 = u[3] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    P12 = 2.0 * u[5] - ρ * v1 * v2
    P22 = 2.0 * u[6] - ρ * v2 * v2
    return SVector(ρ, v1, v2, P11, P12, P22)
end

@inbounds @inline function tenmom_prim2con(prim)
    ρ, v1, v2, P11, P12, P22 = prim

    ρv1 = ρ * v1
    ρv2 = ρ * v2
    E11 = 0.5 * (P11 + ρ * v1 * v1)
    E12 = 0.5 * (P12 + ρ * v1 * v2)
    E22 = 0.5 * (P22 + ρ * v2 * v2)

    return SVector(ρ, ρv1, ρv2, E11, E12, E22)
end

@inbounds @inline prim2con(eq::TenMoment2D, prim) = tenmom_prim2con(prim)

# The flux is given by
# (ρ v1, P11 + ρ v1^2, P12 + ρ v1*v2, (E + P) ⊗ v)
@inbounds @inline function flux(x, y, u, eq::TenMoment2D, orientation)
    r, v1, v2, P11, P12, P22 = con2prim(eq, u)

    if orientation == 1
        f1 = r * v1
        f2 = P11 + r * v1 * v1
        f3 = P12 + r * v1 * v2
        f4 = (u[4] + P11) * v1
        f5 = u[5] * v1 + 0.5 * (P11 * v2 + P12 * v1)
        f6 = u[6] * v1 + P12 * v2
        return SVector(f1, f2, f3, f4, f5, f6)
    else
        g1 = r * v2
        g2 = P12 + r * v1 * v2
        g3 = P22 + r * v2 * v2
        g4 = u[4] * v2 + P12 * v1
        g5 = u[5] * v2 + 0.5 * (P12 * v2 + P22 * v1)
        g6 = (u[6] + P22) * v2
        return SVector(g1, g2, g3, g4, g5, g6)
    end
end

@inbounds @inline function flux(x, y, u, eq::TenMoment2D)
    r, v1, v2, P11, P12, P22 = con2prim(eq, u)

    f1 = r * v1
    f2 = P11 + r * v1 * v1
    f3 = P12 + r * v1 * v2
    f4 = (u[4] + P11) * v1
    f5 = u[5] * v1 + 0.5 * (P11 * v2 + P12 * v1)
    f6 = u[6] * v1 + P12 * v2
    f = SVector(f1, f2, f3, f4, f5, f6)

    g1 = r * v2
    g2 = P12 + r * v1 * v2
    g3 = P22 + r * v2 * v2
    g4 = u[4] * v2 + P12 * v1
    g5 = u[5] * v2 + 0.5 * (P12 * v2 + P22 * v1)
    g6 = (u[6] + P22) * v2
    g = SVector(g1, g2, g3, g4, g5, g6)

    return f, g
end

@inbounds @inline function hll_speeds_min_max(eq::TenMoment2D, ul, ur, dir)
    # Get conservative variables
    rl, v1l, v2l, P11l, P12l, P22l = con2prim(eq, ul)
    rr, v1r, v2r, P11r, P12r, P22r = con2prim(eq, ur)

    if dir == 1 # x-direction
        vel_L = v1l
        vel_R = v1r
    elseif dir == 2 # y-direction
        vel_L = v2l
        vel_R = v2r
    end

    # TODO - Should this be T22? Because that is maximum eigenvalue?
    # Maybe it is the normal direction.
    T11l = P11l / rl
    cl1 = cl2 = sqrt(3.0 * T11l)
    # cl2 = P12l / sqrt(rl * P22l) # TODO - Is this also for 2-D
    cl = max(cl1, cl2)
    min_l = vel_L - cl
    max_l = vel_L + cl

    T11r = P11r / rr
    cr1 = cr2 = sqrt(3.0 * T11r)
    # cr2 = P12r / sqrt(rr * P22r) # TODO - Is this also for 2-D?
    cr = max(cr1, cr2)
    min_r = vel_R - cr
    max_r = vel_R + cr

    sl = min(min_l, min_r)
    sr = max(max_l, max_r)

    # Average state
    # TODO - Roe average? It is not in the SSW code, but is in Trixi.jl, EqEuler2D.jl
    r_avg = 0.5(rl + rr)
    vel_avg = 0.5 * (vel_L + vel_R)
    T11_avg = 0.5 * (T11l + T11r)
    P12_avg = 0.5 * (P12l + P12r)
    P22_avg = 0.5 * (P22l + P22r)
    c1_avg = c2_avg = sqrt(3.0 * T11_avg)
    # c2_avg = P12_avg / sqrt(r_avg * P22_avg) # TODO - Is this also for 2-D?
    c_avg = max(c1_avg, c2_avg)
    min_avg = vel_avg - c_avg
    max_avg = vel_avg + c_avg

    sl = min(sl, min_avg)
    sr = max(sr, max_avg)

    return sl, sr
end

@inbounds @inline function hll(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment2D, dir,
                               wave_speeds::Function)
    sl, sr = wave_speeds(eq, ual, uar, dir)
    if sl > 0.0
        return Fl
    elseif sr < 0.0
        return Fr
    else
        return (sr * Fl - sl * Fr + sl * sr * (Ur - Ul)) / (sr - sl)
    end
end

@inbounds @inline hll(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment2D, dir) = hll(x, ual,
                                                                               uar, Fl,
                                                                               Fr, Ul,
                                                                               Ur,
                                                                               eq::TenMoment2D,
                                                                               dir,
                                                                               hll_speeds_min_max)

@inbounds @inline function hllc3(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment2D, dir,
                                 wave_speeds::Function)
    sl, sr = wave_speeds(eq, ual, uar, dir)
    # Supersonic cases
    # Fl, Fr = flux(x, ual, eq), flux(x, uar, eq) # TEMPORARY (hopefully)
    # Ul, Ur = ual, uar # TEMPORARY (hopefully)
    if sl > 0.0
        return Fl
    elseif sr < 0.0
        return Fr
    end
    ml = Fl - sl * Ul
    mr = Fr - sr * Ur
    v1s = (mr[2] - ml[2]) / (mr[1] - ml[1])
    v2s = (mr[3] - ml[3]) / (mr[1] - ml[1])
    P11s = (ml[2] * mr[1] - ml[1] * mr[2]) / (mr[1] - ml[1]) # TODO - Can use a simpler expression
    P12s = (ml[3] * mr[1] - ml[1] * mr[3]) / (mr[1] - ml[1])
    dsl, dsr = v1s - sl, v1s - sr
    @assert dsl > 0.0&&dsr < 0.0 "Middle contact v1s=$v1s outside [sl, sr]=[$sl,$sr]"
    if v1s > 0.0
        rhosl = ml[1] / dsl
        E11sl = (ml[4] - v1s * P11s) / dsl
        E12sl = (ml[5] - 0.5 * (P11s * v2s + P12s * v1s)) / dsl
        E22sl = (ml[6] - P12s * v2s) / dsl
        if dir == 1
            Usl2 = rhosl * v1s
            Usl3 = rhosl * v2s
        else
            Usl2 = rhosl * v2s
            Usl3 = rhosl * v1s
        end
        Usl = SVector(rhosl, Usl2, Usl3, E11sl, E12sl, E22sl)
        return Fl + sl * (Usl - Ul)
    else
        rhosr = mr[1] / dsr
        E11sr = (mr[4] - v1s * P11s) / dsr
        E12sr = (mr[5] - 0.5 * (P11s * v2s + P12s * v1s)) / dsr
        E22sr = (mr[6] - P12s * v2s) / dsr
        if dir == 1
            Usr2 = rhosr * v1s
            Usr3 = rhosr * v2s
        else
            Usr2 = rhosr * v2s
            Usr3 = rhosr * v1s
        end
        Usr = SVector(rhosr, Usr2, Usr3, E11sr, E12sr, E22sr)
        return Fr + sr * (Usr - Ur)
    end
end

@inbounds @inline hllc3(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment2D, dir) = hllc3(x,
                                                                                   ual,
                                                                                   uar,
                                                                                   Fl,
                                                                                   Fr,
                                                                                   Ul,
                                                                                   Ur,
                                                                                   eq::TenMoment2D,
                                                                                   dir,
                                                                                   hll_speeds_min_max)

@inbounds @inline function max_abs_eigen_value(eq::TenMoment2D, u, dir)
    ρ = u[1]
    if dir == 1
        v1 = u[2] / ρ
        P11 = 2.0 * u[4] - ρ * v1 * v1
        T11 = P11 / ρ
        return abs(v1) + sqrt(3.0 * T11)
    else
        v2 = u[3] / ρ
        P22 = 2.0 * u[6] - ρ * v2 * v2
        T22 = P22 / ρ
        return abs(v2) + sqrt(3.0 * T22)
    end
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment2D, dir)
    λ = max(max_abs_eigen_value(eq, ual, dir), max_abs_eigen_value(eq, uar, dir)) # local wave speed
    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function compute_time_step(eq::TenMoment2D, problem, grid, aux, op, cfl, u1, ua)
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
    end

    dt = cfl / den

    return dt
    end # timer
end

#-------------------------------------------------------------------------------
# Limiters
#-------------------------------------------------------------------------------
# Admissibility constraints
@inline @inbounds density_constraint(eq::TenMoment2D, u) = u[1]
@inline @inbounds function trace_constraint(eq::TenMoment2D, u)
    ρ = u[1]
    v1 = u[2] / ρ
    v2 = u[3] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    P22 = 2.0 * u[6] - ρ * v2 * v2
    return P11 + P22
end

@inline @inbounds function det_constraint(eq::TenMoment2D, u)
    ρ = u[1]
    v1 = u[2] / ρ
    v2 = u[3] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    P12 = 2.0 * u[5] - ρ * v1 * v2
    P22 = 2.0 * u[6] - ρ * v2 * v2
    return P11 * P22 - P12 * P12
end

function Tenkai.is_admissible(eq::TenMoment2D, u::AbstractVector)
    return (density_constraint(eq, u) > 0.0
            && trace_constraint(eq, u) > 0.0
            && det_constraint(eq, u) > 0.0)
end

# Looping over a tuple can be made type stable following this
# https://github.com/trixi-framework/Trixi.jl/blob/0fd86e4bd856d894de6a7514edcb9758bf6f8e1e/src/callbacks_stage/positivity_zhang_shu.jl#L39
function iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                          u1, aux, variables::NTuple{N, Any}) where {N}
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    correct_variable!(eq, variable, op, aux, grid, u1, ua)
    iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                     u1, aux, remaining_variables)
    return nothing
end

function iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                          u1, aux, variables::Tuple{})
    return nothing
end

function Tenkai.apply_bound_limiter!(eq::TenMoment2D, grid, scheme, param, op, ua,
                                     u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end
    variables = (density_constraint, trace_constraint, det_constraint)
    iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua, u1, aux,
                                     variables)
    return nothing
end

function eigmatrix(eq::TenMoment2D, u)
    nvar = nvariables(eq)

    rho = density(eq, u)
    E11, E12, E22 = energy_components(eq, u)
    v1, v2 = velocity(eq, u)
    p11 = 2.0 * E11 - rho * v1 * v1
    p12 = 2.0 * E12 - rho * v1 * v2
    p22 = 2.0 * E22 - rho * v2 * v2
    cx = sqrt(3.0 * abs(p11 / rho))
    cy = sqrt(3.0 * abs(p22 / rho))

    # Right eigenvectors

    Rx00 = rho * p11
    Rx10 = rho * v1 * p11 - cx * rho * p11
    Rx20 = rho * v2 * p11 - cx * rho * p12
    Rx30 = (rho * v1 * v1 * p11) / 2.0 - cx * rho * v1 * p11 + (3.0 * p11 * p11) / 2.0
    Rx40 = (rho * v1 * v2 * p11) / 2.0 - (cx * rho * v2 * p11) / 2.0 -
           (cx * rho * v1 * p12) / 2.0 + (3.0 * p11 * p12) / 2.0
    Rx50 = (rho * v2 * v2 * p11) / 2.0 - p12 * cx * rho * v2 + (p11 * p22) / 2.0 +
           p12 * p12

    Rx01 = 0.0
    Rx11 = 0.0
    Rx21 = -cx * rho / sqrt(3.0)
    Rx31 = 0.0
    Rx41 = (-cx * rho * v1) / (2.0 * sqrt(3.0)) + p11 / 2.0
    Rx51 = (-cx * rho * v2) / sqrt(3.0) + p12

    Rx02 = 1.0
    Rx12 = v1
    Rx22 = v2
    Rx32 = v1 * v1 / 2.0
    Rx42 = v1 * v2 / 2.0
    Rx52 = v2 * v2 / 2.0

    Rx03 = 0.0
    Rx13 = 0.0
    Rx23 = 0.0
    Rx33 = 0.0
    Rx43 = 0.0
    Rx53 = 0.5

    Rx04 = 0.0
    Rx14 = 0.0
    Rx24 = (cx * rho) / sqrt(3.0)
    Rx34 = 0.0
    Rx44 = (cx * rho * v1) / (2.0 * sqrt(3.0)) + p11 / 2.0
    Rx54 = (cx * rho * v2) / sqrt(3.0) + p12

    Rx05 = rho * p11
    Rx15 = rho * v1 * p11 + cx * rho * p11
    Rx25 = rho * v2 * p11 + cx * rho * p12
    Rx35 = (rho * v1 * v1 * p11) / 2.0 + cx * rho * v1 * p11 + (3.0 * p11 * p11) / 2.0
    Rx45 = (rho * v1 * v2 * p11) / 2.0 + (cx * rho * v2 * p11) / 2.0 +
           (cx * rho * v1 * p12) / 2.0 + (3.0 * p11 * p12) / 2.0
    Rx55 = (rho * v2 * v2 * p11) / 2.0 + p12 * cx * rho * v2 + (p11 * p22) / 2.0 +
           p12 * p12

    Rx = SMatrix{nvar, nvar}(Rx00, Rx01, Rx02, Rx03, Rx04, Rx05,
                             Rx10, Rx11, Rx12, Rx13, Rx14, Rx15,
                             Rx20, Rx21, Rx22, Rx23, Rx24, Rx25,
                             Rx30, Rx31, Rx32, Rx33, Rx34, Rx35,
                             Rx40, Rx41, Rx42, Rx43, Rx44, Rx45,
                             Rx50, Rx51, Rx52, Rx53, Rx54, Rx55)

    Ry00 = rho * p22
    Ry10 = rho * v1 * p22 - cy * rho * p12
    Ry20 = rho * v2 * p22 - cy * rho * p22
    Ry30 = (rho * v1 * v1 * p22) / 2.0 - cy * rho * v1 * p12 + (p11 * p22) / 2.0 +
           p12 * p12
    Ry40 = (rho * v1 * v2 * p22) / 2.0 - (cy * rho * v1 * p22) / 2.0 -
           (cy * rho * v2 * p12) / 2.0 + (3.0 * p22 * p12) / 2.0
    Ry50 = (rho * v2 * v2 * p22) / 2.0 - p22 * cy * rho * v2 + (3 * p22 * p22) / 2.0

    Ry01 = 0
    Ry11 = -cy * rho / sqrt(3.0)
    Ry21 = 0
    Ry31 = (-cy * rho * v1) / sqrt(3.0) + p12
    Ry41 = (-cy * rho * v2) / (2.0 * sqrt(3.0)) + p22 / 2.0
    Ry51 = 0

    Ry02 = 1.0
    Ry12 = v1
    Ry22 = v2
    Ry32 = v1 * v1 / 2.0
    Ry42 = v1 * v2 / 2.0
    Ry52 = v2 * v2 / 2.0

    Ry03 = 0.0
    Ry13 = 0.0
    Ry23 = 0.0
    Ry33 = 0.5
    Ry43 = 0.0
    Ry53 = 0.0

    Ry04 = 0
    Ry14 = (cy * rho) / sqrt(3.0)
    Ry24 = 0
    Ry34 = (cy * rho * v1) / sqrt(3.0) + p12
    Ry44 = (cy * rho * v2) / (2.0 * sqrt(3.0)) + p22 / 2.0
    Ry54 = 0

    Ry05 = rho * p22
    Ry15 = rho * v1 * p22 + cy * rho * p12
    Ry25 = rho * v2 * p22 + cy * rho * p22
    Ry35 = (rho * v1 * v1 * p22) / 2.0 + cy * rho * v1 * p12 + (p11 * p22) / 2.0 +
           p12 * p12
    Ry45 = (rho * v1 * v2 * p22) / 2.0 + (cy * rho * v1 * p22) / 2.0 +
           (cy * rho * v2 * p12) / 2.0 + (3.0 * p12 * p22) / 2.0
    Ry55 = (rho * v2 * v2 * p22) / 2.0 + p22 * cy * rho * v2 + (3 * p22 * p22) / 2.0

    Ry = SMatrix{nvar, nvar}(Ry00, Ry01, Ry02, Ry03, Ry04, Ry05,
                             Ry10, Ry11, Ry12, Ry13, Ry14, Ry15,
                             Ry20, Ry21, Ry22, Ry23, Ry24, Ry25,
                             Ry30, Ry31, Ry32, Ry33, Ry34, Ry35,
                             Ry40, Ry41, Ry42, Ry43, Ry44, Ry45,
                             Ry50, Ry51, Ry52, Ry53, Ry54, Ry55)

    # Left eigenvectors

    Lx00 = (-3.0 * v1 * p11 / rho - cx * v1 * v1) / (-6.0 * cx * p11 * p11)
    Lx10 = (v1 * p11 * p12 / rho - v2 * p11 * p11 / rho +
            cx * v1 * v1 * p12 / sqrt(3.0) - cx * v1 * v2 * p11 / sqrt(3.0)) /
           (-2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx20 = (3.0 * p11 - rho * v1 * v1) / (3.0 * p11)
    Lx30 = (4.0 * v1 * v1 * p12 * p12 - v1 * v1 * p11 * p22 -
            6.0 * v1 * v2 * p11 * p12 + 3.0 * v2 * v2 * p11 * p11) /
           (3.0 * p11 * p11)
    Lx40 = (v1 * p11 * p12 / rho - v2 * p11 * p11 / rho -
            cx * v1 * v1 * p12 / sqrt(3.0) + cx * v1 * v2 * p11 / sqrt(3.0)) /
           (2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx50 = (-3.0 * v1 * p11 / rho + cx * v1 * v1) / (6.0 * cx * p11 * p11)

    Lx01 = (3.0 * p11 / rho + 2.0 * cx * v1) / (-6.0 * cx * p11 * p11)
    Lx11 = (-p11 * p12 / rho - 2.0 * v1 * p12 * cx / sqrt(3.0) +
            cx * v2 * p11 / sqrt(3.0)) /
           (-2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx21 = 2.0 * v1 * rho / (3.0 * p11)
    Lx31 = (-8.0 * v1 * p12 * p12 + 2.0 * v1 * p11 * p22 + 6.0 * v2 * p11 * p12) /
           (3.0 * p11 * p11)
    Lx41 = (-p11 * p12 / rho + 2.0 * v1 * p12 * cx / sqrt(3.0) -
            cx * v2 * p11 / sqrt(3.0)) /
           (2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx51 = (3.0 * p11 / rho - 2.0 * cx * v1) / (6.0 * cx * p11 * p11)

    Lx02 = 0.0
    Lx12 = (p11 * p11 / rho + v1 * cx * p11 / sqrt(3.0)) /
           (-2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx22 = 0.0
    Lx32 = (6.0 * v1 * p11 * p12 - 6.0 * v2 * p11 * p11) / (3.0 * p11 * p11)
    Lx42 = (p11 * p11 / rho - v1 * cx * p11 / sqrt(3.0)) /
           (2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx52 = 0.0

    Lx03 = -2.0 * cx / (-6.0 * cx * p11 * p11)
    Lx13 = (2 * cx * p12 / sqrt(3.0)) / (-2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx23 = -2.0 * rho / (3 * p11)
    Lx33 = (8.0 * p12 * p12 - 2.0 * p11 * p22) / (3.0 * p11 * p11)
    Lx43 = (-2 * cx * p12 / sqrt(3.0)) / (2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx53 = 2.0 * cx / (6.0 * cx * p11 * p11)

    Lx04 = 0.0
    Lx14 = (-2 * cx * p11 / sqrt(3.0)) / (-2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx24 = 0.0
    Lx34 = -12.0 * p11 * p12 / (3.0 * p11 * p11)
    Lx44 = (2 * cx * p11 / sqrt(3.0)) / (2.0 * cx * p11 * p11 / sqrt(3.0))
    Lx54 = 0.0

    Lx05 = 0.0
    Lx15 = 0.0
    Lx25 = 0.0
    Lx35 = 2.0
    Lx45 = 0.0
    Lx55 = 0.0

    Lx = SMatrix{nvar, nvar}(Lx00, Lx01, Lx02, Lx03, Lx04, Lx05,
                             Lx10, Lx11, Lx12, Lx13, Lx14, Lx15,
                             Lx20, Lx21, Lx22, Lx23, Lx24, Lx25,
                             Lx30, Lx31, Lx32, Lx33, Lx34, Lx35,
                             Lx40, Lx41, Lx42, Lx43, Lx44, Lx45,
                             Lx50, Lx51, Lx52, Lx53, Lx54, Lx55)

    Ly00 = v2 / (2.0 * cy * p22 * rho) + v2 * v2 / (6.0 * p22 * p22)
    Ly01 = 0
    Ly02 = -1.0 / (2.0 * p22 * cy * rho) - v2 / (3.0 * p22 * p22)
    Ly03 = 0
    Ly04 = 0
    Ly05 = 1.0 / (3.0 * p22 * p22)

    Ly10 = sqrt(3.0) * v1 / (2.0 * cy * rho) -
           sqrt(3.0) * p12 * v2 / (2.0 * cy * p22 * rho) +
           v1 * v2 / (2.0 * p22) - p12 * v2 * v2 / (2.0 * p22 * p22)
    Ly11 = -sqrt(3.0) / (2.0 * cy * rho) - v2 / (2.0 * p22)
    Ly12 = sqrt(3.0) * p12 / (2.0 * cy * p22 * rho) - v1 / (2.0 * p22) +
           p12 * v2 / (p22 * p22)
    Ly13 = 0
    Ly14 = 1.0 / p22
    Ly15 = -p12 / (p22 * p22)

    Ly20 = 1.0 - rho * v2 * v2 / (3.0 * p22)
    Ly21 = 0
    Ly22 = 2.0 * rho * v2 / (3.0 * p22)
    Ly23 = 0
    Ly24 = 0
    Ly25 = -2.0 * rho / (3.0 * p22)

    Ly30 = v1 * v1 - 2.0 * p12 * v1 * v2 / p22 +
           4.0 * p12 * p12 * v2 * v2 / (3.0 * p22 * p22) - p11 * v2 * v2 / (3.0 * p22)
    Ly31 = -2.0 * v1 + 2.0 * p12 * v2 / p22
    Ly32 = 2.0 * p12 * v1 / p22 - 8.0 * p12 * p12 * v2 / (3.0 * p22 * p22) +
           2.0 * p11 * v2 / (3.0 * p22)
    Ly33 = 2.0
    Ly34 = -4.0 * p12 / p22
    Ly35 = 8.0 * p12 * p12 / (3.0 * p22 * p22) - 2.0 * p11 / (3.0 * p22)

    Ly40 = -sqrt(3.0) * v1 / (2.0 * cy * rho) +
           sqrt(3.0) * p12 * v2 / (2.0 * cy * p22 * rho) +
           v1 * v2 / (2.0 * p22) - p12 * v2 * v2 / (2.0 * p22 * p22)
    Ly41 = sqrt(3.0) / (2.0 * cy * rho) - v2 / (2.0 * p22)
    Ly42 = -sqrt(3.0) * p12 / (2.0 * cy * p22 * rho) - v1 / (2.0 * p22) +
           p12 * v2 / (p22 * p22)
    Ly43 = 0
    Ly44 = 1.0 / p22
    Ly45 = -p12 / (p22 * p22)

    Ly50 = -v2 / (2.0 * cy * p22 * rho) + v2 * v2 / (6.0 * p22 * p22)
    Ly51 = 0
    Ly52 = 1.0 / (2.0 * cy * p22 * rho) - v2 / (3.0 * p22 * p22)
    Ly53 = 0
    Ly54 = 0
    Ly55 = 1.0 / (3.0 * p22 * p22)

    Ly = SMatrix{nvar, nvar}(Ly00, Ly01, Ly02, Ly03, Ly04, Ly05,
                             Ly10, Ly11, Ly12, Ly13, Ly14, Ly15,
                             Ly20, Ly21, Ly22, Ly23, Ly24, Ly25,
                             Ly30, Ly31, Ly32, Ly33, Ly34, Ly35,
                             Ly40, Ly41, Ly42, Ly43, Ly44, Ly45,
                             Ly50, Ly51, Ly52, Ly53, Ly54, Ly55)

    return Lx, Ly, Rx, Ry
    # return Id, Id, Id, Id
end

function minmod_β(a, b)
    s1, s2 = sign(a), sign(b)
    if (s1 != s2)
        return 0.0
    else
        slope = s1 * min(abs(a), abs(b))
        return slope
    end
end

function minmod_β(a, b, c, Mdx2)
    if abs(a) < Mdx2
        return a
    end
    s1, s2, s3 = sign(a), sign(b), sign(c)
    if (s1 != s2) || (s2 != s3)
        return 0.0
    else
        # slope = s1 * slope
        # return slope
        slope = s1 * min(abs(a), abs(b), abs(c))
        return slope
    end
end

function Tenkai.apply_tvb_limiterβ!(eq::TenMoment2D, problem, scheme, grid, param,
                                    op, ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack dx, dy = grid
    @unpack tvbM, cache, beta = scheme.limiter
    nvar = nvariables(eq)
    nd = length(wg)

    refresh!(u) = fill!(u, zero(eltype(u)))
    # Pre-allocate for each thread

    # Loop over cells
    @threaded for ij in CartesianIndices((1:nx, 1:ny))
        id = Threads.threadid()
        el_x, el_y = ij[1], ij[2]
        # face averages
        (ul, ur, ud, uu,
        dux, dur, duy, duu,
        dual, duar, duad, duau,
        char_dux, char_dur, char_duy, char_duu,
        char_dual, char_duar, char_duad, char_duau,
        duxm, durm, duym, duum) = cache[id]
        u1_ = @view u1[:, :, :, el_x, el_y]
        ua_, ual, uar, uad, uau = (get_node_vars(ua, eq, el_x, el_y),
                                   get_node_vars(ua, eq, el_x - 1, el_y),
                                   get_node_vars(ua, eq, el_x + 1, el_y),
                                   get_node_vars(ua, eq, el_x, el_y - 1),
                                   get_node_vars(ua, eq, el_x, el_y + 1))
        Lx, Ly, Rx, Ry = eigmatrix(eq, ua_)
        refresh!.((ul, ur, ud, uu))
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_ = get_node_vars(u1_, eq, i, j)
            multiply_add_to_node_vars!(ul, Vl[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ur, Vr[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ud, Vl[j] * wg[i], u_, eq, 1)
            multiply_add_to_node_vars!(uu, Vr[j] * wg[i], u_, eq, 1)
        end
        # KLUDGE - Give better names to these quantities
        # slopes b/w centres and faces
        ul_, ur_ = get_node_vars(ul, eq, 1), get_node_vars(ur, eq, 1)
        ud_, uu_ = get_node_vars(ud, eq, 1), get_node_vars(uu, eq, 1)
        ual_, uar_ = get_node_vars(ual, eq, 1), get_node_vars(uar, eq, 1)
        uad_, uau_ = get_node_vars(uad, eq, 1), get_node_vars(uau, eq, 1)

        multiply_add_set_node_vars!(dux, 1.0, ur_, -1.0, ul_, eq, 1)
        multiply_add_set_node_vars!(duy, 1.0, uu_, -1.0, ud_, eq, 1)

        multiply_add_set_node_vars!(dual, 1.0, ua_, -1.0, ual_, eq, 1)
        multiply_add_set_node_vars!(duar, 1.0, uar_, -1.0, ua_, eq, 1)
        multiply_add_set_node_vars!(duad, 1.0, ua_, -1.0, uad_, eq, 1)
        multiply_add_set_node_vars!(duau, 1.0, uau_, -1.0, ua_, eq, 1)

        dux_ = get_node_vars(dux, eq, 1)
        dual_, duar_ = get_node_vars(dual, eq, 1), get_node_vars(duar, eq, 1)
        duy_ = get_node_vars(duy, eq, 1)
        duad_, duau_ = get_node_vars(duad, eq, 1), get_node_vars(duau, eq, 1)

        # Convert to characteristic variables
        # mul!(char_dux, Lx, dux_)
        # mul!(char_dual, Lx, dual_)
        # mul!(char_duar, Lx, duar_)
        # mul!(char_duy, Ly, duy_)
        # mul!(char_duad, Ly, duad_)
        # mul!(char_duau, Ly, duau_)

        # Convert to primitive variables
        # set_node_vars!(char_dux, con2prim(eq, dux_), eq, 1)
        # set_node_vars!(char_dual, con2prim(eq, dual_), eq, 1)
        # set_node_vars!(char_duar, con2prim(eq, duar_), eq, 1)
        # set_node_vars!(char_duy, con2prim(eq, duy_), eq, 1)
        # set_node_vars!(char_duad, con2prim(eq, duad_), eq, 1)
        # set_node_vars!(char_duau, con2prim(eq, duau_), eq, 1)

        # Keep variables as they are
        set_node_vars!(char_dux, dux_, eq, 1)
        set_node_vars!(char_dual, dual_, eq, 1)
        set_node_vars!(char_duar, duar_, eq, 1)
        set_node_vars!(char_duy, duy_, eq, 1)
        set_node_vars!(char_duad, duad_, eq, 1)
        set_node_vars!(char_duau, duau_, eq, 1)

        Mdx2, Mdy2 = tvbM * dx[el_x]^2, tvbM * dy[el_y]^2
        for n in Base.OneTo(nvar)
            duxm[n] = minmod_β(char_dux[n], beta * char_dual[n],
                               beta * char_duar[n], Mdx2)
            duym[n] = minmod_β(char_duy[n], beta * char_duad[n],
                               beta * char_duau[n], Mdy2)
        end

        jump_x = jump_y = 0.0
        duxm_ = get_node_vars(duxm, eq, 1)
        duym_ = get_node_vars(duym, eq, 1)
        for n in 1:nvar
            jump_x += abs(char_dux[n] - duxm_[n])
            jump_y += abs(char_duy[n] - duym_[n])
        end
        jump_x /= nvar
        jump_y /= nvar
        if jump_x + jump_y > 1e-10
            # @show grid.xc[el_x], grid.yc[el_y], jump_x, jump_y

            # From characteristic to conservative
            # mul!(dux, Rx, duxm_)
            # mul!(duy, Ry, duym_)

            # From primitive to conservative
            # set_node_vars!(dux, prim2con(eq, duxm_), eq, 1)
            # set_node_vars!(duy, prim2con(eq, duym_), eq, 1)

            # Conservative to conservative
            set_node_vars!(dux, duxm_, eq, 1)
            set_node_vars!(duy, duym_, eq, 1)

            # dux_ = duy_ = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            dux_, duy_ = get_node_vars(dux, eq, 1), get_node_vars(duy, eq, 1)
            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                multiply_add_set_node_vars!(u1_,
                                            1.0, ua_,
                                            xg[i] - 0.5,
                                            dux_,
                                            xg[j] - 0.5,
                                            duy_,
                                            eq, i, j)
            end
        end
    end
    return nothing
    end # timer
end

function Tenkai.apply_tvb_limiter!(eq::TenMoment2D, problem, scheme, grid, param, op,
                                   ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack dx, dy = grid
    @unpack tvbM, cache, beta = scheme.limiter
    nvar = nvariables(eq)
    nd = length(wg)

    refresh!(u) = fill!(u, zero(eltype(u)))
    # Pre-allocate for each thread

    # Loop over cells
    @threaded for ij in CartesianIndices((1:nx, 1:ny))
        id = Threads.threadid()
        el_x, el_y = ij[1], ij[2]
        # face averages
        (ul, ur, ud, uu,
        dul, dur, dud, duu,
        dual, duar, duad, duau,
        char_dul, char_dur, char_dud, char_duu,
        char_dual, char_duar, char_duad, char_duau,
        dulm, durm, dudm, duum,
        duxm, duym, dux, duy) = cache[id]
        u1_ = @view u1[:, :, :, el_x, el_y]
        ua_, ual, uar, uad, uau = (get_node_vars(ua, eq, el_x, el_y),
                                   get_node_vars(ua, eq, el_x - 1, el_y),
                                   get_node_vars(ua, eq, el_x + 1, el_y),
                                   get_node_vars(ua, eq, el_x, el_y - 1),
                                   get_node_vars(ua, eq, el_x, el_y + 1))
        Lx, Ly, Rx, Ry = eigmatrix(eq, ua_)
        refresh!.((ul, ur, ud, uu))
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_ = get_node_vars(u1_, eq, i, j)
            multiply_add_to_node_vars!(ul, Vl[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ur, Vr[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ud, Vl[j] * wg[i], u_, eq, 1)
            multiply_add_to_node_vars!(uu, Vr[j] * wg[i], u_, eq, 1)
        end
        # KLUDGE - Give better names to these quantities
        # slopes b/w centres and faces
        ul_, ur_ = get_node_vars(ul, eq, 1), get_node_vars(ur, eq, 1)
        ud_, uu_ = get_node_vars(ud, eq, 1), get_node_vars(uu, eq, 1)
        ual_, uar_ = get_node_vars(ual, eq, 1), get_node_vars(uar, eq, 1)
        uad_, uau_ = get_node_vars(uad, eq, 1), get_node_vars(uau, eq, 1)

        multiply_add_set_node_vars!(dul, 1.0, ua_, -1.0, ul_, eq, 1)
        multiply_add_set_node_vars!(dur, 1.0, ur_, -1.0, ua_, eq, 1)
        multiply_add_set_node_vars!(dud, 1.0, ua_, -1.0, ud_, eq, 1)
        multiply_add_set_node_vars!(duu, 1.0, uu_, -1.0, ua_, eq, 1)

        multiply_add_set_node_vars!(dual, 1.0, ua_, -1.0, ual_, eq, 1)
        multiply_add_set_node_vars!(duar, 1.0, uar_, -1.0, ua_, eq, 1)
        multiply_add_set_node_vars!(duad, 1.0, ua_, -1.0, uad_, eq, 1)
        multiply_add_set_node_vars!(duau, 1.0, uau_, -1.0, ua_, eq, 1)

        dul_, dur_ = get_node_vars(dul, eq, 1), get_node_vars(dur, eq, 1)
        dual_, duar_ = get_node_vars(dual, eq, 1), get_node_vars(duar, eq, 1)
        dud_, duu_ = get_node_vars(dud, eq, 1), get_node_vars(duu, eq, 1)
        duad_, duau_ = get_node_vars(duad, eq, 1), get_node_vars(duau, eq, 1)

        # Convert to characteristic variables
        mul!(char_dul, Lx, dul_)
        mul!(char_dur, Lx, dur_)
        mul!(char_dual, Lx, dual_)
        mul!(char_duar, Lx, duar_)
        mul!(char_dud, Ly, dud_)
        mul!(char_duu, Ly, duu_)
        mul!(char_duad, Ly, duad_)
        mul!(char_duau, Ly, duau_)
        Mdx2, Mdy2 = tvbM * dx[el_x]^2, tvbM * dy[el_y]^2
        for n in Base.OneTo(nvar)
            dulm[n] = minmod(char_dul[n], char_dual[n], char_duar[n], Mdx2)
            durm[n] = minmod(char_dur[n], char_dual[n], char_duar[n], Mdx2)
            dudm[n] = minmod(char_dud[n], char_duad[n], char_duau[n], Mdy2)
            duum[n] = minmod(char_duu[n], char_duad[n], char_duau[n], Mdy2)
        end

        jump_x = jump_y = 0.0
        dulm_, durm_ = get_node_vars(dulm, eq, 1), get_node_vars(durm, eq, 1)
        dudm_, duum_ = get_node_vars(dudm, eq, 1), get_node_vars(duum, eq, 1)
        for n in 1:nvar
            jump_x += 0.5 *
                      (abs(char_dul[n] - dulm_[n]) + abs(char_dur[n] - durm_[n]))
            jump_y += 0.5 *
                      (abs(char_dud[n] - dudm_[n]) + abs(char_duu[n] - duum_[n]))
        end
        jump_x /= nvar
        jump_y /= nvar
        if jump_x + jump_y > 1e-10
            dulm_, durm_, dudm_, duum_ = (get_node_vars(dulm, eq, 1),
                                          get_node_vars(durm, eq, 1),
                                          get_node_vars(dudm, eq, 1),
                                          get_node_vars(duum, eq, 1))
            multiply_add_set_node_vars!(duxm,
                                        0.5, dulm_, 0.5, durm_,
                                        eq, 1)
            multiply_add_set_node_vars!(duym,
                                        0.5, dudm_, 0.5, duum_,
                                        eq, 1)
            duxm_, duym_ = get_node_vars(duxm, eq, 1), get_node_vars(duym, eq, 1)
            mul!(dux, Rx, duxm_)
            mul!(duy, Ry, duym_)
            dux_, duy_ = get_node_vars(dux, eq, 1), get_node_vars(duy, eq, 1)
            dux_ = duy_ = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                multiply_add_set_node_vars!(u1_,
                                            1.0, ua_,
                                            2.0 * (xg[i] - 0.5),
                                            dux_,
                                            2.0 * (xg[j] - 0.5),
                                            duy_,
                                            eq, i, j)
            end
        end
    end
    return nothing
    end # timer
end

#-------------------------------------------------------------------------------
# Blending Limiter
#-------------------------------------------------------------------------------

@inbounds @inline function rho_p_indicator!(un, eq::TenMoment2D)
    for ix in 1:size(un, 2) # loop over dofs and faces
        u_node = get_node_vars(un, eq, ix)
        p = det_constraint(eq, u_node)
        un[1, ix] *= p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

function Tenkai.zhang_shu_flux_fix(eq::TenMoment2D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    uhigh = uprev - c * (Fn - fn_inner) # First candidate for high order update
    ρ_low, ρ_high = density_constraint(eq, ulow), density_constraint(eq, uhigh)
    eps = 0.1 * ρ_low
    ratio = abs(eps - ρ_low) / (abs(ρ_high - ρ_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Second candidate for flux
    end

    uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
    p_low, p_high = trace_constraint(eq, ulow), trace_constraint(eq, uhigh)
    eps = 0.1 * p_low
    ratio = abs(eps - p_low) / (abs(p_high - p_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Final flux
    end

    uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
    p_low, p_high = det_constraint(eq, ulow), det_constraint(eq, uhigh)
    eps = 0.1 * p_low
    ratio = abs(eps - p_low) / (abs(p_high - p_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Final flux
    end

    return Fn
end

function Tenkai.limit_slope(eq::TenMoment2D, slope, ufl, u_star_ll, ufr, u_star_rr,
                            ue, xl, xr, el_x = nothing, el_y = nothing)

    # The MUSCL-Hancock scheme is guaranteed to be admissibility preserving if
    # slope is chosen so that
    # u_star_l = ue + 2.0*slope*xl, u_star_r = ue+2.0*slope*xr are admissible
    # ue is already admissible and we know we can find sequences of thetas
    # to make theta*u_star_l+(1-theta)*ue is admissible.
    # This is equivalent to replacing u_star_l by
    # u_star_l = ue + 2.0*theta*s*xl.
    # Thus, we simply have to update the slope by multiplying by theta.

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, density_constraint, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, trace_constraint, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, det_constraint, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    ufl = ue + slope * xl
    ufr = ue + slope * xr

    return ufl, ufr, slope
end

function Tenkai.update_ghost_values_lwfr!(problem, scheme, eq::TenMoment2D,
                                          grid, aux, op, cache, t, dt,
                                          scaling_factor = 1.0)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, Ub, ua = cache
    update_ghost_values_periodic!(eq, problem, Fb, Ub)

    @unpack periodic_x, periodic_y = problem
    if periodic_x && periodic_y
        return nothing
    end

    nx, ny = grid.size
    nvar = nvariables(eq)
    @unpack degree, xg, wg = op
    nd = degree + 1
    @unpack dx, dy, xf, yf = grid
    @unpack boundary_condition, boundary_value = problem
    left, right, bottom, top = boundary_condition

    refresh!(u) = fill!(u, zero(eltype(u)))

    pre_allocated = cache.ghost_cache

    # Julia bug occuring here. Below, we have unnecessarily named
    # x1,y1, x2, y2,.... We should have been able to just call them x,y
    # Otherwise we were getting a type instability and variables were
    # called Core.Box. This issue is probably related to
    # https://discourse.julialang.org/t/type-instability-of-nested-function/57007
    # https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
    # https://github.com/JuliaLang/julia/issues/15276
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured-1

    dt_scaled = scaling_factor * dt
    wg_scaled = scaling_factor * wg

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    if left == dirichlet
        x1 = xf[1]
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                y1 = yf[j] + xg[k] * dy[j]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ub_value = problem.boundary_value(x1, y1, tq)
                    fb_value = flux(x1, y1, ub_value, eq, 1)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ub_value, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fb_value, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 2, 0, j)
                set_node_vars!(Fb, fb_node, eq, k, 2, 0, j)

                # # Put hllc flux values
                # Ul = get_node_vars(Ub, eq, k, 2, 0, j)
                # Fl = get_node_vars(Fb, eq, k, 2, 0, j)
                # Ur = get_node_vars(Ub, eq, k, 1, 1, j)
                # Fr = get_node_vars(Fb, eq, k, 1, 1, j)
                # ual, uar = get_node_vars(ua, eq, 0, j), get_node_vars(ua, eq, 1, j)
                # X = SVector{2}(x1, y1)
                # Fn = hllc(X, ual, uar, Fl, Fr, Ul, Ur, eq, 1)
                # set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
                # set_node_vars!(Fb, Fn     , eq, k, 1, 1, j)

                # Purely upwind at boundary
                # if abs(y1) < 0.055
                set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
                set_node_vars!(Fb, fb_node, eq, k, 1, 1, j)
                # end
            end
        end
    elseif left in (neumann, reflect)
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 1, 1, j)
                Fb_node = get_node_vars(Fb, eq, k, 1, 1, j)
                set_node_vars!(Ub, Ub_node, eq, k, 2, 0, j)
                set_node_vars!(Fb, Fb_node, eq, k, 2, 0, j)
                if left == reflect
                    Ub[2, k, 2, 0, j] *= -1.0 # ρ*u1
                    Fb[1, k, 2, 0, j] *= -1.0 # ρ*u1
                    Fb[3, k, 2, 0, j] *= -1.0 # ρ*u1*u2
                    Fb[4, k, 2, 0, j] *= -1.0 # (ρ_e + p) * u1
                end
            end
        end
    else
        println("Incorrect bc specified at left.")
        @assert false
    end

    if right == dirichlet
        x2 = xf[nx + 1]
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                y2 = yf[j] + xg[k] * dy[j]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x2, y2, tq)
                    fbvalue = flux(x2, y2, ubvalue, eq, 1)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 1, nx + 1, j)
                set_node_vars!(Fb, fb_node, eq, k, 1, nx + 1, j)

                # Purely upwind
                # set_node_vars!(Ub, ub_node, eq, k, 2, nx, j)
                # set_node_vars!(Fb, fb_node, eq, k, 2, nx, j)
            end
        end
    elseif right in (reflect, neumann)
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 2, nx, j)
                Fb_node = get_node_vars(Fb, eq, k, 2, nx, j)
                set_node_vars!(Ub, Ub_node, eq, k, 1, nx + 1, j)
                set_node_vars!(Fb, Fb_node, eq, k, 1, nx + 1, j)

                if right == reflect
                    Ub[2, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[1, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[3, k, 1, nx + 1, j] *= -1.0 # ρ*u1*u2
                    Fb[4, k, 1, nx + 1, j] *= -1.0 # (ρ_e + p) * u1
                end
            end
        end
    else
        println("Incorrect bc specified at right.")
        @assert false
    end

    if bottom == dirichlet
        y3 = yf[1]
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                x3 = xf[i] + xg[k] * dx[i]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x3, y3, tq)
                    fbvalue = flux(x3, y3, ubvalue, eq, 2)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 4, i, 0)
                set_node_vars!(Fb, fb_node, eq, k, 4, i, 0)

                # Purely upwind

                # set_node_vars!(Ub, ub, eq, k, 3, i, 1)
                # set_node_vars!(Fb, fb, eq, k, 3, i, 1)
            end
        end
    elseif bottom in (reflect, neumann)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 3, i, 1)
                Fb_node = get_node_vars(Fb, eq, k, 3, i, 1)
                set_node_vars!(Ub, Ub_node, eq, k, 4, i, 0)
                set_node_vars!(Fb, Fb_node, eq, k, 4, i, 0)
                if bottom == reflect
                    Ub[3, k, 4, i, 0] *= -1.0 # ρ * vel_y
                    Fb[1, k, 4, i, 0] *= -1.0 # ρ * vel_y
                    Fb[2, k, 4, i, 0] *= -1.0 # ρ * vel_x * vel_y
                    Fb[4, k, 4, i, 0] *= -1.0 # (ρ_e + p) * vel_y
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert typeof(bottom) <: Tuple{Any, Any, Any}
        bc! = bottom[1]
        bc!(grid, eq, op, Fb, Ub, aux)
    end

    if top == dirichlet
        y4 = yf[ny + 1]
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                x4 = xf[i] + xg[k] * dx[i]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x4, y4, tq)
                    fbvalue = flux(x4, y4, ubvalue, eq, 2)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 3, i, ny + 1)
                set_node_vars!(Fb, fb_node, eq, k, 3, i, ny + 1)

                # Purely upwind
                # set_node_vars!(Ub, ub_node, eq, k, 4, i, ny)
                # set_node_vars!(Fb, fb_node, eq, k, 4, i, ny)
            end
        end
    elseif top in (reflect, neumann)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 4, i, ny)
                Fb_node = get_node_vars(Fb, eq, k, 4, i, ny)
                set_node_vars!(Ub, Ub_node, eq, k, 3, i, ny + 1)
                set_node_vars!(Fb, Fb_node, eq, k, 3, i, ny + 1)
                if top == reflect
                    Ub[3, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
                    Fb[1, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
                    Fb[2, k, 3, i, ny + 1] *= -1.0 # ρ * vel_x * vel_y
                    Fb[4, k, 3, i, ny + 1] *= -1.0 # (ρ_e + p) * vel_y
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert false "Incorrect bc specific at top"
    end

    return nothing
    end # timer
end

function Tenkai.update_ghost_values_rkfr!(problem, scheme, eq::TenMoment2D, grid, aux,
                                          op, cache, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, ub, ua = cache
    update_ghost_values_periodic!(eq, problem, Fb, ub)
    end # timer
end

function Tenkai.update_ghost_values_rkfr!(problem, scheme, eq::TenMoment2D,
                                          grid, aux, op, cache, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, ub = cache
    update_ghost_values_periodic!(eq, problem, Fb, ub)

    @unpack periodic_x, periodic_y = problem
    if periodic_x && periodic_y
        return nothing
    end

    nx, ny = grid.size
    @unpack degree, xg = op
    nd = degree + 1
    nvar = nvariables(eq)
    @unpack dx, dy, xf, yf = grid
    @unpack boundary_value, boundary_condition = problem
    left, right, bottom, top = boundary_condition

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    if left == dirichlet
        @threaded for j in 1:ny
            x1 = xf[1]
            for k in Base.OneTo(nd)
                y1 = yf[j] + xg[k] * dy[j]
                ub_value = boundary_value(x1, y1, t)
                set_node_vars!(ub, ub_value, eq, k, 2, 0, j)
                fb_value = flux(x1, y1, ub_value, eq, 1)
                set_node_vars!(Fb, fb_value, eq, k, 2, 0, j)

                # Purely upwind at boundary
                # set_node_vars!(ub, ub_value, eq, k, 1, 1, j)
                # set_node_vars!(Fb, fb_value, eq, k, 1, 1, j)
            end
        end
    elseif left in [neumann, reflect]
        @threaded for j in 1:ny
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 2, 0, j] = ub[n, k, 1, 1, j]
                    Fb[n, k, 2, 0, j] = Fb[n, k, 1, 1, j]
                end
                if left == reflect
                    ub[2, k, 2, 0, j] *= -1.0
                    Fb[1, k, 2, 0, j] *= -1.0
                    Fb[3, k, 2, 0, j] *= -1.0
                    Fb[4, k, 2, 0, j] *= -1.0
                end
            end
        end
    else
        println("Incorrect bc specified at left.")
        @assert false
    end

    if right == dirichlet
        @threaded for j in 1:ny
            x2 = xf[nx + 1]
            for k in 1:nd
                y2 = yf[j] + xg[k] * dy[j]
                ub_value = boundary_value(x2, y2, t)
                fb_value = flux(x2, y2, ub_value, eq, 1)
                for n in 1:nvar
                    ub[n, k, 2, nx, j] = ub[n, k, 1, nx + 1, j] = ub_value[n] # upwind
                    Fb[n, k, 2, nx, j] = Fb[n, k, 1, nx + 1, j] = fb_value[n] # upwind
                end
            end
        end
    elseif right in [neumann, reflect]
        @threaded for j in 1:ny
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 1, nx + 1, j] = ub[n, k, 2, nx, j]
                    Fb[n, k, 1, nx + 1, j] = Fb[n, k, 2, nx, j]
                end
                if right == reflect
                    ub[2, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[1, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[3, k, 1, nx + 1, j] *= -1.0 # ρ*u1*u2
                    Fb[4, k, 1, nx + 1, j] *= -1.0 # (ρ_e + p) * u1
                end
            end
        end
    else
        println("Incorrect bc specified at right.")
        @assert false
    end

    if bottom == dirichlet # in [dirichlet, reflect]
        @threaded for i in 1:nx
            y3 = yf[1]
            for k in Base.OneTo(nd)
                x3 = xf[i] + xg[k] * dx[i]
                ub_value = boundary_value(x3, y3, t)
                fb_value = flux(x3, y3, ub_value, eq, 2)
                for n in 1:nvar
                    ub[n, k, 3, i, 1] = ub[n, k, 4, i, 0] = ub_value[n] # upwind
                    Fb[n, k, 3, i, 1] = Fb[n, k, 4, i, 0] = fb_value[n] # upwind
                end
            end
        end
    elseif bottom in [neumann, reflect]
        @threaded for i in 1:nx
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 4, i, 0] = ub[n, k, 3, i, 1]
                    Fb[n, k, 4, i, 0] = Fb[n, k, 3, i, 1]
                end
                if bottom == reflect
                    ub[3, k, 4, i, 0] *= -1.0
                    Fb[1, k, 4, i, 0] *= -1.0
                    Fb[2, k, 4, i, 0] *= -1.0
                    Fb[4, k, 4, i, 0] *= -1.0
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert typeof(bottom) <: Tuple{Any, Any}
        bc! = bottom[1]
        bc!(grid, eq, op, Fb, ub)
    end

    if top == dirichlet
        @threaded for i in 1:nx
            y4 = yf[ny + 1]
            for k in 1:nd
                x4 = xf[i] + xg[k] * dx[i]
                ub_value = boundary_value(x4, y4, t)
                fb_value = flux(x4, y4, ub_value, eq, 2)
                for n in 1:nvar
                    ub[n, k, 4, i, ny] = ub[n, k, 3, i, ny + 1] = ub_value[n] # upwind
                    Fb[n, k, 4, i, ny] = Fb[n, k, 3, i, ny + 1] = fb_value[n] # upwind
                    # ub[n, k, 3, i, ny+1] = ub_value[n] # upwind
                    # Fb[n, k, 3, i, ny+1] = fb_value[n] # upwind
                end
            end
        end
    elseif top in [neumann, reflect]
        @threaded for i in 1:nx
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 3, i, ny + 1] = ub[n, k, 4, i, ny]
                    Fb[n, k, 3, i, ny + 1] = Fb[n, k, 4, i, ny]
                end
                if top == reflect
                    ub[3, k, 3, i, ny + 1] *= -1.0
                    Fb[1, k, 3, i, ny + 1] *= -1.0
                    Fb[2, k, 3, i, ny + 1] *= -1.0
                    Fb[4, k, 3, i, ny + 1] *= -1.0
                end
            end
        end
    else
        @assert periodic_y "Incorrect bc specified at top"
    end

    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    return nothing
    end # timer
end

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

varnames(eq::TenMoment2D) = eq.varnames
varnames(eq::TenMoment2D, i::Int) = eq.varnames[i]

function Tenkai.initialize_plot(eq::TenMoment2D, op, grid, problem, scheme, timer, u1,
                                ua)
    return nothing
end

function write_poly(eq::TenMoment2D, grid, op, u1, fcount)
    filename = get_filename("output/sol", 3, fcount)
    @show filename
    @unpack xf, yf, dx, dy = grid
    nx, ny = grid.size
    @unpack degree, xg = op
    nd = degree + 1
    # Clear and re-create output directory

    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    Mx, My = nx * nu, ny * nu
    grid_x = zeros(Mx)
    grid_y = zeros(My)
    for i in 1:nx
        i_min = (i - 1) * nu + 1
        i_max = i_min + nu - 1
        # grid_x[i_min:i_max] .= LinRange(xf[i], xf[i+1], nu)
        grid_x[i_min:i_max] .= xf[i] .+ dx[i] * xg
    end

    for j in 1:ny
        j_min = (j - 1) * nu + 1
        j_max = j_min + nu - 1
        # grid_y[j_min:j_max] .= LinRange(yf[j], yf[j+1], nu)
        grid_y[j_min:j_max] .= yf[j] .+ dy[j] * xg
    end

    vtk_sol = vtk_grid(filename, grid_x, grid_y)

    u_equi = zeros(Mx, My)
    u = zeros(nu)
    for j in 1:ny
        for i in 1:nx
            # to get values in the equispaced thing
            for jy in 1:nd
                i_min = (i - 1) * nu + 1
                i_max = i_min + nu - 1
                u_ = @view u1[1, :, jy, i, j]
                mul!(u, Vu, u_)
                j_index = (j - 1) * nu + jy
                u_equi[i_min:i_max, j_index] .= @view u1[1, :, jy, i, j]
            end
        end
    end

    vtk_sol["sol"] = u_equi

    println("Wrote pointwise solution to $filename")

    out = vtk_save(vtk_sol)
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::TenMoment2D, grid,
                            problem, param, op, ua, u1, aux, ndigits = 3)
    @timeit aux.timer "Write solution" begin
    #! format: noindent
    @unpack final_time = problem
    # Clear and re-create output directory
    if fcount == 0
        run(`rm -rf output`)
        run(`mkdir output`)
        save_mesh_file(grid, "output")
    end

    nx, ny = grid.size
    @unpack exact_solution = problem
    exact(x) = exact_solution(x[1], x[2], time)
    @unpack xc, yc = grid
    filename = get_filename("output/avg", ndigits, fcount)
    # filename = string("output/", filename)
    vtk = vtk_grid(filename, xc, yc)
    xy = [[xc[i], yc[j]] for i in 1:nx, j in 1:ny]
    # KLUDGE - Do it efficiently
    prim = @views copy(ua[:, 1:nx, 1:ny])
    exact_data = exact.(xy)
    for j in 1:ny, i in 1:nx
        @views prim[:, i, j] .= con2prim(eq, ua[:, i, j])
    end
    density_arr = prim[1, 1:nx, 1:ny]
    velx_arr = prim[2, 1:nx, 1:ny]
    vely_arr = prim[3, 1:nx, 1:ny]
    P11_arr = prim[4, 1:nx, 1:ny]
    P12_arr = prim[5, 1:nx, 1:ny]
    P22_arr = prim[6, 1:nx, 1:ny]
    vtk["sol"] = density_arr
    vtk["rho"] = density_arr
    vtk["vx"] = velx_arr
    vtk["vy"] = vely_arr
    vtk["P11"] = P11_arr
    vtk["P12"] = P12_arr
    vtk["P22"] = P22_arr
    for j in 1:ny, i in 1:nx
        prim[:, i, j] .= con2prim(eq, exact_data[i, j])
    end
    @views vtk["Exact rho"] = prim[1, 1:nx, 1:ny]
    @views vtk["Exact vx"] = prim[2, 1:nx, 1:ny]
    @views vtk["Exact vy"] = prim[3, 1:nx, 1:ny]
    @views vtk["Exact P11"] = prim[4, 1:nx, 1:ny]
    @views vtk["Exact P12"] = prim[5, 1:nx, 1:ny]
    @views vtk["Exact P22"] = prim[6, 1:nx, 1:ny]
    # @views vtk["Exact Density"] = exact_data[1:nx,1:ny][1]
    # @views vtk["Exact Velocity_x"] = exact_data[1:nx,1:ny][2]
    # @views vtk["Exact Velocity_y"] = exact_data[1:nx,1:ny][3]
    # @views vtk["Exact Pressure"] = exact_data[1:nx,1:ny][4]
    vtk["CYCLE"] = iter
    vtk["TIME"] = time
    out = vtk_save(vtk)
    println("Wrote file ", out[1])
    write_poly(eq, grid, op, u1, fcount)
    if final_time - time < 1e-10
        cp("$filename.vtr", "./output/avg.vtr")
        println("Wrote final average solution to avg.vtr.")
    end

    fcount += 1

    # HDF5 file
    element_variables = Dict()
    element_variables[:density] = vec(density_arr)
    element_variables[:velocity_x] = vec(velx_arr)
    element_variables[:velocity_y] = vec(vely_arr)
    element_variables[:pressure] = vec(P11_arr)
    # element_variables[:indicator_shock_capturing] = vec(aux.blend.cache.alpha[1:nx,1:ny])
    # filename = save_solution_file(u1, time, dt, iter, grid, eq, op,
    #                               element_variables) # Save h5 file
    # println("Wrote ", filename)
    return fcount
    end # timer
end

function rp(x, priml, primr, x0)
    if x < 0.0
        return tenmom_prim2con(priml)
    else
        return tenmom_prim2con(primr)
    end
end

function riemann_problem(x, y, prim_ur, prim_ul, prim_dl, prim_dr)
    if x >= 0.5 && y >= 0.5
        return tenmom_prim2con(prim_ur)
    elseif x <= 0.5 && y >= 0.5
        return tenmom_prim2con(prim_ul)
    elseif x <= 0.5 && y <= 0.5
        return tenmom_prim2con(prim_dl)
    elseif x >= 0.5 && y <= 0.5
        return tenmom_prim2con(prim_dr)
    end
end

sod1d_iv(x, y) = riemann_problem(x, y,
                                 (0.125, 0.0, 0.0, 0.2, 0.1, 0.2),
                                 (1.0, 0.0, 0.0, 2.0, 0.05, 0.6),
                                 (1.0, 0.0, 0.0, 2.0, 0.05, 0.6),
                                 (0.125, 0.0, 0.0, 0.2, 0.1, 0.2))

two_shock1d_iv(x, y) = riemann_problem(x, y,
                                       (1.0, -1.0, -1.0, 1.0, 0.0, 1.0),
                                       (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                                       (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                                       (1.0, -1.0, -1.0, 1.0, 0.0, 1.0))

two_rare1d_iv(x, y) = riemann_problem(x, y,
                                      (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                                      (2.0, -0.5, -0.5, 1.5, 0.5, 1.5),
                                      (2.0, -0.5, -0.5, 1.5, 0.5, 1.5),
                                      (1.0, 1.0, 1.0, 1.0, 0.0, 1.0))

two_rare_vacuum1d_iv(x, y) = riemann_problem(x, y,
                                             (1.0, 5.0, 0.0, 2.0, 0.0, 2.0),
                                             (1.0, -5.0, 0.0, 2.0, 0.0, 2.0),
                                             (1.0, -5.0, 0.0, 2.0, 0.0, 2.0),
                                             (1.0, 5.0, 0.0, 2.0, 0.0, 2.0))

function ten_moment_source_x(u, x, y, t, Wx_, equations::TenMoment2D)
    rho = u[1]
    rho_v1 = u[2]
    rho_v2 = u[3]
    Wx = Wx_(x, y, t)
    return SVector(0.0, -0.5 * rho * Wx, 0.0, -0.5 * rho_v1 * Wx, -0.25 * rho_v2 * Wx,
                   0.0)
end

function ten_moment_source_y(u, x, y, t, Wy_, equations::TenMoment2D)
    rho = u[1]
    rho_v1 = u[2]
    rho_v2 = u[3]
    Wy = Wy_(x, y, t)
    return SVector(0.0, 0.0, -0.5 * rho * Wy, 0.0, -0.25 * rho_v1 * Wy,
                   -0.5 * rho_v2 * Wy)
end

function ten_moment_source(u, x, y, t, Wx, Wy, equations::TenMoment2D)
    source_x = ten_moment_source_x(u, x, y, t, Wx, equations)
    source_y = ten_moment_source_y(u, x, y, t, Wy, equations)

    return source_x + source_y
end

get_equation() = TenMoment2D(["rho", "v1", "v2", "P11", "P12", "P22"],
                             "Ten moment problem")
end # @muladd

end # module
