module EqTenMoment1D

using TimerOutputs
using StaticArrays
using LinearAlgebra
using SimpleUnPack
using Plots
using Printf
using JSON3
using GZip
using DelimitedFiles

using Tenkai
using Tenkai.Basis
using Tenkai: test_variable_bound_limiter!
using Tenkai: newton_solver_tenkai, sum_node_vars_1d, find_theta

(import Tenkai: flux, prim2con, con2prim, limit_slope, zhang_shu_flux_fix,
                apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln,
                correct_variable_bound_limiter!, limit_variable_slope)

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

struct TenMoment1D <: AbstractEquations{1, 6}
    varnames::Vector{String}
    name::String
end

@inline tenmom_density(::TenMoment1D, u) = u[1]

@inline energy11(::TenMoment1D, u) = u[4]
@inline energy12(::TenMoment1D, u) = u[5]
@inline energy22(::TenMoment1D, u) = u[6]

@inline energy_components(::TenMoment1D, u) = u[4], u[5], u[6]

@inline velocity(::TenMoment1D, u) = u[2] / u[1], u[3] / u[1]

@inline function pressure_components(eq::TenMoment1D, u)
    ρ = tenmom_density(eq, u)
    v1, v2 = velocity(eq, u)
    E11, E12, E22 = energy_components(eq, u)
    P11 = 2.0 * E11 - ρ * v1 * v2
    P12 = 2.0 * E12 - ρ * v1 * v2
    P22 = 2.0 * E22 - ρ * v2 * v2
    return P11, P12, P22
end

# The primitive variables are
# ρ, v1, v2, P11, P12, P22.
# Tensor P is defined so that tensor E = 0.5 * (P + ρ*v⊗v)
function con2prim(eq::TenMoment1D, u)
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

@inbounds @inline prim2con(eq::TenMoment1D, prim) = tenmom_prim2con(prim)

# The flux is given by
# (ρ v1, P11 + ρ v1^2, P12 + ρ v1*v2, (E + P) ⊗ v)
@inbounds @inline function flux(x, u, eq::TenMoment1D)
    r, v1, v2, P11, P12, P22 = con2prim(eq, u)

    f1 = r * v1
    f2 = P11 + r * v1 * v1
    f3 = P12 + r * v1 * v2
    f4 = (u[4] + P11) * v1
    f5 = u[5] * v1 + 0.5 * (P11 * v2 + P12 * v1)
    f6 = u[6] * v1 + P12 * v2

    return SVector(f1, f2, f3, f4, f5, f6)
end

# Compute flux directly from the primitive variables
@inbounds @inline function prim2flux(x, prim, eq::TenMoment1D)
    r, v1, v2, P11, P12, P22 = prim

    E11 = 0.5 * (P11 + r * v1 * v1)
    E12 = 0.5 * (P12 + r * v1 * v2)
    E22 = 0.5 * (P22 + r * v2 * v2)

    f1 = r * v1
    f2 = P11 + r * v1 * v1
    f3 = P12 + r * v1 * v2
    f4 = (E11 + P11) * v1
    f5 = E12 * v1 + 0.5 * (P11 * v2 + P12 * v1)
    f6 = E22 * v1 + P12 * v2

    return SVector(f1, f2, f3, f4, f5, f6)
end

function Tenkai.is_admissible(eq::TenMoment1D, u::AbstractVector)
    return (density_constraint(eq, u) > 0.0
            && trace_constraint(eq, u) > 0.0
            && det_constraint(eq, u) > 0.0)
end

@inbounds @inline function hll_speeds_min_max(eq::TenMoment1D, ul, ur)
    # Get conservative variables
    rl, v1l, v2l, P11l, P12l, P22l = con2prim(eq, ul)
    rr, v1r, v2r, P11r, P12r, P22r = con2prim(eq, ur)

    T11l = P11l / rl
    cl1 = sqrt(3.0 * T11l)
    cl2 = P12l / sqrt(rl * P22l)
    cl = max(cl1, cl2)
    min_l = v1l - cl
    max_l = v1l + cl

    T11r = P11r / rr
    cr1 = sqrt(3.0 * T11r)
    cr2 = P12r / sqrt(rr * P22r)
    cr = max(cr1, cr2)
    min_r = v1r - cr
    max_r = v1r + cr

    sl = min(min_l, min_r)
    sr = max(max_l, max_r)

    # Roe average state
    r = 0.5(rl + rr)
    srl = sqrt(rl)
    srr = sqrt(rr)
    f1 = 1.0 / (srl + srr)
    v1 = f1 * (srl * v1l + srr * v1r)
    T11 = f1 * (srl * T11l + srr * T11r) +
          (srl * srr * (v1r - v1l)^2) / (3.0 * (srl + srr)^2)

    sl = min(sl, v1 - sqrt(3.0 * T11))
    sr = max(sr, v1 + sqrt(3.0 * T11))

    return sl, sr
end

@inbounds @inline function hll(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir,
                               wave_speeds::Function)
    sl, sr = wave_speeds(eq, ual, uar)
    if sl > 0.0
        return Fl
    elseif sr < 0.0
        return Fr
    else
        return (sr * Fl - sl * Fr + sl * sr * (Ur - Ul)) / (sr - sl)
    end
end

@inbounds @inline hll(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir) = hll(x, ual,
                                                                               uar, Fl,
                                                                               Fr, Ul,
                                                                               Ur,
                                                                               eq::TenMoment1D,
                                                                               dir,
                                                                               hll_speeds_min_max)

@inbounds @inline function hllc3(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir,
                                 wave_speeds::Function)
    sl, sr = wave_speeds(eq, ual, uar)
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
        Usl = SVector(rhosl, rhosl * v1s, rhosl * v2s, E11sl, E12sl, E22sl)
        return Fl + sl * (Usl - Ul)
    else
        rhosr = mr[1] / dsr
        E11sr = (mr[4] - v1s * P11s) / dsr
        E12sr = (mr[5] - 0.5 * (P11s * v2s + P12s * v1s)) / dsr
        E22sr = (mr[6] - P12s * v2s) / dsr
        Usr = SVector(rhosr, rhosr * v1s, rhosr * v2s, E11sr, E12sr, E22sr)
        return Fr + sr * (Usr - Ur)
    end
end

@inbounds @inline hllc3(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir) = hllc3(x,
                                                                                   ual,
                                                                                   uar,
                                                                                   Fl,
                                                                                   Fr,
                                                                                   Ul,
                                                                                   Ur,
                                                                                   eq::TenMoment1D,
                                                                                   dir,
                                                                                   hll_speeds_min_max)

function max_abs_eigen_value(eq::TenMoment1D, u)
    ρ = u[1]
    v1 = u[2] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    T11 = P11 / ρ
    return abs(v1) + sqrt(3.0 * T11)
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir)
    # rl, v1l, v2l, P11l, P12l, P22l = con2prim(eq, ual)
    # rr, v1r, v2r, P11r, P12r, P22r = con2prim(eq, uar)

    # T11l = P11l / rl
    # T11r = P11r / rr
    # cl, cr = sqrt(3.0 * T11l), sqrt(3.0 * T11r)
    # λ = max(abs(v1l), abs(v1r)) + max(cl, cr)

    λ = max(max_abs_eigen_value(eq, ual), max_abs_eigen_value(eq, uar)) # local wave speed

    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function compute_time_step(eq::TenMoment1D, problem, grid, aux, op, cfl, u1, ua)
    @unpack source_terms = problem
    nx = grid.size
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        u = get_node_vars(ua, eq, i)
        smax = max_abs_eigen_value(eq, u)
        den = max(den, smax / dx[i])
    end
    dt_source = cfl *
                compute_source_time_step(eq, source_terms, grid, aux, op, cfl, u1, ua)
    dt = cfl / den
    return min(dt, dt_source)
end

compute_source_time_step(eq, source_terms, grid, aux, op, cfl, u1, ua) = 1e20

#-------------------------------------------------------------------------------
# Limiters
#-------------------------------------------------------------------------------
# Admissibility constraints
@inline @inbounds density_constraint(eq::TenMoment1D, u) = u[1]

@inline @inbounds function P11_constraint(eq::TenMoment1D, u)
    ρ = u[1]
    v1 = u[2] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    return P11
end

@inline @inbounds function P22_constraint(eq::TenMoment1D, u)
    ρ = u[1]
    v2 = u[3] / ρ
    P22 = 2.0 * u[6] - ρ * v2 * v2
    return P22
end

@inline @inbounds function trace_constraint(eq::TenMoment1D, u)
    ρ = u[1]
    v1 = u[2] / ρ
    v2 = u[3] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    P22 = 2.0 * u[6] - ρ * v2 * v2
    return P11 + P22
end

@inline @inbounds function det_constraint(eq::TenMoment1D, u)
    ρ = u[1]
    v1 = u[2] / ρ
    v2 = u[3] / ρ
    P11 = 2.0 * u[4] - ρ * v1 * v1
    P12 = 2.0 * u[5] - ρ * v1 * v2
    P22 = 2.0 * u[6] - ρ * v2 * v2
    return P11 * P22 - P12 * P12
end

# Looping over a tuple can be made type stable following this
# https://github.com/trixi-framework/Trixi.jl/blob/0fd86e4bd856d894de6a7514edcb9758bf6f8e1e/src/callbacks_stage/positivity_zhang_shu.jl#L39
function iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                          u1, aux, variables::NTuple{N, Any}) where {N}
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    correct_variable_bound_limiter!(variable, eq, grid, op, ua, u1)

    # test_variable_bound_limiter!(variable, eq, grid, op, ua, u1)
    iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                     u1, aux, remaining_variables)
    return nothing
end

function iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                          u1, aux, variables::Tuple{})
    return nothing
end

function correct_variable_bound_limiter!(variable::typeof(det_constraint),
                                         eq::AbstractEquations{1}, grid, op, ua, u1)
    @unpack Vl, Vr = op
    nx = grid.size
    nd = op.degree + 1
    eps = 1e-10 # TODO - Get a better one
    # Loop to find var_avg_min
    var_avg_min = 1e20
    for element in 1:nx
        ua_ = get_node_vars(ua, eq, element)
        var = variable(eq, ua_)
        var_avg_min = min(var_avg_min, var)
    end
    @assert var_avg_min>0.0 "The average value of the variable is non-positive"
    eps = min(eps, 0.1 * var_avg_min)
    for element in 1:nx
        local theta = 1.0
        ua_node = get_node_vars(ua, eq, element)
        for i in Base.OneTo(nd)
            u_node = get_node_vars(u1, eq, i, element)
            theta = min(theta, find_theta(eq, variable, u_node, ua_node, eps))
        end

        # In order to correct the solution at the faces, we need to extrapolate it to faces
        # and then correct it.
        ul = sum_node_vars_1d(Vl, u1, eq, 1:nd, element) # ul = ∑ Vl*u
        ur = sum_node_vars_1d(Vr, u1, eq, 1:nd, element) # ur = ∑ Vr*u

        theta = min(theta, find_theta(eq, variable, ul, ua_node, eps))
        theta = min(theta, find_theta(eq, variable, ur, ua_node, eps))

        theta = min(theta, 1.0)
        if theta < 0.0
            @warn "Negative theta = $theta"
            theta = 0.0
        end

        if theta < 1.0
            for i in 1:nd
                u_node = get_node_vars(u1, eq, i, element)
                multiply_add_set_node_vars!(u1,
                                            theta, u_node,
                                            1 - theta, ua_node,
                                            eq, i, element)
            end
        end
    end
end

function Tenkai.apply_bound_limiter!(eq::TenMoment1D, grid, scheme, param, op, ua,
                                     u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end
    variables = (density_constraint, trace_constraint, det_constraint, P11_constraint,
                 P22_constraint)
    iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua, u1, aux,
                                     variables)
    return nothing
end

#-------------------------------------------------------------------------------
# Blending Limiter
#-------------------------------------------------------------------------------

@inbounds @inline function rho_p_indicator!(un, eq::TenMoment1D)
    for ix in 1:size(un, 2) # loop over dofs and faces
        u_node = get_node_vars(un, eq, ix)
        p = det_constraint(eq, u_node)
        un[1, ix] *= p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

function Tenkai.zhang_shu_flux_fix(eq::TenMoment1D,
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
    eps = min(0.1 * p_low, 1e-10)
    theta = find_theta(eq, det_constraint, uhigh, ulow, eps)
    Fn = theta * Fn + (1.0 - theta) * fn # Final flux

    return Fn
end

function limit_variable_slope(eq::TenMoment1D, variable::typeof(det_constraint), slope,
                              u_star_ll, u_star_rr, ue, xl, xr)
    # By Jensen's inequality, we can find theta's directly for the primitives
    var_star_ll, var_star_rr = variable(eq, u_star_ll), variable(eq, u_star_rr)
    var_low = variable(eq, ue)
    threshold = 0.1 * var_low
    eps = 1e-10
    if var_star_ll < eps || var_star_rr < eps
        theta_ll = find_theta(eq, variable, u_star_ll, ue, threshold)
        theta_rr = find_theta(eq, variable, u_star_rr, ue, threshold)
        theta = min(theta_ll, theta_rr, 1.0)
        slope *= theta
        u_star_ll = ue + 2.0 * xl * slope
        u_star_rr = ue + 2.0 * xr * slope
    end
    return slope, u_star_ll, u_star_rr
end

function Tenkai.limit_slope(eq::TenMoment1D, slope, ufl, u_star_ll, ufr, u_star_rr,
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

#-------------------------------------------------------------------------------
# TVB limiter
#-------------------------------------------------------------------------------

function eigmatrix(eq::TenMoment1D, u)
    nvar = nvariables(eq)

    rho = tenmom_density(eq, u)
    E11, E12, E22 = energy_components(eq, u)
    v1, v2 = velocity(eq, u)
    p11 = 2.0 * E11 - rho * v1 * v1
    p12 = 2.0 * E12 - rho * v1 * v2
    p22 = 2.0 * E22 - rho * v2 * v2
    cx = sqrt(3.0 * abs(p11 / rho))

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

    return Rx, Lx
end

function Tenkai.apply_tvb_limiter!(eq::TenMoment1D, problem, scheme, grid, param, op,
                                   ua,
                                   u1, aux)
    @timeit aux.timer "TVB limiter" begin
    #! format: noindent
    nx = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack limiter = scheme
    @unpack tvbM, cache = limiter
    left_bc, right_bc = problem.boundary_condition
    nd = length(wg)
    nvar = nvariables(eq)
    # face values
    (uimh, uiph, Δul, Δur, Δual, Δuar, char_Δul, char_Δur, char_Δual, char_Δuar,
    dulm, durm, du) = cache

    # Loop over cells
    beta = limiter.beta
    beta_ = beta / 2.0
    for cell in 1:nx
        ual, ua_, uar = (get_node_vars(ua, eq, cell - 1),
                         get_node_vars(ua, eq, cell),
                         get_node_vars(ua, eq, cell + 1))

        # Needed for characteristic limiting
        R, L = eigmatrix(eq, ua_)
        fill!(uimh, zero(eltype(uimh)))
        fill!(uiph, zero(eltype(uiph)))
        Mdx2 = tvbM * grid.dx[cell]^2
        if left_bc == neumann && right_bc == neumann && (cell == 1 || cell == nx)
            Mdx2 = 0.0 # Force TVD on boundary for Shu-Osher
        end
        # end # timer
        for ii in 1:nd
            u_ = get_node_vars(u1, eq, ii, cell)
            multiply_add_to_node_vars!(uimh, Vl[ii], u_, eq, 1)
            multiply_add_to_node_vars!(uiph, Vr[ii], u_, eq, 1)
        end
        # Get views of needed cell averages
        # slopes b/w centres and faces

        uimh_ = get_node_vars(uimh, eq, 1)
        uiph_ = get_node_vars(uiph, eq, 1)

        # We will set
        # Δul[n] = ua_[n] - uimh[n]
        # Δur[n] = uiph[n] - ua_[n]
        # Δual[n] = ua_[n] - ual[n]
        # Δuar[n] = uar[n] - ua_[n]

        set_node_vars!(Δul, ua_, eq, 1)
        set_node_vars!(Δur, uiph_, eq, 1)
        set_node_vars!(Δual, ua_, eq, 1)
        set_node_vars!(Δuar, uar, eq, 1)

        subtract_from_node_vars!(Δul, uimh_, eq)
        subtract_from_node_vars!(Δur, ua_, eq)
        subtract_from_node_vars!(Δual, ual, eq)
        subtract_from_node_vars!(Δuar, ua_, eq)

        Δul_ = get_node_vars(Δul, eq, 1)
        Δur_ = get_node_vars(Δur, eq, 1)
        Δual_ = get_node_vars(Δual, eq, 1)
        Δuar_ = get_node_vars(Δuar, eq, 1)

        # Uncomment this part for characteristic limiting
        # mul!(char_Δul, L, Δul_)   # char_Δul = L*Δul
        # mul!(char_Δur, L, Δur_)   # char_Δur = L*Δur
        # mul!(char_Δual, L, Δual_) # char_Δual = L*Δual
        # mul!(char_Δuar, L, Δuar_) # char_Δuar = L*Δuar

        # Use primitive variables
        # set_node_vars!(char_Δul, con2prim(eq, Δul_), eq, 1)
        # set_node_vars!(char_Δur, con2prim(eq, Δur_), eq, 1)
        # set_node_vars!(char_Δual, con2prim(eq, Δual_), eq, 1)
        # set_node_vars!(char_Δuar, con2prim(eq, Δuar_), eq, 1)

        # Keep conservative variables
        set_node_vars!(char_Δul, Δul_, eq, 1)
        set_node_vars!(char_Δur, Δur_, eq, 1)
        set_node_vars!(char_Δual, Δual_, eq, 1)
        set_node_vars!(char_Δuar, Δuar_, eq, 1)

        char_Δul_ = get_node_vars(char_Δul, eq, 1)
        char_Δur_ = get_node_vars(char_Δur, eq, 1)
        char_Δual_ = get_node_vars(char_Δual, eq, 1)
        char_Δuar_ = get_node_vars(char_Δuar, eq, 1)
        for n in eachvariable(eq)
            dulm[n] = minmod(char_Δul_[n], beta_ * char_Δual_[n],
                             beta_ * char_Δuar_[n], Mdx2)
            durm[n] = minmod(char_Δur_[n], beta_ * char_Δual_[n],
                             beta_ * char_Δuar_[n], Mdx2)
        end

        # limit if jumps are detected
        dulm_ = get_node_vars(dulm, eq, 1)
        durm_ = get_node_vars(durm, eq, 1)
        jump_l = jump_r = 0.0
        for n in 1:nvar
            jump_l += abs(char_Δul_[n] - dulm_[n])
            jump_r += abs(char_Δur_[n] - durm_[n])
        end
        jump_l /= nvar
        jump_r /= nvar

        if jump_l > 1e-06 || jump_r > 1e-06
            add_to_node_vars!(durm, dulm_, eq, 1) # durm = durm + dulm
            # We want durm = 0.5 * (dul + dur), we adjust 0.5 later

            # Uncomment this for characteristic variables
            # mul!(du, R, durm)            # du = R * (dulm+durm)

            # Conservative / primitive variables
            # durm_ = prim2con(eq, get_node_vars(durm, eq, 1)) # Keep for primitive
            durm_ = get_node_vars(durm, eq, 1) # Keep for conservative
            set_node_vars!(du, durm_, eq, 1)
            for ii in Base.OneTo(nd)
                du_ = get_node_vars(du, eq, 1)
                set_node_vars!(u1, ua_ + (xg[ii] - 0.5) * du_, # 2.0 adjusted with 0.5 above
                               eq, ii,
                               cell)
            end
        end
    end
    return nothing
    end # timer
end

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

varnames(eq::TenMoment1D) = eq.varnames
varnames(eq::TenMoment1D, i::Int) = eq.varnames[i]

function Tenkai.initialize_plot(eq::TenMoment1D, op, grid, problem, scheme, timer, u1,
                                ua)
    @timeit timer "Write solution" begin
    #! format: noindent
    @timeit timer "Initialize write solution" begin
    #! format: noindent
    # Clear and re-create output directory
    rm("output", force = true, recursive = true)
    mkdir("output")

    xc = grid.xc
    nx = grid.size
    @unpack xg = op
    nd = op.degree + 1
    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    xf = grid.xf
    nvar = nvariables(eq)
    # Create plot objects to be later collected as subplots

    # Creating a subplot for title
    p_title = plot(title = "Cell averages plot, $nx cells, t = 0.0",
                   grid = false, showaxis = false, bottom_margin = 0Plots.px)
    p_ua, p_u1 = [plot() for _ in 1:nvar], [plot() for _ in 1:nvar]
    labels = varnames(eq)
    y = zeros(nx) # put dummy to fix plotly bug with OffsetArrays
    for n in 1:nvar
        @views plot!(p_ua[n], xc, y, label = "Approximate",
                     linestyle = :dot, seriestype = :scatter,
                     color = :blue, markerstrokestyle = :dot,
                     markershape = :circle, markersize = 2,
                     markerstrokealpha = 0)
        xlabel!(p_ua[n], "x")
        ylabel!(p_ua[n], labels[n])
    end
    l_super = @layout[a{0.01h}; b c d; e f g] # Selecting layout for p_title being title
    p_ua = plot(p_title, p_ua..., layout = l_super,
                size = (1500, 500)) # Make subplots

    # Set up p_u1 to contain polynomial approximation as a different curve
    # for each cell
    x = LinRange(xf[1], xf[2], nu)
    up1 = zeros(nvar, nd)
    u = zeros(nu)
    for ii in 1:nd
        u_node = get_node_vars(u1, eq, ii, 1)
        up1[:, ii] .= con2prim(eq, u_node)
    end

    for n in 1:nvar
        u = @views Vu * up1[n, :]
        plot!(p_u1[n], x, u, color = :red, legend = false)
        xlabel!(p_u1[n], "x")
        ylabel!(p_u1[n], labels[n])
    end

    for i in 2:nx
        for ii in 1:nd
            u_node = get_node_vars(u1, eq, ii, i)
            up1[:, ii] .= con2prim(eq, u_node)
        end
        x = LinRange(xf[i], xf[i + 1], nu)
        for n in 1:nvar
            u = @views Vu * up1[n, :]
            plot!(p_u1[n], x, u, color = :red, label = nothing, legend = false)
        end
    end

    l = @layout[a{0.01h}; b c d; e f g] # Selecting layout for p_title being title
    p_u1 = plot(p_title, p_u1..., layout = l,
                size = (1700, 500)) # Make subplots

    anim_ua, anim_u1 = Animation(), Animation() # Initialize animation objects
    plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1)
    return plot_data
    end # timer
    end # timer
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::TenMoment1D, grid,
                            problem, param, op, ua, u1, aux, ndigits = 3)
    @timeit aux.timer "Write solution" begin
    #! format: noindent
    @unpack plot_data = aux
    avg_filename = get_filename("output/avg", ndigits, fcount)
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    @unpack final_time = problem
    xc = grid.xc
    nx = grid.size
    @unpack xg = op
    nd = op.degree + 1
    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    nvar = nvariables(eq)
    @unpack save_time_interval, save_iter_interval, animate = param
    avg_file = open("$avg_filename.txt", "w")
    up_ = zeros(nvar)
    ylims = [[Inf, -Inf] for _ in 1:nvar] # set ylims for plots of all variables
    for i in 1:nx
        ua_node = get_node_vars(ua, eq, i)
        up_ .= con2prim(eq, ua_node)
        @printf(avg_file, "%e %e %e %e %e %e %e \n", xc[i], up_...)
        # TOTHINK - Check efficiency of printf
        for n in eachvariable(eq)
            p_ua[n + 1][1][:y][i] = @views up_[n]    # Update y-series
            ylims[n][1] = min(ylims[n][1], up_[n]) # Compute ymin
            ylims[n][2] = max(ylims[n][2], up_[n]) # Compute ymax
        end
    end
    close(avg_file)
    for n in 1:nvar # set ymin, ymax for ua, u1 plots
        ylims!(p_ua[n + 1], (ylims[n][1] - 0.1, ylims[n][2] + 0.1))
        ylims!(p_u1[n + 1], (ylims[n][1] - 0.1, ylims[n][2] + 0.1))
    end
    t = round(time; digits = 3)
    title!(p_ua[1], "Cell averages plot, $nx cells, t = $t")
    sol_filename = get_filename("output/sol", ndigits, fcount)
    sol_file = open(sol_filename * ".txt", "w")
    up1 = zeros(nvar, nd)

    u = zeros(nvar, nu)
    x = zeros(nu)
    for i in 1:nx
        for ii in 1:nd
            u_node = get_node_vars(u1, eq, ii, i)
            up1[:, ii] .= con2prim(eq, u_node)
        end
        @. x = grid.xf[i] + grid.dx[i] * xu
        @views mul!(u, up1, Vu')
        for n in 1:nvar
            p_u1[n + 1][i][:y] = u[n, :]
        end
        for ii in 1:nu
            u_node = get_node_vars(u, eq, ii)
            @printf(sol_file, "%e %e %e %e %e %e %e \n", x[ii], u_node...)
        end
    end
    close(sol_file)
    title!(p_u1[1], "Numerical Solution, $nx cells, t = $t")
    println("Wrote $sol_filename.txt, $avg_filename.txt")
    if problem.final_time - time < 1e-10
        cp("$avg_filename.txt", "./output/avg.txt", force = true)
        cp("$sol_filename.txt", "./output/sol.txt", force = true)
        println("Wrote final solution to avg.txt, sol.txt.")
    end
    if animate == true
        if abs(time - final_time) < 1.0e-10
            frame(anim_ua, p_ua)
            frame(anim_u1, p_u1)
        end
        if save_iter_interval > 0
            animate_iter_interval = save_iter_interval
            if mod(iter, animate_iter_interval) == 0
                frame(anim_ua, p_ua)
                frame(anim_u1, p_u1)
            end
        elseif save_time_interval > 0
            animate_time_interval = save_time_interval
            k1, k2 = ceil(time / animate_time_interval),
                     floor(time / animate_time_interval)
            if (abs(time - k1 * animate_time_interval) < 1e-10 ||
                abs(time - k2 * animate_time_interval) < 1e-10)
                frame(anim_ua, p_ua)
                frame(anim_u1, p_u1)
            end
        end
    end
    fcount += 1
    return fcount
    end # timer
end

function rp(x, priml, primr, x0)
    if x < x0
        return tenmom_prim2con(priml)
    else
        return tenmom_prim2con(primr)
    end
end

sod_iv(x) = rp(x,
               (1.0, 0.0, 0.0, 2.0, 0.05, 0.6),
               (0.125, 0.0, 0.0, 0.2, 0.1, 0.2),
               0.0)
two_shock_iv(x) = rp(x,
                     (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                     (1.0, -1.0, -1.0, 1.0, 0.0, 1.0),
                     0.0)

two_rare_iv(x) = rp(x,
                    (2.0, -0.5, -0.5, 1.5, 0.5, 1.5),
                    (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                    0.0)

two_rare_vacuum_iv(x) = rp(x,
                           (1.0, -5.0, 0.0, 2.0, 0.0, 2.0),
                           (1.0, 5.0, 0.0, 2.0, 0.0, 2.0),
                           0.0)

exact_solution_data(iv::Function) = nothing
function exact_solution_data(iv::typeof(sod_iv))
    exact_filename = joinpath(Tenkai.data_dir, "10mom_sod.gz")
    file = GZip.open(exact_filename)
    exact_data = readdlm(file)
    return exact_data
end

function exact_solution_data(iv::typeof(two_shock_iv))
    exact_filename = joinpath(Tenkai.data_dir, "10mom_two_shock.gz")
    file = GZip.open(exact_filename)
    exact_data = readdlm(file)
    return exact_data
end

function exact_solution_data(iv::typeof(two_rare_iv))
    exact_filename = joinpath(Tenkai.data_dir, "10mom_two_rare.gz")
    file = GZip.open(exact_filename)
    exact_data = readdlm(file)
    return exact_data
end

function post_process_soln(eq::TenMoment1D, aux, problem, param, scheme)
    @unpack timer, error_file = aux
    @timeit timer "Write solution" begin
    #! format: noindent
    println("Post processing solution")
    @unpack plot_data = aux
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    @unpack animate, saveto = param
    @unpack initial_value = problem

    exact_data = exact_solution_data(initial_value)
    @show exact_data
    if exact_data !== nothing
        for n in eachvariable(eq)
            @views plot!(p_ua[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact",
                         color = :black)
            @views plot!(p_u1[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact",
                         color = :black, legend = true)
            ymin = min(minimum(p_ua[n + 1][1][:y]), minimum(exact_data[:, n + 1]))
            ymax = max(maximum(p_ua[n + 1][1][:y]), maximum(exact_data[:, n + 1]))
            ylims!(p_ua[n + 1], (ymin - 0.1, ymax + 0.1))
            ylims!(p_u1[n + 1], (ymin - 0.1, ymax + 0.1))
        end
    end
    savefig(p_ua, "output/avg.png")
    savefig(p_u1, "output/sol.png")
    savefig(p_ua, "output/avg.html")
    savefig(p_u1, "output/sol.html")
    if animate == true
        gif(anim_ua, "output/avg.mp4", fps = 5)
        gif(anim_u1, "output/sol.mp4", fps = 5)
    end
    println("Wrote avg, sol in gif,html,png format to output directory.")
    plot(p_ua)
    plot(p_u1)

    close(error_file)
    if saveto != "none"
        if saveto[end] == "/"
            saveto = saveto[1:(end - 1)]
        end
        mkpath(saveto)
        for file in readdir("./output")
            cp("./output/$file", "$saveto/$file", force = true)
        end
        cp("./error.txt", "$saveto/error.txt", force = true)
        println("Saved output files to $saveto")
    end
    end # timer

    # Print timer data on screen
    print_timer(aux.timer, sortby = :firstexec)
    print("\n")
    show(aux.timer)
    print("\n")
    println("Time outside write_soln = "
            *
            "$(( TimerOutputs.tottime(timer)
                - TimerOutputs.time(timer["Write solution"]) ) * 1e-9)s")
    println("─────────────────────────────────────────────────────────────────────────────────────────")
    timer_file = open("./output/timer.json", "w")
    JSON3.write(timer_file, TimerOutputs.todict(timer))
    close(timer_file)
    return nothing
end

struct TenMoment1DSourceTerms{WxType}
    Wx::WxType
end

function compute_source_time_step(eq, source_terms::TenMoment1DSourceTerms, grid, aux,
                                  op, cfl,
                                  u1, ua)
    nx = grid.size
    L = 1e20
    dx = grid.dx
    xc = grid.xc
    @unpack xg = op
    nd = length(xg)
    dummy_t = 0.0
    for cell in 1:nx
        Wx = source_terms.Wx(xc[cell], dummy_t) + 1e-6
        ua_node = get_node_vars(ua, eq, cell)
        rho, v1, v2, P11, P12, P22 = con2prim(eq, ua_node)
        a = 0.5 * sqrt(P11 / rho) / abs(Wx)
        b_sqr = det_constraint(eq, ua_node) / (rho * P22)
        b = 0.5 * sqrt(b_sqr) / abs(Wx)
        L = min(L, a, b)
        # if L < 1e-12
        #     @assert false rho, v1, v2, P11, P12, P22,Wx,a,b
        # end
        # @show L, a, b, Wx
        for i in 1:nd
            x = xc[cell] - 0.5 * dx[cell] + xg[i] * dx[cell]
            u_node = get_node_vars(u1, eq, i, cell)
            Wx = source_terms.Wx(x, dummy_t) + 1e-6
            a = 0.5 * sqrt(P11 / rho) / abs(Wx)
            b_sqr = det_constraint(eq, u_node) / (rho * P22)
            b = 0.5 * sqrt(b_sqr) / abs(Wx)
            L = min(L, a, b)
            # @show L
            # if L < 1e-12
            #     @assert false rho, v1, v2, P11, P12, P22,Wx,a,b
            # end
        end
    end
    # @show L
    return max(L, 1e-10)
end

function (tem_moment_1d_source::TenMoment1DSourceTerms)(u, x, t, equations::TenMoment1D)
    rho = u[1]
    rho_v1 = u[2]
    rho_v2 = u[3]
    Wx = tem_moment_1d_source.Wx(x, t)
    return SVector(0.0, -0.5 * rho * Wx, 0.0, -0.5 * rho_v1 * Wx, -0.25 * rho_v2 * Wx,
                   0.0)
end

function ten_moment_source_x(u, x, t, Wx_, equations::TenMoment1D)
    rho = u[1]
    rho_v1 = u[2]
    rho_v2 = u[3]
    Wx = Wx_(x, t)
    return SVector(0.0, -0.5 * rho * Wx, 0.0, -0.5 * rho_v1 * Wx, -0.25 * rho_v2 * Wx,
                   0.0)
end

get_equation() = TenMoment1D(["rho", "v1", "v2", "P11", "P12", "P22"],
                             "Ten moment problem")
end # @muladd

end # module
