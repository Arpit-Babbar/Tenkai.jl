module EqTenMoment2D

using TimerOutputs
using StaticArrays
using LinearAlgebra
using UnPack
using Plots
using Printf
using JSON3
using GZip
using DelimitedFiles
using WriteVTK

using Tenkai
using Tenkai.Basis
using Tenkai.FR2D: correct_variable!
using Tenkai.FR: limit_variable_slope

using Tenkai.CartesianGrids: CartesianGrid2D, save_mesh_file

(import Tenkai: flux, prim2con, con2prim, limit_slope, zhang_shu_flux_fix,
                apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln)

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
function con2prim(eq::TenMoment2D, u)
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
    P11s = (ml[2]*mr[1] - ml[1]*mr[2]) / (mr[1] - ml[1]) # TODO - Can use a simpler expression
    P12s = (ml[3]*mr[1] - ml[1]*mr[3]) / (mr[1] - ml[1])
    dsl, dsr = v1s - sl, v1s - sr
    @assert dsl > 0.0 && dsr < 0.0 "Middle contact v1s=$v1s outside [sl, sr]=[$sl,$sr]"
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

@inbounds @inline hllc3(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment2D, dir) = hllc3(
    x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment2D, dir, hll_speeds_min_max)

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

function Tenkai.compute_time_step(eq::TenMoment2D, grid, aux, op, cfl, u1, ua)
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
        sx, sy = max_abs_eigen_value(eq, u_node, 1), max_abs_eigen_value(eq, u_node, 2)
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
        prim[:, i,j] .= con2prim(eq, exact_data[i, j])
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

sod1d_iv(x,y) = riemann_problem(x,y,
                                (0.125, 0.0, 0.0, 0.2, 0.1, 0.2),
                                (1.0, 0.0, 0.0, 2.0, 0.05, 0.6),
                                (1.0, 0.0, 0.0, 2.0, 0.05, 0.6),
                                (0.125, 0.0, 0.0, 0.2, 0.1, 0.2))

two_shock1d_iv(x,y) = riemann_problem(x, y,
                                     (1.0, -1.0, -1.0, 1.0, 0.0, 1.0),
                                     (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                                     (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                                     (1.0, -1.0, -1.0, 1.0, 0.0, 1.0))

two_rare1d_iv(x,y) = riemann_problem(x, y,
                                     (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                                     (2.0, -0.5, -0.5, 1.5, 0.5, 1.5),
                                     (2.0, -0.5, -0.5, 1.5, 0.5, 1.5),
                                     (1.0, 1.0, 1.0, 1.0, 0.0, 1.0))

two_rare_vacuum1d_iv(x, y) = riemann_problem(x,y,
                                            (1.0, 5.0, 0.0, 2.0, 0.0, 2.0),
                                            (1.0, -5.0, 0.0, 2.0, 0.0, 2.0),
                                            (1.0, -5.0, 0.0, 2.0, 0.0, 2.0),
                                            (1.0, 5.0, 0.0, 2.0, 0.0, 2.0))


get_equation() = TenMoment2D(["rho", "v1", "v2", "P11", "P12", "P22"],
                             "Ten moment problem")
end # @muladd

end # module
