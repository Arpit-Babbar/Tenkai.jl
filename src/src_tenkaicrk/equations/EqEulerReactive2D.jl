module EqEulerReactive2D
#! format: noindent

using Tenkai

import GZip
using Tenkai.DelimitedFiles
using Plots
using LinearAlgebra
using Tenkai.SimpleUnPack
using Printf
using TimerOutputs
using StaticArrays
using Tenkai.WriteVTK
using Tenkai.Polyester
using Tenkai.LoopVectorization
using Tenkai.JSON3
using Tenkai.EqEuler2D: h5open, attributes

using Tenkai
using Tenkai.Basis

using EllipsisNotation

import Tenkai: admissibility_tolerance, rho_p_indicator!

(import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
                eigmatrix,
                limit_slope, zhang_shu_flux_fix,
                apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln)

using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
              get_node_vars,
              set_node_vars!,
              nvariables, eachvariable,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, correct_variable!,
              limit_variable_slope

using Tenkai.CartesianGrids: save_mesh_file

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The conservative variables are (rho, rho*v1, rho*v2, E, rho*Y) where Y is the mass fraction
# of the reactant. and Z is the unreacted mass fraction, probably given by 1 - Y, and hopefully
# won't be needed in our implementation.
struct EulerReactive2D <: AbstractEquations{2, 5}
    gamma::Float64 # specific heat ratio
    gamma_minus_one::Float64 # gamma - 1
    inv_gamma_minus_one::Float64 # gamma - 1
    q::Float64 # heat release of reaction
    tK::Float64 # a constant
    tT::Float64 # activation constant temperature
    nvar::Int64
    name::String
    numfluxes::Dict{String, Function}
end

#--------------------------------------------------------------------------------------------------

# Extending the flux function
@inline @inbounds function flux(x, y, u, eq::EulerReactive2D, orientation::Integer)
    @unpack gamma_minus_one = eq
    rho, rho_v1, rho_v2, rho_e, rho_Y = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = pressure(eq, u)
    if orientation == 1
        F1 = rho_v1
        F2 = rho_v1 * v1 + p
        F3 = rho_v1 * v2
        F4 = (rho_e + p) * v1
        F5 = rho_Y * v1
        return SVector(F1, F2, F3, F4, F5)
    else
        G1 = rho_v2
        G2 = rho_v2 * v1
        G3 = rho_v2 * v2 + p
        G4 = (rho_e + p) * v2
        G5 = rho_Y * v2
        return SVector(G1, G2, G3, G4, G5)
    end
end

@inline @inbounds flux(U, eq::EulerReactive2D, orientation::Integer) = flux(1.0, 1.0, U,
                                                                            eq,
                                                                            orientation)

# Extending the flux function
@inline @inbounds function flux(x, y, U, eq::EulerReactive2D)
    return flux(x, y, U, eq, 1), flux(x, y, U, eq, 2)
end

@inline @inbounds flux(U, eq::EulerReactive2D) = flux(1.0, 1.0, U, eq)

function prim2con(eq::EulerReactive2D, u)
    @unpack q, inv_gamma_minus_one = eq
    rho, v1, v2, p, Y = u
    E = p * inv_gamma_minus_one + 0.5 * rho * (v1^2 + v2^2) + rho * q * Y
    return SVector(rho, rho * v1, rho * v2, E, rho * Y)
end

function prim2con!(eq::EulerReactive2D, u)
    con = prim2con(eq, u)
    u .= con
    return nothing
end

function con2prim(eq::EulerReactive2D, u)
    @unpack gamma_minus_one, q = eq
    rho, rho_v1, rho_v2, _, rho_Y = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = pressure(eq, u)
    Y = rho_Y / rho
    return SVector(rho, v1, v2, p, Y)
end

function con2prim!(eq::EulerReactive2D, prim, u)
    u .= con2prim(eq, prim)
    return nothing
end

function density(eq::EulerReactive2D, u)
    return u[1]
end

function pressure(eq::EulerReactive2D, u)
    @unpack gamma_minus_one, q = eq
    return gamma_minus_one * (u[4] - 0.5 * (u[2]^2 + u[3]^2) / u[1] - u[5] * q)
end

function reactant_mass(eq::EulerReactive2D, u)
    return u[5]
end

function is_admissible(eq::EulerReactive2D, u)
    return density(eq, u) > 0.0 && pressure(eq, u) > 0.0 && reactant_mass(eq, u) >= 0.0
end

function max_abs_eigen_value(eq::EulerReactive2D, orientation, u)
    @unpack gamma, q = eq
    rho = u[1]
    p = pressure(eq, u)
    v = u[1 + orientation] / rho
    c = sqrt(gamma * p / rho)
    # Source = Wang, Zhang, Shu, Ning (2013) - Robust high order DG for 2D gaseous detonations
    return abs(v) + c
end

function compute_time_step(eq::EulerReactive2D, problem, grid, aux, op, cfl, u1, ua)
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
        sx, sy = max_abs_eigen_value(eq, 1, u_node),
                 max_abs_eigen_value(eq, 2, u_node)
        den = max(den, sx / dx[el_x] + sy / dy[el_y] + 1e-12)
    end

    dt = cfl / den
    return dt
    end # timer
end

#-------------------------------------------------------------------------------
# Numerical Fluxes
#-------------------------------------------------------------------------------
function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::EulerReactive2D, dir)
    cl, cr = max_abs_eigen_value(eq, dir, ual), max_abs_eigen_value(eq, dir, uar)
    c = max(cl, cr)
    return 0.5 * (Fl + Fr) - 0.5 * c * (Ur - Ul)
end

#------------------------------------------------------------------------------
# Limiters
#------------------------------------------------------------------------------

function apply_bound_limiter!(eq::EulerReactive2D, grid, scheme, param, op, ua, u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end

    @timeit aux.timer "Bound limiter" begin
    #! format: noindent

    @unpack eps = param
    correct_variable!(eq, density, op, aux, grid, u1, ua, eps)
    correct_variable!(eq, pressure, op, aux, grid, u1, ua, eps)
    correct_variable!(eq, reactant_mass, op, aux, grid, u1, ua, eps)

    return nothing
    end # timer
end

function eigmatrix(eq::EulerReactive2D, U)
    Id = SMatrix{5, 5}(1.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 1.0)
    return Id, Id, Id, Id
end

function Tenkai.apply_tvb_limiter!(eq::EulerReactive2D, problem, scheme, grid, param,
                                   op,
                                   ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack dx, dy = grid
    @unpack tvbM, cache, beta = scheme.limiter
    @unpack nvar = eq
    nd = length(wg)

    refresh!(u) = fill!(u, zero(eltype(u)))
    # Pre-allocate for each thread

    beta = limiter.beta
    beta_ = beta / 2.0

    # Loop over cells
    for ij in CartesianIndices((1:nx, 1:ny))
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
            dulm[n] = minmod(beta_ * char_dul[n], beta_ * char_dual[n],
                             beta_ * char_duar[n], Mdx2)
            durm[n] = minmod(beta_ * char_dur[n], beta_ * char_dual[n],
                             beta_ * char_duar[n], Mdx2)
            dudm[n] = minmod(beta_ * char_dud[n], beta_ * char_duad[n],
                             beta_ * char_duau[n], Mdy2)
            duum[n] = minmod(beta_ * char_duu[n], beta_ * char_duad[n],
                             beta_ * char_duau[n], Mdy2)
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

@inbounds @inline function rho_p_indicator!(un, eq::EulerReactive2D)
    nd_p2 = size(un, 2) # nd + 2
    for iy in 1:nd_p2, ix in 1:nd_p2 # loop over dofs and faces
        u_node = get_node_vars(un, eq, ix, iy)
        p = pressure(eq, u_node)
        un[1, ix, iy] *= p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

function zhang_shu_flux_fix(eq::EulerReactive2D,
                            uprev,    # Solution at previous time level
                            ulow,     # low order update
                            Fn,       # Blended flux candidate
                            fn_inner, # Inner part of flux
                            fn,       # low order flux
                            c)
    uhigh = uprev - c * (Fn - fn_inner) # First candidate for high order update
    ρ_low, ρ_high = density(eq, ulow), density(eq, uhigh)
    eps = 0.1 * ρ_low
    ratio = abs(eps - ρ_low) / (abs(ρ_high - ρ_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Second candidate for flux
    end

    uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
    p_low, p_high = pressure(eq, ulow), pressure(eq, uhigh)
    eps = 0.1 * p_low
    ratio = abs(eps - p_low) / (abs(p_high - p_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Final flux
    end

    uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
    p_low, p_high = reactant_mass(eq, ulow), reactant_mass(eq, uhigh)
    eps = 0.1 * p_low
    ratio = abs(eps - p_low) / (abs(p_high - p_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Final flux
    end

    return Fn
end

function Tenkai.limit_slope(eq::EulerReactive2D, slope, ufl, u_star_ll, ufr, u_star_rr,
                            ue, xl, xr, el_x = nothing, el_y = nothing)

    # The MUSCL-Hancock scheme is guaranteed to be admissibility preserving if
    # slope is chosen so that
    # u_star_l = ue + 2.0*slope*xl, u_star_r = ue+2.0*slope*xr are admissible
    # ue is already admissible and we know we can find sequences of thetas
    # to make theta*u_star_l+(1-theta)*ue is admissible.
    # This is equivalent to replacing u_star_l by
    # u_star_l = ue + 2.0*theta*s*xl.
    # Thus, we simply have to update the slope by multiplying by theta.

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, density, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, pressure, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, reactant_mass, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    ufl = ue + slope * xl
    ufr = ue + slope * xr

    return ufl, ufr, slope
end

# TODO - Delete this!
# function update_ghost_values_lwfr!(problem, scheme, eq::EulerReactive2D,
#                                    grid, aux, op, cache, t, dt,
#                                    scaling_factor = 1.0)
#     @timeit aux.timer "Update ghost values" begin
#     #! format: noindent
#     @unpack Fb, Ub, ua = cache
#     update_ghost_values_periodic!(eq, problem, Fb, Ub)

#     @unpack periodic_x, periodic_y = problem
#     if periodic_x && periodic_y
#         return nothing
#     end

#     nx, ny = grid.size
#     nvar = nvariables(eq)
#     @unpack degree, xg, wg = op
#     nd = degree + 1
#     @unpack dx, dy, xf, yf = grid
#     @unpack boundary_condition, boundary_value = problem
#     left, right, bottom, top = boundary_condition

#     refresh!(u) = fill!(u, zero(eltype(u)))

#     pre_allocated = cache.ghost_cache

#     # Julia bug occuring here. Below, we have unnecessarily named
#     # x1,y1, x2, y2,.... We should have been able to just call them x,y
#     # Otherwise we were getting a type instability and variables were
#     # called Core.Box. This issue is probably related to
#     # https://discourse.julialang.org/t/type-instability-of-nested-function/57007
#     # https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
#     # https://github.com/JuliaLang/julia/issues/15276
#     # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured-1

#     dt_scaled = scaling_factor * dt
#     wg_scaled = scaling_factor * wg

#     # For Dirichlet bc, use upwind flux at faces by assigning both physical
#     # and ghost cells through the bc.
#     if left == dirichlet
#         x1 = xf[1]
#         @threaded for j in 1:ny
#             for k in Base.OneTo(nd)
#                 y1 = yf[j] + xg[k] * dy[j]
#                 ub, fb = pre_allocated[Threads.threadid()]
#                 refresh!.((ub, fb))
#                 for l in Base.OneTo(nd)
#                     tq = t + xg[l] * dt_scaled
#                     ub_value = problem.boundary_value(x1, y1, tq)
#                     fb_value = flux(x1, y1, ub_value, eq, 1)
#                     multiply_add_to_node_vars!(ub, wg_scaled[l], ub_value, eq, 1)
#                     multiply_add_to_node_vars!(fb, wg_scaled[l], fb_value, eq, 1)
#                 end
#                 ub_node = get_node_vars(ub, eq, 1)
#                 fb_node = get_node_vars(fb, eq, 1)
#                 set_node_vars!(Ub, ub_node, eq, k, 2, 0, j)
#                 set_node_vars!(Fb, fb_node, eq, k, 2, 0, j)

#                 # # Put hllc flux values
#                 # Ul = get_node_vars(Ub, eq, k, 2, 0, j)
#                 # Fl = get_node_vars(Fb, eq, k, 2, 0, j)
#                 # Ur = get_node_vars(Ub, eq, k, 1, 1, j)
#                 # Fr = get_node_vars(Fb, eq, k, 1, 1, j)
#                 # ual, uar = get_node_vars(ua, eq, 0, j), get_node_vars(ua, eq, 1, j)
#                 # X = SVector{2}(x1, y1)
#                 # Fn = hllc(X, ual, uar, Fl, Fr, Ul, Ur, eq, 1)
#                 # set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
#                 # set_node_vars!(Fb, Fn     , eq, k, 1, 1, j)

#                 # Purely upwind at boundary
#                 # if abs(y1) < 0.055
#                 set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
#                 set_node_vars!(Fb, fb_node, eq, k, 1, 1, j)
#                 # end
#             end
#         end
#     elseif left in (neumann, reflect)
#         @threaded for j in 1:ny
#             for k in Base.OneTo(nd)
#                 Ub_node = get_node_vars(Ub, eq, k, 1, 1, j)
#                 Fb_node = get_node_vars(Fb, eq, k, 1, 1, j)
#                 set_node_vars!(Ub, Ub_node, eq, k, 2, 0, j)
#                 set_node_vars!(Fb, Fb_node, eq, k, 2, 0, j)
#                 if left == reflect
#                     Ub[2, k, 2, 0, j] *= -1.0 # ρ*u1
#                     Fb[1, k, 2, 0, j] *= -1.0 # ρ*u1
#                     Fb[3, k, 2, 0, j] *= -1.0 # ρ*u1*u2
#                     Fb[4, k, 2, 0, j] *= -1.0 # (ρ_e + p) * u1
#                 end
#             end
#         end
#     else
#         println("Incorrect bc specified at left.")
#         @assert false
#     end

#     if right == dirichlet
#         x2 = xf[nx + 1]
#         @threaded for j in 1:ny
#             for k in Base.OneTo(nd)
#                 y2 = yf[j] + xg[k] * dy[j]
#                 ub, fb = pre_allocated[Threads.threadid()]
#                 refresh!.((ub, fb))
#                 for l in Base.OneTo(nd)
#                     tq = t + xg[l] * dt_scaled
#                     ubvalue = boundary_value(x2, y2, tq)
#                     fbvalue = flux(x2, y2, ubvalue, eq, 1)
#                     multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
#                     multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
#                 end
#                 ub_node = get_node_vars(ub, eq, 1)
#                 fb_node = get_node_vars(fb, eq, 1)
#                 set_node_vars!(Ub, ub_node, eq, k, 1, nx + 1, j)
#                 set_node_vars!(Fb, fb_node, eq, k, 1, nx + 1, j)

#                 # Purely upwind
#                 # set_node_vars!(Ub, ub_node, eq, k, 2, nx, j)
#                 # set_node_vars!(Fb, fb_node, eq, k, 2, nx, j)
#             end
#         end
#     elseif right in (reflect, neumann)
#         @threaded for j in 1:ny
#             for k in Base.OneTo(nd)
#                 Ub_node = get_node_vars(Ub, eq, k, 2, nx, j)
#                 Fb_node = get_node_vars(Fb, eq, k, 2, nx, j)
#                 set_node_vars!(Ub, Ub_node, eq, k, 1, nx + 1, j)
#                 set_node_vars!(Fb, Fb_node, eq, k, 1, nx + 1, j)

#                 if right == reflect
#                     Ub[2, k, 1, nx + 1, j] *= -1.0 # ρ*u1
#                     Fb[1, k, 1, nx + 1, j] *= -1.0 # ρ*u1
#                     Fb[3, k, 1, nx + 1, j] *= -1.0 # ρ*u1*u2
#                     Fb[4, k, 1, nx + 1, j] *= -1.0 # (ρ_e + p) * u1
#                 end
#             end
#         end
#     else
#         println("Incorrect bc specified at right.")
#         @assert false
#     end

#     if bottom == dirichlet
#         y3 = yf[1]
#         @threaded for i in 1:nx
#             for k in Base.OneTo(nd)
#                 x3 = xf[i] + xg[k] * dx[i]
#                 ub, fb = pre_allocated[Threads.threadid()]
#                 refresh!.((ub, fb))
#                 for l in Base.OneTo(nd)
#                     tq = t + xg[l] * dt_scaled
#                     ubvalue = boundary_value(x3, y3, tq)
#                     fbvalue = flux(x3, y3, ubvalue, eq, 2)
#                     multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
#                     multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
#                 end
#                 ub_node = get_node_vars(ub, eq, 1)
#                 fb_node = get_node_vars(fb, eq, 1)
#                 set_node_vars!(Ub, ub_node, eq, k, 4, i, 0)
#                 set_node_vars!(Fb, fb_node, eq, k, 4, i, 0)

#                 # Purely upwind

#                 # set_node_vars!(Ub, ub, eq, k, 3, i, 1)
#                 # set_node_vars!(Fb, fb, eq, k, 3, i, 1)
#             end
#         end
#     elseif bottom in (reflect, neumann)
#         @threaded for i in 1:nx
#             for k in Base.OneTo(nd)
#                 Ub_node = get_node_vars(Ub, eq, k, 3, i, 1)
#                 Fb_node = get_node_vars(Fb, eq, k, 3, i, 1)
#                 set_node_vars!(Ub, Ub_node, eq, k, 4, i, 0)
#                 set_node_vars!(Fb, Fb_node, eq, k, 4, i, 0)
#                 if bottom == reflect
#                     Ub[3, k, 4, i, 0] *= -1.0 # ρ * vel_y
#                     Fb[1, k, 4, i, 0] *= -1.0 # ρ * vel_y
#                     Fb[2, k, 4, i, 0] *= -1.0 # ρ * vel_x * vel_y
#                     Fb[4, k, 4, i, 0] *= -1.0 # (ρ_e + p) * vel_y
#                 end
#             end
#         end
#     elseif periodic_y
#         nothing
#     else
#         @assert typeof(bottom) <: Tuple{Any, Any, Any}
#         bc! = bottom[1]
#         bc!(grid, eq, op, Fb, Ub, aux)
#     end

#     if top == dirichlet
#         y4 = yf[ny + 1]
#         @threaded for i in 1:nx
#             for k in Base.OneTo(nd)
#                 x4 = xf[i] + xg[k] * dx[i]
#                 ub, fb = pre_allocated[Threads.threadid()]
#                 refresh!.((ub, fb))
#                 for l in Base.OneTo(nd)
#                     tq = t + xg[l] * dt_scaled
#                     ubvalue = boundary_value(x4, y4, tq)
#                     fbvalue = flux(x4, y4, ubvalue, eq, 2)
#                     multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
#                     multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
#                 end
#                 ub_node = get_node_vars(ub, eq, 1)
#                 fb_node = get_node_vars(fb, eq, 1)
#                 set_node_vars!(Ub, ub_node, eq, k, 3, i, ny + 1)
#                 set_node_vars!(Fb, fb_node, eq, k, 3, i, ny + 1)

#                 # Purely upwind
#                 # set_node_vars!(Ub, ub_node, eq, k, 4, i, ny)
#                 # set_node_vars!(Fb, fb_node, eq, k, 4, i, ny)
#             end
#         end
#     elseif top in (reflect, neumann)
#         @threaded for i in 1:nx
#             for k in Base.OneTo(nd)
#                 Ub_node = get_node_vars(Ub, eq, k, 4, i, ny)
#                 Fb_node = get_node_vars(Fb, eq, k, 4, i, ny)
#                 set_node_vars!(Ub, Ub_node, eq, k, 3, i, ny + 1)
#                 set_node_vars!(Fb, Fb_node, eq, k, 3, i, ny + 1)
#                 if top == reflect
#                     Ub[3, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
#                     Fb[1, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
#                     Fb[2, k, 3, i, ny + 1] *= -1.0 # ρ * vel_x * vel_y
#                     Fb[4, k, 3, i, ny + 1] *= -1.0 # (ρ_e + p) * vel_y
#                 end
#             end
#         end
#     elseif periodic_y
#         nothing
#     else
#         @assert false "Incorrect bc specific at top"
#     end

#     return nothing
#     end # timer
# end

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

varnames(eq::EulerReactive2D) = eq.varnames
varnames(eq::EulerReactive2D, i::Int) = eq.varnames[i]

function initialize_plot(eq::EulerReactive2D, op, grid, problem, scheme, timer,
                         u1, ua)
    return nothing
end

function write_poly(eq::EulerReactive2D, grid, op, u1, fcount)
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

function write_soln!(base_name, fcount, iter, time, dt, eq::EulerReactive2D, grid,
                     problem, param, op, z, u1, aux, ndigits = 3)
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
    prim = @views copy(z[:, 1:nx, 1:ny])
    exact_data = exact.(xy)
    for j in 1:ny, i in 1:nx
        @views con2prim!(eq, z[:, i, j], prim[:, i, j])
    end
    density_arr = prim[1, 1:nx, 1:ny]
    velx_arr = prim[2, 1:nx, 1:ny]
    vely_arr = prim[3, 1:nx, 1:ny]
    pres_arr = prim[4, 1:nx, 1:ny]
    raction_mass_arr = prim[4, 1:nx, 1:ny]
    vtk["sol"] = density_arr
    vtk["Density"] = density_arr
    vtk["Velocity_x"] = velx_arr
    vtk["Velocity_y"] = vely_arr
    vtk["Pressure"] = pres_arr
    vtk["Reaction mass"] = raction_mass_arr
    for j in 1:ny, i in 1:nx
        @views con2prim!(eq, exact_data[i, j], prim[:, i, j])
    end
    @views vtk["Exact Density"] = prim[1, 1:nx, 1:ny]
    @views vtk["Exact Velocity_x"] = prim[2, 1:nx, 1:ny]
    @views vtk["Exact Velocity_y"] = prim[3, 1:nx, 1:ny]
    @views vtk["Exact Pressure"] = prim[4, 1:nx, 1:ny]
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
    element_variables[:pressure] = vec(pres_arr)
    # element_variables[:indicator_shock_capturing] = vec(aux.blend.cache.alpha[1:nx,1:ny])
    filename = save_solution_file(u1, time, dt, iter, grid, eq, op,
                                  element_variables) # Save h5 file
    println("Wrote ", filename)
    return fcount
    end # timer
end

function save_solution_file(u_, time, dt, iter,
                            mesh,
                            equations, op,
                            element_variables = Dict{Symbol, Any}();
                            system = "")
    # Filename without extension based on current time step
    output_directory = "output"
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%06d.h5", iter))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%06d.h5", system, iter))
    end

    solution_variables(u) = con2prim(equations, u) # For broadcasting

    nx, ny = mesh.size
    u = @view u_[:, :, :, 1:nx, 1:ny] # Don't plot ghost cells

    # Convert to different set of variables if requested
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    # OffsetArray(reinterpret(eltype(ua), con2prim_.(reinterpret(SVector{nvariables(equation), eltype(ua)}, ua))))
    u_static_reinter = reinterpret(SVector{nvariables(equations), eltype(u)}, u)
    data = Array(reinterpret(eltype(u), solution_variables.(u_static_reinter)))

    # Find out variable count by looking at output from `solution_variables` function
    n_vars = size(data, 1)

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = 2
        attributes(file)["equations"] = "2D Euler Equations"
        attributes(file)["polydeg"] = op.degree
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nx * ny
        attributes(file)["mesh_type"] = "StructuredMesh" # For Trixi2Vtk
        attributes(file)["mesh_file"] = "mesh.h5"
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = iter

        # Store each variable of the solution data
        var_names = ("Density", "Velocity x", "Velocity y", "Pressure", "Reaction mass")
        for v in 1:n_vars
            # Convert to 1D array
            file["variables_$v"] = vec(data[v, .., :])

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = var_names[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            file["element_variables_$v"] = element_variable

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

#------------------------------------------------------------

function omega_reactive_euler(eq::EulerReactive2D, u)
    T = pressure(eq, u) / u[1]
    exponent = -eq.tT / T
    coeff = -eq.tK * u[5]
    return coeff * exp(exponent)
end

function source_term_arrhenius(x, u, eq::EulerReactive2D)
    omega = omega_reactive_euler(eq, u)
    return SVector(0.0, 0.0, 0.0, 0.0, omega)
end

function get_equation(gamma, q, tK, tT)
    gamma_minus_one = gamma - 1.0
    inv_gamma_minus_one = 1.0 / gamma_minus_one
    numfluxes = Dict("rusanov" => rusanov)

    nvar = 5
    name = "2d reactive Euler Equations"

    eq = EulerReactive2D(gamma, gamma_minus_one, inv_gamma_minus_one, q, tK, tT, nvar,
                         name,
                         numfluxes)

    return eq
end
end # muladd
end # module
