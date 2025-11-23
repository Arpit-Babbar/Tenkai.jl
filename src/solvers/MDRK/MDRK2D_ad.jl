using ..Tenkai: evaluate, extrapolate,
                get_node_vars, set_node_vars!,
                add_to_node_vars!, subtract_from_node_vars!,
                multiply_add_to_node_vars!, multiply_add_set_node_vars!,
                comp_wise_mutiply_node_vars!

using SimpleUnPack
using TimerOutputs
using Polyester
using MuladdMacro
using OffsetArrays
using StaticArrays
using LoopVectorization

using Tenkai: calc_source, calc_source_t_N34

using Tenkai: @threaded
import Tenkai: extrap_bflux!

import Tenkai: compute_cell_residual_mdrk_1!, compute_cell_residual_mdrk_2!,
               setup_arrays_mdrk

@muladd begin
#! format: noindent

function setup_arrays(grid, scheme::Scheme{<:MDRKEnzymeTower},
                      eq::AbstractEquations{2})
    function gArray(nvar, nx, ny)
        OffsetArray(zeros(RealT, nvar, nx + 2, ny + 2),
                    OffsetArrays.Origin(1, 0, 0))
    end
    function gArray(nvar, n1, n2, nx, ny)
        OffsetArray(zeros(RealT, nvar, n1, n2, nx + 2, ny + 2),
                    OffsetArrays.Origin(1, 1, 1, 0, 0))
    end
    # Allocate memory
    @unpack degree, bflux = scheme
    @unpack bflux_ind = bflux
    @unpack nvar = eq
    nd = degree + 1
    nx, ny = grid.size
    RealT = eltype(grid.xc)
    u1 = gArray(nvar, nd, nd, nx, ny)
    us = gArray(nvar, nd, nd, nx, ny)
    F2 = zeros(RealT, nvar, nd, nd, nx, ny)
    G2 = zeros(RealT, nvar, nd, nd, nx, ny)
    U2 = zeros(RealT, nvar, nd, nd, nx, ny)
    S2 = zeros(RealT, nvar, nd, nd, nx, ny)
    ua = gArray(nvar, nx, ny)
    res = gArray(nvar, nd, nd, nx, ny)
    Fb = gArray(nvar, nd, 4, nx, ny)
    Fb2 = gArray(nvar, nd, 4, nx, ny)
    Ub = gArray(nvar, nd, 4, nx, ny)
    Ub2 = gArray(nvar, nd, 4, nx, ny)

    # Cell residual cache

    nt = Threads.nthreads()

    # Construct `cache_size` number of objects with `constructor`
    # and store them in an SVector
    function alloc(constructor, cache_size)
        SVector{cache_size}(constructor(undef) for _ in Base.OneTo(cache_size))
    end

    # Create the result of `alloc` for each thread. Basically,
    # for each thread, construct `cache_size` number of objects with
    # `constructor` and store them in an SVector
    function alloc_for_threads(constructor, cache_size)
        nt = Threads.nthreads()
        SVector{nt}([alloc(constructor, cache_size) for _ in Base.OneTo(nt)])
    end

    MArr = MArray{Tuple{nvariables(eq), nd, nd}, RealT}
    cell_arrays = alloc_for_threads(MArr, 8)

    MEval = MArray{Tuple{nvariables(eq), nd}, RealT}
    eval_data_big = alloc_for_threads(MEval, 8)

    eval_data = (; eval_data_big)

    # Ghost values cache

    Marr = MArray{Tuple{nvariables(eq), 1}, RealT}

    ghost_cache = alloc_for_threads(Marr, 2)

    # KLUDGE - Rename this to LWFR cache
    cache = (; u1, us, U2, F2, G2, S2, ua, res, Fb, Fb2, Ub, Ub2, eval_data,
             cell_arrays, ghost_cache)
    return cache
end

function get_cfl(eq::AbstractEquations{2}, scheme::Scheme{<:MDRKADSolver}, param)
    os_vector(v) = OffsetArray(v, OffsetArrays.Origin(0))
    @unpack solver, degree, correction_function = scheme
    @unpack cfl_safety_factor, cfl_style = param
    diss = scheme.dissipation
    @assert (degree >= 0&&degree < 5) "Invalid degree"
    cfl_radau = os_vector([1.0, 0.333, 0.170, 0.107, 0.069])
    cfl_g2 = os_vector([1.0, 1.000, 0.333, 0.224, 0.103])
    # Reduce this cfl by a small amount
    if correction_function == "radau"
        return cfl_safety_factor * cfl_radau[degree]
    elseif correction_function == "g2"
        return cfl_safety_factor * cfl_g2[degree]
    else
        println("get_cfl: unknown correction function")
        @assert false
    end
end

@inline @inbounds function eval_bflux_mdrk_ad!(eq::AbstractEquations{2}, grid,
                                               cell_data, eval_data_big,
                                               xg, Vl, Vr, F, G, F2, G2, Fb, Fb2, aux)
    @unpack nvar = eq
    nd = length(xg)

    # Load pre-allocated arrays
    u, el_x, el_y = cell_data

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    ul, ur, uu, ud, utl, utr, utd, utu = eval_data_big
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for i in 1:nd
        x, y = xl + dx * xg[i], yd + dy * xg[i]

        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)

        utl_node = get_node_vars(utl, eq, i)
        utr_node = get_node_vars(utr, eq, i)
        utd_node = get_node_vars(utd, eq, i)
        utu_node = get_node_vars(utu, eq, i)

        fl = flux(xl, y, ul_node, eq, 1)
        fr = flux(xr, y, ur_node, eq, 1)
        gd = flux(x, yd, ud_node, eq, 2)
        gu = flux(x, yu, uu_node, eq, 2)

        # KLUDGE - Indices order needs to be changed!!
        set_node_vars!(Fb, 0.5 * fl, eq, i, 1)
        set_node_vars!(Fb2, fl, eq, i, 1)
        set_node_vars!(Fb, 0.5 * fr, eq, i, 2)
        set_node_vars!(Fb2, fr, eq, i, 2)
        set_node_vars!(Fb, 0.5 * gd, eq, i, 3)
        set_node_vars!(Fb2, gd, eq, i, 3)
        set_node_vars!(Fb, 0.5 * gu, eq, i, 4)
        set_node_vars!(Fb2, gu, eq, i, 4)

        ftl_node = derivative_bundle(flux_x, (ul_node, utl_node))
        ftr_node = derivative_bundle(flux_x, (ur_node, utr_node))

        multiply_add_to_node_vars!(Fb, 0.125, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 0.125, ftr_node, eq, i, 2)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, ftr_node, eq, i, 2)

        gtd_node = derivative_bundle(flux_y, (ud_node, utd_node))
        gtu_node = derivative_bundle(flux_y, (uu_node, utu_node))

        multiply_add_to_node_vars!(Fb, 0.125, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 0.125, gtu_node, eq, i, 4)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, gtu_node, eq, i, 4)
    end
end

@inbounds @inline function eval_bflux_mdrk_ad!(eq::AbstractEquations{2}, grid,
                                               cell_data, eval_data_big,
                                               xg, Vl, Vr, F, G, Fb, Fb2, aux,
                                               ::Nothing)
    @unpack nvar = eq
    nd = length(xg)

    ul, ur, ud, uu, utl, utr, utd, utu = eval_data_big

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    for i in 1:nd

        # KLUDGE - Indices order needs to be changed to avoid cache misses
        Fb2l = get_node_vars(Fb2, eq, i, 1)
        Fb2r = get_node_vars(Fb2, eq, i, 2)
        Fb2d = get_node_vars(Fb2, eq, i, 3)
        Fb2u = get_node_vars(Fb2, eq, i, 4)

        set_node_vars!(Fb, Fb2l, eq, i, 1)
        set_node_vars!(Fb, Fb2r, eq, i, 2)
        set_node_vars!(Fb, Fb2d, eq, i, 3)
        set_node_vars!(Fb, Fb2u, eq, i, 4)

        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)

        utl_node = get_node_vars(utl, eq, i)
        utr_node = get_node_vars(utr, eq, i)
        utd_node = get_node_vars(utd, eq, i)
        utu_node = get_node_vars(utu, eq, i)

        ftl_node = derivative_bundle(flux_x, (ul_node, utl_node))
        ftr_node = derivative_bundle(flux_x, (ur_node, utr_node))

        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, ftr_node, eq, i, 2)

        gtd_node = derivative_bundle(flux_y, (ud_node, utd_node))
        gtu_node = derivative_bundle(flux_y, (uu_node, utu_node))

        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, gtu_node, eq, i, 4)
    end
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=3 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_mdrk_1!(eq::AbstractEquations{2}, grid, op,
                                       problem, scheme::Scheme{<:MDRKEnzymeTower},
                                       aux, t, dt, u1, res, Fb, Ub,
                                       cache)
    @unpack nvar = eq
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size

    @unpack source_terms = problem

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    # Select boundary flux
    @unpack blend_cell_residual! = aux.blend.subroutines
    get_dissipation_node_vars = scheme.dissipation
    @unpack F2, G2, U2, S2, Fb2, Ub2, cell_arrays, eval_data = cache
    refresh!.((res, Fb, Fb2, Ub, Ub2)) # Reset previously used variables to zero

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        # Some local variables
        id = Threads.threadid()
        F, G, F2_loc, G2_loc, ut, U, U2_loc, S = cell_arrays[id]

        eval_data_big = eval_data.eval_data_big[id]
        refresh!.(eval_data_big)
        ul, ur, uu, ud, utl, utr, utd, utu = eval_data_big

        refresh!(ut)

        u1_ = @view u1[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            flux1, flux2 = flux(x, y, u_node, eq)
            set_node_vars!(F, 0.5 * flux1, eq, i, j)
            set_node_vars!(G, 0.5 * flux2, eq, i, j)
            set_node_vars!(F2_loc, flux1, eq, i, j)
            set_node_vars!(G2_loc, flux2, eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] * f[i,j] (sum over i)
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(ut, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
            set_node_vars!(U, 0.5 * u_node, eq, i, j)
            set_node_vars!(U2_loc, u_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            x = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            s_node = calc_source(u_node, x, t, source_terms, eq)
            set_node_vars!(S, 0.5 * s_node, eq, i, j)
            set_node_vars!(S2, s_node, eq, i, j, el_x, el_y)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)

            # For efficient computation of time averaged flux at interfaces
            # ul = u * V
            # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
            multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)

            # ud = u * V
            # ud[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
            multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

            multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, j)
            multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, j)
            multiply_add_to_node_vars!(utd, Vl[j], ut_node, eq, i)
            multiply_add_to_node_vars!(utu, Vr[j], ut_node, eq, i)

            ft = derivative_bundle(flux_x, (u_node, ut_node))
            gt = derivative_bundle(flux_y, (u_node, ut_node))

            multiply_add_to_node_vars!(F, 0.125, ft, eq, i, j)
            multiply_add_to_node_vars!(G, 0.125, gt, eq, i, j)
            multiply_add_to_node_vars!(U, 0.125, ut_node, eq, i, j)
            multiply_add_to_node_vars!(F2_loc, 1.0 / 6.0, ft, eq, i, j)
            multiply_add_to_node_vars!(G2_loc, 1.0 / 6.0, gt, eq, i, j)
            multiply_add_to_node_vars!(U2_loc, 1.0 / 6.0, ut_node, eq, i, j)
            F_node = get_node_vars(F, eq, i, j)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq,
                                           ii, j, el_x, el_y)
            end
            G_node = get_node_vars(G, eq, i, j)
            for jj in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamy * D1[jj, j], G_node, eq,
                                           i, jj, el_x, el_y)
            end

            U_ = get_dissipation_node_vars(u1_, U, eq, i, j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(Ub, Vl[i], U_, eq, j, 1, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[i], U_, eq, j, 2, el_x, el_y)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(Ub, Vl[j], U_, eq, i, 3, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[j], U_, eq, i, 4, el_x, el_y)

            # TODO - Source term computation to be added
            # st = calc_source_t_N34(up, um,
            #                        X, t, dt, source_terms, eq)
            # multiply_add_to_node_vars!(S, 0.125, st, eq, i, j)

            # multiply_add_to_node_vars!(S2, 1.0/6.0, st, eq, i, j, el_x, el_y)

            S_node = get_node_vars(S, eq, i, j)

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, j, el_x, el_y)
        end

        @turbo F2[:, :, :, el_x, el_y] .= F2_loc
        @turbo G2[:, :, :, el_x, el_y] .= G2_loc
        @turbo U2[:, :, :, el_x, el_y] .= U2_loc

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res,
                             0.5)
        @views cell_data = (u1_, el_x, el_y)
        @views eval_bflux_mdrk_ad!(eq, grid, cell_data, eval_data_big, xg, Vl, Vr,
                                   F, G, F2_loc, G2_loc, Fb[:, :, :, el_x, el_y],
                                   Fb2[:, :, :, el_x, el_y], aux)
        # @views extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #    F, G, F2_loc, G2_loc, Fb[:, :, :, el_x, el_y], Fb2[:, :, :, el_x, el_y], aux)
    end
    return nothing
end

function compute_cell_residual_mdrk_2!(eq::AbstractEquations{2}, grid, op, problem,
                                       scheme::Scheme{<:MDRKEnzymeTower},
                                       aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack nvar = eq
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size

    @unpack source_terms = problem

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    # Select boundary flux
    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack us, F2, G2, U2, Fb2, S2, cell_arrays, eval_data = cache
    refresh!.((res, Fb, Ub)) # Reset previously used variables to zero

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        # Some local variables
        id = Threads.threadid()
        F, G, ust, U = cell_arrays[id]

        eval_data_big = eval_data.eval_data_big[id]
        refresh!.(eval_data_big)
        ul, ur, ud, uu, utl, utr, utd, utu = eval_data_big

        F2_ = @view F2[:, :, :, el_x, el_y]
        G2_ = @view G2[:, :, :, el_x, el_y]
        U2_ = @view U2[:, :, :, el_x, el_y]
        @turbo F .= F2_
        @turbo G .= G2_
        @turbo U .= U2_

        refresh!(ust)

        u1_ = @view u1[:, :, :, el_x, el_y]
        us_ = @view us[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            us_node = get_node_vars(us_, eq, i, j)
            flux1, flux2 = flux(x, y, us_node, eq)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] * f[i,j] (sum over i)
                multiply_add_to_node_vars!(ust, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(ust, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
        end

        for j in 1:nd, i in 1:nd
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)
            u_node = get_node_vars(us, eq, i, j, el_x, el_y)
            s_node = calc_source(u_node, X, t + 0.5 * dt, source_terms, eq)
            multiply_add_to_node_vars!(ust, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            us_node = get_node_vars(us_, eq, i, j)
            ust_node = get_node_vars(ust, eq, i, j)

            multiply_add_to_node_vars!(ul, Vl[i], us_node, eq, j)
            multiply_add_to_node_vars!(ur, Vr[i], us_node, eq, j)
            multiply_add_to_node_vars!(ud, Vl[j], us_node, eq, i)
            multiply_add_to_node_vars!(uu, Vr[j], us_node, eq, i)

            multiply_add_to_node_vars!(utl, Vl[i], ust_node, eq, j)
            multiply_add_to_node_vars!(utr, Vr[i], ust_node, eq, j)
            multiply_add_to_node_vars!(utd, Vl[j], ust_node, eq, i)
            multiply_add_to_node_vars!(utu, Vr[j], ust_node, eq, i)

            ft = derivative_bundle(flux_x, (us_node, ust_node))
            gt = derivative_bundle(flux_y, (us_node, ust_node))

            multiply_add_to_node_vars!(F, 1.0 / 3.0, ft, eq, i, j)
            multiply_add_to_node_vars!(G, 1.0 / 3.0, gt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 3.0, ust_node, eq, i, j)

            # TODO - source term computation to be added
            # st = calc_source_t_N34(us_node, up, upp, um, umm, X, t+0.5*dt, dt, source_terms, eq)
            # multiply_add_to_node_vars!(S2, 1.0/3.0, st, eq, i, j, el_x, el_y)

            S_node = get_node_vars(S2, eq, i, j, el_x, el_y)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, j, el_x, el_y)

            F_node = get_node_vars(F, eq, i, j)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq,
                                           ii, j, el_x, el_y)
            end
            G_node = get_node_vars(G, eq, i, j)
            for jj in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamy * D1[jj, j], G_node, eq,
                                           i, jj, el_x, el_y)
            end

            U_ = get_dissipation_node_vars(u1_, U, eq, i, j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(Ub, Vl[i], U_, eq, j, 1, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[i], U_, eq, j, 2, el_x, el_y)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(Ub, Vl[j], U_, eq, i, 3, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[j], U_, eq, i, 4, el_x, el_y)
        end

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        cell_data = (el_x, el_y)
        @views eval_bflux_mdrk_ad!(eq, grid, cell_data, eval_data_big, xg, Vl, Vr,
                                   F, G, Fb[:, :, :, el_x, el_y],
                                   Fb2[:, :, :, el_x, el_y],
                                   aux, nothing)
    end
    return nothing
end
end # muladd
