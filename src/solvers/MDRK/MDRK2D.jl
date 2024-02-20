using ..Tenkai: periodic, dirichlet, neumann, reflect,
                evaluate, extrapolate,
                get_node_vars, set_node_vars!,
                add_to_node_vars!, subtract_from_node_vars!,
                multiply_add_to_node_vars!, multiply_add_set_node_vars!,
                comp_wise_mutiply_node_vars!

using UnPack
using TimerOutputs
using Polyester
using MuladdMacro
using OffsetArrays
using StaticArrays
using LoopVectorization

using Tenkai.LWFR: calc_source, calc_source_t_N12, calc_source_t_N34,
                   calc_source_tt_N23, calc_source_tt_N4, calc_source_ttt_N34,
                   calc_source_tttt_N4

using ..FR: @threaded
using ..FR2D: update_ghost_values_periodic!, update_ghost_values_fn_blend!
import Tenkai.LWFR2D: extrap_bflux!

using ..Equations: AbstractEquations, nvariables, eachvariable
@muladd begin
#! format: noindent

@inline @inbounds function refresh!(u)
    @turbo u .= zero(eltype(u))
end

function setup_arrays_mdrk(grid, scheme, eq::AbstractEquations{2})
    function gArray(nvar, nx, ny)
        OffsetArray(zeros(nvar, nx + 2, ny + 2),
                    OffsetArrays.Origin(1, 0, 0))
    end
    function gArray(nvar, n1, n2, nx, ny)
        OffsetArray(zeros(nvar, n1, n2, nx + 2, ny + 2),
                    OffsetArrays.Origin(1, 1, 1, 0, 0))
    end
    # Allocate memory
    @unpack degree, bflux = scheme
    @unpack bflux_ind = bflux
    @unpack nvar = eq
    nd = degree + 1
    nx, ny = grid.size
    u1 = gArray(nvar, nd, nd, nx, ny)
    us = gArray(nvar, nd, nd, nx, ny)
    F2 = zeros(nvar, nd, nd, nx, ny)
    G2 = zeros(nvar, nd, nd, nx, ny)
    U2 = zeros(nvar, nd, nd, nx, ny)
    S2 = zeros(nvar, nd, nd, nx, ny)
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

    MArr = MArray{Tuple{nvariables(eq), nd, nd}, Float64}
    cell_arrays = alloc_for_threads(MArr, 8)

    MEval = MArray{Tuple{nvariables(eq), nd}, Float64}
    eval_data_big = alloc_for_threads(MEval, 20)

    eval_data = (; eval_data_big)

    # Ghost values cache

    Marr = MArray{Tuple{nvariables(eq), 1}, Float64}

    ghost_cache = alloc_for_threads(Marr, 2)

    # KLUDGE - Rename this to LWFR cache
    cache = (; u1, us, U2, F2, G2, S2, ua, res, Fb, Fb2, Ub, Ub2, eval_data,
             cell_arrays, ghost_cache)
    return cache
end

@inline @inbounds function eval_bflux_mdrk!(eq::AbstractEquations{2}, grid, cell_data, eval_data_big,
                          xg, Vl, Vr, F, G, F2, G2, Fb, Fb2, aux)
    @unpack nvar = eq
    nd = length(xg)

    # Load pre-allocated arrays
    u, el_x, el_y = cell_data

    ul, ur, uu, ud, upl, upr, upd, upu, uml, umr, umd, umu = eval_data_big
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for i in 1:nd
        x, y = xl + dx * xg[i], yd + dy * xg[i]

        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)

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

        upl_node = get_node_vars(upl, eq, i)
        upr_node = get_node_vars(upr, eq, i)
        upd_node = get_node_vars(upd, eq, i)
        upu_node = get_node_vars(upu, eq, i)

        fpl = flux(xl, y, upl_node, eq, 1)
        fpr = flux(xr, y, upr_node, eq, 1)
        gpd = flux(x, yd, upd_node, eq, 2)
        gpu = flux(x, yu, upu_node, eq, 2)

        uml_node = get_node_vars(uml, eq, i)
        umr_node = get_node_vars(umr, eq, i)
        umd_node = get_node_vars(umd, eq, i)
        umu_node = get_node_vars(umu, eq, i)

        fml = flux(xl, y, uml_node, eq, 1)
        fmr = flux(xr, y, umr_node, eq, 1)
        gmd = flux(x, yd, umd_node, eq, 2)
        gmu = flux(x, yu, umu_node, eq, 2)

        ftl_node = 0.5 * (fpl - fml)
        ftr_node = 0.5 * (fpr - fmr)

        multiply_add_to_node_vars!(Fb, 0.125, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 0.125, ftr_node, eq, i, 2)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, ftr_node, eq, i, 2)

        gtd_node = 0.5 * (gpd - gmd)
        gtu_node = 0.5 * (gpu - gmu)

        multiply_add_to_node_vars!(Fb, 0.125, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 0.125, gtu_node, eq, i, 4)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, gtu_node, eq, i, 4)
    end
end

@inbounds @inline function eval_bflux_mdrk!(eq::AbstractEquations{2}, grid, cell_data, eval_data_big,
                          xg, Vl, Vr, F, G, Fb, Fb2, aux, ::Nothing)
    @unpack nvar = eq
    nd = length(xg)

    # Load pre-allocated arrays
    el_x, el_y = cell_data

    (_, _, _, _, upl, uppl, upr, uppr, upd, uppd, upu, uppu, uml, umml,
    umr, ummr, umd, ummd, umu, ummu) = eval_data_big
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for i in 1:nd
        x, y = xl + dx * xg[i], yd + dy * xg[i]

        # KLUDGE - Indices order needs to be changed to avoid cache misses
        Fb2l = get_node_vars(Fb2, eq, i, 1)
        Fb2r = get_node_vars(Fb2, eq, i, 2)
        Fb2d = get_node_vars(Fb2, eq, i, 3)
        Fb2u = get_node_vars(Fb2, eq, i, 4)

        set_node_vars!(Fb, Fb2l, eq, i, 1)
        set_node_vars!(Fb, Fb2r, eq, i, 2)
        set_node_vars!(Fb, Fb2d, eq, i, 3)
        set_node_vars!(Fb, Fb2u, eq, i, 4)

        upl_node = get_node_vars(upl, eq, i)
        upr_node = get_node_vars(upr, eq, i)
        upd_node = get_node_vars(upd, eq, i)
        upu_node = get_node_vars(upu, eq, i)

        fpl = flux(xl, y, upl_node, eq, 1)
        fpr = flux(xr, y, upr_node, eq, 1)
        gpd = flux(x, yd, upd_node, eq, 2)
        gpu = flux(x, yu, upu_node, eq, 2)

        uml_node = get_node_vars(uml, eq, i)
        umr_node = get_node_vars(umr, eq, i)
        umd_node = get_node_vars(umd, eq, i)
        umu_node = get_node_vars(umu, eq, i)

        fml = flux(xl, y, uml_node, eq, 1)
        fmr = flux(xr, y, umr_node, eq, 1)
        gmd = flux(x, yd, umd_node, eq, 2)
        gmu = flux(x, yu, umu_node, eq, 2)

        uppl_node = get_node_vars(uppl, eq, i)
        uppr_node = get_node_vars(uppr, eq, i)
        uppd_node = get_node_vars(uppd, eq, i)
        uppu_node = get_node_vars(uppu, eq, i)

        fppl = flux(xl, y, uppl_node, eq, 1)
        fppr = flux(xr, y, uppr_node, eq, 1)
        gppd = flux(x, yd, uppd_node, eq, 2)
        gppu = flux(x, yu, uppu_node, eq, 2)

        umml_node = get_node_vars(umml, eq, i)
        ummr_node = get_node_vars(ummr, eq, i)
        ummd_node = get_node_vars(ummd, eq, i)
        ummu_node = get_node_vars(ummu, eq, i)

        fmml = flux(xl, y, umml_node, eq, 1)
        fmmr = flux(xr, y, ummr_node, eq, 1)
        gmmd = flux(x, yd, ummd_node, eq, 2)
        gmmu = flux(x, yu, ummu_node, eq, 2)

        ftl_node = 1.0 / 12.0 * (-fppl + 8.0 * fpl - 8.0 * fml + fmml)
        ftr_node = 1.0 / 12.0 * (-fppr + 8.0 * fpr - 8.0 * fmr + fmmr)

        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, ftr_node, eq, i, 2)

        gtd_node = 1.0 / 12.0 * (-gppd + 8.0 * gpd - 8.0 * gmd + gmmd)
        gtu_node = 1.0 / 12.0 * (-gppu + 8.0 * gpu - 8.0 * gmu + gmmu)

        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, gtu_node, eq, i, 4)
    end
end

@inbounds @inline function extrap_bflux!(eq::AbstractEquations{2}, grid, cell_data, eval_data,
                       xg, Vl, Vr, F, G, F2, G2, Fb, Fb2, aux, n = nothing)
    @unpack nvar = eq
    nd = length(xg)
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        F_node = get_node_vars(F, eq, i, j)
        # Fb = FT * V
        # Fb[j] += ∑_i F[i,j] * V[i]
        multiply_add_to_node_vars!(Fb, Vl[i], F_node, eq, j, 1)
        multiply_add_to_node_vars!(Fb, Vr[i], F_node, eq, j, 2)

        G_node = get_node_vars(G, eq, i, j)
        # Fb = g * V
        # Fb[i] += ∑_j g[i,j]*V[j]
        multiply_add_to_node_vars!(Fb, Vl[j], G_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, Vr[j], G_node, eq, i, 4)
    end
end

@inbounds @inline function extrap_bflux!(eq::AbstractEquations{2}, grid, cell_data, eval_data,
                       xg, Vl, Vr, F, G, Fb, Fb2, aux, n = nothing)
    @unpack nvar = eq
    nd = length(xg)
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        F_node = get_node_vars(F, eq, i, j)
        # Fb = FT * V
        # Fb[j] += ∑_i F[i,j] * V[i]
        multiply_add_to_node_vars!(Fb, Vl[i], F_node, eq, j, 1)
        multiply_add_to_node_vars!(Fb, Vr[i], F_node, eq, j, 2)

        G_node = get_node_vars(G, eq, i, j)
        # Fb = g * V
        # Fb[i] += ∑_j g[i,j]*V[j]
        multiply_add_to_node_vars!(Fb, Vl[j], G_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, Vr[j], G_node, eq, i, 4)
    end
end

function compute_cell_residual_mdrk_1!(eq::AbstractEquations{2}, grid, op,
                                       problem, scheme, aux, t, dt, u1, res, Fb, Ub,
                                       cache)
    @unpack nvar = eq
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size

    @unpack source_terms = problem

    # Select boundary flux
    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
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
        (F, G, F2_loc, G2_loc, ut, U, U2_loc, S) = cell_arrays[id]

        eval_data_big = eval_data.eval_data_big[id]
        refresh!.(eval_data_big)
        ul, ur, uu, ud, upl, upr, upd, upu, uml, umr, umd, umu = eval_data_big

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
            set_node_vars!(S,  0.5 * s_node, eq, i, j)
            set_node_vars!(S2, s_node, eq, i, j, el_x, el_y)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)

            um = u_node - ut_node
            up = u_node + ut_node

            # For efficient computation of time averaged flux at interfaces
            # ul = u * V
            # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
            multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)

            # ud = u * V
            # ud[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
            multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

            multiply_add_to_node_vars!(upl, Vl[i], up, eq, j)
            multiply_add_to_node_vars!(upr, Vr[i], up, eq, j)
            multiply_add_to_node_vars!(upd, Vl[j], up, eq, i)
            multiply_add_to_node_vars!(upu, Vr[j], up, eq, i)

            multiply_add_to_node_vars!(uml, Vl[i], um, eq, j)
            multiply_add_to_node_vars!(umr, Vr[i], um, eq, j)
            multiply_add_to_node_vars!(umd, Vl[j], um, eq, i)
            multiply_add_to_node_vars!(umu, Vr[j], um, eq, i)

            fm, gm = flux(x, y, um, eq)
            fp, gp = flux(x, y, up, eq)

            ft = 0.5 * (fp - fm)
            gt = 0.5 * (gp - gm)
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

            st = calc_source_t_N12(up, um,
                                   X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.125, st, eq, i, j)

            multiply_add_to_node_vars!(S2, 1.0/6.0, st, eq, i, j, el_x, el_y)

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
        @views compute_bflux!(eq, grid, cell_data, eval_data_big, xg, Vl, Vr,
                              F, G, F2_loc, G2_loc, Fb[:, :, :, el_x, el_y],
                              Fb2[:, :, :, el_x, el_y], aux)
        # @views extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #    F, G, F2_loc, G2_loc, Fb[:, :, :, el_x, el_y], Fb2[:, :, :, el_x, el_y], aux)
    end
    return nothing
end

function compute_cell_residual_mdrk_2!(eq::AbstractEquations{2}, grid, op, problem,
                                       scheme, aux, t, dt, u1, res, Fb, Ub,
                                       cache)
    @unpack nvar = eq
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size

    @unpack source_terms = problem

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
        (_, _, _, _, upl, uppl, upr, uppr, upd, uppd, upu, uppu, uml, umml,
        umr, ummr, umd, ummd, umu, ummu) = eval_data_big

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
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)
            us_node = get_node_vars(us_, eq, i, j)
            ust_node = get_node_vars(ust, eq, i, j)

            um = us_node - ust_node
            up = us_node + ust_node
            umm = us_node - 2.0 * ust_node
            upp = us_node + 2.0 * ust_node

            multiply_add_to_node_vars!(upl, Vl[i], up, eq, j)
            multiply_add_to_node_vars!(upr, Vr[i], up, eq, j)
            multiply_add_to_node_vars!(upd, Vl[j], up, eq, i)
            multiply_add_to_node_vars!(upu, Vr[j], up, eq, i)

            multiply_add_to_node_vars!(uml, Vl[i], um, eq, j)
            multiply_add_to_node_vars!(umr, Vr[i], um, eq, j)
            multiply_add_to_node_vars!(umd, Vl[j], um, eq, i)
            multiply_add_to_node_vars!(umu, Vr[j], um, eq, i)

            multiply_add_to_node_vars!(uppl, Vl[i], upp, eq, j)
            multiply_add_to_node_vars!(uppr, Vr[i], upp, eq, j)
            multiply_add_to_node_vars!(uppd, Vl[j], upp, eq, i)
            multiply_add_to_node_vars!(uppu, Vr[j], upp, eq, i)

            multiply_add_to_node_vars!(umml, Vl[i], umm, eq, j)
            multiply_add_to_node_vars!(ummr, Vr[i], umm, eq, j)
            multiply_add_to_node_vars!(ummd, Vl[j], umm, eq, i)
            multiply_add_to_node_vars!(ummu, Vr[j], umm, eq, i)

            fm, gm = flux(x, y, um, eq)
            fp, gp = flux(x, y, up, eq)
            fmm, gmm = flux(x, y, umm, eq)
            fpp, gpp = flux(x, y, upp, eq)

            #  ft_node = get_node_vars(ft, eq, i, j)
            ft = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
            gt = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)
            multiply_add_to_node_vars!(F, 1.0 / 3.0, ft, eq, i, j)
            multiply_add_to_node_vars!(G, 1.0 / 3.0, gt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 3.0, ust_node, eq, i, j)

            st = calc_source_t_N34(us_node, up, upp, um, umm, X, t+0.5*dt, dt, source_terms, eq)

            multiply_add_to_node_vars!(S2, 1.0/3.0, st, eq, i, j, el_x, el_y)

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
        @views compute_bflux!(eq, grid, cell_data, eval_data_big, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], Fb2[:, :, :, el_x, el_y],
                              aux, nothing)
        # @views extrap_bflux!(eq, grid, cell_data, eval_data_big, xg, Vl, Vr,
        #    F, G, Fb[:, :, :, el_x, el_y], Fb2[:, :, :, el_x, el_y],
        #    aux, nothing)
    end
    return nothing
end
end # muladd
