using ..Tenkai: periodic, dirichlet, neumann, reflect, extrapolate, evaluate,
                update_ghost_values_periodic!,
                update_ghost_values_fn_blend!,
                get_node_vars, set_node_vars!,
                add_to_node_vars!, subtract_from_node_vars!,
                multiply_add_to_node_vars!, multiply_add_set_node_vars!,
                comp_wise_mutiply_node_vars!, flux,
                @threaded, alloc_for_threads

using SimpleUnPack
using TimerOutputs
using Polyester
using MuladdMacro
using OffsetArrays
using StaticArrays

import Tenkai: setup_arrays_mdrk

using ..Equations: AbstractEquations, nvariables, eachvariable

#-------------------------------------------------------------------------------
# Allocate solution arrays needed by MDRK in 1d
#-------------------------------------------------------------------------------
function setup_arrays_mdrk(grid, scheme, eq::AbstractEquations{1})
    gArray(nvar, nx) = OffsetArray(zeros(nvar, nx + 2),
                                   OffsetArrays.Origin(1, 0))
    function gArray(nvar, n1, nx)
        OffsetArray(zeros(nvar, n1, nx + 2),
                    OffsetArrays.Origin(1, 1, 0))
    end
    # Allocate memory
    @unpack degree = scheme
    nd = degree + 1
    nx = grid.size
    nvar = nvariables(eq)
    u1 = gArray(nvar, nd, nx)
    F2 = zeros(nvar, nd, nx)
    U2 = zeros(nvar, nd, nx)
    S2 = zeros(nvar, nd, nx)
    us = gArray(nvar, nd, nx)
    ua = gArray(nvar, nx)
    res = gArray(nvar, nd, nx)
    Fb = gArray(nvar, 2, nx)
    Fb2 = gArray(nvar, 2, nx)
    Ub = gArray(nvar, 2, nx)
    Ub2 = gArray(nvar, 2, nx)

    if degree == 1
        cell_data_size = 6
        eval_data_size = 6
    elseif degree == 2
        cell_data_size = 8
        eval_data_size = 6
    elseif degree == 3
        cell_data_size = 13
        eval_data_size = 16
    elseif degree == 4
        cell_data_size = 15
        eval_data_size = 18
    else
        @assert false "Degree not implemented"
    end

    MArr = MArray{Tuple{nvariables(eq), nd}, Float64}
    cell_data = alloc_for_threads(MArr, cell_data_size)

    MArr = MArray{Tuple{nvariables(eq), 1}, Float64}
    eval_data = alloc_for_threads(MArr, eval_data_size)

    cache = (; u1, U2, F2, S2, ua, us, res, Fb, Fb2, Ub, Ub2, cell_data, eval_data)
    return cache
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=3 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_mdrk_1!(eq::AbstractEquations{1}, grid, op, problem,
                                       scheme, aux, t, dt, u1, res, Fb, Ub,
                                       cache)
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    refresh!(u) = fill!(u, 0.0)

    @unpack source_terms = problem

    @unpack F2, U2, S2, Fb2, Ub2, cell_data, eval_data = cache

    F, U, F2_loc, U2_loc, S, ut = cell_data[Threads.threadid()]

    refresh!.((res, Fb, Fb2, Ub, Ub2))

    @inbounds for cell in 1:nx # Loop over cells
        ul, ur, upl, upr, uml, umr, uppl, uppr, umml, ummr = (zero(get_node_vars(u1, eq, 1,
                                                                                 cell)) for _ in 1:10)

        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        refresh!(ut)
        # Solution points
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)
            set_node_vars!(F, 0.5 * flux1, eq, i)
            set_node_vars!(F2_loc, flux1, eq, i)
            for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
                multiply_add_to_node_vars!(ut, -lamx * Dm[ix, i], flux1, eq, ix)
            end
            set_node_vars!(U, 0.5 * u_node, eq, i)
            set_node_vars!(U2_loc, u_node, eq, i)
        end

        # Add source term contribution to ut and some to S
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            s_node = calc_source(u_node, x_, t, source_terms, eq)
            set_node_vars!(S, 0.5 * s_node, eq, i)
            set_node_vars!(S2, s_node, eq, i, cell)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i)
        end

        # computes and stores ft, gt and puts them in respective place
        for i in Base.OneTo(nd) # Loop over solution points
            x_ = xc - 0.5 * dx + xg[i] * dx

            ut_node = get_node_vars(ut, eq, i)
            u_node = get_node_vars(u1, eq, i, cell)

            um = u_node - ut_node
            up = u_node + ut_node
            umm = u_node - 2.0 * ut_node
            upp = u_node + 2.0 * ut_node

            # For high accurate time averaged flux at interfaces
            # TODO - Is this facter than storing them in an array?
            ul += Vl[i] * u_node
            ur += Vr[i] * u_node
            upl += Vl[i] * up
            upr += Vr[i] * up
            uml += Vl[i] * um
            umr += Vr[i] * um
            uppl += Vl[i] * upp
            uppr += Vr[i] * upp
            umml += Vl[i] * umm
            ummr += Vr[i] * umm

            fm = flux(x_, um, eq)
            fp = flux(x_, up, eq)
            fmm = flux(x_, umm, eq)
            fpp = flux(x_, upp, eq)

            ft = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm) # This ft is actually Δt * ft
            multiply_add_to_node_vars!(F, 0.125, ft, eq, i) # F += 0.125*dt*ft
            multiply_add_to_node_vars!(U, 0.125, ut_node, eq, i) # U += 0.125*dt*ut
            multiply_add_to_node_vars!(F2_loc, 1.0 / 6.0, ft, eq, i) # F2 += 1/6 * dt*ft
            multiply_add_to_node_vars!(U2_loc, 1.0 / 6.0, ut_node, eq, i) # U2 += 1/6 * dt*ut

            st = calc_source_t_N34(u_node, up, upp, um, umm, x_, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.125, st, eq, i)

            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)

            multiply_add_to_node_vars!(S2, 1.0 / 6.0, st, eq, i, cell)
        end

        for i in 1:nd # Loop over solution points
            F_node = get_node_vars(F, eq, i)
            for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq,
                                           ix, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, nothing, r,
                                   0.5)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)

            multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, 1, cell)
            multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, 2, cell)
        end
        F2[:, :, cell] .= F2_loc
        U2[:, :, cell] .= U2_loc
        if bflux_ind == extrapolate
            for i in Base.OneTo(nd)
                F_node = get_node_vars(F, eq, i)
                F2_node = get_node_vars(F2_loc, eq, i)

                multiply_add_to_node_vars!(Fb, Vl[i], F_node, eq, 1, cell)
                multiply_add_to_node_vars!(Fb, Vr[i], F_node, eq, 2, cell)
                multiply_add_to_node_vars!(Fb2, Vl[i], F2_node, eq, 1, cell)
                multiply_add_to_node_vars!(Fb2, Vr[i], F2_node, eq, 2, cell)
            end
        else
            xl, xr = grid.xf[cell], grid.xf[cell + 1]

            fl, fr = flux(xl, ul, eq), flux(xr, ur, eq)

            set_node_vars!(Fb, 0.5 * fl, eq, 1, cell)
            set_node_vars!(Fb, 0.5 * fr, eq, 2, cell)
            set_node_vars!(Fb2, fl, eq, 1, cell)
            set_node_vars!(Fb2, fr, eq, 2, cell)

            fml, fmr = flux(xl, uml, eq), flux(xr, umr, eq)
            fpl, fpr = flux(xl, upl, eq), flux(xr, upr, eq)
            fmml, fmmr = flux(xl, umml, eq), flux(xr, ummr, eq)
            fppl, fppr = flux(xl, uppl, eq), flux(xr, uppr, eq)

            ftl = 1.0 / 12.0 * (-fppl + 8.0 * fpl - 8.0 * fml + fmml)
            ftr = 1.0 / 12.0 * (-fppr + 8.0 * fpr - 8.0 * fmr + fmmr)

            multiply_add_to_node_vars!(Fb, 0.125, ftl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 0.125, ftr, eq, 2, cell)
            multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, ftl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb2, 1.0 / 6.0, ftr, eq, 2, cell)
        end
    end
    return nothing
end

function compute_cell_residual_mdrk_2!(eq::AbstractEquations{1}, grid, op, problem,
                                       scheme, aux, t, dt, u1, res, Fb, Ub,
                                       cache)
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack us = cache
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    refresh!(u) = fill!(u, 0.0)

    @unpack source_terms = problem

    @unpack U2, F2, S2, Fb2, cell_data, eval_data = cache

    F, U, ust = cell_data[Threads.threadid()]

    refresh!.((res, Fb, Ub))

    @inbounds for cell in 1:nx # Loop over cells
        ul, ur, upl, upr, uml, umr, uppl, uppr, umml, ummr = (zero(get_node_vars(u1, eq, 1,
                                                                                 cell)) for _ in 1:10)

        F .= @view F2[:, :, cell]
        U .= @view U2[:, :, cell]
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        refresh!(ust)
        # Solution points
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            us_node = get_node_vars(us, eq, i, cell)
            # Compute flux at all solution points
            flux_ = flux(x_, us_node, eq)
            for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
                multiply_add_to_node_vars!(ust, -lamx * Dm[ix, i], flux_, eq, ix)
            end
        end

        # Add source term contribution to ust
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(us, eq, i, cell)
            s_node = calc_source(u_node, x_, t + 0.5 * dt, source_terms, eq)
            multiply_add_to_node_vars!(ust, dt, s_node, eq, i)
        end

        # computes and stores ft, gt and puts them in respective place
        for i in Base.OneTo(nd) # Loop over solution points
            x_ = xc - 0.5 * dx + xg[i] * dx

            us_node = get_node_vars(us, eq, i, cell) # u(t+0.5*dt)
            ust_node = get_node_vars(ust, eq, i)

            um = us_node - ust_node # u(t - 0.5*dt)
            up = us_node + ust_node # u(t + 1.5*dt)
            umm = us_node - 2.0 * ust_node # u(t - 2.0*dt)
            upp = us_node + 2.0 * ust_node # u(t + 3.5*dt)
            fm = flux(x_, um, eq)
            fp = flux(x_, up, eq)
            fmm = flux(x_, umm, eq)
            fpp = flux(x_, upp, eq)

            # For high accurate time averaged flux at interfaces
            ul += Vl[i] * us_node
            ur += Vr[i] * us_node
            upl += Vl[i] * up
            upr += Vr[i] * up
            uml += Vl[i] * um
            umr += Vr[i] * um
            uppl += Vl[i] * upp
            uppr += Vr[i] * upp
            umml += Vl[i] * umm
            ummr += Vr[i] * umm

            ft_s = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)

            multiply_add_to_node_vars!(F, 1.0 / 3.0, ft_s, eq, i)    # F += 1/3 *dt*ft

            multiply_add_to_node_vars!(U, 1.0 / 3.0, ust_node, eq, i)    # U += 1/6 * dt * ut

            st = calc_source_t_N34(us_node, up, upp, um, umm, x_, t + 0.5 * dt, dt,
                                   source_terms, eq)

            multiply_add_to_node_vars!(S2, 1.0 / 3.0, st, eq, i, cell)

            S_node = get_node_vars(S2, eq, i, cell)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end
        for i in 1:nd # Loop over solution points
            F_node = get_node_vars(F, eq, i)
            for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq,
                                           ix, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, nothing, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
            # U_node = get_node_vars(u1, eq, i, cell)
            multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, 1, cell)
            multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, 2, cell)
        end
        if bflux_ind == extrapolate
            for i in Base.OneTo(nd)
                Fl_node = get_node_vars(F, eq, i)
                Fr_node = get_node_vars(F, eq, i)
                multiply_add_to_node_vars!(Fb, Vl[i], Fl_node, eq, 1, cell)
                multiply_add_to_node_vars!(Fb, Vr[i], Fr_node, eq, 2, cell)
            end
        else
            xl, xr = grid.xf[cell], grid.xf[cell + 1]

            Fbl, Fbr = get_node_vars(Fb2, eq, 1, cell), get_node_vars(Fb2, eq, 2, cell)

            set_node_vars!(Fb, Fbl, eq, 1, cell)
            set_node_vars!(Fb, Fbr, eq, 2, cell)

            fml, fmr = flux(xl, uml, eq), flux(xr, umr, eq)
            fpl, fpr = flux(xl, upl, eq), flux(xr, upr, eq)
            fmml, fmmr = flux(xl, umml, eq), flux(xr, ummr, eq)
            fppl, fppr = flux(xl, uppl, eq), flux(xr, uppr, eq)

            ftl = 1.0 / 12.0 * (-fppl + 8.0 * fpl - 8.0 * fml + fmml)
            ftr = 1.0 / 12.0 * (-fppr + 8.0 * fpr - 8.0 * fmr + fmmr)

            multiply_add_to_node_vars!(Fb, 1.0 / 3.0, ftl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 1.0 / 3.0, ftr, eq, 2, cell)
        end
    end
    return nothing
end
