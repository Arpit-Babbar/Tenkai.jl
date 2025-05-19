@muladd begin
#! format: noindent

function setup_arrays(grid, scheme::Scheme{<:LWEnzymeTower}, eq::AbstractEquations{1})
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
    ua = gArray(nvar, nx)
    res = gArray(nvar, nd, nx)
    Fb = gArray(nvar, 2, nx)
    Ub = gArray(nvar, 2, nx)

    if degree == 0
        cell_data_size = 0
        eval_data_size = 0
    elseif degree == 1
        cell_data_size = 7
        eval_data_size = 6
    elseif degree == 2
        cell_data_size = 9
        eval_data_size = 6
    elseif degree == 3
        cell_data_size = 14
        eval_data_size = 16
    elseif degree == 4
        cell_data_size = 16
        eval_data_size = 18
    elseif degree == 5
        cell_data_size = 16
        eval_data_size = 18
    else
        @assert false "Degree not implemented"
    end

    MArr = MArray{Tuple{nvariables(eq), nd}, Float64}
    cell_data = alloc_for_threads(MArr, cell_data_size)

    MArr = MArray{Tuple{nvariables(eq), 1}, Float64}
    eval_data = alloc_for_threads(MArr, eval_data_size)

    cache = (; u1, ua, res, Fb, Ub, cell_data, eval_data)
    return cache
end

function get_cfl(eq::AbstractEquations{1}, scheme::Scheme{<:LWEnzymeTower}, param)
    @unpack solver, degree, correction_function = scheme
    @unpack cfl_safety_factor, cfl_style = param
    @unpack dissipation = scheme
    @assert (degree >= 0&&degree < 6) "Invalid degree"
    os_vector(v) = OffsetArray(v, OffsetArrays.Origin(0))
    if dissipation == get_second_node_vars # Diss 2
        cfl_radau = os_vector([1.0, 0.333, 0.170, 0.103, 0.069, 0.02419])
        cfl_g2 = os_vector([1.0, 1.000, 0.333, 0.170, 0.103, 0.02482])
        if solver == "rkfr"
            println("Using LW-D2 CFL with RKFR")
        else
            println("Using LW-D2 CFL with LW-D2")
        end
    elseif dissipation == get_first_node_vars # Diss 1
        cfl_radau = os_vector([1.0, 0.226, 0.117, 0.072, 0.049, 0.01988])
        cfl_g2 = os_vector([1.0, 0.465, 0.204, 0.116, 0.060, 0.0205])
        if solver == "rkfr"
            println("Using LW-D1 CFL with RKFR")
        else
            println("Using LW-D1 CFL with LW-D1")
        end
    end

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

#------------------------------------------------------------------------------
# Use automatic differentiation to compute the second derivative
#------------------------------------------------------------------------------
function compute_second_derivative_enzyme_1d(u, du, ddu, equations)
    df(x, dx) = autodiff(Forward, flux, Duplicated(x, dx), Const(equations))[1]
    ddf(x, dx, ddx) = autodiff(Forward, df, Duplicated(x, dx), Duplicated(dx, ddx))[1]

    return ddf(u, du, ddu)
end

function compute_third_derivative_enzyme_1d(u, du, ddu, dddu, equations)
    df(x, dx) = autodiff(Forward, flux, Duplicated(x, dx), Const(equations))[1]
    ddf(x, dx, ddx) = autodiff(Forward, df, Duplicated(x, dx), Duplicated(dx, ddx))[1]
    dddf(x, dx, ddx, dddx) = autodiff(Forward, ddf, Duplicated(x, dx),
                                      Duplicated(dx, ddx), Duplicated(ddx, dddx))[1]

    return dddf(u, du, ddu, dddu)
end

function compute_fourth_derivative_enzyme_1d(u, du, ddu, dddu, ddddu, equations)
    df(x, dx) = autodiff(Forward, flux, Duplicated(x, dx), Const(equations))[1]
    ddf(x, dx, ddx) = autodiff(Forward, df, Duplicated(x, dx), Duplicated(dx, ddx))[1]
    dddf(x, dx, ddx, dddx) = autodiff(Forward, ddf, Duplicated(x, dx),
                                      Duplicated(dx, ddx), Duplicated(ddx, dddx))[1]
    ddddf(x, dx, ddx, dddx, ddddx) = autodiff(Forward, dddf, Duplicated(x, dx),
                                              Duplicated(dx, ddx),
                                              Duplicated(ddx, dddx),
                                              Duplicated(dddx, ddddx))[1]

    return ddddf(u, du, ddu, dddu, ddddu)
end

function compute_fifth_derivative_enzyme_1d(u, du, ddu, dddu, ddddu, dddddu, equations)
    df(x, dx) = autodiff(Forward, flux, Duplicated, Duplicated(x, dx),
                         Const(equations))[1]
    ddf(x, dx, ddx) = autodiff(Forward, df, Duplicated, Duplicated(x, dx),
                               Duplicated(dx, ddx))[1]
    dddf(x, dx, ddx, dddx) = autodiff(Forward, ddf, Duplicated, Duplicated(x, dx),
                                      Duplicated(dx, ddx), Duplicated(ddx, dddx))[1]
    ddddf(x, dx, ddx, dddx, ddddx) = autodiff(Forward, dddf, Duplicated,
                                              Duplicated(x, dx),
                                              Duplicated(dx, ddx),
                                              Duplicated(ddx, dddx),
                                              Duplicated(dddx, ddddx))[1]
    dddddf(x, dx, ddx, dddx, ddddx, dddddx) = autodiff(Forward, ddddf, Duplicated,
                                                       Duplicated(x, dx),
                                                       Duplicated(dx, ddx),
                                                       Duplicated(ddx, dddx),
                                                       Duplicated(dddx, ddddx),
                                                       Duplicated(ddddx, dddddx))[1]

    return dddddf(u, du, ddu, dddu, ddddu, dddddu)
end

function compute_cell_residual_1!(eq::AbstractEquations{1}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    refresh!(u) = fill!(u, 0.0)
    # Pre-allocate local variables

    @unpack cell_data, eval_data = cache

    F, f, U, ut, S = cell_data[Threads.threadid()]
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inline @inbounds flux_(u_node) = flux(nothing, u_node, eq)

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        refresh!(ut)

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)
            set_node_vars!(F, flux1, eq, i)
            set_node_vars!(f, flux1, eq, i)
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii)
            end
            set_node_vars!(U, u_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            ft = derivative_bundle(flux_, (u_node, ut_node))
            multiply_add_to_node_vars!(F, 0.5, ft, eq, i)
            ut_node = get_node_vars(ut, eq, i)
            multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            ul, ur, utl, utr = eval_data[Threads.threadid()]
            refresh!.((ul, ur, utl, utr))
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u, eq, i)
                ut_node = get_node_vars(ut, eq, i)

                # ul = u * V
                # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
                multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, 1)
                multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, 1)

                multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, 1)
                multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, 1)
            end

            ul_node = get_node_vars(ul, eq, 1)
            ur_node = get_node_vars(ur, eq, 1)

            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)

            set_node_vars!(Fb, fl, eq, 1, cell)
            set_node_vars!(Fb, fr, eq, 2, cell)

            utl_node = get_node_vars(utl, eq, 1)
            utr_node = get_node_vars(utr, eq, 1)

            ftl = derivative_bundle(flux_, (ul_node, utl_node))
            ftr = derivative_bundle(flux_, (ur_node, utr_node))

            multiply_add_to_node_vars!(Fb, 0.5, ftl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 0.5, ftr, eq, 2, cell)
            # @assert false "Not implemented"
        end
    end
    return nothing
end

# TODO - Change from Euler1D to general AbstractEquations
function compute_cell_residual_2!(eq::AbstractEquations{1}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    refresh!(u) = fill!(u, 0.0)
    # Pre-allocate local variables

    @unpack cell_data, eval_data = cache

    F, f, U, ut, utt = cell_data[Threads.threadid()]
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inline @inbounds flux_(u_node) = flux(nothing, u_node, eq)

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        refresh!.((ut, utt))

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)
            set_node_vars!(F, flux1, eq, i)
            set_node_vars!(f, flux1, eq, i)
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii)
            end
            set_node_vars!(U, u_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            ft = derivative_bundle(flux_, (u_node, ut_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft, eq, ii)
            end

            multiply_add_to_node_vars!(F, 0.5, ft, eq, i)
            multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            ftt = derivative_bundle(flux_, (u_node, ut_node, utt_node))

            multiply_add_to_node_vars!(F, 1.0 / 6.0, ftt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq, ii, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            ul, ur, utl, utr, uttl, uttr = eval_data[Threads.threadid()]
            refresh!.((ul, ur, utl, utr, uttl, uttr))
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u, eq, i)
                ut_node = get_node_vars(ut, eq, i)
                utt_node = get_node_vars(ut, eq, i)

                # ul = u * V
                # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
                multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, 1)
                multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, 1)

                multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, 1)
                multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, 1)

                multiply_add_to_node_vars!(uttl, Vl[i], ut_node, eq, 1)
                multiply_add_to_node_vars!(uttr, Vr[i], ut_node, eq, 1)
            end

            ul_node = get_node_vars(ul, eq, 1)
            ur_node = get_node_vars(ur, eq, 1)

            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)

            set_node_vars!(Fb, fl, eq, 1, cell)
            set_node_vars!(Fb, fr, eq, 2, cell)

            utl_node = get_node_vars(utl, eq, 1)
            utr_node = get_node_vars(utr, eq, 1)

            ftl = derivative_bundle(flux_, (ul_node, utl_node))
            ftr = derivative_bundle(flux_, (ur_node, utr_node))

            multiply_add_to_node_vars!(Fb, 0.5, ftl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 0.5, ftr, eq, 2, cell)

            uttl_node = get_node_vars(uttl, eq, 1)
            uttr_node = get_node_vars(uttr, eq, 1)

            fttl = derivative_bundle(flux_, (ul_node, utl_node, uttl_node))
            fttr = derivative_bundle(flux_, (ur_node, utr_node, uttr_node))

            multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttr, eq, 2, cell)
        end
    end
    return nothing
end

function compute_cell_residual_3!(eq::AbstractEquations{1}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    refresh!(u) = fill!(u, 0.0)
    # Pre-allocate local variables

    @unpack cell_data, eval_data = cache

    F, f, U, ut, utt, uttt = cell_data[Threads.threadid()]
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inline @inbounds flux_(u_node) = flux(nothing, u_node, eq)

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        refresh!.((ut, utt, uttt))

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)
            set_node_vars!(F, flux1, eq, i)
            set_node_vars!(f, flux1, eq, i)
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii)
            end
            set_node_vars!(U, u_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            ft = derivative_bundle(flux_, (u_node, ut_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft, eq, ii)
            end

            multiply_add_to_node_vars!(F, 0.5, ft, eq, i)
            multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            ftt = derivative_bundle(flux_, (u_node, ut_node, utt_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(uttt, -lamx * Dm[ii, i], ftt, eq, ii)
            end

            multiply_add_to_node_vars!(F, 1.0 / 6.0, ftt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            uttt_node = get_node_vars(uttt, eq, i)
            fttt = derivative_bundle(flux_, (u_node, ut_node, utt_node, uttt_node))

            multiply_add_to_node_vars!(F, 1.0 / 24.0, fttt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq, ii, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            ul, ur, utl, utr, uttl, uttr, utttl, utttr = eval_data[Threads.threadid()]
            refresh!.((ul, ur, utl, utr, uttl, uttr, utttl, utttr))
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u, eq, i)
                ut_node = get_node_vars(ut, eq, i)
                utt_node = get_node_vars(ut, eq, i)
                uttt_node = get_node_vars(ut, eq, i)

                # ul = u * V
                # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
                multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, 1)
                multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, 1)

                multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, 1)
                multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, 1)

                multiply_add_to_node_vars!(uttl, Vl[i], ut_node, eq, 1)
                multiply_add_to_node_vars!(uttr, Vr[i], ut_node, eq, 1)

                multiply_add_to_node_vars!(utttl, Vl[i], ut_node, eq, 1)
                multiply_add_to_node_vars!(utttr, Vr[i], ut_node, eq, 1)
            end

            ul_node = get_node_vars(ul, eq, 1)
            ur_node = get_node_vars(ur, eq, 1)

            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)

            set_node_vars!(Fb, fl, eq, 1, cell)
            set_node_vars!(Fb, fr, eq, 2, cell)

            utl_node = get_node_vars(utl, eq, 1)
            utr_node = get_node_vars(utr, eq, 1)

            ftl = derivative_bundle(flux_, (ul_node, utl_node))
            ftr = derivative_bundle(flux_, (ur_node, utr_node))

            multiply_add_to_node_vars!(Fb, 0.5, ftl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 0.5, ftr, eq, 2, cell)

            uttl_node = get_node_vars(uttl, eq, 1)
            uttr_node = get_node_vars(uttr, eq, 1)

            fttl = derivative_bundle(flux_, (ul_node, utl_node, uttl_node))
            fttr = derivative_bundle(flux_, (ur_node, utr_node, uttr_node))

            multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttr, eq, 2, cell)

            utttl_node = get_node_vars(utttl, eq, 1)
            utttr_node = get_node_vars(utttr, eq, 1)

            ftttl = derivative_bundle(flux_, (ul_node, utl_node, uttl_node, utttl_node))
            ftttr = derivative_bundle(flux_, (ur_node, utr_node, uttr_node, utttr_node))

            multiply_add_to_node_vars!(Fb, 1.0 / 24.0, ftttl, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 1.0 / 24.0, ftttr, eq, 2, cell)
        end
    end
    return nothing
end

function compute_cell_residual_4!(eq::AbstractEquations{1}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    refresh!(u) = fill!(u, 0.0)
    # Pre-allocate local variables

    @unpack cell_data, eval_data = cache

    F, f, U, ut, utt, uttt, utttt = cell_data[Threads.threadid()]
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inline @inbounds flux_(u_node) = flux(nothing, u_node, eq)

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        refresh!.((ut, utt, uttt, utttt))

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)
            set_node_vars!(F, flux1, eq, i)
            set_node_vars!(f, flux1, eq, i)
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii)
            end
            set_node_vars!(U, u_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            ft = derivative_bundle(flux_, (u_node, ut_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft, eq, ii)
            end

            multiply_add_to_node_vars!(F, 0.5, ft, eq, i)
            multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            ftt = derivative_bundle(flux_, (u_node, ut_node, utt_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(uttt, -lamx * Dm[ii, i], ftt, eq, ii)
            end

            multiply_add_to_node_vars!(F, 1.0 / 6.0, ftt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            uttt_node = get_node_vars(uttt, eq, i)
            fttt = derivative_bundle(flux_, (u_node, ut_node, utt_node, uttt_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(utttt, -lamx * Dm[ii, i], fttt, eq, ii)
            end

            multiply_add_to_node_vars!(F, 1.0 / 24.0, fttt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            uttt_node = get_node_vars(uttt, eq, i)
            utttt_node = get_node_vars(utttt, eq, i)
            ftttt = derivative_bundle(flux_, (u_node, ut_node, utt_node, uttt_node))

            multiply_add_to_node_vars!(F, 1.0 / 120.0, ftttt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq, ii, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            @assert false "Not implemented"
        end
    end
    return nothing
end

function compute_cell_residual_5!(eq::AbstractEquations{1}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    refresh!(u) = fill!(u, 0.0)
    # Pre-allocate local variables

    @unpack cell_data, eval_data = cache

    F, f, U, ut, utt, uttt, utttt, uttttt = cell_data[Threads.threadid()]
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inline @inbounds flux_(u_node) = flux(nothing, u_node, eq)

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        refresh!.((ut, utt, uttt, utttt, uttttt))

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)
            set_node_vars!(F, flux1, eq, i)
            set_node_vars!(f, flux1, eq, i)
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii)
            end
            set_node_vars!(U, u_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            ft = derivative_bundle(flux_, (u_node, ut_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft, eq, ii)
            end

            multiply_add_to_node_vars!(F, 0.5, ft, eq, i)
            multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            ftt = derivative_bundle(flux_, (u_node, ut_node, utt_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(uttt, -lamx * Dm[ii, i], ftt, eq, ii)
            end

            multiply_add_to_node_vars!(F, 1.0 / 6.0, ftt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            uttt_node = get_node_vars(uttt, eq, i)
            fttt = derivative_bundle(flux_, (u_node, ut_node, utt_node, uttt_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(utttt, -lamx * Dm[ii, i], fttt, eq, ii)
            end

            multiply_add_to_node_vars!(F, 1.0 / 24.0, fttt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            uttt_node = get_node_vars(uttt, eq, i)
            utttt_node = get_node_vars(utttt, eq, i)
            ftttt = derivative_bundle(flux_, (u_node, ut_node, utt_node, uttt_node))
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(uttttt, -lamx * Dm[ii, i], ftttt, eq, ii)
            end

            multiply_add_to_node_vars!(F, 1.0 / 120.0, ftttt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            ut_node = get_node_vars(ut, eq, i)
            utt_node = get_node_vars(utt, eq, i)
            uttt_node = get_node_vars(uttt, eq, i)
            utttt_node = get_node_vars(utttt, eq, i)
            uttttt_node = get_node_vars(utttt, eq, i)
            fttttt = derivative_bundle(flux_,
                                       (u_node, ut_node, utt_node, uttt_node,
                                        utttt_node))
            multiply_add_to_node_vars!(F, 1.0 / 720.0, fttttt, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 720.0, uttttt_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq, ii, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            @assert false "Not implemented"
        end
    end
    return nothing
end
end # muladd
