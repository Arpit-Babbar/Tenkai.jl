using TowerOfEnzyme
using TowerOfEnzyme.Enzyme

@muladd begin
#! format: noindent

@inline @inbounds df_primal(x, dx, equations, orientation) = autodiff(Enzyme.set_abi(ForwardWithPrimal,
                                                                                     Enzyme.InlineABI),
                                                                      flux,
                                                                      DuplicatedNoNeed(x,
                                                                                       dx),
                                                                      Const(equations),
                                                                      Const(orientation))

@inline @inbounds ddf_primal(x, dx, ddx, equations, orientation) = autodiff(Enzyme.set_abi(ForwardWithPrimal,
                                                                                           Enzyme.InlineABI),
                                                                            df_primal,
                                                                            DuplicatedNoNeed(x,
                                                                                             dx),
                                                                            DuplicatedNoNeed(dx,
                                                                                             ddx),
                                                                            Const(equations),
                                                                            Const(orientation))

@inline @inbounds dddf_primal(x, dx, ddx, dddx, equations, orientation) = autodiff(Enzyme.set_abi(ForwardWithPrimal,
                                                                                                  Enzyme.InlineABI),
                                                                                   ddf_primal,
                                                                                   Duplicated,
                                                                                   Duplicated(x,
                                                                                              dx),
                                                                                   Duplicated(dx,
                                                                                              ddx),
                                                                                   Duplicated(ddx,
                                                                                              dddx),
                                                                                   Const(equations),
                                                                                   Const(orientation))

@inline @inbounds ddddf_primal(x, dx, ddx, dddx, ddddx, equations, orientation) = autodiff(Enzyme.set_abi(ForwardWithPrimal,
                                                                                                          Enzyme.InlineABI),
                                                                                           dddf_primal,
                                                                                           Duplicated,
                                                                                           Duplicated(x,
                                                                                                      dx),
                                                                                           Duplicated(dx,
                                                                                                      ddx),
                                                                                           Duplicated(ddx,
                                                                                                      dddx),
                                                                                           Duplicated(dddx,
                                                                                                      ddddx),
                                                                                           Const(equations),
                                                                                           Const(orientation))

@inline @inbounds dddddf_primal(x, dx, ddx, dddx, ddddx, dddddx, equations, orientation) = autodiff(Enzyme.set_abi(ForwardWithPrimal,
                                                                                                                   Enzyme.InlineABI),
                                                                                                    ddddf_primal,
                                                                                                    Duplicated,
                                                                                                    Duplicated(x,
                                                                                                               dx),
                                                                                                    Duplicated(dx,
                                                                                                               ddx),
                                                                                                    Duplicated(ddx,
                                                                                                               dddx),
                                                                                                    Duplicated(dddx,
                                                                                                               ddddx),
                                                                                                    Duplicated(dddx,
                                                                                                               dddddx),
                                                                                                    Const(equations),
                                                                                                    Const(orientation))

@inline @inbounds function compute_second_derivative_enzyme_2d_primal(u, du, ddu,
                                                                      equations,
                                                                      orientation)
    arr = ddf_primal(u, du, ddu, equations, orientation)
    return (arr[1][1], arr[1][2], arr[2][2]) # ftt, ft, f
end

function compute_third_derivative_enzyme_2d_primal(u, du, ddu, dddu, equations,
                                                   orientation)
    arr = dddf_primal(u, du, ddu, dddu, equations, orientation)
    return (arr[1][1][1], arr[1][1][2], arr[1][2][2], arr[2][2][2]) # fttt, ftt, ft, f
end

function compute_fourth_derivative_enzyme_2d_primal(u, du, ddu, dddu, ddddu, equations,
                                                    orientation)
    arr = ddddf_primal(u, du, ddu, dddu, ddddu, equations, orientation)
    # ftttt, fttt, fttt, ft, f
    return (arr[1][1][1][1], arr[1][1][1][2], arr[1][1][2][2], arr[1][2][2][2],
            arr[2][2][2][2])
end

function compute_fifth_derivative_enzyme_2d_primal(u, du, ddu, dddu, ddddu, dddddu,
                                                   equations, orientation)
    arr = dddddf_primal(u, du, ddu, dddu, ddddu, dddddu, equations, orientation)
    # fttttt, ftttt, ftttt, fttt, ft, f
    return (arr[1][1][1][1][1], arr[1][1][1][1][2], arr[1][1][1][2][2],
            arr[1][1][2][2][2], arr[1][2][2][2][2], arr[2][2][2][2][2])
end

@inline function bflux_enzyme_1!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver, compute_bflux!::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux)
end

@inline function bflux_enzyme_2!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver, compute_bflux!::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux)
end

@inline function bflux_enzyme_3!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver,
                                 compute_bflux!::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux)
end

@inline function bflux_enzyme_4!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver,
                                 compute_bflux!::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux)
end

@inline function bflux_enzyme_5!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver,
                                 compute_bflux!::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux)
end

function get_cfl(eq::AbstractEquations{2}, scheme::Scheme{<:LWADSolver}, param)
    @unpack solver, degree, correction_function = scheme
    @unpack cfl_safety_factor, cfl_style = param
    @unpack dissipation = scheme
    @assert (degree >= 0&&degree < 6) "Invalid degree"
    os_vector(v) = OffsetArray(v, OffsetArrays.Origin(0))
    cfl_radau = os_vector([1.0, 0.259, 0.170, 0.103, 0.069, 0.02419])
    cfl_g2 = os_vector([1.0, 0.511, 0.333, 0.170, 0.103])
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

function setup_arrays(grid, scheme::Scheme{<:LWEnzymeTower},
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
    nvar = nvariables(eq)
    nd = degree + 1
    nx, ny = grid.size
    RealT = eltype(grid.xc)
    u1 = gArray(nvar, nd, nd, nx, ny)
    ua = gArray(nvar, nx, ny)
    res = gArray(nvar, nd, nd, nx, ny)
    Fb = gArray(nvar, nd, 4, nx, ny)
    Ub = gArray(nvar, nd, 4, nx, ny)

    # Cell residual cache

    nt = Threads.nthreads()
    cell_array_sizes = Dict(1 => 11, 2 => 12, 3 => 15, 4 => 16, 5 => 17)
    big_eval_data_sizes = Dict(1 => 12, 2 => 32, 3 => 40, 4 => 56, 5 => 72)
    small_eval_data_sizes = Dict(1 => 4, 2 => 4, 3 => 4, 4 => 4, 5 => 4)
    if bflux_ind == extrapolate
        cell_array_size = cell_array_sizes[degree]
        big_eval_data_size = 2
        small_eval_data_size = 2
    elseif bflux_ind == evaluate
        cell_array_size = cell_array_sizes[degree]
        big_eval_data_size = big_eval_data_sizes[degree]
        small_eval_data_size = small_eval_data_sizes[degree]
    else
        @assert false "Incorrect bflux"
    end

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
    cell_arrays = alloc_for_threads(MArr, cell_array_size)

    nt = Threads.nthreads()

    MEval = MArray{Tuple{nvariables(eq), nd}, RealT}
    eval_data_big = alloc_for_threads(MEval, big_eval_data_size)

    MEval_small = MArray{Tuple{nvariables(eq), 1}, RealT}
    eval_data_small = alloc_for_threads(MEval_small, small_eval_data_size)

    eval_data = (; eval_data_big, eval_data_small)

    # Ghost values cache

    Marr = MArray{Tuple{nvariables(eq), 1}, RealT}

    ghost_cache = alloc_for_threads(Marr, 2)

    # KLUDGE - Rename this to LWFR cache
    cache = (; u1, ua, res, Fb, Ub,
             eval_data, cell_arrays,
             ghost_cache)
    return cache
end

@inline function bflux_enzyme_1!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver::LWEnzymeTower,
                                 compute_bflux!::typeof(eval_bflux1!))
    u, ut, el_x, el_y = cell_data
    refresh!(u) = fill!(u, zero(eltype(u)))

    nd = length(xg)
    nvar = nvariables(eq)
    eval_data_big = eval_data.eval_data_big[Threads.threadid()]
    refresh!.(eval_data_big)
    ul, ur, ud, uu, utl, utr, utd, utu = eval_data_big

    # It should be faster to use @turbo, but we do this for
    # similar scheme to FD in Tenkai.jl
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        ut_node = get_node_vars(ut, eq, i, j)

        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utd, Vl[j], ut_node, eq, i)
        multiply_add_to_node_vars!(utu, Vr[j], ut_node, eq, i)
    end

    for i in Base.OneTo(nd)
        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)
        utl_node = get_node_vars(utl, eq, i)
        utr_node = get_node_vars(utr, eq, i)
        utd_node = get_node_vars(utd, eq, i)
        utu_node = get_node_vars(utu, eq, i)

        ftl, fl = df_primal(ul_node, utl_node, eq, 1)
        ftr, fr = df_primal(ur_node, utr_node, eq, 1)
        ftd, fd = df_primal(ud_node, utd_node, eq, 2)
        ftu, fu = df_primal(uu_node, utu_node, eq, 2)

        Fbl_local = fl + 0.5 * ftl
        Fbr_local = fr + 0.5 * ftr
        Fbd_local = fd + 0.5 * ftd
        Fbu_local = fu + 0.5 * ftu

        set_node_vars!(Fb_local, Fbl_local, eq, i, 1)
        set_node_vars!(Fb_local, Fbr_local, eq, i, 2)
        set_node_vars!(Fb_local, Fbd_local, eq, i, 3)
        set_node_vars!(Fb_local, Fbu_local, eq, i, 4)
    end
end

function compute_cell_residual_1!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nd_val = Val(nd)
    nx, ny = grid.size
    refresh!(u) = fill!(u, zero(eltype(u)))

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack solver = scheme
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack eval_data, cell_arrays = cache

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy

        id = Threads.threadid()
        f, g, F, G, ut, U,         # up, um,
        ft, gt, S = cell_arrays[id]

        refresh!(ut)
        refresh!(ft)
        refresh!(gt)

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]
        for j in 1:nd, i in 1:nd
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            flux1, flux2 = flux(x, y, u_node, eq)
            set_node_vars!(F, flux1, eq, i, j)
            set_node_vars!(G, flux2, eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(ut, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
            # set_node_vars!(um, u_node, eq, i, j)
            # set_node_vars!(up, u_node, eq, i, j)
            set_node_vars!(U, u_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            x = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            s_node = calc_source(u_node, x, t, source_terms, eq)
            set_node_vars!(S, s_node, eq, i, j)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)
            ft = derivative_bundle(flux_x, (u_node, ut_node))
            gt = derivative_bundle(flux_y, (u_node, ut_node))
            ft_node = ft
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = gt
            multiply_add_to_node_vars!(G,
                                       0.5, gt_node,
                                       eq, i, j)
            F_node = get_node_vars(F, eq, i, j)
            G_node = get_node_vars(G, eq, i, j)
            for ii in Base.OneTo(nd)
                # res              += -lam * D * F for each variable
                # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
                multiply_add_to_node_vars!(r1, lamx * D1[ii, i], F_node, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(r1, lamy * D1[jj, j], G_node, eq, i, jj)
            end

            X = SVector(x, y)
            st = calc_source_t_N12(u_node, nothing, X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)

            S_node = get_node_vars(S, eq, i, j)

            # TODO - add blend source term function here

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, j, el_x, el_y)

            # KLUDGE - update to v1.8 and call with @inline
            # Give u1_ or U depending on dissipation model
            U_node = get_dissipation_node_vars(u1_, U, eq, i, j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(Ub_, Vl[i], U_node, eq, j, 1)
            multiply_add_to_node_vars!(Ub_, Vr[i], U_node, eq, j, 2)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(Ub_, Vl[j], U_node, eq, i, 3)
            multiply_add_to_node_vars!(Ub_, Vr[j], U_node, eq, i, 4)
        end
        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, f, res)
        # Interpolate to faces
        @views cell_data = (u1_, ut, el_x, el_y)
        # @views eval_bflux_ad_1!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #                       F, G, Fb[:, :, :, el_x, el_y], aux)
        cell_data = (u, ut, el_x, el_y)
        # @views eval_bflux_ad_1!(eq, grid, cell_data, eval_data, xg, op, nd_val,
        #                         F, G, Fb[:, :, :, el_x, el_y], aux)
        Fb_local = @view Fb[:, :, :, el_x, el_y]
        bflux_enzyme_1!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux,
                        solver, compute_bflux!)
        # @views extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #                       F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    return nothing
end

@inline function bflux_enzyme_2!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver::LWEnzymeTower,
                                 compute_bflux!::typeof(eval_bflux2!))
    u, ut, utt, el_x, el_y = cell_data
    refresh!(u) = fill!(u, zero(eltype(u)))

    nd = length(xg)
    nvar = nvariables(eq)
    eval_data_big = eval_data.eval_data_big[Threads.threadid()]
    refresh!.(eval_data_big)
    ul, ur, ud, uu, utl, utr, utd, utu, uttl, uttr, uttd, uttu = eval_data_big

    # It should be faster to use @turbo, but we do this for
    # similar scheme to FD in Tenkai.jl
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        ut_node = get_node_vars(ut, eq, i, j)
        utt_node = get_node_vars(utt, eq, i, j)

        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utd, Vl[j], ut_node, eq, i)
        multiply_add_to_node_vars!(utu, Vr[j], ut_node, eq, i)

        multiply_add_to_node_vars!(uttl, Vl[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttr, Vr[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttd, Vl[j], utt_node, eq, i)
        multiply_add_to_node_vars!(uttu, Vr[j], utt_node, eq, i)
    end

    for i in Base.OneTo(nd)
        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)
        utl_node = get_node_vars(utl, eq, i)
        utr_node = get_node_vars(utr, eq, i)
        utd_node = get_node_vars(utd, eq, i)
        utu_node = get_node_vars(utu, eq, i)
        uttl_node = get_node_vars(uttl, eq, i)
        uttr_node = get_node_vars(uttr, eq, i)
        uttd_node = get_node_vars(uttd, eq, i)
        uttu_node = get_node_vars(uttu, eq, i)

        fttl, ftl, fl = compute_second_derivative_enzyme_2d_primal(ul_node, utl_node,
                                                                   uttl_node, eq, 1)
        fttr, ftr, fr = compute_second_derivative_enzyme_2d_primal(ur_node, utr_node,
                                                                   uttr_node, eq, 1)
        fttd, ftd, fd = compute_second_derivative_enzyme_2d_primal(ud_node, utd_node,
                                                                   uttd_node, eq, 2)
        fttu, ftu, fu = compute_second_derivative_enzyme_2d_primal(uu_node, utu_node,
                                                                   uttu_node, eq, 2)

        Fbl_local = fl + 0.5 * ftl + 1.0 / 6.0 * fttl
        Fbr_local = fr + 0.5 * ftr + 1.0 / 6.0 * fttr
        Fbd_local = fd + 0.5 * ftd + 1.0 / 6.0 * fttd
        Fbu_local = fu + 0.5 * ftu + 1.0 / 6.0 * fttu

        set_node_vars!(Fb_local, Fbl_local, eq, i, 1)
        set_node_vars!(Fb_local, Fbr_local, eq, i, 2)
        set_node_vars!(Fb_local, Fbd_local, eq, i, 3)
        set_node_vars!(Fb_local, Fbu_local, eq, i, 4)
    end
end

function compute_cell_residual_2!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, DmT, D1T, Vl, Vr = op
    # nd = length(xg)
    nd = scheme_degree_plus_one(scheme)
    nx, ny = grid.size
    refresh!(u) = fill!(u, zero(eltype(u)))

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack solver = scheme
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack eval_data, cell_arrays = cache

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero
    @inbounds @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy

        # Some local variables
        id = Threads.threadid()
        F, G, ut, utt, U, S = cell_arrays[id]

        refresh!.((ut, utt))

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            flux1, flux2 = flux(x, y, u_node, eq)
            set_node_vars!(F, flux1, eq, i, j)
            set_node_vars!(G, flux2, eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(ut, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
            set_node_vars!(U, u_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            x = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            s_node = calc_source(u_node, x, t, source_terms, eq)
            set_node_vars!(S, s_node, eq, i, j)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i, j)
            ft_node = derivative_bundle(flux_x, (u_node, ut_node))
            gt_node = derivative_bundle(flux_y, (u_node, ut_node))
            multiply_add_to_node_vars!(F, 0.5, ft_node, eq, i, j)
            multiply_add_to_node_vars!(G, 0.5, gt_node, eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft_node, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(utt, -lamy * Dm[jj, j], gt_node, eq, i, jj)
            end
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            # Add source term contribution to utt
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            # TODO - IMPLEMENT THIS
            st = calc_source_t_N12(u_node, nothing, X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)
            multiply_add_to_node_vars!(utt, dt, st, eq, i, j) # has no jacobian factor
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node,
                                       eq, i, j)
            ftt_node = derivative_bundle(flux_x, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(F, 1.0 / 6.0, ftt_node,
                                       eq, i, j)
            gtt_node = derivative_bundle(flux_y, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 6.0, gtt_node,
                                       eq, i, j)
            F_node = get_node_vars(F, eq, i, j)
            G_node = get_node_vars(G, eq, i, j)
            for ii in Base.OneTo(nd)
                # res              += -lam * D * F for each variable
                # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
                multiply_add_to_node_vars!(r1, lamx * D1[ii, i], F_node, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(r1, lamy * D1[jj, j], G_node, eq, i, jj)
            end

            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            X = SVector(x, y)
            stt = calc_source_tt_N23(u_node, nothing, nothing, X, t, dt, source_terms,
                                     eq)
            multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, eq, i, j)

            S_node = get_node_vars(S, eq, i, j)

            # TODO - add blend source term function here

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, j, el_x, el_y)

            # KLUDGE - update to v1.8 and call with @inline
            # Give u1_ or U depending on dissipation model
            U_node = get_dissipation_node_vars(u1_, U, eq, i, j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(Ub_, Vl[i], U_node, eq, j, 1)
            multiply_add_to_node_vars!(Ub_, Vr[i], U_node, eq, j, 2)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(Ub_, Vl[j], U_node, eq, i, 3)
            multiply_add_to_node_vars!(Ub_, Vr[j], U_node, eq, i, 4)
        end
        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # computes ftt, gtt and puts them in respective place; no need to store
        cell_data = (u1_, ut, utt, el_x, el_y)
        Fb_local = @view Fb[:, :, :, el_x, el_y]
        bflux_enzyme_2!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux,
                        solver, compute_bflux!)
        # @views eval_bflux_ad_2!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #                       F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    return nothing
end

@inline function bflux_enzyme_3!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver::LWEnzymeTower,
                                 compute_bflux!::typeof(eval_bflux3!))
    u, ut, utt, uttt, el_x, el_y = cell_data
    refresh!(u) = fill!(u, zero(eltype(u)))

    nd = length(xg)
    nvar = nvariables(eq)
    eval_data_big = eval_data.eval_data_big[Threads.threadid()]
    refresh!.(eval_data_big)
    ul, ur, ud, uu, utl, utr, utd, utu, uttl, uttr, uttd, uttu,
    utttl, utttr, utttd, utttu = eval_data_big

    # It should be faster to use @turbo, but we do this for
    # similar scheme to FD in Tenkai.jl
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        ut_node = get_node_vars(ut, eq, i, j)
        utt_node = get_node_vars(utt, eq, i, j)
        uttt_node = get_node_vars(uttt, eq, i, j)

        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utd, Vl[j], ut_node, eq, i)
        multiply_add_to_node_vars!(utu, Vr[j], ut_node, eq, i)

        multiply_add_to_node_vars!(uttl, Vl[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttr, Vr[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttd, Vl[j], utt_node, eq, i)
        multiply_add_to_node_vars!(uttu, Vr[j], utt_node, eq, i)

        multiply_add_to_node_vars!(utttl, Vl[i], uttt_node, eq, j)
        multiply_add_to_node_vars!(utttr, Vr[i], uttt_node, eq, j)
        multiply_add_to_node_vars!(utttd, Vl[j], uttt_node, eq, i)
        multiply_add_to_node_vars!(utttu, Vr[j], uttt_node, eq, i)
    end

    for i in Base.OneTo(nd)
        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)
        utl_node = get_node_vars(utl, eq, i)
        utr_node = get_node_vars(utr, eq, i)
        utd_node = get_node_vars(utd, eq, i)
        utu_node = get_node_vars(utu, eq, i)
        uttl_node = get_node_vars(uttl, eq, i)
        uttr_node = get_node_vars(uttr, eq, i)
        uttd_node = get_node_vars(uttd, eq, i)
        uttu_node = get_node_vars(uttu, eq, i)
        utttl_node = get_node_vars(utttl, eq, i)
        utttr_node = get_node_vars(utttr, eq, i)
        utttd_node = get_node_vars(utttd, eq, i)
        utttu_node = get_node_vars(utttu, eq, i)

        ftttl, fttl, ftl, fl = compute_third_derivative_enzyme_2d_primal(ul_node,
                                                                         utl_node,
                                                                         uttl_node,
                                                                         utttl_node, eq,
                                                                         1)
        ftttr, fttr, ftr, fr = compute_third_derivative_enzyme_2d_primal(ur_node,
                                                                         utr_node,
                                                                         uttr_node,
                                                                         utttr_node, eq,
                                                                         1)
        ftttd, fttd, ftd, fd = compute_third_derivative_enzyme_2d_primal(ud_node,
                                                                         utd_node,
                                                                         uttd_node,
                                                                         utttd_node, eq,
                                                                         2)
        ftttu, fttu, ftu, fu = compute_third_derivative_enzyme_2d_primal(uu_node,
                                                                         utu_node,
                                                                         uttu_node,
                                                                         utttu_node, eq,
                                                                         2)

        Fbl_local = fl + 0.5 * ftl + 1.0 / 6.0 * fttl + 1.0 / 24.0 * ftttl
        Fbr_local = fr + 0.5 * ftr + 1.0 / 6.0 * fttr + 1.0 / 24.0 * ftttr
        Fbd_local = fd + 0.5 * ftd + 1.0 / 6.0 * fttd + 1.0 / 24.0 * ftttd
        Fbu_local = fu + 0.5 * ftu + 1.0 / 6.0 * fttu + 1.0 / 24.0 * ftttu

        set_node_vars!(Fb_local, Fbl_local, eq, i, 1)
        set_node_vars!(Fb_local, Fbr_local, eq, i, 2)
        set_node_vars!(Fb_local, Fbd_local, eq, i, 3)
        set_node_vars!(Fb_local, Fbu_local, eq, i, 4)
    end
end

function compute_cell_residual_3!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    nvar = nvariables(eq)
    @unpack source_terms = problem
    @unpack solver = scheme
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    refresh!(u) = fill!(u, zero(eltype(u)))

    # Select boundary flux
    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack eval_data, cell_arrays = cache
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        # Some local variables
        id = Threads.threadid()
        (f, g, ft, gt, F, G, ut, utt, uttt, U, S) = cell_arrays[id]

        refresh!(ut)
        refresh!(utt)
        refresh!(uttt)
        refresh!(ft)
        refresh!(gt)

        u1_ = @view u1[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            flux1, flux2 = flux(x, y, u_node, eq)
            set_node_vars!(f, flux1, eq, i, j)
            set_node_vars!(g, flux2, eq, i, j)
            set_node_vars!(F, flux1, eq, i, j)
            set_node_vars!(G, flux2, eq, i, j)
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
            set_node_vars!(U, u_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            x = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            s_node = calc_source(u_node, x, t, source_terms, eq)
            set_node_vars!(S, s_node, eq, i, j)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)

            ft_node = derivative_bundle(flux_x, (u_node, ut_node))
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = derivative_bundle(flux_y, (u_node, ut_node))
            multiply_add_to_node_vars!(G,
                                       0.5, gt_node,
                                       eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft_node, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(utt, -lamy * Dm[jj, j], gt_node, eq, i, jj)
            end
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            # Add source term contribution to utt
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            st = calc_source_t_N34(u_node, nothing, nothing, nothing, nothing, X, t, dt,
                                   source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)
            multiply_add_to_node_vars!(utt, dt, st, eq, i, j) # has no jacobian factor
        end

        ftt, gtt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node,
                                       eq, i, j)
            ftt_node = derivative_bundle(flux_x, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 6.0, ftt_node,
                                       eq, i, j)

            gtt_node = derivative_bundle(flux_y, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 6.0, gtt_node,
                                       eq, i, j)

            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(uttt, -lamx * Dm[ii, i], ftt_node, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(uttt, -lamy * Dm[jj, j], gtt_node, eq, i, jj)
            end
        end

        # Add source term contribution to uttt and some to S
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x_, y_)
            # Add source term contribution to uttt
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            stt = calc_source_tt_N23(u_node, ut_node, utt_node, X, t, dt, source_terms,
                                     eq)
            multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, eq, i, j)
            multiply_add_to_node_vars!(uttt, dt, stt, eq, i, j) # has no jacobian factor
        end

        fttt, gttt = ft, gt

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            uttt_node = get_node_vars(uttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node,
                                       eq, i, j)
            fttt_node = derivative_bundle(flux_x,
                                          (u_node, ut_node, utt_node, uttt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 24.0, fttt_node,
                                       eq, i, j)
            gttt_node = derivative_bundle(flux_y,
                                          (u_node, ut_node, utt_node, uttt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 24.0, gttt_node,
                                       eq, i, j)
            F_node = get_node_vars(F, eq, i, j)
            G_node = get_node_vars(G, eq, i, j)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq, ii, j,
                                           el_x, el_y)
            end

            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(res, lamy * D1[jj, j], G_node, eq, i, jj,
                                           el_x, el_y)
            end

            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            X = SVector(x, y)
            sttt = calc_source_ttt_N34(u_node, nothing, nothing, nothing, nothing,
                                       X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, eq, i, j)

            S_node = get_node_vars(S, eq, i, j)

            # TODO - add blend source term function here

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, j, el_x, el_y)

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
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, f, res)
        @views cell_data = (u1_, ut, utt, uttt, el_x, el_y)
        # @views eval_bflux_ad_3!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #                        F, G, Fb[:, :, :, el_x, el_y], aux)
        Fb_local = @view Fb[:, :, :, el_x, el_y]
        bflux_enzyme_3!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux,
                        solver, compute_bflux!)
    end
    return nothing
end

@inline function bflux_enzyme_4!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver::LWEnzymeTower,
                                 compute_bflux!::typeof(eval_bflux4!))
    u, ut, utt, uttt, utttt, el_x, el_y = cell_data
    refresh!(u) = fill!(u, zero(eltype(u)))

    nd = length(xg)
    nvar = nvariables(eq)
    eval_data_big = eval_data.eval_data_big[Threads.threadid()]
    refresh!.(eval_data_big)
    ul, ur, ud, uu, utl, utr, utd, utu, uttl, uttr, uttd, uttu,
    utttl, utttr, utttd, utttu, uttttl, uttttr, uttttd, uttttu = eval_data_big

    @inline flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    # It should be faster to use @turbo, but we do this for
    # similar scheme to FD in Tenkai.jl
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        ut_node = get_node_vars(ut, eq, i, j)
        utt_node = get_node_vars(utt, eq, i, j)
        uttt_node = get_node_vars(uttt, eq, i, j)
        utttt_node = get_node_vars(utttt, eq, i, j)

        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utd, Vl[j], ut_node, eq, i)
        multiply_add_to_node_vars!(utu, Vr[j], ut_node, eq, i)

        multiply_add_to_node_vars!(uttl, Vl[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttr, Vr[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttd, Vl[j], utt_node, eq, i)
        multiply_add_to_node_vars!(uttu, Vr[j], utt_node, eq, i)

        multiply_add_to_node_vars!(utttl, Vl[i], uttt_node, eq, j)
        multiply_add_to_node_vars!(utttr, Vr[i], uttt_node, eq, j)
        multiply_add_to_node_vars!(utttd, Vl[j], uttt_node, eq, i)
        multiply_add_to_node_vars!(utttu, Vr[j], uttt_node, eq, i)

        multiply_add_to_node_vars!(uttttl, Vl[i], utttt_node, eq, j)
        multiply_add_to_node_vars!(uttttr, Vr[i], utttt_node, eq, j)
        multiply_add_to_node_vars!(uttttd, Vl[j], utttt_node, eq, i)
        multiply_add_to_node_vars!(uttttu, Vr[j], utttt_node, eq, i)
    end

    for i in Base.OneTo(nd)
        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)

        utl_node = get_node_vars(utl, eq, i)
        utr_node = get_node_vars(utr, eq, i)
        utd_node = get_node_vars(utd, eq, i)
        utu_node = get_node_vars(utu, eq, i)

        uttl_node = get_node_vars(uttl, eq, i)
        uttr_node = get_node_vars(uttr, eq, i)
        uttd_node = get_node_vars(uttd, eq, i)
        uttu_node = get_node_vars(uttu, eq, i)

        utttl_node = get_node_vars(utttl, eq, i)
        utttr_node = get_node_vars(utttr, eq, i)
        utttd_node = get_node_vars(utttd, eq, i)
        utttu_node = get_node_vars(utttu, eq, i)

        uttttl_node = get_node_vars(uttttl, eq, i)
        uttttr_node = get_node_vars(uttttr, eq, i)
        uttttd_node = get_node_vars(uttttd, eq, i)
        uttttu_node = get_node_vars(uttttu, eq, i)

        fttttl, ftttl, fttl, ftl, fl = compute_fourth_derivative_enzyme_2d_primal(ul_node,
                                                                                  utl_node,
                                                                                  uttl_node,
                                                                                  utttl_node,
                                                                                  uttttl_node,
                                                                                  eq, 1)
        fttttr, ftttr, fttr, ftr, fr = compute_fourth_derivative_enzyme_2d_primal(ur_node,
                                                                                  utr_node,
                                                                                  uttr_node,
                                                                                  utttr_node,
                                                                                  uttttr_node,
                                                                                  eq, 1)
        fttttd, ftttd, fttd, ftd, fd = compute_fourth_derivative_enzyme_2d_primal(ud_node,
                                                                                  utd_node,
                                                                                  uttd_node,
                                                                                  utttd_node,
                                                                                  uttttd_node,
                                                                                  eq, 2)
        fttttu, ftttu, fttu, ftu, fu = compute_fourth_derivative_enzyme_2d_primal(uu_node,
                                                                                  utu_node,
                                                                                  uttu_node,
                                                                                  utttu_node,
                                                                                  uttttu_node,
                                                                                  eq, 2)

        Fbl_local = fl + 0.5 * ftl + 1.0 / 6.0 * fttl + 1.0 / 24.0 * ftttl +
                    1.0 / 120.0 * fttttl
        Fbr_local = fr + 0.5 * ftr + 1.0 / 6.0 * fttr + 1.0 / 24.0 * ftttr +
                    1.0 / 120.0 * fttttr
        Fbd_local = fd + 0.5 * ftd + 1.0 / 6.0 * fttd + 1.0 / 24.0 * ftttd +
                    1.0 / 120.0 * fttttd
        Fbu_local = fu + 0.5 * ftu + 1.0 / 6.0 * fttu + 1.0 / 24.0 * ftttu +
                    1.0 / 120.0 * fttttu

        set_node_vars!(Fb_local, Fbl_local, eq, i, 1)
        set_node_vars!(Fb_local, Fbr_local, eq, i, 2)
        set_node_vars!(Fb_local, Fbd_local, eq, i, 3)
        set_node_vars!(Fb_local, Fbu_local, eq, i, 4)
    end
end

function compute_cell_residual_4!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack source_terms = problem
    refresh!(u) = fill!(u, zero(eltype(u)))

    # Select boundary flux
    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack solver = scheme
    @unpack eval_data, cell_arrays = cache
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        # Some local variables
        id = Threads.threadid()
        (f, g, F, G, U, ft, gt, ut, utt, uttt, utttt, S) = cell_arrays[id]

        refresh!.((ut, utt, uttt, utttt))

        u = @view u1[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            flux1, flux2 = flux(x, y, u_node, eq)
            set_node_vars!(f, flux1, eq, i, j)
            set_node_vars!(g, flux2, eq, i, j)
            set_node_vars!(F, flux1, eq, i, j)
            set_node_vars!(G, flux2, eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(ut, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
            set_node_vars!(U, u_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            x = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            s_node = calc_source(u_node, x, t, source_terms, eq)
            set_node_vars!(S, s_node, eq, i, j)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)

            ft_node = derivative_bundle(flux_x, (u_node, ut_node))
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = derivative_bundle(flux_y, (u_node, ut_node))
            multiply_add_to_node_vars!(G,
                                       0.5, gt_node,
                                       eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft_node, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(utt, -lamy * Dm[jj, j], gt_node, eq, i, jj)
            end
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            # Add source term contribution to utt
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            st = calc_source_t_N34(u_node, nothing, nothing, nothing, nothing,
                                   X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)
            multiply_add_to_node_vars!(utt, dt, st, eq, i, j) # has no jacobian factor
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node,
                                       eq, i, j)
            ftt_node = derivative_bundle(flux_x, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 6.0, ftt_node,
                                       eq, i, j)
            gtt_node = derivative_bundle(flux_y, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 6.0, gtt_node,
                                       eq, i, j)

            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(uttt, -lamx * Dm[ii, i], ftt_node, eq, ii, j)
            end

            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(uttt, -lamy * Dm[jj, j], gtt_node, eq, i, jj)
            end
        end

        # Add source term contribution to uttt and some to S
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x_, y_)
            # Add source term contribution to uttt
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            stt = calc_source_tt_N23(u_node, ut_node, utt_node, X, t,
                                     dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, eq, i, j)
            multiply_add_to_node_vars!(uttt, dt, stt, eq, i, j) # has no jacobian factor
        end

        fttt, gttt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            uttt_node = get_node_vars(uttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node,
                                       eq, i, j)

            fttt_node = derivative_bundle(flux_x,
                                          (u_node, ut_node, utt_node, uttt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 24.0, fttt_node,
                                       eq, i, j)
            gttt_node = derivative_bundle(flux_y,
                                          (u_node, ut_node, utt_node, uttt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 24.0, gttt_node,
                                       eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(utttt, -lamx * Dm[ii, i], fttt_node, eq, ii,
                                           j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(utttt, -lamy * Dm[jj, j], gttt_node, eq, i,
                                           jj)
            end
        end

        # Add source term contribution to utttt and some to S
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)
            # Add source term contribution to utttt
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)

            sttt = calc_source_ttt_N34(u_node, nothing, nothing, nothing, nothing,
                                       X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, eq, i, j)
            multiply_add_to_node_vars!(utttt, dt, sttt, eq, i, j) # has no jacobian factor
        end

        ftttt, gtttt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            uttt_node = get_node_vars(uttt, eq, i, j)
            utttt_node = get_node_vars(utttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node,
                                       eq, i, j)

            ftttt_node = derivative_bundle(flux_x,
                                           (u_node, ut_node, utt_node, uttt_node,
                                            utttt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 120.0, ftttt_node,
                                       eq, i, j)
            gtttt_node = derivative_bundle(flux_y,
                                           (u_node, ut_node, utt_node, uttt_node,
                                            utttt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 120.0, gtttt_node,
                                       eq, i, j)
            F_node = get_node_vars(F, eq, i, j)
            G_node = get_node_vars(G, eq, i, j)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq, ii, j,
                                           el_x, el_y)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(res, lamy * D1[jj, j], G_node, eq, i, jj,
                                           el_x, el_y)
            end

            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            X = SVector(x, y)
            stttt = calc_source_tttt_N4(u_node, nothing, nothing, nothing, nothing,
                                        X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 120.0, stttt, eq, i, j)

            S_node = get_node_vars(S, eq, i, j)

            # TODO - add blend source term function here

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, j, el_x, el_y)

            U_node = get_dissipation_node_vars(u, U, eq, i, j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, j, 1, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, j, 2, el_x, el_y)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(Ub, Vl[j], U_node, eq, i, 3, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[j], U_node, eq, i, 4, el_x, el_y)
        end
        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, f, res)

        @views cell_data = (u, ut, utt, uttt, utttt, el_x, el_y)
        Fb_local = @view Fb[:, :, :, el_x, el_y]
        bflux_enzyme_4!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux,
                        solver, compute_bflux!)
    end
    return nothing
end

@inline function bflux_enzyme_5!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G,
                                 Fb_local, aux,
                                 solver::LWEnzymeTower,
                                 compute_bflux!::typeof(eval_bflux5!))
    u, ut, utt, uttt, utttt, uttttt, el_x, el_y = cell_data
    refresh!(u) = fill!(u, zero(eltype(u)))

    nd = length(xg)
    nvar = nvariables(eq)
    eval_data_big = eval_data.eval_data_big[Threads.threadid()]
    refresh!.(eval_data_big)
    ul, ur, ud, uu, utl, utr, utd, utu, uttl, uttr, uttd, uttu,
    utttl, utttr, utttd, utttu, uttttl, uttttr, uttttd, uttttu,
    utttttl, utttttr, utttttd, utttttu = eval_data_big

    @inline flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    # It should be faster to use @turbo, but we do this for
    # similar scheme to FD in Tenkai.jl
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        ut_node = get_node_vars(ut, eq, i, j)
        utt_node = get_node_vars(utt, eq, i, j)
        uttt_node = get_node_vars(uttt, eq, i, j)
        utttt_node = get_node_vars(utttt, eq, i, j)
        uttttt_node = get_node_vars(uttttt, eq, i, j)

        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(utl, Vl[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utr, Vr[i], ut_node, eq, j)
        multiply_add_to_node_vars!(utd, Vl[j], ut_node, eq, i)
        multiply_add_to_node_vars!(utu, Vr[j], ut_node, eq, i)

        multiply_add_to_node_vars!(uttl, Vl[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttr, Vr[i], utt_node, eq, j)
        multiply_add_to_node_vars!(uttd, Vl[j], utt_node, eq, i)
        multiply_add_to_node_vars!(uttu, Vr[j], utt_node, eq, i)

        multiply_add_to_node_vars!(utttl, Vl[i], uttt_node, eq, j)
        multiply_add_to_node_vars!(utttr, Vr[i], uttt_node, eq, j)
        multiply_add_to_node_vars!(utttd, Vl[j], uttt_node, eq, i)
        multiply_add_to_node_vars!(utttu, Vr[j], uttt_node, eq, i)

        multiply_add_to_node_vars!(uttttl, Vl[i], utttt_node, eq, j)
        multiply_add_to_node_vars!(uttttr, Vr[i], utttt_node, eq, j)
        multiply_add_to_node_vars!(uttttd, Vl[j], utttt_node, eq, i)
        multiply_add_to_node_vars!(uttttu, Vr[j], utttt_node, eq, i)

        multiply_add_to_node_vars!(utttttl, Vl[i], uttttt_node, eq, j)
        multiply_add_to_node_vars!(utttttr, Vr[i], uttttt_node, eq, j)
        multiply_add_to_node_vars!(utttttd, Vl[j], uttttt_node, eq, i)
        multiply_add_to_node_vars!(utttttu, Vr[j], uttttt_node, eq, i)
    end

    for i in Base.OneTo(nd)
        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)

        utl_node = get_node_vars(utl, eq, i)
        utr_node = get_node_vars(utr, eq, i)
        utd_node = get_node_vars(utd, eq, i)
        utu_node = get_node_vars(utu, eq, i)

        uttl_node = get_node_vars(uttl, eq, i)
        uttr_node = get_node_vars(uttr, eq, i)
        uttd_node = get_node_vars(uttd, eq, i)
        uttu_node = get_node_vars(uttu, eq, i)

        utttl_node = get_node_vars(utttl, eq, i)
        utttr_node = get_node_vars(utttr, eq, i)
        utttd_node = get_node_vars(utttd, eq, i)
        utttu_node = get_node_vars(utttu, eq, i)

        uttttl_node = get_node_vars(uttttl, eq, i)
        uttttr_node = get_node_vars(uttttr, eq, i)
        uttttd_node = get_node_vars(uttttd, eq, i)
        uttttu_node = get_node_vars(uttttu, eq, i)

        utttttl_node = get_node_vars(utttttl, eq, i)
        utttttr_node = get_node_vars(utttttr, eq, i)
        utttttd_node = get_node_vars(utttttd, eq, i)
        utttttu_node = get_node_vars(utttttu, eq, i)

        ftttttl, fttttl, ftttl, fttl, ftl, fl = compute_fifth_derivative_enzyme_2d_primal(ul_node,
                                                                                          utl_node,
                                                                                          uttl_node,
                                                                                          utttl_node,
                                                                                          uttttl_node,
                                                                                          utttttl_node,
                                                                                          eq,
                                                                                          1)
        ftttttr, fttttr, ftttr, fttr, ftr, fr = compute_fifth_derivative_enzyme_2d_primal(ur_node,
                                                                                          utr_node,
                                                                                          uttr_node,
                                                                                          utttr_node,
                                                                                          uttttr_node,
                                                                                          utttttr_node,
                                                                                          eq,
                                                                                          1)
        ftttttd, fttttd, ftttd, fttd, ftd, fd = compute_fifth_derivative_enzyme_2d_primal(ud_node,
                                                                                          utd_node,
                                                                                          uttd_node,
                                                                                          utttd_node,
                                                                                          uttttd_node,
                                                                                          utttttd_node,
                                                                                          eq,
                                                                                          2)
        ftttttu, fttttu, ftttu, fttu, ftu, fu = compute_fifth_derivative_enzyme_2d_primal(uu_node,
                                                                                          utu_node,
                                                                                          uttu_node,
                                                                                          utttu_node,
                                                                                          uttttu_node,
                                                                                          utttttu_node,
                                                                                          eq,
                                                                                          2)

        Fbl_local = fl + 0.5 * ftl + 1.0 / 6.0 * fttl + 1.0 / 24.0 * ftttl +
                    1.0 / 120.0 * fttttl + 1.0 / 720.0 * ftttttl
        Fbr_local = fr + 0.5 * ftr + 1.0 / 6.0 * fttr + 1.0 / 24.0 * ftttr +
                    1.0 / 120.0 * fttttr + 1.0 / 720.0 * ftttttr
        Fbd_local = fd + 0.5 * ftd + 1.0 / 6.0 * fttd + 1.0 / 24.0 * ftttd +
                    1.0 / 120.0 * fttttd + 1.0 / 720.0 * ftttttd
        Fbu_local = fu + 0.5 * ftu + 1.0 / 6.0 * fttu + 1.0 / 24.0 * ftttu +
                    1.0 / 120.0 * fttttu + 1.0 / 720.0 * ftttttu

        set_node_vars!(Fb_local, Fbl_local, eq, i, 1)
        set_node_vars!(Fb_local, Fbr_local, eq, i, 2)
        set_node_vars!(Fb_local, Fbd_local, eq, i, 3)
        set_node_vars!(Fb_local, Fbu_local, eq, i, 4)
    end
end

function compute_cell_residual_5!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWEnzymeTower},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    nvar = nvariables(eq)
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack source_terms = problem
    refresh!(u) = fill!(u, zero(eltype(u)))

    # Select boundary flux
    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack solver = scheme
    @unpack eval_data, cell_arrays = cache

    @inline @inbounds flux_x(u) = flux(1.0, 1.0, u, eq, 1)
    @inline @inbounds flux_y(u) = flux(1.0, 1.0, u, eq, 2)

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        # Some local variables
        id = Threads.threadid()
        (f, g, F, G, U, ft, gt, ut, utt, uttt, utttt, uttttt, S) = cell_arrays[id]

        refresh!.((ut, utt, uttt, utttt, uttttt))

        u = @view u1[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            flux1, flux2 = flux(x, y, u_node, eq)
            set_node_vars!(f, flux1, eq, i, j)
            set_node_vars!(g, flux2, eq, i, j)
            set_node_vars!(F, flux1, eq, i, j)
            set_node_vars!(G, flux2, eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(ut, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
            set_node_vars!(U, u_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            x = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            s_node = calc_source(u_node, x, t, source_terms, eq)
            set_node_vars!(S, s_node, eq, i, j)
            multiply_add_to_node_vars!(ut, dt, s_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)

            ft_node = derivative_bundle(flux_x, (u_node, ut_node))
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = derivative_bundle(flux_y, (u_node, ut_node))
            multiply_add_to_node_vars!(G,
                                       0.5, gt_node,
                                       eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(utt, -lamx * Dm[ii, i], ft_node, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(utt, -lamy * Dm[jj, j], gt_node, eq, i, jj)
            end
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            # Add source term contribution to utt
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x_, y_)
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            st = calc_source_t_N34(u_node, nothing, nothing, nothing, nothing,
                                   X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)
            multiply_add_to_node_vars!(utt, dt, st, eq, i, j) # has no jacobian factor
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node,
                                       eq, i, j)
            ftt_node = derivative_bundle(flux_x, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 6.0, ftt_node,
                                       eq, i, j)
            gtt_node = derivative_bundle(flux_y, (u_node, ut_node, utt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 6.0, gtt_node,
                                       eq, i, j)

            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(uttt, -lamx * Dm[ii, i], ftt_node, eq, ii, j)
            end

            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(uttt, -lamy * Dm[jj, j], gtt_node, eq, i, jj)
            end
        end

        # Add source term contribution to uttt and some to S
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            y_ = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x_, y_)
            # Add source term contribution to uttt
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            stt = calc_source_tt_N23(u_node, ut_node, utt_node, X, t,
                                     dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, eq, i, j)
            multiply_add_to_node_vars!(uttt, dt, stt, eq, i, j) # has no jacobian factor
        end

        fttt, gttt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            uttt_node = get_node_vars(uttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node,
                                       eq, i, j)

            fttt_node = derivative_bundle(flux_x,
                                          (u_node, ut_node, utt_node, uttt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 24.0, fttt_node,
                                       eq, i, j)
            gttt_node = derivative_bundle(flux_y,
                                          (u_node, ut_node, utt_node, uttt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 24.0, gttt_node,
                                       eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(utttt, -lamx * Dm[ii, i], fttt_node, eq, ii,
                                           j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(utttt, -lamy * Dm[jj, j], gttt_node, eq, i,
                                           jj)
            end
        end

        # Add source term contribution to utttt and some to S
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)
            # Add source term contribution to utttt
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)

            sttt = calc_source_ttt_N34(u_node, nothing, nothing, nothing, nothing,
                                       X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, eq, i, j)
            multiply_add_to_node_vars!(utttt, dt, sttt, eq, i, j) # has no jacobian factor
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            uttt_node = get_node_vars(uttt, eq, i, j)
            utttt_node = get_node_vars(utttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node,
                                       eq, i, j)

            ftttt_node = derivative_bundle(flux_x,
                                           (u_node, ut_node, utt_node, uttt_node,
                                            utttt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 120.0, ftttt_node,
                                       eq, i, j)
            gtttt_node = derivative_bundle(flux_y,
                                           (u_node, ut_node, utt_node, uttt_node,
                                            utttt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 120.0, gtttt_node,
                                       eq, i, j)
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * ft for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
                multiply_add_to_node_vars!(uttttt, -lamx * Dm[ii, i], ftttt_node, eq,
                                           ii,
                                           j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*gt*Dm' for each variable
                # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(uttttt, -lamy * Dm[jj, j], gtttt_node, eq, i,
                                           jj)
            end
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            ut_node = get_node_vars(ut, eq, i, j)
            utt_node = get_node_vars(utt, eq, i, j)
            uttt_node = get_node_vars(uttt, eq, i, j)
            utttt_node = get_node_vars(utttt, eq, i, j)
            uttttt_node = get_node_vars(uttttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 720.0, uttttt_node,
                                       eq, i, j)

            fttttt_node = derivative_bundle(flux_x,
                                            (u_node, ut_node, utt_node, uttt_node,
                                             utttt_node, uttttt_node))
            multiply_add_to_node_vars!(F,
                                       1.0 / 720.0, fttttt_node,
                                       eq, i, j)
            gttttt_node = derivative_bundle(flux_y,
                                            (u_node, ut_node, utt_node, uttt_node,
                                             utttt_node, uttttt_node))
            multiply_add_to_node_vars!(G,
                                       1.0 / 720.0, gttttt_node,
                                       eq, i, j)
            F_node = get_node_vars(F, eq, i, j)
            G_node = get_node_vars(G, eq, i, j)
            for ii in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ii, i], F_node, eq, ii, j,
                                           el_x, el_y)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(res, lamy * D1[jj, j], G_node, eq, i, jj,
                                           el_x, el_y)
            end

            U_node = get_dissipation_node_vars(u, U, eq, i, j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, j, 1, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, j, 2, el_x, el_y)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(Ub, Vl[j], U_node, eq, i, 3, el_x, el_y)
            multiply_add_to_node_vars!(Ub, Vr[j], U_node, eq, i, 4, el_x, el_y)
        end
        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, f, res)

        @views cell_data = (u, ut, utt, uttt, utttt, uttttt, el_x, el_y)
        Fb_local = @view Fb[:, :, :, el_x, el_y]
        bflux_enzyme_5!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_local, aux,
                        solver, compute_bflux!)
    end
    return nothing
end
end # muladd
