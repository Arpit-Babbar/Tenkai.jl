# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function get_cfl(eq::AbstractEquations{2}, scheme::Scheme{<:cRKSolver}, param)
    @unpack solver, degree, correction_function = scheme
    @unpack cfl_safety_factor, cfl_style = param
    @unpack dissipation = scheme
    @assert (degree >= 0&&degree < 5) "Invalid degree"
    os_vector(v) = OffsetArray(v, OffsetArrays.Origin(0))
    cfl_radau = os_vector([1.0, 0.259, 0.170, 0.103, 0.069])
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

function setup_arrays(grid, scheme::Scheme{<:cRKSolver}, eq::AbstractEquations{2})
    RealT = eltype(grid.xc)
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
    u1 = gArray(nvar, nd, nd, nx, ny)
    ua = gArray(nvar, nx, ny)
    res = gArray(nvar, nd, nd, nx, ny)
    Fb = gArray(nvar, nd, 4, nx, ny)
    Ub = gArray(nvar, nd, 4, nx, ny)
    u1_b = copy(Ub)
    ub_N = gArray(nvar, nd, 4, nx, ny) # The final stage of cRK before communication

    # Cell residual cache

    nt = Threads.nthreads()
    cell_array_sizes = Dict(0 => 0, 1 => 11, 2 => 12, 3 => 15, 4 => 16)
    big_eval_data_sizes = Dict(0 => 0, 1 => 12, 2 => 32, 3 => 40, 4 => 56)
    small_eval_data_sizes = Dict(0 => 0, 1 => 4, 2 => 4, 3 => 4, 4 => 4)
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

    RealT = eltype(grid.xc)
    MArr = MArray{Tuple{nvariables(eq), nd, nd}, RealT}
    cell_arrays = alloc_for_threads(MArr, cell_array_size)

    MEval = MArray{Tuple{nvariables(eq), nd}, RealT}
    eval_data_big = alloc_for_threads(MEval, big_eval_data_size)

    MEval_small = MArray{Tuple{nvariables(eq), 1}, RealT}
    eval_data_small = alloc_for_threads(MEval_small, small_eval_data_size)

    eval_data = (; eval_data_big, eval_data_small)

    # Ghost values cache

    Marr = MArray{Tuple{nvariables(eq), 1}, RealT}

    ghost_cache = alloc_for_threads(Marr, 2)

    # KLUDGE - Rename this to LWFR cache
    cache = (; u1, ua, ub_N, res, Fb, Ub, u1_b, eval_data, cell_arrays, ghost_cache)
    return cache
end

function prolong_solution_to_face_and_ghosts!(u1, cache, eq::AbstractEquations{2}, grid,
                                              op,
                                              problem, scheme, aux, t, dt)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack u1_b = cache
    refresh!(u1_b)
    @unpack degree, Vl, Vr = op
    nd = degree + 1
    @threaded for element in CartesianIndices((0:(nx + 1), 0:(ny + 1))) # Loop over cells
        el_x, el_y = element[1], element[2]
        for j in 1:nd, i in 1:nd
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            multiply_add_to_node_vars!(u1_b, Vl[i], u_node, eq, j, 1, el_x, el_y)
            multiply_add_to_node_vars!(u1_b, Vr[i], u_node, eq, j, 2, el_x, el_y)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(u1_b, Vl[j], u_node, eq, i, 3, el_x, el_y)
            multiply_add_to_node_vars!(u1_b, Vr[j], u_node, eq, i, 4, el_x, el_y)
        end
    end

    return nothing
    end # timer
end

function get_bflux_function(solver::cRKSolver, degree, bflux)
    if bflux == extrapolate
        return extrap_bflux!
    else
        return eval_bflux!
    end
end

function extrap_bflux!(eq::AbstractEquations{2}, scheme::Scheme{<:cRKSolver}, grid,
                       cell_data, eval_data, xg, Vl, Vr, F, G, Fb, aux)
    extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb, aux)
end

function eval_bflux!(eq::AbstractEquations{2}, scheme::Scheme{<:cRK22}, grid,
                     cell_data, eval_data, xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
    nd = length(xg)
    refresh!(u) = fill!(u, zero(eltype(u)))

    (u, u2, el_x, el_y) = cell_data

    id = Threads.threadid()

    eval_data_big, eval_data_small = (eval_data.eval_data_big[id],
                                      eval_data.eval_data_small[id])

    refresh!.(eval_data_big)
    u2l, u2r, u2d, u2u = eval_data_big  # Pre-allocated arrays
    ftl, ftr, gtd, gtu = eval_data_small
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u2_node = get_node_vars(u2, eq, i, j)

        multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, j)
        multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, j)
        multiply_add_to_node_vars!(u2d, Vl[j], u2_node, eq, i)
        multiply_add_to_node_vars!(u2u, Vr[j], u2_node, eq, i)
    end

    for i in 1:nd
        x, y = xl + dx * xg[i], yd + dy * xg[i]

        u2l_node = get_node_vars(u2l, eq, i)
        u2r_node = get_node_vars(u2r, eq, i)
        u2d_node = get_node_vars(u2d, eq, i)
        u2u_node = get_node_vars(u2u, eq, i)

        fl = flux(xl, y, u2l_node, eq, 1)
        fr = flux(xr, y, u2r_node, eq, 1)
        gd = flux(x, yd, u2d_node, eq, 2)
        gu = flux(x, yu, u2u_node, eq, 2)

        # KLUDGE - Indices order needs to be changed, or something else
        # needs to be done to avoid cache misses
        set_node_vars!(Fb, fl, eq, i, 1)
        set_node_vars!(Fb, fr, eq, i, 2)
        set_node_vars!(Fb, gd, eq, i, 3)
        set_node_vars!(Fb, gu, eq, i, 4)
    end
end

function eval_bflux!(eq::AbstractEquations{2}, scheme::Scheme{<:cRK33}, grid,
                     cell_data, eval_data, xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
    nd = length(xg)
    refresh!(u) = fill!(u, zero(eltype(u)))

    (u, u3, el_x, el_y) = cell_data

    id = Threads.threadid()

    eval_data_big, eval_data_small = (eval_data.eval_data_big[id],
                                      eval_data.eval_data_small[id])

    refresh!.(eval_data_big)
    ul, ur, ud, uu, u3l, u3r, u3d, u3u = eval_data_big  # Pre-allocated arrays
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        u3_node = get_node_vars(u3, eq, i, j)

        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, j)
        multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, j)
        multiply_add_to_node_vars!(u3d, Vl[j], u3_node, eq, i)
        multiply_add_to_node_vars!(u3u, Vr[j], u3_node, eq, i)
    end

    for i in 1:nd
        x, y = xl + dx * xg[i], yd + dy * xg[i]

        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)

        u3l_node = get_node_vars(u3l, eq, i)
        u3r_node = get_node_vars(u3r, eq, i)
        u3d_node = get_node_vars(u3d, eq, i)
        u3u_node = get_node_vars(u3u, eq, i)

        fl = flux(xl, y, ul_node, eq, 1)
        fr = flux(xr, y, ur_node, eq, 1)
        gd = flux(x, yd, ud_node, eq, 2)
        gu = flux(x, yu, uu_node, eq, 2)

        f3l = flux(xl, y, u3l_node, eq, 1)
        f3r = flux(xr, y, u3r_node, eq, 1)
        g3d = flux(x, yd, u3d_node, eq, 2)
        g3u = flux(x, yu, u3u_node, eq, 2)

        # KLUDGE - Indices order needs to be changed, or something else
        # needs to be done to avoid cache misses
        multiply_add_to_node_vars!(Fb, 0.25, fl, 0.75, f3l, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 0.25, fr, 0.75, f3r, eq, i, 2)
        multiply_add_to_node_vars!(Fb, 0.25, gd, 0.75, g3d, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 0.25, gu, 0.75, g3u, eq, i, 4)
    end
end

function eval_bflux!(eq::AbstractEquations{2}, scheme::Scheme{<:cRK44}, grid,
                     cell_data, eval_data, xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
    nd = length(xg)
    refresh!(u) = fill!(u, zero(eltype(u)))

    u, u2, u3, u4, el_x, el_y = cell_data

    id = Threads.threadid()

    eval_data_big, eval_data_small = (eval_data.eval_data_big[id],
                                      eval_data.eval_data_small[id])

    refresh!.(eval_data_big)
    (ul, ur, ud, uu, u2l, u2r, u2d, u2u, u3l, u3r, u3d, u3u,
    u4l, u4r, u4d, u4u) = eval_data_big  # Pre-allocated arrays
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        u2_node = get_node_vars(u2, eq, i, j)
        u3_node = get_node_vars(u3, eq, i, j)
        u4_node = get_node_vars(u4, eq, i, j)

        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, j)
        multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, j)
        multiply_add_to_node_vars!(u2d, Vl[j], u2_node, eq, i)
        multiply_add_to_node_vars!(u2u, Vr[j], u2_node, eq, i)

        multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, j)
        multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, j)
        multiply_add_to_node_vars!(u3d, Vl[j], u3_node, eq, i)
        multiply_add_to_node_vars!(u3u, Vr[j], u3_node, eq, i)

        multiply_add_to_node_vars!(u4l, Vl[i], u4_node, eq, j)
        multiply_add_to_node_vars!(u4r, Vr[i], u4_node, eq, j)
        multiply_add_to_node_vars!(u4d, Vl[j], u4_node, eq, i)
        multiply_add_to_node_vars!(u4u, Vr[j], u4_node, eq, i)
    end

    for i in 1:nd
        x, y = xl + dx * xg[i], yd + dy * xg[i]

        ul_node = get_node_vars(ul, eq, i)
        ur_node = get_node_vars(ur, eq, i)
        ud_node = get_node_vars(ud, eq, i)
        uu_node = get_node_vars(uu, eq, i)

        u2l_node = get_node_vars(u2l, eq, i)
        u2r_node = get_node_vars(u2r, eq, i)
        u2d_node = get_node_vars(u2d, eq, i)
        u2u_node = get_node_vars(u2u, eq, i)

        u3l_node = get_node_vars(u3l, eq, i)
        u3r_node = get_node_vars(u3r, eq, i)
        u3d_node = get_node_vars(u3d, eq, i)
        u3u_node = get_node_vars(u3u, eq, i)

        u4l_node = get_node_vars(u4l, eq, i)
        u4r_node = get_node_vars(u4r, eq, i)
        u4d_node = get_node_vars(u4d, eq, i)
        u4u_node = get_node_vars(u4u, eq, i)

        fl = flux(xl, y, ul_node, eq, 1)
        fr = flux(xr, y, ur_node, eq, 1)
        gd = flux(x, yd, ud_node, eq, 2)
        gu = flux(x, yu, uu_node, eq, 2)

        f2l = flux(xl, y, u2l_node, eq, 1)
        f2r = flux(xr, y, u2r_node, eq, 1)
        g2d = flux(x, yd, u2d_node, eq, 2)
        g2u = flux(x, yu, u2u_node, eq, 2)

        f3l = flux(xl, y, u3l_node, eq, 1)
        f3r = flux(xr, y, u3r_node, eq, 1)
        g3d = flux(x, yd, u3d_node, eq, 2)
        g3u = flux(x, yu, u3u_node, eq, 2)

        f4l = flux(xl, y, u4l_node, eq, 1)
        f4r = flux(xr, y, u4r_node, eq, 1)
        g4d = flux(x, yd, u4d_node, eq, 2)
        g4u = flux(x, yu, u4u_node, eq, 2)

        # KLUDGE - Indices order needs to be changed, or something else
        # needs to be done to avoid cache misses
        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fl, 1.0 / 3.0, f2l, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, f3l, 1.0 / 6.0, f4l, eq, i, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fr, 1.0 / 3.0, f2r, eq, i, 2)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, f3r, 1.0 / 6.0, f4r, eq, i, 2)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gd, 1.0 / 3.0, g2d, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, g3d, 1.0 / 6.0, g4d, eq, i, 3)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gu, 1.0 / 3.0, g2u, eq, i, 4)
        multiply_add_to_node_vars!(Fb, 1.0 / 3.0, g3u, 1.0 / 6.0, g4u, eq, i, 4)
    end
end

function compute_face_residual!(eq::AbstractEquations{2}, grid, op, cache, problem,
                                scheme::Scheme{<:cRKSolver}, param, aux, t, dt, u1,
                                Fb, Ub, ua, res, scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack bl, br, xg, wg, degree = op
    nd = degree + 1
    nx, ny = grid.size
    @unpack dx, dy, xf, yf = grid
    @unpack numerical_flux = scheme
    @unpack blend = aux
    @unpack blend_face_residual_x!, blend_face_residual_y!, get_element_alpha = blend.subroutines
    @unpack u1_b = cache

    # Vertical faces, x flux
    @threaded for element in CartesianIndices((1:(nx + 1), 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        # This is face between elements (el_x-1, el_y), (el_x, el_y)
        x = xf[el_x]
        # ul, ur = get_node_vars(ua, eq, el_x - 1, el_y),
        #            get_node_vars(ua, eq, el_x, el_y)
        for jy in Base.OneTo(nd)
            y = yf[el_y] + xg[jy] * dy[el_y]
            ul, ur = (get_node_vars(u1_b, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(u1_b, eq, jy, 1, el_x, el_y))
            Fl, Fr = (get_node_vars(Fb, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(Fb, eq, jy, 1, el_x, el_y))
            Ul, Ur = (get_node_vars(Ub, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(Ub, eq, jy, 1, el_x, el_y))
            X = SVector{2}(x, y)
            Fn = numerical_flux(X, ul, ur, Fl, Fr, Ul, Ur, eq, 1)
            Fn, blend_factors = blend_face_residual_x!(el_x, el_y, jy, x, y, u1, ua,
                                                       eq, dt, grid, op,
                                                       scheme, param, Fn, aux,
                                                       res, scaling_factor)

            # These quantities won't be used so we can store numerical flux here
            set_node_vars!(Fb, Fn, eq, jy, 2, el_x - 1, el_y)
            set_node_vars!(Fb, Fn, eq, jy, 1, el_x, el_y)
            # for ix in Base.OneTo(nd)
            #    multiply_add_to_node_vars!(res,
            #                               blend_factors[1] * dt/dx[el_x-1] * br[ix], Fn,
            #                               eq,
            #                               ix, jy, el_x-1, el_y )

            #    multiply_add_to_node_vars!(res,
            #                               blend_factors[2] * dt/dx[el_x]   * bl[ix], Fn,
            #                               eq,
            #                               ix, jy, el_x, el_y )
            # end

            # r = @view res[:, :, jy, el_x-1, el_y]
            # multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
            #                            alpha[el_x-1,el_y]*dt/(dx[el_x-1]*wg[nd]), Fn,
            #                            eq, nd)

            # r = @view res[:, :, jy, el_x, el_y]
            # multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
            #                            - alpha[el_x,el_y]*dt/(dx[el_x]*wg[1]), Fn,
            #                            eq, 1)
        end
    end

    # Horizontal faces, y flux
    @threaded for element in CartesianIndices((1:nx, 1:(ny + 1))) # Loop over cells
        el_x, el_y = element[1], element[2]
        # This is the face between elements (el_x,el_y-1) and (el_x,el_y)
        y = yf[el_y]
        # ul, ur = get_node_vars(ua, eq, el_x, el_y - 1),
        #            get_node_vars(ua, eq, el_x, el_y)
        for ix in Base.OneTo(nd)
            x = xf[el_x] + xg[ix] * dx[el_x]
            ul, ur = get_node_vars(u1_b, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(u1_b, eq, ix, 3, el_x, el_y)
            Fl, Fr = get_node_vars(Fb, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(Fb, eq, ix, 3, el_x, el_y)
            Ul, Ur = get_node_vars(Ub, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(Ub, eq, ix, 3, el_x, el_y)
            X = SVector{2}(x, y)
            Fn = numerical_flux(X, ul, ur, Fl, Fr, Ul, Ur, eq, 2)
            Fn, blend_factors = blend_face_residual_y!(el_x, el_y, ix, x, y,
                                                       u1, ua, eq, dt, grid, op,
                                                       scheme, param, Fn, aux,
                                                       res, scaling_factor)
            # These quantities won't be used so we can store numerical flux here
            set_node_vars!(Fb, Fn, eq, ix, 4, el_x, el_y - 1)
            set_node_vars!(Fb, Fn, eq, ix, 3, el_x, el_y)
            # for jy in Base.OneTo(nd)
            #    multiply_add_to_node_vars!(res,
            #                               blend_factors[1] * dt/dy[el_y-1] * br[jy], Fn,
            #                               eq,
            #                               ix, jy, el_x, el_y-1 )
            #    multiply_add_to_node_vars!(res,
            #                               blend_factors[2] * dt/dy[el_y]   * bl[jy], Fn,
            #                               eq,
            #                               ix, jy, el_x, el_y   )
            # end

            # r = @view res[:,ix,:,el_x,el_y-1]

            # multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
            #                            alpha[el_x,el_y-1] * dt/(dy[el_y-1]*wg[nd]),
            #                            Fn,
            #                            eq, nd
            #                            )

            # r = @view res[:,ix,:,el_x,el_y]

            # multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
            #                            - alpha[el_x,el_y] * dt/(dy[el_y]*wg[1]),
            #                            Fn,
            #                            eq, 1)
        end
    end

    # This loop is slow with Threads.@threads so we use Polyster.jl threads
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        alpha = get_element_alpha(blend, el_x, el_y) # TODO - Use a function to get this
        one_m_alp = 1.0 - alpha
        for ix in Base.OneTo(nd)
            for jy in Base.OneTo(nd)
                Fl = get_node_vars(Fb, eq, jy, 1, el_x, el_y)
                Fr = get_node_vars(Fb, eq, jy, 2, el_x, el_y)
                Fd = get_node_vars(Fb, eq, ix, 3, el_x, el_y)
                Fu = get_node_vars(Fb, eq, ix, 4, el_x, el_y)
                # multiply_add_to_node_vars!(res,
                #                            one_m_alp * dt/dy[el_y] * br[jy], Fu,
                #                            eq,
                #                            ix, jy, el_x, el_y)
                # multiply_add_to_node_vars!(res,
                #                            one_m_alp * dt/dy[el_y] * bl[jy], Fd,
                #                            eq,
                #                            ix, jy, el_x, el_y)

                # multiply_add_to_node_vars!(res,
                #                            one_m_alp * dt/dx[el_x] * br[ix], Fr,
                #                            eq,
                #                            ix, jy, el_x, el_y )
                # multiply_add_to_node_vars!(res,
                #                            one_m_alp * dt/dx[el_x] * bl[ix], Fl,
                #                            eq,
                #                            ix, jy, el_x, el_y )
                for n in eachvariable(eq)
                    res[n, ix, jy, el_x, el_y] += one_m_alp * dt / dy[el_y] *
                                                  br[jy] * Fu[n]
                    res[n, ix, jy, el_x, el_y] += one_m_alp * dt / dy[el_y] *
                                                  bl[jy] * Fd[n]
                    res[n, ix, jy, el_x, el_y] += one_m_alp * dt / dx[el_x] *
                                                  br[ix] * Fr[n]
                    res[n, ix, jy, el_x, el_y] += one_m_alp * dt / dx[el_x] *
                                                  bl[ix] * Fl[n]
                end
            end
        end
    end

    add_low_order_face_residual!(get_element_alpha, eq, grid, op, aux, dt, Fb, res)
    return nothing
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractEquations{2}, grid, op,
                                    problem, scheme::Scheme{<:cRK22}, aux, t, dt, cache)
    @timeit aux.timer "Cell Residual" begin
        @unpack source_terms = problem
        @unpack xg, wg, Dm, D1, Vl, Vr = op
        nd = length(xg)
        nx, ny = grid.size
        refresh!(u) = fill!(u, zero(eltype(u)))
        @unpack bflux_ind = scheme.bflux
        @unpack blend = aux

        @unpack blend_cell_residual! = aux.blend.subroutines
        @unpack compute_bflux! = scheme.bflux
        get_dissipation_node_vars = scheme.dissipation
        @unpack eval_data, cell_arrays, ua, u1, res, Fb, Ub, = cache

        refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

        @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
            el_x, el_y = element[1], element[2]
            dx, dy = grid.dx[el_x], grid.dy[el_y]
            xc, yc = grid.xc[el_x], grid.yc[el_y]
            lamx, lamy = dt / dx, dt / dy

            id = Threads.threadid()
            u2, F, G, U, S = cell_arrays[id]

            u2 .= @view u1[:, :, :, el_x, el_y]

            u1_ = @view u1[:, :, :, el_x, el_y]
            r1 = @view res[:, :, :, el_x, el_y]
            Ub_ = @view Ub[:, :, :, el_x, el_y]

            # Solution points
            for j in 1:nd, i in 1:nd
                x = xc - 0.5 * dx + xg[i] * dx
                y = yc - 0.5 * dy + xg[j] * dy
                u_node = get_node_vars(u1_, eq, i, j)
                flux1, flux2 = flux(x, y, u_node, eq)
                for ii in Base.OneTo(nd)
                    # ut              += -lam * D * f for each variable
                    # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                    multiply_add_to_node_vars!(u2, -0.5 * lamx * Dm[ii, i], flux1, eq,
                                               ii, j)
                end
                for jj in Base.OneTo(nd)
                    # C += -lam*g*Dm' for each variable
                    # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                    multiply_add_to_node_vars!(u2, -0.5 * lamy * Dm[jj, j], flux2, eq,
                                               i, jj)
                end
            end

            # Add source term contribution to u2 and some to S
            for j in 1:nd, i in 1:nd
                x = xc - 0.5 * dx + xg[i] * dx
                y = yc - 0.5 * dy + xg[j] * dy
                u_node = get_node_vars(u1_, eq, i, j)
                X = SVector(x, y)
                s_node = calc_source(u_node, X, t, source_terms, eq)
                multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i, j)
            end

            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                x = xc - 0.5 * dx + xg[i] * dx
                y = yc - 0.5 * dy + xg[j] * dy

                u2_node = get_node_vars(u2, eq, i, j)

                flux1, flux2 = flux(x, y, u2_node, eq)

                set_node_vars!(U, u2_node, eq, i, j)
                set_node_vars!(F, flux1, eq, i, j)
                set_node_vars!(G, flux2, eq, i, j)

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

                u2_node = get_node_vars(u2, eq, i, j)
                X = SVector(x, y)
                S_node = calc_source(u2_node, X, t + 0.5 * dt, source_terms, eq)
                multiply_add_to_node_vars!(r1, -dt, S_node, eq, i, j)

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
            blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                                 dy,
                                 grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
            # Interpolate to faces
            @views cell_data = (u1_, u2, el_x, el_y)
            @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                                  F, G, Fb[:, :, :, el_x, el_y], aux)
        end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractEquations{2}, grid, op,
                                    problem, scheme::Scheme{<:cRK33}, aux, t,
                                    dt, cache)
    @timeit aux.timer "Cell Residual" begin
        @unpack source_terms = problem
        @unpack xg, wg, Dm, D1, Vl, Vr = op
        nd = length(xg)
        nx, ny = grid.size
        refresh!(u) = fill!(u, zero(eltype(u)))
        @unpack bflux_ind = scheme.bflux
        @unpack blend = aux

        @unpack blend_cell_residual! = aux.blend.subroutines
        @unpack compute_bflux! = scheme.bflux
        get_dissipation_node_vars = scheme.dissipation
        @unpack eval_data, cell_arrays, ua, u1, res, Fb, Ub, = cache

        refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

        @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
            el_x, el_y = element[1], element[2]
            dx, dy = grid.dx[el_x], grid.dy[el_y]
            xc, yc = grid.xc[el_x], grid.yc[el_y]
            lamx, lamy = dt / dx, dt / dy

            u1_ = @view u1[:, :, :, el_x, el_y]
            r1 = @view res[:, :, :, el_x, el_y]
            Ub_ = @view Ub[:, :, :, el_x, el_y]

            id = Threads.threadid()
            u2, u3, F, G, U, S = cell_arrays[id]

            u2 .= u1_
            u3 .= u2

            # Solution points
            for j in 1:nd, i in 1:nd
                x = xc - 0.5 * dx + xg[i] * dx
                y = yc - 0.5 * dy + xg[j] * dy
                u_node = get_node_vars(u1_, eq, i, j)
                flux1, flux2 = flux(x, y, u_node, eq)
                set_node_vars!(F, 0.25 * flux1, eq, i, j)
                set_node_vars!(G, 0.25 * flux2, eq, i, j)
                set_node_vars!(U, 0.25 * u_node, eq, i, j)
                for ii in Base.OneTo(nd)
                    # ut              += -lam * D * f for each variable
                    # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                    multiply_add_to_node_vars!(u2, -lamx / 3.0 * Dm[ii, i], flux1, eq,
                                               ii, j)
                end
                for jj in Base.OneTo(nd)
                    # C += -lam*g*Dm' for each variable
                    # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                    multiply_add_to_node_vars!(u2, -lamy / 3.0 * Dm[jj, j], flux2, eq,
                                               i, jj)
                end
            end

            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                x = xc - 0.5 * dx + xg[i] * dx
                y = yc - 0.5 * dy + xg[j] * dy

                u2_node = get_node_vars(u2, eq, i, j)

                flux1, flux2 = flux(x, y, u2_node, eq)

                for ii in Base.OneTo(nd)
                    # ut              += -lam * D * f for each variable
                    # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                    multiply_add_to_node_vars!(u3, -2.0 * lamx / 3.0 * Dm[ii, i], flux1,
                                               eq, ii, j)
                end
                for jj in Base.OneTo(nd)
                    # C += -lam*g*Dm' for each variable
                    # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                    multiply_add_to_node_vars!(u3, -2.0 * lamy / 3.0 * Dm[jj, j], flux2,
                                               eq, i, jj)
                end
            end

            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                x = xc - 0.5 * dx + xg[i] * dx
                y = yc - 0.5 * dy + xg[j] * dy

                u3_node = get_node_vars(u3, eq, i, j)

                flux1, flux2 = flux(x, y, u3_node, eq)

                multiply_add_to_node_vars!(F, 0.75, flux1, eq, i, j)
                multiply_add_to_node_vars!(G, 0.75, flux2, eq, i, j)
                multiply_add_to_node_vars!(U, 0.75, u3_node, eq, i, j)

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
            blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                                 dy,
                                 grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
            # Interpolate to faces
            @views cell_data = (u1_, u3, el_x, el_y)
            @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                                  F, G, Fb[:, :, :, el_x, el_y], aux)
        end
    end # timer
end

# function compute_cell_residual_cRK!(eq::AbstractEquations{2}, grid, op,
#                                     problem, scheme::Scheme{<:cRK44}, aux, t,
#                                     dt, cache)
#     @timeit aux.timer "Cell Residual" begin
#         @unpack source_terms = problem
#         @unpack xg, wg, Dm, D1, Vl, Vr = op
#         nd = length(xg)
#         nx, ny = grid.size
#         refresh!(u) = fill!(u, zero(eltype(u)))
#         @unpack bflux_ind = scheme.bflux
#         @unpack blend = aux

#         @unpack blend_cell_residual! = aux.blend.subroutines
#         @unpack compute_bflux! = scheme.bflux
#         get_dissipation_node_vars = scheme.dissipation
#         @unpack eval_data, cell_arrays, ua, u1, res, Fb, Ub, = cache

#         refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

#         @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
#             el_x, el_y = element[1], element[2]
#             dx, dy = grid.dx[el_x], grid.dy[el_y]
#             xc, yc = grid.xc[el_x], grid.yc[el_y]
#             lamx, lamy = dt / dx, dt / dy

#             u1_ = @view u1[:, :, :, el_x, el_y]
#             r1 = @view res[:, :, :, el_x, el_y]
#             Ub_ = @view Ub[:, :, :, el_x, el_y]

#             id = Threads.threadid()
#             u2, u3, u4, F, G, U, S = cell_arrays[id]

#             u2 .= u1_
#             u3 .= u1_
#             u4 .= u1_

#             # Solution points
#             for j in 1:nd, i in 1:nd
#                 x = xc - 0.5 * dx + xg[i] * dx
#                 y = yc - 0.5 * dy + xg[j] * dy
#                 u_node = get_node_vars(u1_, eq, i, j)
#                 flux1, flux2 = flux(x, y, u_node, eq)
#                 set_node_vars!(F, flux1 / 6.0, eq, i, j)
#                 set_node_vars!(G, flux2 / 6.0, eq, i, j)
#                 set_node_vars!(U, u_node / 6.0, eq, i, j)
#                 for ii in Base.OneTo(nd)
#                     # ut              += -lam * D * f for each variable
#                     # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
#                     multiply_add_to_node_vars!(u2, -0.5 * lamx * Dm[ii, i], flux1, eq,
#                                                ii, j)
#                 end
#                 for jj in Base.OneTo(nd)
#                     # C += -lam*g*Dm' for each variable
#                     # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
#                     multiply_add_to_node_vars!(u2, -0.5 * lamy * Dm[jj, j], flux2, eq,
#                                                i, jj)
#                 end
#             end

#             # Add source term contribution to u2 and some to S
#             for j in 1:nd, i in 1:nd
#                 x = xc - 0.5 * dx + xg[i] * dx
#                 y = yc - 0.5 * dy + xg[j] * dy
#                 u_node = get_node_vars(u1_, eq, i, j)
#                 X = SVector(x, y)
#                 s_node = calc_source(u_node, X, t, source_terms, eq)
#                 multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i, j)
#                 set_node_vars!(S, s_node / 6.0, eq, i, j)
#             end

#             for j in Base.OneTo(nd), i in Base.OneTo(nd)
#                 x = xc - 0.5 * dx + xg[i] * dx
#                 y = yc - 0.5 * dy + xg[j] * dy

#                 u2_node = get_node_vars(u2, eq, i, j)

#                 flux1, flux2 = flux(x, y, u2_node, eq)

#                 multiply_add_to_node_vars!(F, 1.0 / 3.0, flux1, eq, i, j)
#                 multiply_add_to_node_vars!(G, 1.0 / 3.0, flux2, eq, i, j)
#                 multiply_add_to_node_vars!(U, 1.0 / 3.0, u2_node, eq, i, j)

#                 for ii in Base.OneTo(nd)
#                     # ut              += -lam * D * f for each variable
#                     # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
#                     multiply_add_to_node_vars!(u3, -0.5 * lamx * Dm[ii, i], flux1, eq,
#                                                ii, j)
#                 end
#                 for jj in Base.OneTo(nd)
#                     # C += -lam*g*Dm' for each variable
#                     # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
#                     multiply_add_to_node_vars!(u3, -0.5 * lamy * Dm[jj, j], flux2, eq,
#                                                i, jj)
#                 end
#             end

#             # Add source term contribution to u3 and some to S
#             for j in 1:nd, i in 1:nd
#                 x = xc - 0.5 * dx + xg[i] * dx
#                 y = yc - 0.5 * dy + xg[j] * dy
#                 u2_node = get_node_vars(u2, eq, i, j)
#                 X = SVector(x, y)
#                 s2_node = calc_source(u2_node, X, t + 0.5 * dt, source_terms, eq)
#                 multiply_add_to_node_vars!(u3, 0.5 * dt, s2_node, eq, i, j)
#                 multiply_add_to_node_vars!(S, 1.0 / 3.0, s2_node, eq, i, j)
#             end

#             for j in Base.OneTo(nd), i in Base.OneTo(nd)
#                 x = xc - 0.5 * dx + xg[i] * dx
#                 y = yc - 0.5 * dy + xg[j] * dy

#                 u3_node = get_node_vars(u3, eq, i, j)

#                 flux1, flux2 = flux(x, y, u3_node, eq)

#                 multiply_add_to_node_vars!(F, 1.0 / 3.0, flux1, eq, i, j)
#                 multiply_add_to_node_vars!(G, 1.0 / 3.0, flux2, eq, i, j)
#                 multiply_add_to_node_vars!(U, 1.0 / 3.0, u3_node, eq, i, j)

#                 for ii in Base.OneTo(nd)
#                     # ut              += -lam * D * f for each variable
#                     # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
#                     multiply_add_to_node_vars!(u4, -lamx * Dm[ii, i], flux1, eq, ii, j)
#                 end
#                 for jj in Base.OneTo(nd)
#                     # C += -lam*g*Dm' for each variable
#                     # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
#                     multiply_add_to_node_vars!(u4, -lamy * Dm[jj, j], flux2, eq, i, jj)
#                 end
#             end

#             # Add source term contribution to u4 and some to S
#             for j in 1:nd, i in 1:nd
#                 x = xc - 0.5 * dx + xg[i] * dx
#                 y = yc - 0.5 * dy + xg[j] * dy
#                 u3_node = get_node_vars(u3, eq, i, j)
#                 X = SVector(x, y)
#                 s3_node = calc_source(u3_node, X, t + 0.5 * dt, source_terms, eq)
#                 multiply_add_to_node_vars!(u4, dt, s3_node, eq, i, j)
#                 multiply_add_to_node_vars!(S, 1.0 / 3.0, s3_node, eq, i, j)
#             end

#             for j in Base.OneTo(nd), i in Base.OneTo(nd)
#                 x = xc - 0.5 * dx + xg[i] * dx
#                 y = yc - 0.5 * dy + xg[j] * dy

#                 u4_node = get_node_vars(u4, eq, i, j)

#                 flux1, flux2 = flux(x, y, u4_node, eq)

#                 multiply_add_to_node_vars!(F, 1.0 / 6.0, flux1, eq, i, j)
#                 multiply_add_to_node_vars!(G, 1.0 / 6.0, flux2, eq, i, j)
#                 multiply_add_to_node_vars!(U, 1.0 / 6.0, u4_node, eq, i, j)

#                 F_node = get_node_vars(F, eq, i, j)
#                 G_node = get_node_vars(G, eq, i, j)
#                 for ii in Base.OneTo(nd)
#                     # res              += -lam * D * F for each variable
#                     # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
#                     multiply_add_to_node_vars!(r1, lamx * D1[ii, i], F_node, eq, ii, j)
#                 end
#                 for jj in Base.OneTo(nd)
#                     # C += -lam*g*Dm' for each variable
#                     # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
#                     multiply_add_to_node_vars!(r1, lamy * D1[jj, j], G_node, eq, i, jj)
#                 end

#                 X = SVector(x, y)
#                 s4_node = calc_source(u4_node, X, t + dt, source_terms, eq)
#                 multiply_add_to_node_vars!(S, 1.0 / 6.0, s4_node, eq, i, j)
#                 S_node = get_node_vars(S, eq, i, j)
#                 multiply_add_to_node_vars!(r1, -dt, S_node, eq, i, j)

#                 # KLUDGE - update to v1.8 and call with @inline
#                 # Give u1_ or U depending on dissipation model
#                 U_node = get_dissipation_node_vars(u1_, U, eq, i, j)

#                 # Ub = UT * V
#                 # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
#                 multiply_add_to_node_vars!(Ub_, Vl[i], U_node, eq, j, 1)
#                 multiply_add_to_node_vars!(Ub_, Vr[i], U_node, eq, j, 2)

#                 # Ub = U * V
#                 # Ub[i] += ∑_j U[i,j]*V[j]
#                 multiply_add_to_node_vars!(Ub_, Vl[j], U_node, eq, i, 3)
#                 multiply_add_to_node_vars!(Ub_, Vr[j], U_node, eq, i, 4)
#             end

#             u = @view u1[:, :, :, el_x, el_y]
#             blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
#                                  dy,
#                                  grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
#             # Interpolate to faces
#             @views cell_data = (u1_, u2, u3, u4, el_x, el_y)
#             @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
#                                   F, G, Fb[:, :, :, el_x, el_y], aux)
#         end
#     end # timer
# end
end # muladd
