using TaylorDiff

# struct DerivativeBundleCache{Constants,TaylorArrays, N}
#     # Cache fields for derivatives up to order N
#   constants::Constants
#.  taylor_arrays::TaylorArrays
# end

# TODO: Add as DerivativeBundleCache{N}. For now, it will be a NamedTuple
function derivative_bundle!(func_out1, func1!, bundle::NTuple{N}, cache) where {N}
    # Use bundle values to set up cache values

    in_array = get_u_array(cache, Val(N))
    set_arr_A_B!(in_array.value, bundle[1])
    for i in 1:N
        set_arr_A_B!(in_array.partials[i], bundle[i + 1])
    end

    func1!(taylor_arrays, bundle, cache.constants...)
end

function derivative_bundle!(func_out1, func_out2, func1!, func2!, bundle::NTuple{N},
                            cache) where {N}
    # Use bundle values to set up cache values
end

@inline function Tenkai.set_node_vars!(u, u_node::TaylorScalar{<:Any}, eq, indices...)
    for v in eachvariable(eq)
        u[v, indices...] = u_node[v]
    end
    return nothing
end

function reset_f_u_ut!(f::StaticArray, u::StaticArray, ut::StaticArray)
    z = zero(eltype(f))
    @turbo for i in eachindex(f)
        f[i] = z
        u[i] = z
        ut[i] = z
    end
end

function reset_arr!(arr)
    z = zero(eltype(arr))
    @turbo for i in eachindex(arr)
        arr[i] = z
    end
end

function reset_Fb_Ub!(Fb::AbstractArray, Ub::AbstractArray)
    z = zero(eltype(Fb))
    @turbo for i in eachindex(Fb)
        Fb[i] = z
        Ub[i] = z
    end
end

function set_arr_A_B!(A::StaticArray, B::StaticArray)
    @turbo for i in eachindex(A)
        A[i] = B[i]
    end
end

@inline function eachpoint(scheme)
    nd = scheme_degree_plus_one(scheme)
    return Iterators.product(Base.OneTo(nd), Base.OneTo(nd)) # TODO - Can this be made better?
end

@inline @inbounds function FGS2res!(res, F, G, S, D1, dt, lamx, lamy, ::Val{nd},
                                    ::Val{nvar}) where {
                                                        nd, nvar}

    # TODO - Should these be loops be merged?

    @turbo for j in Base.OneTo(nd), i in Base.OneTo(nd), ii in Base.OneTo(nd),
               n in Base.OneTo(nvar)

        res[n, ii, j] += lamx * D1[ii, i] * F[n, i, j]
        # res[n, i, ii] += lamy * D1[ii, j] * G[n, i, j]
    end

    @turbo for j in Base.OneTo(nd), ii in Base.OneTo(nd), i in Base.OneTo(nd),
               n in Base.OneTo(nvar)
        # res[n, ii, j] += lamx * D1[ii, i] * F[n, i, j]
        res[n, i, ii] += lamy * D1[ii, j] * G[n, i, j]
    end

    # @turbo for index in eachindex(S)
    #     res[index] -= dt * S[index]
    # end

    @turbo for j in Base.OneTo(nd), i in Base.OneTo(nd), n in Base.OneTo(nvar)
        res[n, i, j] -= dt * S[n, i, j]
    end
end

@inline @inbounds function compute_fluxes_and_sources_array!(f_g_s, u, local_grid,
                                                             op, source_terms, eq, scheme)
    dx, dy, xc, yc, _, _, t, dt = local_grid
    @inbounds for (i, j) in eachpoint(scheme)
        @unpack xg = op
        x = xc - 0.5 * dx + dx * xg[i]
        y = yc - 0.5 * dy + dy * xg[j]

        u_node = get_node_vars(u, eq, i, j)
        flux1, flux2 = Tenkai.flux(x, y, u_node, eq)
        set_node_vars!(f_g_s, flux1, eq, 1, i, j)
        set_node_vars!(f_g_s, flux2, eq, 2, i, j)

        # TODO - Also needs to be differentiated w.r.t t
        s_node = calc_source(u_node, (x, y), t, source_terms, eq)

        set_node_vars!(f_g_s, s_node, eq, 3, i, j)
    end
end

@inline @inbounds function set_U_u!(U::StaticArray, u_marr::StaticArray, u_arr)
    @turbo for i in eachindex(u_marr)
        U[i] = u_arr[i]
        u_marr[i] = u_arr[i]
    end
end

@inline @inbounds function set_F_G_S!(F::StaticArray, G::StaticArray, S::StaticArray,
                                      f::StaticArray, g::StaticArray, s::StaticArray)
    @turbo for i in eachindex(f)
        F[i] = f[i]
        G[i] = g[i]
        S[i] = s[i]
    end
end

@inline @inbounds function set_F_G_S!(F::StaticArray, G::StaticArray, S::StaticArray,
                                      f_g_s, ::Val{nd}, ::Val{nvar}) where {nd, nvar}
    @turbo for j in Base.OneTo(nd), i in Base.OneTo(nd), n in Base.OneTo(nvar)
        F[n, i, j] = f_g_s[n, 1, i, j]
        G[n, i, j] = f_g_s[n, 2, i, j]
        S[n, i, j] = f_g_s[n, 3, i, j]
    end
end

@inline @inbounds function compute_ut!(ut, f_g_s, op, local_grid, eq,
                                       nd_val::Val{nd}, nvar_val::Val{nvar},
                                       scaling_factor = 1.0) where {nd, nvar}
    @unpack Dm = op
    _, _, _, _, lamx, lamy, t, dt = local_grid
    # @turbo for j in Base.OneTo(nd), i in Base.OneTo(nd), n in Base.OneTo(nvar)
    # Surprisingly, this is faster than LoopVectorization which doesn't make
    # sense to me. However, in that case, it should be merged with flux computation loop.
    reset_arr!(ut)
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        flux1 = get_node_vars(f_g_s, eq, 1, i, j)
        flux2 = get_node_vars(f_g_s, eq, 2, i, j)
        s_node = get_node_vars(f_g_s, eq, 3, i, j)
        for ii in Base.OneTo(nd) # TODO - This inner loop causes cache misses
            # TODO - This looks like cache misses. Maybe they can be kept in separate arrays?
            # Well, this will be better than separate arrays.
            # ut[n, ii, j] = ut[n, ii, j] - lamx * Dm[ii, i] * f_g_s[n, 1, i, j]
            # ut[n, i, ii] = ut[n, i, ii] - lamy * Dm[ii, j] * f_g_s[n, 2, i, j]
            multiply_add_to_node_vars!(ut, -scaling_factor * lamx * Dm[ii, i], flux1, eq,
                                       ii, j)
            # TODO - Is it actually beneficial to have merged this loops?
            multiply_add_to_node_vars!(ut, -scaling_factor * lamy * Dm[ii, j], flux2, eq, i,
                                       ii)
        end
        # ut[n, i, j] = ut[n, i, j] + dt * f_g_s[n, 3, i, j]
        multiply_add_to_node_vars!(ut, scaling_factor * dt, s_node, eq, i, j)
    end
end

@inline @inbounds function add_to_F_G_S!(F, G, S, f, g,
                                         s, fac::Real)
    @turbo for i in eachindex(f)
        F[i] = F[i] + fac * f[i]
        G[i] = G[i] + fac * g[i]
        S[i] = S[i] + fac * s[i]
    end
end

@inline @inbounds function add_to_F_G_S!(F, G, S, f_g_s, fac::Real, ::Val{nd},
                                         ::Val{nvar}) where {nd, nvar}
    @turbo for j in Base.OneTo(nd), i in Base.OneTo(nd), n in Base.OneTo(nvar)
        F[n, i, j] = F[n, i, j] + fac * f_g_s[n, 1, i, j]
        G[n, i, j] = G[n, i, j] + fac * f_g_s[n, 2, i, j]
        S[n, i, j] = S[n, i, j] + fac * f_g_s[n, 3, i, j]
    end
end

@inline @inbounds function add_to_U!(U, ut::StaticArray, fac::Real)
    @turbo for i in eachindex(ut)
        U[i] = U[i] + fac * ut[i]
    end
end

@inline @inbounds function extrapolate_to_faces!(Ub, U, op, ::Val{nd},
                                                 ::Val{nvar}) where {nd, nvar}
    @unpack Vl, Vr = op
    # @turbo for j in Base.OneTo(nd), i in Base.OneTo(nd), n in Base.OneTo(nvar)
    @turbo for n in axes(U, 1), j in axes(U, 2), i in axes(U, 2)
        Ub[n, j, 1] += Vl[i] * U[n, i, j]
        Ub[n, j, 2] += Vr[i] * U[n, i, j]

        Ub[n, i, 3] += Vl[j] * U[n, i, j]
        Ub[n, i, 4] += Vr[j] * U[n, i, j]
    end
end

function construct_taylor_arrays(ArrayType, degree::Val{N}) where {N}
    nt = Threads.nthreads()
    # The first one is without derivatives
    constructor() = zero(ArrayType)
    cache = (SVector{nt}((constructor() for _ in Base.OneTo(nt))),)
    for i in 1:N
        cache = (cache...,
                 SVector{nt}((TaylorArray{i}(constructor())
                              for _ in Base.OneTo(nt))))
    end
    return cache
end

function setup_arrays(grid, scheme::Scheme{<:LWTDEltWise}, eq::AbstractEquations{2})
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
    nvar = nvariables(eq)
    nd = degree + 1
    nx, ny = grid.size
    u1 = gArray(nvar, nd, nd, nx, ny)
    ua = gArray(nvar, nx, ny)
    res = gArray(nvar, nd, nd, nx, ny)
    Fb = gArray(nvar, nd, 4, nx, ny)
    Ub = gArray(nvar, nd, 4, nx, ny)

    # Cell residual cache

    nt = Threads.nthreads()
    cell_array_sizes = Dict(1 => 11, 2 => 12, 3 => 15, 4 => 16)
    big_eval_data_sizes = Dict(1 => 12, 2 => 32, 3 => 40, 4 => 56)
    small_eval_data_sizes = Dict(1 => 4, 2 => 4, 3 => 4, 4 => 4)
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

    MArr = MArray{Tuple{nvariables(eq), nd, nd}, Float64}
    cell_arrays = alloc_for_threads(MArr, cell_array_size)

    nt = Threads.nthreads()

    f_taylor_constructor = MArray{Tuple{nvariables(eq), 3, nd, nd}, Float64}
    f_taylor_arrays = construct_taylor_arrays(f_taylor_constructor, Val(degree))

    u_taylor_constructor = MArray{Tuple{nvar, nd, nd}, Float64}
    u_taylor_arrays = construct_taylor_arrays(u_taylor_constructor, Val(degree))

    fb_eval() = zero(MArray{Tuple{nvariables(eq), 4, nd}, Float64})
    ub_eval() = zero(MArray{Tuple{nvariables(eq), 4, nd}, Float64})
    fb_array() = TaylorArray{degree}(fb_eval())
    ub_array() = TaylorArray{degree}(ub_eval())
    fb_arrays = SVector{nt}((fb_array() for _ in Base.OneTo(nt)))
    ub_arrays = SVector{nt}((ub_array() for _ in Base.OneTo(nt)))

    f_g_s_ut() = SVector{4}(MArr(undef) for _ in Base.OneTo(4))

    f_g_s_ut_tuple() = SVector{nd}(f_g_s_ut() for _ in Base.OneTo(nd))

    f_g_s_ut_tuples = SVector{nt}((f_g_s_ut_tuple() for _ in Base.OneTo(nt)))

    MEval = MArray{Tuple{nvariables(eq), nd}, Float64}
    eval_data_big = alloc_for_threads(MEval, big_eval_data_size)

    MEval_small = MArray{Tuple{nvariables(eq), 1}, Float64}
    eval_data_small = alloc_for_threads(MEval_small, small_eval_data_size)

    eval_data = (; eval_data_big, eval_data_small)

    # Ghost values cache

    Marr = MArray{Tuple{nvariables(eq), 1}, Float64}

    ghost_cache = alloc_for_threads(Marr, 2)

    # KLUDGE - Rename this to LWFR cache
    cache = (; u1, ua, res, Fb, Ub,
             eval_data, cell_arrays,
             f_taylor_arrays,
             u_taylor_arrays,
             f_g_s_ut_tuples, ub_arrays,
             ghost_cache, fb_arrays)
    return cache
end

function compute_cell_residual_1!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWTDEltWise},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = scheme_degree_plus_one(scheme)
    nd_val = Val(nd)
    nvar = nvariables(eq)
    nvar_val = Val(nvar)
    nx, ny = grid.size

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    @unpack eval_data, cell_arrays, f_taylor_arrays, u_taylor_arrays = cache

    f_g_s_arrays, df_g_s_arrays = f_taylor_arrays
    _, u_du_arrays = u_taylor_arrays

    reset_arr!(res)
    reset_Fb_Ub!(Fb, Ub)
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (dx, dy, xc, yc, lamx, lamy, t, dt)

        id = Threads.threadid()
        F, G, U, S = cell_arrays[id]

        f_g_s = f_g_s_arrays[id]
        df_g_s = df_g_s_arrays[id]
        u_du_array = u_du_arrays[id]

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        u = u_du_array.value
        set_U_u!(U, u, u1_)

        compute_fluxes_and_sources_array!(f_g_s, u, local_grid, op, source_terms, eq, scheme)

        # TODO - Is this the best we can do for performance? I think it is best if we pass f_g_s.
        set_F_G_S!(F, G, S, f_g_s, nd_val, nvar_val)

        # This will be done with loop vectorization
        ut = u_du_array.partials[1]
        compute_ut!(ut, f_g_s, op, local_grid, eq, nd_val, nvar_val)

        add_to_U!(U, ut, 0.5)

        compute_fluxes_and_sources_array!(df_g_s, u_du_array, local_grid, op,
                                          source_terms, eq, scheme)

        add_to_F_G_S!(F, G, S, df_g_s.partials[1], 0.5, nd_val, nvar_val)

        FGS2res!(r1, F, G, S, D1, dt, lamx, lamy, nd_val, nvar_val)

        extrapolate_to_faces!(Ub_, U, op, nd_val, nvar_val)

        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u1_, nothing, res)

        fb = cache.fb_arrays[id]
        ub = cache.ub_arrays[id]
        Fb_loc = @view Fb[:, :, :, el_x, el_y]
        bflux_data = (Fb_loc, fb, ub, u_du_array, local_grid, op, nd_val, nvar_val)
        compute_bflux_1!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                         F, G, Fb_loc, aux, compute_bflux!)
    end
    return nothing
end

function compute_cell_residual_2!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWTDEltWise},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = scheme_degree_plus_one(scheme)
    nd_val = Val(nd)
    nvar = nvariables(eq)
    nvar_val = Val(nvar)
    nx, ny = grid.size

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    @unpack eval_data, cell_arrays, f_taylor_arrays, u_taylor_arrays = cache

    f_g_s_arrays, df_g_s_arrays, ddf_g_s_arrays = f_taylor_arrays
    _, u_du_arrays, u_du_ddu_arrays = u_taylor_arrays

    reset_arr!(res)
    reset_Fb_Ub!(Fb, Ub)
    @inbounds @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (dx, dy, xc, yc, lamx, lamy, t, dt)

        id = Threads.threadid()

        el_x, el_y = element[1], element[2]
        id = Threads.threadid()
        F, G, U, S = cell_arrays[id]

        f_g_s = f_g_s_arrays[id]
        df_g_s = df_g_s_arrays[id]
        ddf_g_s = ddf_g_s_arrays[id]
        u_du_array = u_du_arrays[id]
        u_du_ddu_array = u_du_ddu_arrays[id]

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        u = u_du_array.value
        set_U_u!(U, u, u1_)
        set_arr_A_B!(u_du_ddu_array.value, u)

        compute_fluxes_and_sources_array!(f_g_s, u, local_grid, op, source_terms, eq,
                                          scheme)

        set_F_G_S!(F, G, S, f_g_s, nd_val, nvar_val)

        ut = u_du_array.partials[1]
        compute_ut!(ut, f_g_s, op, local_grid, eq, nd_val, nvar_val)

        # TODO - I wish that u_du_ddu_array.partials was the same as the previous
        # guy
        set_arr_A_B!(u_du_ddu_array.partials[1], ut)

        add_to_U!(U, ut, 0.5)

        compute_fluxes_and_sources_array!(df_g_s, u_du_array, local_grid, op,
                                          source_terms, eq, scheme)
        add_to_F_G_S!(F, G, S, df_g_s.partials[1], 0.5, nd_val, nvar_val)

        utt = u_du_ddu_array.partials[2]
        compute_ut!(utt, df_g_s.partials[1], op, local_grid, eq, nd_val, nvar_val, 0.5)

        compute_fluxes_and_sources_array!(ddf_g_s, u_du_ddu_array, local_grid, op,
                                          source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, ddf_g_s.partials[2], 1.0 / 3.0, nd_val, nvar_val)
        add_to_U!(U, utt, 1.0 / 3.0)

        FGS2res!(r1, F, G, S, D1, dt, lamx, lamy, nd_val, nvar_val)

        extrapolate_to_faces!(Ub_, U, op, nd_val, nvar_val)

        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)

        # cell_data  = (u, ut, utt, el_x, el_y)
        # @views extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #                      F, G, Fb[:, :, :, el_x, el_y], aux)

        fb = cache.fb_arrays[id]
        ub = cache.ub_arrays[id]
        Fb_loc = @view Fb[:, :, :, el_x, el_y]
        bflux_data = (Fb_loc, fb, ub, u_du_ddu_array, local_grid, op, nd_val, nvar_val)
        compute_bflux_2!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                         F, G, Fb_loc, aux, compute_bflux!)
        # @views eval_bflux_ad_2!(eq, grid, cell_data, eval_data, xg, op, nd_val,
        #                         F, G, Fb[:, :, :, el_x, el_y], aux)

    end
    return nothing
end

function compute_cell_residual_3!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWTDEltWise},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = scheme_degree_plus_one(scheme)
    nd_val = Val(nd)
    nvar = nvariables(eq)
    nvar_val = Val(nvar)
    nx, ny = grid.size

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    @unpack eval_data, cell_arrays, u_taylor_arrays, f_taylor_arrays = cache

    f_g_s_arrays, df_g_s_arrays, ddf_g_s_arrays, dddf_g_s_arrays = f_taylor_arrays
    _, u_du_arrays, u_du_ddu_arrays, u_du_ddu_dddu_arrays = u_taylor_arrays

    reset_arr!(res)
    reset_Fb_Ub!(Fb, Ub)
    @inbounds @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (dx, dy, xc, yc, lamx, lamy, t, dt)

        id = Threads.threadid()

        el_x, el_y = element[1], element[2]
        id = Threads.threadid()
        F, G, U, S = cell_arrays[id]

        f_g_s = f_g_s_arrays[id]
        df_g_s = df_g_s_arrays[id]
        ddf_g_s = ddf_g_s_arrays[id]
        dddf_g_s = dddf_g_s_arrays[id]
        u_du_array = u_du_arrays[id]
        u_du_ddu_array = u_du_ddu_arrays[id]
        u_du_ddu_dddu_array = u_du_ddu_dddu_arrays[id]

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        u = u_du_array.value
        set_U_u!(U, u, u1_)
        set_arr_A_B!(u_du_ddu_array.value, u)
        set_arr_A_B!(u_du_ddu_dddu_array.value, u)

        compute_fluxes_and_sources_array!(f_g_s, u, local_grid, op, source_terms, eq,
                                          scheme)

        set_F_G_S!(F, G, S, f_g_s, nd_val, nvar_val)

        ut = u_du_array.partials[1]
        compute_ut!(ut, f_g_s, op, local_grid, eq, nd_val, nvar_val)

        # TODO - I wish that u_du_ddu_array.partials was the same as the previous
        # guy
        set_arr_A_B!(u_du_ddu_array.partials[1], ut)
        set_arr_A_B!(u_du_ddu_dddu_array.partials[1], ut)

        add_to_U!(U, ut, 0.5)

        compute_fluxes_and_sources_array!(df_g_s, u_du_array, local_grid, op,
                                          source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, df_g_s.partials[1], 0.5, nd_val, nvar_val)

        utt = u_du_ddu_array.partials[2]
        compute_ut!(utt, df_g_s.partials[1], op, local_grid, eq, nd_val, nvar_val, 0.5)
        set_arr_A_B!(u_du_ddu_dddu_array.partials[2], utt)

        compute_fluxes_and_sources_array!(ddf_g_s, u_du_ddu_array, local_grid, op,
                                          source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, ddf_g_s.partials[2], 1.0 / 3.0, nd_val, nvar_val)
        add_to_U!(U, utt, 1.0 / 3.0)

        uttt = u_du_ddu_dddu_array.partials[3]
        compute_ut!(uttt, ddf_g_s.partials[2], op, local_grid, eq, nd_val, nvar_val,
                    1.0 / 3.0) # 2! / 3!
        compute_fluxes_and_sources_array!(dddf_g_s, u_du_ddu_dddu_array, local_grid,
                                          op, source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, dddf_g_s.partials[3], 1.0 / 4.0, nd_val, nvar_val)
        add_to_U!(U, uttt, 1.0 / 4.0)

        FGS2res!(r1, F, G, S, D1, dt, lamx, lamy, nd_val, nvar_val)

        extrapolate_to_faces!(Ub_, U, op, nd_val, nvar_val)

        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # cell_data = (u, ut, utt, uttt, el_x, el_y)
        # @views eval_bflux_ad_3!(eq, grid, cell_data, eval_data, xg, op, nd_val,
        #                         F, G, Fb[:, :, :, el_x, el_y], aux)
        # @views extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
        #                       F, G, Fb[:, :, :, el_x, el_y], aux)

        fb = cache.fb_arrays[id]
        ub = cache.ub_arrays[id]
        Fb_loc = @view Fb[:, :, :, el_x, el_y]
        bflux_data = (Fb_loc, fb, ub, u_du_ddu_dddu_array, local_grid, op, nd_val, nvar_val)
        compute_bflux_3!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                         F, G, Fb_loc, aux, compute_bflux!)
    end
    return nothing
end

function compute_cell_residual_4!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme::Scheme{<:LWTDEltWise, <:Any, <:Function},
                                  aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = scheme_degree_plus_one(scheme)
    nd_val = Val(nd)
    nvar = nvariables(eq)
    nvar_val = Val(nvar)
    nx, ny = grid.size

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    @unpack eval_data, cell_arrays, f_taylor_arrays, u_taylor_arrays = cache

    f_g_s_arrays, df_g_s_arrays, ddf_g_s_arrays, dddf_g_s_arrays, ddddf_g_s_arrays = f_taylor_arrays
    _, u_du_arrays, u_du_ddu_arrays, u_du_ddu_dddu_arrays, u_du_ddu_dddu_ddddu_arrays = u_taylor_arrays

    reset_arr!(res)
    reset_Fb_Ub!(Fb, Ub)
    @inbounds @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (dx, dy, xc, yc, lamx, lamy, t, dt)

        id = Threads.threadid()

        F, G, U, S = cell_arrays[id]

        f_g_s = f_g_s_arrays[id]
        df_g_s = df_g_s_arrays[id]
        ddf_g_s = ddf_g_s_arrays[id]
        dddf_g_s = dddf_g_s_arrays[id]
        ddddf_g_s = ddddf_g_s_arrays[id]
        u_du_array = u_du_arrays[id]
        u_du_ddu_array = u_du_ddu_arrays[id]
        u_du_ddu_dddu_array = u_du_ddu_dddu_arrays[id]
        u_du_ddu_dddu_ddddu_array = u_du_ddu_dddu_ddddu_arrays[id]

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        u = u_du_array.value
        set_U_u!(U, u, u1_)
        set_arr_A_B!(u_du_ddu_array.value, u)
        set_arr_A_B!(u_du_ddu_dddu_array.value, u)
        set_arr_A_B!(u_du_ddu_dddu_ddddu_array.value, u)
        # t_val1 = TaylorScalar{1}(t, (dt,))

        compute_fluxes_and_sources_array!(f_g_s, u, local_grid, op, source_terms, eq,
                                          scheme)

        set_F_G_S!(F, G, S, f_g_s, nd_val, nvar_val)

        ut = u_du_array.partials[1]
        compute_ut!(ut, f_g_s, op, local_grid, eq, nd_val, nvar_val)

        # TODO - I wish that u_du_ddu_array.partials was the same as the previous
        # guy
        set_arr_A_B!(u_du_ddu_array.partials[1], ut)
        set_arr_A_B!(u_du_ddu_dddu_array.partials[1], ut)
        set_arr_A_B!(u_du_ddu_dddu_ddddu_array.partials[1], ut)

        add_to_U!(U, ut, 0.5)

        compute_fluxes_and_sources_array!(df_g_s, u_du_array, local_grid, op,
                                          source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, df_g_s.partials[1], 0.5, nd_val, nvar_val)

        utt = u_du_ddu_array.partials[2]
        compute_ut!(utt, df_g_s.partials[1], op, local_grid, eq, nd_val, nvar_val,
                    0.5) # 1! / 2!
        set_arr_A_B!(u_du_ddu_dddu_array.partials[2], utt)
        set_arr_A_B!(u_du_ddu_dddu_ddddu_array.partials[2], utt)

        compute_fluxes_and_sources_array!(ddf_g_s, u_du_ddu_array, local_grid,
                                          op, source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, ddf_g_s.partials[2],
                      1.0 / 3.0,
                      nd_val, nvar_val)
        add_to_U!(U, utt, 1.0 / 3.0)

        uttt = u_du_ddu_dddu_array.partials[3]
        compute_ut!(uttt, ddf_g_s.partials[2], op, local_grid, eq, nd_val, nvar_val,
                    1.0 / 3.0) # 2! / 3!
        set_arr_A_B!(u_du_ddu_dddu_ddddu_array.partials[3], uttt)
        compute_fluxes_and_sources_array!(dddf_g_s, u_du_ddu_dddu_array, local_grid, op, source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, dddf_g_s.partials[3],
                      1.0 / 4.0,
                      nd_val, nvar_val)
        add_to_U!(U, uttt, 1.0 / 4.0)

        utttt = u_du_ddu_dddu_ddddu_array.partials[4]
        compute_ut!(utttt, dddf_g_s.partials[3], op, local_grid, eq, nd_val, nvar_val,
                    1.0 / 4.0) # 3! / 4!

        compute_fluxes_and_sources_array!(ddddf_g_s, u_du_ddu_dddu_ddddu_array,
                                          local_grid, op, source_terms,
                                          eq, scheme)
        add_to_F_G_S!(F, G, S, ddddf_g_s.partials[4], 1.0 / 5.0, nd_val, nvar_val)
        add_to_U!(U, utttt, 1.0 / 5.0)

        FGS2res!(r1, F, G, S, D1, dt, lamx, lamy, nd_val, nvar_val)

        extrapolate_to_faces!(Ub_, U, op, nd_val, nvar_val)

        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        fb = cache.fb_arrays[id]
        ub = cache.ub_arrays[id]
        Fb_loc = @view Fb[:, :, :, el_x, el_y]
        bflux_data = (Fb_loc, fb, ub, u_du_ddu_dddu_ddddu_array, local_grid, op, nd_val,
                      nvar_val)
        # @views eval_bflux_efficient_4!(cell_data, op, eq, nd_val, nvar_val)
        # extrap_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr, F, G, Fb_loc, aux)
        compute_bflux_4!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                         F, G, Fb_loc, aux, compute_bflux!)
    end
    return nothing
end

@inline @inbounds function extrapolate_to_faces_2!(Ub, U, op, ::Val{nd},
                                                   ::Val{nvar}) where {nd, nvar}
    @unpack Vl, Vr = op
    # @turbo for j in Base.OneTo(nd), i in Base.OneTo(nd), n in Base.OneTo(nvar)
    @turbo for n in axes(U, 1), j in axes(U, 2), i in axes(U, 2)
        Ub[n, 1, j] += Vl[i] * U[n, i, j]
        Ub[n, 2, j] += Vr[i] * U[n, i, j]

        Ub[n, 3, i] += Vl[j] * U[n, i, j]
        Ub[n, 4, i] += Vr[j] * U[n, i, j]
    end
end

function set_bflux_array_4!(ub, u, op,
                            nd_val::Val{nd}, nvar_val::Val{nvar}) where {nd, nvar}
    reset_arr!(ub.value)
    reset_arr!.(ub.partials)

    extrapolate_to_faces_2!(ub.value, u.value, op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[1], u.partials[1], op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[2], u.partials[2], op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[3], u.partials[3], op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[4], u.partials[4], op, nd_val, nvar_val)
end

function set_bflux_array_3!(ub, u, op,
                            nd_val::Val{nd}, nvar_val::Val{nvar}) where {nd, nvar}
    reset_arr!(ub.value)
    reset_arr!.(ub.partials)

    extrapolate_to_faces_2!(ub.value, u.value, op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[1], u.partials[1], op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[2], u.partials[2], op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[3], u.partials[3], op, nd_val, nvar_val)
end

function set_bflux_array_2!(ub, u, op,
                            nd_val::Val{nd}, nvar_val::Val{nvar}) where {nd, nvar}
    reset_arr!(ub.value)
    reset_arr!.(ub.partials)

    extrapolate_to_faces_2!(ub.value, u.value, op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[1], u.partials[1], op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[2], u.partials[2], op, nd_val, nvar_val)
end

function set_bflux_array_1!(ub, u, op,
                            nd_val::Val{nd}, nvar_val::Val{nvar}) where {nd, nvar}
    reset_arr!(ub.value)
    reset_arr!.(ub.partials)

    extrapolate_to_faces_2!(ub.value, u.value, op, nd_val, nvar_val)
    extrapolate_to_faces_2!(ub.partials[1], u.partials[1], op, nd_val, nvar_val)
end

function compute_bflux_array!(fb, ub, eq::AbstractEquations{2}, local_grid, op,
                              val_nd::Val{nd}, val_nvar::Val{nvar}) where {nd, nvar}
    dx, dy, xc, yc, _, _, t, dt = local_grid
    @unpack xg = op

    xl = xc - 0.5 * dx
    yd = yc - 0.5 * dy
    for i in Base.OneTo(nd)
        x = xl + dx * xg[i]
        y = yd + dy * xg[i]
        ul_node = get_node_vars(ub, eq, 1, i)
        ur_node = get_node_vars(ub, eq, 2, i)
        ud_node = get_node_vars(ub, eq, 3, i)
        uu_node = get_node_vars(ub, eq, 4, i)

        fl_node = flux(x, y, ul_node, eq, 1)
        fr_node = flux(x, y, ur_node, eq, 1)
        gd_node = flux(x, y, ud_node, eq, 2)
        gu_node = flux(x, y, uu_node, eq, 2)

        set_node_vars!(fb, fl_node, eq, 1, i)
        set_node_vars!(fb, fr_node, eq, 2, i)
        set_node_vars!(fb, gd_node, eq, 3, i)
        set_node_vars!(fb, gu_node, eq, 4, i)
    end
end

@inline function eval_bflux_efficient_1!(cell_data, op, eq,
                                         nd_val::Val{nd},
                                         nvar_val::Val{nvar}) where {nd, nvar}
    Fb, fb, ub, u, local_grid = cell_data
    set_bflux_array_1!(ub, u, op, nd_val, nvar_val)
    compute_bflux_array!(fb, ub, eq, local_grid, op, nd_val, nvar_val)

    @turbo for i in Base.OneTo(nd), n in Base.OneTo(nvar)
        Fb[n, i, 1] = fb.value[n, 1, i] + 0.5 * fb.partials[1][n, 1, i]
        Fb[n, i, 2] = fb.value[n, 2, i] + 0.5 * fb.partials[1][n, 2, i]
        Fb[n, i, 3] = fb.value[n, 3, i] + 0.5 * fb.partials[1][n, 3, i]
        Fb[n, i, 4] = fb.value[n, 4, i] + 0.5 * fb.partials[1][n, 4, i]
    end
end

@inline function eval_bflux_efficient_2!(cell_data, op, eq,
                                         nd_val::Val{nd},
                                         nvar_val::Val{nvar}) where {nd, nvar}
    Fb, fb, ub, u, local_grid = cell_data
    set_bflux_array_2!(ub, u, op, nd_val, nvar_val)
    compute_bflux_array!(fb, ub, eq, local_grid, op, nd_val, nvar_val)

    @turbo for i in Base.OneTo(nd), n in Base.OneTo(nvar)
        Fb[n, i, 1] = (fb.value[n, 1, i] + 0.5 * fb.partials[1][n, 1, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 1, i])
        Fb[n, i, 2] = (fb.value[n, 2, i] + 0.5 * fb.partials[1][n, 2, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 2, i])
        Fb[n, i, 3] = (fb.value[n, 3, i] + 0.5 * fb.partials[1][n, 3, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 3, i])
        Fb[n, i, 4] = (fb.value[n, 4, i] + 0.5 * fb.partials[1][n, 4, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 4, i])
    end
end

@inline function eval_bflux_efficient_3!(cell_data, op, eq,
                                         nd_val::Val{nd},
                                         nvar_val::Val{nvar}) where {nd, nvar}
    Fb, fb, ub, u, local_grid = cell_data
    set_bflux_array_3!(ub, u, op, nd_val, nvar_val)
    compute_bflux_array!(fb, ub, eq, local_grid, op, nd_val, nvar_val)

    @turbo for i in Base.OneTo(nd), n in Base.OneTo(nvar)
        Fb[n, i, 1] = (fb.value[n, 1, i] + 0.5 * fb.partials[1][n, 1, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 1, i]
                       + 0.25 * fb.partials[3][n, 1, i])
        Fb[n, i, 2] = (fb.value[n, 2, i] + 0.5 * fb.partials[1][n, 2, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 2, i]
                       + 0.25 * fb.partials[3][n, 2, i])
        Fb[n, i, 3] = (fb.value[n, 3, i] + 0.5 * fb.partials[1][n, 3, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 3, i]
                       + 0.25 * fb.partials[3][n, 3, i])
        Fb[n, i, 4] = (fb.value[n, 4, i] + 0.5 * fb.partials[1][n, 4, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 4, i]
                       + 0.25 * fb.partials[3][n, 4, i])
    end
end

@inline function eval_bflux_efficient_4!(cell_data, op, eq,
                                         nd_val::Val{nd},
                                         nvar_val::Val{nvar}) where {nd, nvar}
    Fb, fb, ub, u, local_grid = cell_data
    set_bflux_array_4!(ub, u, op, nd_val, nvar_val)
    compute_bflux_array!(fb, ub, eq, local_grid, op, nd_val, nvar_val)

    @turbo for i in Base.OneTo(nd), n in Base.OneTo(nvar)
        Fb[n, i, 1] = (fb.value[n, 1, i] + 0.5 * fb.partials[1][n, 1, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 1, i]
                       + 0.25 * fb.partials[3][n, 1, i] +
                       1.0 / 5.0 * fb.partials[4][n, 1, i])
        Fb[n, i, 2] = (fb.value[n, 2, i] + 0.5 * fb.partials[1][n, 2, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 2, i]
                       + 0.25 * fb.partials[3][n, 2, i] +
                       1.0 / 5.0 * fb.partials[4][n, 2, i])
        Fb[n, i, 3] = (fb.value[n, 3, i] + 0.5 * fb.partials[1][n, 3, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 3, i]
                       + 0.25 * fb.partials[3][n, 3, i] +
                       1.0 / 5.0 * fb.partials[4][n, 3, i])
        Fb[n, i, 4] = (fb.value[n, 4, i] + 0.5 * fb.partials[1][n, 4, i]
                       + 1.0 / 3.0 * fb.partials[2][n, 4, i]
                       + 0.25 * fb.partials[3][n, 4, i] +
                       1.0 / 5.0 * fb.partials[4][n, 4, i])
    end
end

@inline @inbounds function compute_bflux_4!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, bflux_data, eval_data, xg, Vl, Vr, F, G, Fb_loc, aux)
end

@inline @inbounds function compute_bflux_3!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, bflux_data, eval_data, xg, Vl, Vr, F, G, Fb_loc, aux)
end

@inline @inbounds function compute_bflux_2!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, bflux_data, eval_data, xg, Vl, Vr, F, G, Fb_loc, aux)
end

@inline @inbounds function compute_bflux_1!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.extrap_bflux!))
    extrap_bflux!(eq, grid, bflux_data, eval_data, xg, Vl, Vr, F, G, Fb_loc, aux)
end

@inline @inbounds function compute_bflux_1!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.eval_bflux1!))
    (Fb_loc, fb, ub, u_du_ddu_dddu_ddddu_array, local_grid, op, nd_val, nvar_val) = bflux_data
    eval_bflux_efficient_1!(bflux_data, op, eq, nd_val, nvar_val)
end

@inline @inbounds function compute_bflux_2!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.eval_bflux2!))
    (Fb_loc, fb, ub, u_du_ddu_dddu_ddddu_array, local_grid, op, nd_val, nvar_val) = bflux_data
    eval_bflux_efficient_2!(bflux_data, op, eq, nd_val, nvar_val)
end

@inline @inbounds function compute_bflux_3!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.eval_bflux3!))
    (Fb_loc, fb, ub, u_du_ddu_dddu_ddddu_array, local_grid, op, nd_val, nvar_val) = bflux_data
    eval_bflux_efficient_3!(bflux_data, op, eq, nd_val, nvar_val)
end

@inline @inbounds function compute_bflux_4!(eq, grid, bflux_data, eval_data, xg, Vl, Vr,
                                            F, G, Fb_loc, aux,
                                            bflux::typeof(Tenkai.eval_bflux4!))
    (Fb_loc, fb, ub, u_du_ddu_dddu_ddddu_array, local_grid, op, nd_val, nvar_val) = bflux_data
    eval_bflux_efficient_4!(bflux_data, op, eq, nd_val, nvar_val)
end
