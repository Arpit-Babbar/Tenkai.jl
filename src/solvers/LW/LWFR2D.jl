using ..Tenkai: periodic, dirichlet, neumann, reflect,
                evaluate, extrapolate,
                get_node_vars, set_node_vars!,
                add_to_node_vars!, subtract_from_node_vars!,
                multiply_add_to_node_vars!, multiply_add_set_node_vars!,
                comp_wise_mutiply_node_vars!, flux,
                update_ghost_values_periodic!, update_ghost_values_fn_blend!

using UnPack
using TimerOutputs
using Polyester
using MuladdMacro
using OffsetArrays
using StaticArrays

using ..FR: @threaded

using ..Equations: AbstractEquations, nvariables, eachvariable

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function setup_arrays_lwfr(grid, scheme, eq::AbstractEquations{2})
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

    MEval = MArray{Tuple{nvariables(eq), nd}, Float64}
    eval_data_big = alloc_for_threads(MEval, big_eval_data_size)

    MEval_small = MArray{Tuple{nvariables(eq), 1}, Float64}
    eval_data_small = alloc_for_threads(MEval_small, small_eval_data_size)

    eval_data = (; eval_data_big, eval_data_small)

    # Ghost values cache

    Marr = MArray{Tuple{nvariables(eq), 1}, Float64}

    ghost_cache = alloc_for_threads(Marr, 2)

    # KLUDGE - Rename this to LWFR cache
    cache = (; u1, ua, res, Fb, Ub, eval_data, cell_arrays, ghost_cache)
    return cache
end

function update_ghost_values_lwfr!(problem, scheme, eq::AbstractEquations{2, 1},
                                   grid, aux, op, cache, t, dt, scaling_factor = 1)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, Ub = cache
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

    refresh!(u) = fill!(u, 0.0)

    dt_scaled = scaling_factor * dt
    wg_scaled = scaling_factor * wg

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    if left == dirichlet
        pre_allocated = [(zeros(nvar) for _ in 1:2) for _ in 1:Threads.nthreads()]
        @threaded for j in 1:ny
            x = xf[1]
            for k in 1:nd
                y = yf[j] + xg[k] * dy[j]
                # KLUDGE - Don't allocate so much!
                ub, fb = pre_allocated[Threads.threadid()]
                for l in 1:nd
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x, y, tq)
                    fbvalue = flux(x, y, ubvalue, eq, 1)
                    for n in 1:nvar
                        ub[n] += ubvalue[n] * wg_scaled[l]
                        fb[n] += fbvalue[n] * wg_scaled[l]
                    end
                end
                for n in 1:nvar
                    Ub[n, k, 1, 1, j] = Ub[n, k, 2, 0, j] = ub[n] # upwind
                    Fb[n, k, 1, 1, j] = Fb[n, k, 2, 0, j] = fb[n] # upwind
                end
            end
        end
    elseif left in [neumann, reflect]
        @threaded for j in 1:ny
            for k in 1:nd
                for n in 1:nvar
                    Ub[n, k, 2, 0, j] = Ub[n, k, 1, 1, j]
                    Fb[n, k, 2, 0, j] = Fb[n, k, 1, 1, j]
                end
                if left == reflect
                    Ub[2, k, 2, 0, j] *= -1.0
                    Fb[1, k, 2, 0, j] *= -1.0
                    Fb[3, k, 2, 0, j] *= -1.0
                    Fb[4, k, 2, 0, j] *= -1.0
                end
            end
        end
    else
        println("Incorrect bc specified at left.")
        @assert false
    end

    if right == dirichlet
        pre_allocated = [(zeros(nvar) for _ in 1:2) for _ in 1:Threads.nthreads()]
        @threaded for j in 1:ny
            x = xf[nx + 1]
            for k in 1:nd
                y = yf[j] + xg[k] * dy[j]
                # KLUDGE - Improve
                ub, fb = pre_allocated[Threads.threadid()]
                for l in 1:nd
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x, y, tq)
                    fbvalue = flux(x, y, ubvalue, eq, 1)
                    for n in 1:nvar
                        ub[n] += ubvalue[n] * wg_scaled[l]
                        fb[n] += fbvalue[n] * wg_scaled[l]
                    end
                end
                for n in 1:nvar
                    Ub[n, k, 2, nx, j] = Ub[n, k, 1, nx + 1, j] = ub[n] # upwind
                    Fb[n, k, 2, nx, j] = Fb[n, k, 1, nx + 1, j] = fb[n] # upwind
                end
            end
        end
    elseif right in [reflect, neumann]
        @threaded for j in 1:ny
            for k in 1:nd
                for n in 1:nvar
                    Ub[n, k, 1, nx + 1, j] = Ub[n, k, 2, nx, j]
                    Fb[n, k, 1, nx + 1, j] = Fb[n, k, 2, nx, j]
                end
                if right == reflect
                    Ub[2, k, 1, nx + 1, j] *= -1.0
                    Fb[1, k, 1, nx + 1, j] *= -1.0
                    Fb[3, k, 1, nx + 1, j] *= -1.0
                    Fb[4, k, 1, nx + 1, j] *= -1.0
                end
            end
        end
    else
        println("Incorrect bc specified at right.")
        @assert false
    end

    if bottom == dirichlet
        pre_allocated = [(zeros(nvar) for _ in 1:2) for _ in 1:Threads.nthreads()]
        @threaded for i in 1:nx
            y = yf[1]
            for k in 1:nd
                x = xf[i] + xg[k] * dx[i]
                ub, fb = pre_allocated[Threads.threadid()]
                for l in 1:nd
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x, y, tq)
                    fbvalue = flux(x, y, ubvalue, eq, 2)
                    for n in 1:nvar
                        ub[n] += ubvalue[n] * wg_scaled[l]
                        fb[n] += fbvalue[n] * wg_scaled[l]
                    end
                end
                for n in 1:nvar
                    Ub[n, k, 3, i, 1] = Ub[n, k, 4, i, 0] = ub[n] # upwind
                    Fb[n, k, 3, i, 1] = Fb[n, k, 4, i, 0] = fb[n] # upwind
                end
            end
        end
    elseif bottom in [reflect, neumann, dirichlet]
        @threaded for i in 1:nx
            for k in 1:nd
                for n in 1:nvar
                    Ub[n, k, 4, i, 0] = Ub[n, k, 3, i, 1]
                    Fb[n, k, 4, i, 0] = Fb[n, k, 3, i, 1]
                end
                if bottom == reflect
                    Ub[3, k, 4, i, 0] *= -1.0
                    Fb[1, k, 4, i, 0] *= -1.0
                    Fb[2, k, 4, i, 0] *= -1.0
                    Fb[4, k, 4, i, 0] *= -1.0
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert typeof(bottom) <: Tuple{Any, Any, Any}
        bc! = bottom[1]
        bc!(grid, eq, op, Fb, Ub)
    end
    if top == dirichlet
        pre_allocated = [(zeros(nvar) for _ in 1:2) for _ in 1:Threads.nthreads()]
        @threaded for i in 1:nx
            y = yf[ny + 1]
            for k in 1:nd
                x = xf[i] + xg[k] * dx[i]
                ub, fb = pre_allocated[Threads.threadid()]
                for l in 1:nd
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x, y, tq)
                    fbvalue = flux(x, y, ubvalue, eq, 2)
                    for n in 1:nvar
                        ub[n] += ubvalue[n] * wg_scaled[l]
                        fb[n] += fbvalue[n] * wg_scaled[l]
                    end
                end
                for n in 1:nvar
                    Ub[n, k, 4, i, ny] = Ub[n, k, 3, i, ny + 1] = ub[n] # upwind
                    Fb[n, k, 4, i, ny] = Fb[n, k, 3, i, ny + 1] = fb[n] # upwind
                end
            end
        end
    elseif top in [reflect, neumann]
        @threaded for i in 1:nx
            for k in 1:nd
                for n in 1:nvar
                    Ub[n, k, 3, i, ny + 1] = Ub[n, k, 4, i, ny]
                    Fb[n, k, 3, i, ny + 1] = Fb[n, k, 4, i, ny]
                end
                if top == reflect
                    Ub[3, k, 3, i, ny + 1] *= -1.0
                    Fb[1, k, 3, i, ny + 1] *= -1.0
                    Fb[2, k, 3, i, ny + 1] *= -1.0
                    Fb[4, k, 3, i, ny + 1] *= -1.0
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert typeof(top)<:Tuple{Any, Any, Any} "Incorrect bc specified at top"
    end

    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    return nothing
    end # timer
end

#-------------------------------------------------------------------------------
# Compute boundary flux
#-------------------------------------------------------------------------------
function eval_bflux1!(eq::AbstractEquations{2}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
    nd = length(xg)
    refresh!(u) = fill!(u, zero(eltype(u)))

    (u, up, um, el_x, el_y) = cell_data

    id = Threads.threadid()

    eval_data_big, eval_data_small = (eval_data.eval_data_big[id],
                                      eval_data.eval_data_small[id])

    refresh!.(eval_data_big)
    (ul, ur, uu, ud, upl, upr, upd, upu, uml, umr, umd, umu) = eval_data_big  # Pre-allocated arrays
    ftl, ftr, gtd, gtu = eval_data_small
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        up_node = get_node_vars(up, eq, i, j)
        um_node = get_node_vars(um, eq, i, j)

        # ul = u * V
        # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)

        # ud = u * V
        # ud[i] += ∑_j U[i,j]*V[j]
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, j)
        multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, j)
        multiply_add_to_node_vars!(upd, Vl[j], up_node, eq, i)
        multiply_add_to_node_vars!(upu, Vr[j], up_node, eq, i)

        multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, j)
        multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, j)
        multiply_add_to_node_vars!(umd, Vl[j], um_node, eq, i)
        multiply_add_to_node_vars!(umu, Vr[j], um_node, eq, i)
    end

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

        # KLUDGE - Indices order needs to be changed, or something else
        # needs to be done to avoid cache misses
        set_node_vars!(Fb, fl, eq, i, 1)
        set_node_vars!(Fb, fr, eq, i, 2)
        set_node_vars!(Fb, gd, eq, i, 3)
        set_node_vars!(Fb, gu, eq, i, 4)

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

        multiply_add_set_node_vars!(ftl, 0.5, fpl, -0.5, fml, eq, 1)
        multiply_add_set_node_vars!(ftr, 0.5, fpr, -0.5, fmr, eq, 1)

        ftl_node = get_node_vars(ftl, eq, 1)
        ftr_node = get_node_vars(ftr, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 0.5, ftr_node, eq, i, 2)

        multiply_add_set_node_vars!(gtd, 0.5, gpd, -0.5, gmd, eq, 1)
        multiply_add_set_node_vars!(gtu, 0.5, gpu, -0.5, gmu, eq, 1)

        gtd_node = get_node_vars(gtd, eq, 1)
        gtu_node = get_node_vars(gtu, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 0.5, gtu_node, eq, i, 4)
    end
end

function eval_bflux2!(eq::AbstractEquations{2}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
    nd = length(xg)
    refresh!(u) = fill!(u, zero(eltype(u)))

    u, up, um, el_x, el_y = cell_data
    # Load pre-allocated arrays
    id = Threads.threadid()
    eval_data_big, eval_data_small = (eval_data.eval_data_big[id],
                                      eval_data.eval_data_small[id])

    refresh!.(eval_data_big)
    ul, ur, uu, ud, upl, upr, upd, upu, uml, umr, umd, umu = eval_data_big
    ftl, ftr, gtd, gtu = eval_data_small
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        up_node = get_node_vars(up, eq, i, j)
        um_node = get_node_vars(um, eq, i, j)

        # ul = u * V
        # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)

        # ud = u * V
        # ud[i] += ∑_j U[i,j]*V[j]
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, j)
        multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, j)
        multiply_add_to_node_vars!(upd, Vl[j], up_node, eq, i)
        multiply_add_to_node_vars!(upu, Vr[j], up_node, eq, i)

        multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, j)
        multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, j)
        multiply_add_to_node_vars!(umd, Vl[j], um_node, eq, i)
        multiply_add_to_node_vars!(umu, Vr[j], um_node, eq, i)
    end

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
        set_node_vars!(Fb, fl, eq, i, 1)
        set_node_vars!(Fb, fr, eq, i, 2)
        set_node_vars!(Fb, gd, eq, i, 3)
        set_node_vars!(Fb, gu, eq, i, 4)

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

        multiply_add_set_node_vars!(ftl, 0.5, fpl, -0.5, fml, eq, 1)
        multiply_add_set_node_vars!(ftr, 0.5, fpr, -0.5, fmr, eq, 1)

        ftl_node = get_node_vars(ftl, eq, 1)
        ftr_node = get_node_vars(ftr, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 0.5, ftr_node, eq, i, 2)

        multiply_add_set_node_vars!(gtd, 0.5, gpd, -0.5, gmd, eq, 1)
        multiply_add_set_node_vars!(gtu, 0.5, gpu, -0.5, gmu, eq, 1)

        gtd_node = get_node_vars(gtd, eq, 1)
        gtu_node = get_node_vars(gtu, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 0.5, gtu_node, eq, i, 4)

        fttl, fttr, gttd, gttu = ftl, ftr, gtd, gtu

        # ftt = fm - 2*f + fp
        multiply_add_set_node_vars!(fttl, fml, -2.0, fl, fpl, eq, 1)
        multiply_add_set_node_vars!(fttr, fmr, -2.0, fr, fpr, eq, 1)

        fttl_node = get_node_vars(fttl, eq, 1)
        fttr_node = get_node_vars(fttr, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttr_node, eq, i, 2)

        # gtt = gm - 2*g + gp
        multiply_add_set_node_vars!(gttd, gmd, -2.0, gd, gpd, eq, 1)
        multiply_add_set_node_vars!(gttu, gmu, -2.0, gu, gpu, eq, 1)

        gttd_node = get_node_vars(gttd, eq, 1)
        gttu_node = get_node_vars(gttu, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gttd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gttu_node, eq, i, 4)
    end
end

function eval_bflux3!(eq::AbstractEquations{2}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
    nd = length(xg)
    refresh!(u) = fill!(u, zero(eltype(u)))

    # Load pre-allocated arrays
    u, up, um, upp, umm, el_x, el_y = cell_data
    id = Threads.threadid()
    eval_data_big, eval_data_small = (eval_data.eval_data_big[id],
                                      eval_data.eval_data_small[id])

    refresh!.(eval_data_big)
    (ul, ur, uu, ud, upl, upr, upd, upu, uml, umr, umd, umu, uppl,
    uppr, uppd, uppu, umml, ummr, ummd, ummu) = eval_data_big
    ftl, ftr, gtd, gtu = eval_data_small
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        up_node = get_node_vars(up, eq, i, j)
        um_node = get_node_vars(um, eq, i, j)
        upp_node = get_node_vars(upp, eq, i, j)
        umm_node = get_node_vars(umm, eq, i, j)

        # ul = u * V
        # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)

        # ud = u * V
        # ud[i] += ∑_j U[i,j]*V[j]
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, j)
        multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, j)
        multiply_add_to_node_vars!(upd, Vl[j], up_node, eq, i)
        multiply_add_to_node_vars!(upu, Vr[j], up_node, eq, i)

        multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, j)
        multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, j)
        multiply_add_to_node_vars!(umd, Vl[j], um_node, eq, i)
        multiply_add_to_node_vars!(umu, Vr[j], um_node, eq, i)

        multiply_add_to_node_vars!(uppl, Vl[i], upp_node, eq, j)
        multiply_add_to_node_vars!(uppr, Vr[i], upp_node, eq, j)
        multiply_add_to_node_vars!(uppd, Vl[j], upp_node, eq, i)
        multiply_add_to_node_vars!(uppu, Vr[j], upp_node, eq, i)

        multiply_add_to_node_vars!(umml, Vl[i], umm_node, eq, j)
        multiply_add_to_node_vars!(ummr, Vr[i], umm_node, eq, j)
        multiply_add_to_node_vars!(ummd, Vl[j], umm_node, eq, i)
        multiply_add_to_node_vars!(ummu, Vr[j], umm_node, eq, i)
    end

    @views for i in 1:nd
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
        set_node_vars!(Fb, fl, eq, i, 1)
        set_node_vars!(Fb, fr, eq, i, 2)
        set_node_vars!(Fb, gd, eq, i, 3)
        set_node_vars!(Fb, gu, eq, i, 4)

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

        multiply_add_set_node_vars!(ftl, 1.0 / 12.0,
                                    -1.0, fppl, 8.0, fpl,
                                    -8.0, fml, 1.0, fmml,
                                    eq, 1)
        multiply_add_set_node_vars!(ftr, 1.0 / 12.0,
                                    -1.0, fppr, 8.0, fpr,
                                    -8.0, fmr, 1.0, fmmr,
                                    eq, 1)

        ftl_node = get_node_vars(ftl, eq, 1)
        ftr_node = get_node_vars(ftr, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 0.5, ftr_node, eq, i, 2)

        multiply_add_set_node_vars!(gtd, 1.0 / 12.0,
                                    -1.0, gppd, 8.0, gpd,
                                    -8.0, gmd, 1.0, gmmd,
                                    eq, 1)
        multiply_add_set_node_vars!(gtu, 1.0 / 12.0,
                                    -1.0, gppu, 8.0, gpu,
                                    -8.0, gmu, 1.0, gmmu,
                                    eq, 1)

        gtd_node = get_node_vars(gtd, eq, 1)
        gtu_node = get_node_vars(gtu, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 0.5, gtu_node, eq, i, 4)

        fttl, fttr, gttd, gttu = ftl, ftr, gtd, gtu

        # ftt = fm - 2*f + fp
        multiply_add_set_node_vars!(fttl, fml, -2.0, fl, fpl, eq, 1)
        multiply_add_set_node_vars!(fttr, fmr, -2.0, fr, fpr, eq, 1)

        fttl_node = get_node_vars(fttl, eq, 1)
        fttr_node = get_node_vars(fttr, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttr_node, eq, i, 2)

        # gtt = gm - 2*g + gp
        multiply_add_set_node_vars!(gttd, gmd, -2.0, gd, gpd, eq, 1)
        multiply_add_set_node_vars!(gttu, gmu, -2.0, gu, gpu, eq, 1)

        gttd_node = get_node_vars(gttd, eq, 1)
        gttu_node = get_node_vars(gttu, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gttd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gttu_node, eq, i, 4)

        ftttl, ftttr, gtttd, gtttu = ftl, ftr, gtd, gtu

        multiply_add_set_node_vars!(ftttl, 0.5,
                                    1.0, fppl, -2.0, fpl,
                                    2.0, fml, -1.0, fmml,
                                    eq, 1)
        multiply_add_set_node_vars!(ftttr, 0.5,
                                    1.0, fppr, -2.0, fpr,
                                    2.0, fmr, -1.0, fmmr,
                                    eq, 1)

        ftttl_node = get_node_vars(ftttl, eq, 1)
        ftttr_node = get_node_vars(ftttr, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, ftttl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, ftttr_node, eq, i, 2)

        multiply_add_set_node_vars!(gtttd, 0.5,
                                    1.0, gppd, -2.0, gpd,
                                    2.0, gmd, -1.0, gmmd,
                                    eq, 1)
        multiply_add_set_node_vars!(gtttu, 0.5,
                                    1.0, gppu, -2.0, gpu,
                                    2.0, gmu, -1.0, gmmu,
                                    eq, 1)

        gtttd_node = get_node_vars(gtttd, eq, 1)
        gtttu_node = get_node_vars(gtttu, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, gtttd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, gtttu_node, eq, i, 4)
    end
end

function eval_bflux4!(eq::AbstractEquations{2}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
    nd = length(xg)
    refresh!(u) = fill!(u, zero(eltype(u)))

    # Load pre-allocated arrays
    u, up, um, upp, umm, el_x, el_y = cell_data
    id = Threads.threadid()
    eval_data_big, eval_data_small = (eval_data.eval_data_big[id],
                                      eval_data.eval_data_small[id])

    refresh!.(eval_data_big)
    (ul, ur, uu, ud, upl, upr, upd, upu, uml, umr, umd, umu, uppl,
    uppr, uppd, uppu, umml, ummr, ummd, ummu) = eval_data_big
    ftl, ftr, gtd, gtu = eval_data_small
    xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
    yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
    dx, dy = grid.dx[el_x], grid.dy[el_y]

    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        u_node = get_node_vars(u, eq, i, j)
        up_node = get_node_vars(up, eq, i, j)
        um_node = get_node_vars(um, eq, i, j)
        upp_node = get_node_vars(upp, eq, i, j)
        umm_node = get_node_vars(umm, eq, i, j)

        # ul = u * V
        # ul[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
        multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, j)
        multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, j)

        # ud = u * V
        # ud[i] += ∑_j U[i,j]*V[j]
        multiply_add_to_node_vars!(ud, Vl[j], u_node, eq, i)
        multiply_add_to_node_vars!(uu, Vr[j], u_node, eq, i)

        multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, j)
        multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, j)
        multiply_add_to_node_vars!(upd, Vl[j], up_node, eq, i)
        multiply_add_to_node_vars!(upu, Vr[j], up_node, eq, i)

        multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, j)
        multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, j)
        multiply_add_to_node_vars!(umd, Vl[j], um_node, eq, i)
        multiply_add_to_node_vars!(umu, Vr[j], um_node, eq, i)

        multiply_add_to_node_vars!(uppl, Vl[i], upp_node, eq, j)
        multiply_add_to_node_vars!(uppr, Vr[i], upp_node, eq, j)
        multiply_add_to_node_vars!(uppd, Vl[j], upp_node, eq, i)
        multiply_add_to_node_vars!(uppu, Vr[j], upp_node, eq, i)

        multiply_add_to_node_vars!(umml, Vl[i], umm_node, eq, j)
        multiply_add_to_node_vars!(ummr, Vr[i], umm_node, eq, j)
        multiply_add_to_node_vars!(ummd, Vl[j], umm_node, eq, i)
        multiply_add_to_node_vars!(ummu, Vr[j], umm_node, eq, i)
    end

    @views for i in 1:nd
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
        set_node_vars!(Fb, fl, eq, i, 1)
        set_node_vars!(Fb, fr, eq, i, 2)
        set_node_vars!(Fb, gd, eq, i, 3)
        set_node_vars!(Fb, gu, eq, i, 4)

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

        multiply_add_set_node_vars!(ftl, 1.0 / 12.0,
                                    -1.0, fppl, 8.0, fpl,
                                    -8.0, fml, 1.0, fmml,
                                    eq, 1)
        multiply_add_set_node_vars!(ftr, 1.0 / 12.0,
                                    -1.0, fppr, 8.0, fpr,
                                    -8.0, fmr, 1.0, fmmr,
                                    eq, 1)

        ftl_node = get_node_vars(ftl, eq, 1)
        ftr_node = get_node_vars(ftr, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, ftl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 0.5, ftr_node, eq, i, 2)

        multiply_add_set_node_vars!(gtd, 1.0 / 12.0,
                                    -1.0, gppd, 8.0, gpd,
                                    -8.0, gmd, 1.0, gmmd,
                                    eq, 1)
        multiply_add_set_node_vars!(gtu, 1.0 / 12.0,
                                    -1.0, gppu, 8.0, gpu,
                                    -8.0, gmu, 1.0, gmmu,
                                    eq, 1)

        gtd_node = get_node_vars(gtd, eq, 1)
        gtu_node = get_node_vars(gtu, eq, 1)

        multiply_add_to_node_vars!(Fb, 0.5, gtd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 0.5, gtu_node, eq, i, 4)

        fttl, fttr, gttd, gttu = ftl, ftr, gtd, gtu

        multiply_add_set_node_vars!(fttl, 1.0 / 12.0,
                                    -1.0, fppl, 16.0, fpl,
                                    -30.0, fl,
                                    16.0, fml, -1.0, fmml,
                                    eq, 1)
        multiply_add_set_node_vars!(fttr, 1.0 / 12.0,
                                    -1.0, fppr, 16.0, fpr,
                                    -30.0, fr,
                                    16.0, fmr, -1.0, fmmr,
                                    eq, 1)

        fttl_node = get_node_vars(fttl, eq, 1)
        fttr_node = get_node_vars(fttr, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fttr_node, eq, i, 2)

        multiply_add_set_node_vars!(gttd, 1.0 / 12.0,
                                    -1.0, gppd, 16.0, gpd,
                                    -30.0, gd,
                                    16.0, gmd, -1.0, gmmd,
                                    eq, 1)
        multiply_add_set_node_vars!(gttu, 1.0 / 12.0,
                                    -1.0, gppu, 16.0, gpu,
                                    -30.0, gu,
                                    16.0, gmu, -1.0, gmmu,
                                    eq, 1)

        gttd_node = get_node_vars(gttd, eq, 1)
        gttu_node = get_node_vars(gttu, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gttd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 6.0, gttu_node, eq, i, 4)

        # reusing old arrays
        ftttl, ftttr, gtttd, gtttu = ftl, ftr, gtd, gtu

        multiply_add_set_node_vars!(ftttl, 0.5,
                                    1.0, fppl, -2.0, fpl,
                                    2.0, fml, -1.0, fmml,
                                    eq, 1)
        multiply_add_set_node_vars!(ftttr, 0.5,
                                    1.0, fppr, -2.0, fpr,
                                    2.0, fmr, -1.0, fmmr,
                                    eq, 1)

        ftttl_node = get_node_vars(ftttl, eq, 1)
        ftttr_node = get_node_vars(ftttr, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, ftttl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, ftttr_node, eq, i, 2)

        multiply_add_set_node_vars!(gtttd, 0.5,
                                    1.0, gppd, -2.0, gpd,
                                    2.0, gmd, -1.0, gmmd,
                                    eq, 1)

        multiply_add_set_node_vars!(gtttu, 0.5,
                                    1.0, gppu, -2.0, gpu,
                                    2.0, gmu, -1.0, gmmu,
                                    eq, 1)

        gtttd_node = get_node_vars(gtttd, eq, 1)
        gtttu_node = get_node_vars(gtttu, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, gtttd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 24.0, gtttu_node, eq, i, 4)

        fttttl, fttttr, gttttd, gttttu = ftl, ftr, gtd, gtu # Reusing

        multiply_add_set_node_vars!(fttttl, 0.5,
                                    1.0, fppl, -4.0, fpl,
                                    6.0, fl,
                                    -4.0, fml, 1.0, fmml,
                                    eq, 1)

        multiply_add_set_node_vars!(fttttr, 0.5,
                                    1.0, fppr, -4.0, fpr,
                                    6.0, fr,
                                    -4.0, fmr, 1.0, fmmr,
                                    eq, 1)

        fttttl_node = get_node_vars(fttttl, eq, 1)
        fttttr_node = get_node_vars(fttttr, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 120.0, fttttl_node, eq, i, 1)
        multiply_add_to_node_vars!(Fb, 1.0 / 120.0, fttttr_node, eq, i, 2)

        multiply_add_set_node_vars!(gttttd, 0.5,
                                    1.0, gppd, -4.0, gpd,
                                    6.0, gd,
                                    -4.0, gmd, 1.0, gmmd,
                                    eq, 1)
        multiply_add_set_node_vars!(gttttu, 0.5,
                                    1.0, gppu, -4.0, gpu,
                                    6.0, gu,
                                    -4.0, gmu, 1.0, gmmu,
                                    eq, 1)

        gttttd_node = get_node_vars(gttttd, eq, 1)
        gttttu_node = get_node_vars(gttttu, eq, 1)

        multiply_add_to_node_vars!(Fb, 1.0 / 120.0, gttttd_node, eq, i, 3)
        multiply_add_to_node_vars!(Fb, 1.0 / 120.0, gttttu_node, eq, i, 4)
    end
end

function extrap_bflux!(eq::AbstractEquations{2}, grid, cell_data, eval_data,
                       xg, Vl, Vr, F, G, Fb, aux)
    nvar = nvariables(eq)
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

#-------------------------------------------------------------------------------
# Compute cell residual for degree=1 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_1!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme, aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    refresh!(u) = fill!(u, zero(eltype(u)))

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack eval_data, cell_arrays = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy

        id = Threads.threadid()
        f, g, F, G, ut, U, up, um, ft, gt, S = cell_arrays[id]

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
            set_node_vars!(um, u_node, eq, i, j)
            set_node_vars!(up, u_node, eq, i, j)
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
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)
            add_to_node_vars!(up, ut_node, eq, i, j)
            subtract_from_node_vars!(um, ut_node, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            multiply_add_to_node_vars!(ft, 0.5, fp, -0.5, fm, eq, i, j)
            multiply_add_to_node_vars!(gt, 0.5, gp, -0.5, gm, eq, i, j)
            ft_node = get_node_vars(ft, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = get_node_vars(gt, eq, i, j)
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
            st = calc_source_t_N12(up_node, um_node, X, t, dt, source_terms, eq)
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
        @views cell_data = (u1_, up, um, el_x, el_y)
        @views compute_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=2 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_2!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme, aux, t, dt, u1, res, Fb, Ub, cache)
    @unpack source_terms = problem
    @unpack xg, Dm, D1, DmT, D1T, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    refresh!(u) = fill!(u, zero(eltype(u)))

    @unpack blend_cell_residual! = aux.blend.subroutines
    @unpack compute_bflux! = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack eval_data, cell_arrays = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy

        # Some local variables
        id = Threads.threadid()
        f, g, ft, gt, F, G, ut, utt, U, up, um, S = cell_arrays[id]

        refresh!.((ut, utt))

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]
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
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(ut, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(ut, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
            set_node_vars!(um, u_node, eq, i, j)
            set_node_vars!(up, u_node, eq, i, j)
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
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)
            add_to_node_vars!(up, ut_node, eq, i, j)
            subtract_from_node_vars!(um, ut_node, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            multiply_add_set_node_vars!(ft, 0.5, fp, -0.5, fm, eq, i, j)
            multiply_add_set_node_vars!(gt, 0.5, gp, -0.5, gm, eq, i, j)
            ft_node = get_node_vars(ft, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = get_node_vars(gt, eq, i, j)
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
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            st = calc_source_t_N12(up_node, um_node, X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)
            multiply_add_to_node_vars!(utt, dt, st, eq, i, j) # has no jacobian factor
        end

        ftt, gtt = ft, gt

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            utt_node = get_node_vars(utt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node,
                                       eq, i, j)
            multiply_add_to_node_vars!(up, 0.5, utt_node, eq, i, j)
            multiply_add_to_node_vars!(um, 0.5, utt_node, eq, i, j)
            f_node = get_node_vars(f, eq, i, j)
            g_node = get_node_vars(g, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            multiply_add_set_node_vars!(ftt, 1.0, fp, -2.0, f_node, 1.0, fm, eq, i, j)
            multiply_add_set_node_vars!(gtt, 1.0, gp, -2.0, g_node, 1.0, gm, eq, i, j)
            ftt_node = get_node_vars(ftt, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       1.0 / 6.0, ftt_node,
                                       eq, i, j)
            gtt_node = get_node_vars(gtt, eq, i, j)
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
            stt = calc_source_tt_N23(u_node, up_node, um_node, X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0/6.0, stt, eq, i, j)

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
        # computes ftt, gtt and puts them in respective place; no need to store
        cell_data = (u1_, up, um, el_x, el_y)
        @views compute_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=3 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_3!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme, aux, t, dt, u1, res, Fb, Ub, cache)
    nvar = nvariables(eq)
    @unpack source_terms = problem
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

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        # Some local variables
        id = Threads.threadid()
        (f, g, ft, gt, F, G, ut, utt, uttt, U, up, um, upp, umm, S) = cell_arrays[id]

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
            set_node_vars!(um, u_node, eq, i, j)
            set_node_vars!(up, u_node, eq, i, j)
            set_node_vars!(umm, u_node, eq, i, j)
            set_node_vars!(upp, u_node, eq, i, j)
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
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)
            subtract_from_node_vars!(um, ut_node, eq, i, j)
            add_to_node_vars!(up, ut_node, eq, i, j)
            multiply_add_to_node_vars!(umm, -2.0, ut_node, eq, i, j)
            multiply_add_to_node_vars!(upp, 2.0, ut_node, eq, i, j)

            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            fmm, gmm = flux(x, y, umm_node, eq)
            fpp, gpp = flux(x, y, upp_node, eq)

            multiply_add_set_node_vars!(ft, 1.0 / 12.0,
                                        -1.0, fpp, 8.0, fp,
                                        -8.0, fm, 1.0, fmm,
                                        eq, i, j)
            multiply_add_set_node_vars!(gt, 1.0 / 12.0,
                                        -1.0, gpp, 8.0, gp,
                                        -8.0, gm, 1.0, gmm,
                                        eq, i, j)
            ft_node = get_node_vars(ft, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = get_node_vars(gt, eq, i, j)
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
            um_node = get_node_vars(um, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            st = calc_source_t_N34(u_node, up_node, upp_node, um_node, umm_node,
                                   X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)
            multiply_add_to_node_vars!(utt, dt, st, eq, i, j) # has no jacobian factor
        end

        ftt, gtt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            utt_node = get_node_vars(utt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node,
                                       eq, i, j)
            multiply_add_to_node_vars!(up, 0.5, utt_node, eq, i, j)
            multiply_add_to_node_vars!(um, 0.5, utt_node, eq, i, j)
            multiply_add_to_node_vars!(upp, 2.0, utt_node, eq, i, j)
            multiply_add_to_node_vars!(umm, 2.0, utt_node, eq, i, j)
            f_node = get_node_vars(f, eq, i, j)
            g_node = get_node_vars(g, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            multiply_add_set_node_vars!(ftt, fp, -2.0, f_node, fm, eq, i, j)

            ftt_node = get_node_vars(ftt, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       1.0 / 6.0, ftt_node,
                                       eq, i, j)

            multiply_add_set_node_vars!(gtt, gp, -2.0, g_node, gm, eq, i, j)
            gtt_node = get_node_vars(gtt, eq, i, j)
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
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            stt = calc_source_tt_N23(u_node, up_node, um_node, X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, eq, i, j)
            multiply_add_to_node_vars!(uttt, dt, stt, eq, i, j) # has no jacobian factor
        end

        fttt, gttt = ft, gt

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            uttt_node = get_node_vars(uttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node,
                                       eq, i, j)
            multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, eq, i, j)
            multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, eq, i, j)
            multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, eq, i, j)
            multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            fmm, gmm = flux(x, y, umm_node, eq)
            fpp, gpp = flux(x, y, upp_node, eq)
            multiply_add_set_node_vars!(fttt, 0.5,
                                        1.0, fpp, -2.0, fp,
                                        2.0, fm, -1.0, fmm,
                                        eq, i, j)
            fttt_node = get_node_vars(fttt, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       1.0 / 24.0, fttt_node,
                                       eq, i, j)
            multiply_add_set_node_vars!(gttt, 0.5,
                                        1.0, gpp, -2.0, gp,
                                        2.0, gm, -1.0, gmm,
                                        eq, i, j)
            gttt_node = get_node_vars(gttt, eq, i, j)
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
            sttt = calc_source_ttt_N34(u_node, up_node, um_node, upp_node, umm_node,
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
        @views cell_data = (u1_, up, um, upp, umm, el_x, el_y)
        @views compute_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=4 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_4!(eq::AbstractEquations{2}, grid, op, problem,
                                  scheme, aux, t, dt, u1, res, Fb, Ub, cache)
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
    @unpack eval_data, cell_arrays = cache
    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        # Some local variables
        id = Threads.threadid()
        (f, g, F, G, U, ft, gt, ut, utt, uttt, utttt, up, um, upp, umm, S) = cell_arrays[id]

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
            set_node_vars!(um, u_node, eq, i, j)
            set_node_vars!(up, u_node, eq, i, j)
            set_node_vars!(umm, u_node, eq, i, j)
            set_node_vars!(upp, u_node, eq, i, j)
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
            ut_node = get_node_vars(ut, eq, i, j)
            multiply_add_to_node_vars!(U,
                                       0.5, ut_node,
                                       eq, i, j)
            subtract_from_node_vars!(um, ut_node, eq, i, j)
            add_to_node_vars!(up, ut_node, eq, i, j)
            multiply_add_to_node_vars!(umm, -2.0, ut_node, eq, i, j)
            multiply_add_to_node_vars!(upp, 2.0, ut_node, eq, i, j)

            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            fmm, gmm = flux(x, y, umm_node, eq)
            fpp, gpp = flux(x, y, upp_node, eq)

            multiply_add_set_node_vars!(ft, 1.0 / 12.0,
                                        -1.0, fpp, 8.0, fp,
                                        -8.0, fm, 1.0, fmm,
                                        eq, i, j)
            multiply_add_set_node_vars!(gt, 1.0 / 12.0,
                                        -1.0, gpp, 8.0, gp,
                                        -8.0, gm, 1.0, gmm,
                                        eq, i, j)
            ft_node = get_node_vars(ft, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       0.5, ft_node,
                                       eq, i, j)
            gt_node = get_node_vars(gt, eq, i, j)
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
            um_node = get_node_vars(um, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            st = calc_source_t_N34(u_node, up_node, upp_node, um_node, umm_node,
                                   X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, st, eq, i, j)
            multiply_add_to_node_vars!(utt, dt, st, eq, i, j) # has no jacobian factor
        end

        ftt, gtt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            utt_node = get_node_vars(utt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node,
                                       eq, i, j)
            multiply_add_to_node_vars!(up, 0.5, utt_node, eq, i, j)
            multiply_add_to_node_vars!(um, 0.5, utt_node, eq, i, j)
            multiply_add_to_node_vars!(upp, 2.0, utt_node, eq, i, j)
            multiply_add_to_node_vars!(umm, 2.0, utt_node, eq, i, j)
            f_node = get_node_vars(f, eq, i, j)
            g_node = get_node_vars(g, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)

            umm_node = get_node_vars(umm, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            fmm, gmm = flux(x, y, umm_node, eq)
            fpp, gpp = flux(x, y, upp_node, eq)
            multiply_add_set_node_vars!(ftt, 1.0 / 12.0,
                                        -1.0, fpp, 16.0, fp,
                                        -30.0, f_node,
                                        16.0, fm, -1.0, fmm,
                                        eq, i, j)

            ftt_node = get_node_vars(ftt, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       1.0 / 6.0, ftt_node,
                                       eq, i, j)

            multiply_add_set_node_vars!(gtt, 1.0 / 12.0,
                                        -1.0, gpp, 16.0, gp,
                                        -30.0, g_node,
                                        16.0, gm, -1.0, gmm,
                                        eq, i, j)
            gtt_node = get_node_vars(gtt, eq, i, j)
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
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            stt = calc_source_tt_N4(u_node, up_node, upp_node, um_node, umm_node, X, t,
                                    dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, eq, i, j)
            multiply_add_to_node_vars!(uttt, dt, stt, eq, i, j) # has no jacobian factor
        end

        fttt, gttt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            uttt_node = get_node_vars(uttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node,
                                       eq, i, j)
            multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, eq, i, j)
            multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, eq, i, j)
            multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, eq, i, j)
            multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            fmm, gmm = flux(x, y, umm_node, eq)
            fpp, gpp = flux(x, y, upp_node, eq)
            multiply_add_set_node_vars!(fttt, 0.5,
                                        1.0, fpp, -2.0, fp,
                                        2.0, fm, -1.0, fmm,
                                        eq, i, j)
            fttt_node = get_node_vars(fttt, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       1.0 / 24.0, fttt_node,
                                       eq, i, j)
            multiply_add_set_node_vars!(gttt, 0.5,
                                        1.0, gpp, -2.0, gp,
                                        2.0, gm, -1.0, gmm,
                                        eq, i, j)
            gttt_node = get_node_vars(gttt, eq, i, j)
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
            X = SVector(x,y)
            # Add source term contribution to utttt
            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            um_node = get_node_vars(um, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            sttt = calc_source_ttt_N34(u_node, up_node, upp_node, um_node, umm_node,
                                       X, t, dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, eq, i, j)
            multiply_add_to_node_vars!(utttt, dt, sttt, eq, i, j) # has no jacobian factor
        end

        ftttt, gtttt = ft, gt # reusing old

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            utttt_node = get_node_vars(utttt, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node,
                                       eq, i, j)
            multiply_add_to_node_vars!(um, 1.0 / 24.0, utttt_node, eq, i, j)
            multiply_add_to_node_vars!(up, 1.0 / 24.0, utttt_node, eq, i, j)
            multiply_add_to_node_vars!(umm, 2.0 / 3.0, utttt_node, eq, i, j)
            multiply_add_to_node_vars!(upp, 2.0 / 3.0, utttt_node, eq, i, j)
            f_node = get_node_vars(f, eq, i, j)
            g_node = get_node_vars(g, eq, i, j)
            um_node = get_node_vars(um, eq, i, j)
            up_node = get_node_vars(up, eq, i, j)
            umm_node = get_node_vars(umm, eq, i, j)
            upp_node = get_node_vars(upp, eq, i, j)
            fm, gm = flux(x, y, um_node, eq)
            fp, gp = flux(x, y, up_node, eq)
            fmm, gmm = flux(x, y, umm_node, eq)
            fpp, gpp = flux(x, y, upp_node, eq)
            multiply_add_set_node_vars!(ftttt, 0.5,
                                        1.0, fpp, -4.0, fp,
                                        6.0, f_node,
                                        -4.0, fm, 1.0, fmm,
                                        eq, i, j)
            ftttt_node = get_node_vars(ftttt, eq, i, j)
            multiply_add_to_node_vars!(F,
                                       1.0 / 120.0, ftttt_node,
                                       eq, i, j)
            multiply_add_set_node_vars!(gtttt, 0.5,
                                        1.0, gpp, -4.0, gp,
                                        6.0, g_node,
                                        -4.0, gm, 1.0, gmm,
                                        eq, i, j)
            gtttt_node = get_node_vars(gtttt, eq, i, j)
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
            stttt = calc_source_tttt_N4(u_node, up_node, um_node, upp_node, umm_node,
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

        @views cell_data = (u, up, um, upp, umm, el_x, el_y)
        Fb_ = @view Fb[:, :, :, el_x, el_y]
        compute_bflux!(eq, grid, cell_data, eval_data, xg, Vl, Vr,
                       F, G, Fb_, aux)
    end
    return nothing
end
end # @muladd
