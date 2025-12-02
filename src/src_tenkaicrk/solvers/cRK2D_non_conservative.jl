import Tenkai: compute_face_residual!, compute_cell_residual_cRK!, evolve_solution!,
               trivial_face_residual, blend_cell_residual_fo!,
               blend_face_residual_fo_x!, blend_face_residual_fo_y!,
               multiply_add_to_node_vars!, blend_flux_face_x_residual!,
               blend_flux_face_y_residual!

import Base: *, -, +
import LinearAlgebra: adjoint

using Tenkai: cRKSolver, add_low_order_face_residual!

struct MyZero end

function flux_central_conservative(u_ll, u_rr, orientation, eq)
    x = y = nothing
    return 0.5 * (flux(x, y, u_ll, eq, orientation) +
                  flux(x, y, u_rr, eq, orientation))
end

function flux_central_non_conservative(u_ll, u_rr, orientation, eq)
    x = y = t = nothing
    return calc_non_cons_Bu(u_ll, u_rr, x, y, t, orientation, eq)
end

struct MyVolumeIntegralFluxDifferencing{FluxConservative, FluxNonConservative}
    degree::Int
    flux_conservative::FluxConservative
    flux_non_conservative::FluxNonConservative
end

function MyVolumeIntegralFluxDifferencing(degree, flux_conservative,
                                          flux_non_conservative)
    return MyVolumeIntegralFluxDifferencing(degree, flux_conservative,
                                            flux_non_conservative)
end

*(a::MyZero, ::Any) = a
*(::Any, a::MyZero) = a
-(a::MyZero) = a
+(::MyZero, b) = b
+(b, ::MyZero) = b
adjoint(a::MyZero) = a

@inline function multiply_add_to_node_vars!(u::AbstractArray,
                                            factor::MyZero, u_node::SVector{<:Any},
                                            equations::AbstractEquations,
                                            indices...)
    return nothing
end

function setup_arrays(grid, scheme::Scheme{<:cRKSolver},
                      eq::AbstractNonConservativeEquations{2})
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
    nc_var = nvariables(non_conservative_equation(eq))
    nd = degree + 1
    nx, ny = grid.size
    u1 = gArray(nvar, nd, nd, nx, ny)
    ua = gArray(nvar, nx, ny)
    res = gArray(nvar, nd, nd, nx, ny)
    Bb = OffsetArray(zeros(nvar, nc_var, nd, 4, nx + 2, ny + 2),
                     OffsetArrays.Origin(1, 1, 1, 1, 0, 0))
    Fb = gArray(nvar, nd, 4, nx, ny)
    # TODO - Can Fnum be equal to Fb?
    Fnum = gArray(nvar, nd, 4, nx, ny) # Numerical fluxes at faces of each element
    Ub = gArray(nvar, nd, 4, nx, ny)
    u1_b = copy(Ub)
    ub_N = gArray(nvar, nd, 4, nx, ny) # The final stage of cRK before communication

    # Cell residual cache

    nt = Threads.nthreads()
    cell_array_sizes = Dict(0 => 11, 1 => 11, 2 => 12, 3 => 15, 4 => 16)
    big_eval_data_sizes = Dict(0 => 12, 1 => 32, 2 => 40, 3 => 56, 4 => 56)
    small_eval_data_sizes = Dict(0 => 4, 1 => 4, 2 => 4, 3 => 4, 4 => 4)
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
    cache = (; u1, ua, ub_N, res, Fb, Fnum, Ub, Bb, u1_b, eval_data, cell_arrays,
             ghost_cache)
    return cache
end

function update_ghost_values_Bb!(problem, scheme, eq::AbstractNonConservativeEquations{2},
                                 grid, aux, op, cache, t, dt)
    # TODO - Move this to its right place!!
    @unpack Bb = cache
    nx = size(Bb, 5) - 2
    ny = size(Bb, 6) - 2
    nvar = size(Bb, 1) # Temporary, should take from eq
    nvar_nc = size(Bb, 2)

    @unpack degree, xg, wg = op
    nd = degree + 1
    (; dx, dy, xf, yf) = grid
    nvar = nvariables(eq)
    @unpack boundary_value, boundary_condition = problem
    left, right, bottom, top = boundary_condition

    if problem.periodic_x
        # Left ghost cells
        copyto!(Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 2:2, 0:0, 1:ny)),
                Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 2:2, nx:nx, 1:ny)))

        # Right ghost cells
        copyto!(Bb,
                CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 1:1, (nx + 1):(nx + 1), 1:ny)),
                Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 1:1, 1:1, 1:ny)))
    end

    if problem.periodic_y
        # Bottom ghost cells
        copyto!(Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 3:3, 1:nx, 0:0)),
                Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 3:3, 1:nx, ny:ny)))

        # Top ghost cells
        copyto!(Bb,
                CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 4:4, 1:nx, (ny + 1):(ny + 1))),
                Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:nd, 4:4, 1:nx, 1:1)))
    end

    if problem.periodic_x && problem.periodic_y
        return nothing # No need to update ghost cells
    end

    return nothing
end

function blend_cell_residual_fo!(el_x, el_y, eq::AbstractNonConservativeEquations{2},
                                 problem, scheme,
                                 aux, t, dt, grid, dx, dy, xf, yf, op, u1, u_, f, res,
                                 scaling_factor = 1.0)
    @timeit_debug aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack blend = aux
    @unpack Vl, Vr, xg, wg = op
    @unpack source_terms = problem
    num_flux = scheme.numerical_flux
    nd = length(xg)

    id = Threads.threadid()
    xxf, yyf = blend.cache.subcell_faces[id]
    @unpack fn_low = blend.cache
    alpha = blend.cache.alpha[el_x, el_y]
    blend_resl = @view blend.cache.resl[:, :, :, el_x, el_y]
    blend_resl .= zero(eltype(blend_resl))

    u = @view u1[:, :, :, el_x, el_y]
    r = @view res[:, :, :, el_x, el_y]

    # if alpha < 1e-12
    #     store_low_flux!(u, el_x, el_y, xf, yf, dx, dy, op, blend, eq,
    #                     scaling_factor)
    #     return nothing
    # end

    # limit the higher order part
    lmul!(1.0 - alpha, r)

    # compute subcell faces
    xxf[0], yyf[0] = xf, yf
    for ii in Base.OneTo(nd)
        xxf[ii] = xxf[ii - 1] + dx * wg[ii]
        yyf[ii] = yyf[ii - 1] + dy * wg[ii]
    end

    # loop over vertical inner faces between (ii-1,jj) and (ii,jj)
    for ii in 2:nd # skipping the supercell face for blend_face_residual
        xx = xxf[ii - 1] # Face x coordinate, offset because index starts from 0
        for jj in Base.OneTo(nd)
            yy = yf + dy * xg[jj] # Face y coordinates picked same as soln points
            X = SVector(xx, yy)
            ul, ur = get_node_vars(u, eq, ii - 1, jj), get_node_vars(u, eq, ii, jj)
            fl, fr = flux(xx, yy, ul, eq, 1), flux(xx, yy, ur, eq, 1)
            fn = scaling_factor * num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)
            us = 0.5 * (ul + ur) # For "non-conservative numerical flux"
            Bu_l = calc_non_cons_Bu(ul, us, xx, yy, t, 1, eq)
            Bu_r = calc_non_cons_Bu(ur, us, xx, yy, t, 1, eq)
            fn_l = fn + Bu_l
            fn_r = fn + Bu_r
            multiply_add_to_node_vars!(r, # r[ii-1,jj]+=alpha*dt/(dx*wg[ii-1])*fn
                                       alpha * dt / (dx * wg[ii - 1]),
                                       fn_l, eq, ii - 1, jj)
            multiply_add_to_node_vars!(r, # r[ii,jj]+=alpha*dt/(dx*wg[ii])*fn
                                       -alpha * dt / (dx * wg[ii]),
                                       fn_r, eq, ii, jj)
            multiply_add_to_node_vars!(blend_resl,
                                       dt / (dx * wg[ii - 1]),
                                       fn_l, eq, ii - 1, jj)
            multiply_add_to_node_vars!(blend_resl,
                                       -dt / (dx * wg[ii]),
                                       fn_r, eq, ii, jj)
            # TOTHINK - Can checking this in every step of the loop be avoided
            if ii == 2
                set_node_vars!(fn_low, fn_l, eq, jj, 1, el_x, el_y)
            elseif ii == nd
                set_node_vars!(fn_low, fn_r, eq, jj, 2, el_x, el_y)
            end
        end
    end

    # loop over horizontal inner faces between (ii,jj-1) and (ii,jj)
    for jj in 2:nd
        yy = yyf[jj - 1] # face y coordinate, offset because index starts from 0
        for ii in Base.OneTo(nd)
            xx = xf + dx * xg[ii] # face x coordinate picked same as soln pt
            X = SVector(xx, yy)
            ul, ur = get_node_vars(u, eq, ii, jj - 1), get_node_vars(u, eq, ii, jj)
            fl, fr = flux(xx, yy, ul, eq, 2), flux(xx, yy, ur, eq, 2)
            fn = scaling_factor * num_flux(X, ul, ur, fl, fr, ul, ur, eq, 2)
            us = 0.5 * (ul + ur) # For "non-conservative numerical flux"
            Bu_l = calc_non_cons_Bu(ul, us, xx, yy, t, 2, eq)
            Bu_r = calc_non_cons_Bu(ur, us, xx, yy, t, 2, eq)
            fn_l = fn + Bu_l
            fn_r = fn + Bu_r

            multiply_add_to_node_vars!(r, # r[ii,jj-1]+=alpha*dt/(dy*wg[jj-1])*fn
                                       alpha * dt / (dy * wg[jj - 1]),
                                       fn_l,
                                       eq, ii, jj - 1)
            multiply_add_to_node_vars!(r, # r[ii,jj]+=alpha*dt/(dy*wg[jj])*fn
                                       -alpha * dt / (dy * wg[jj]),
                                       fn_r,
                                       eq, ii, jj)
            multiply_add_to_node_vars!(blend_resl,
                                       dt / (dy * wg[jj - 1]),
                                       fn_l, eq, ii, jj - 1)
            multiply_add_to_node_vars!(blend_resl,
                                       -dt / (dy * wg[jj]),
                                       fn_r, eq, ii, jj)
            # TOTHINK - Can checking this in every step of the loop be avoided
            if jj == 2
                # TOTHINK - Does this need to be doubled? I don't think so...
                set_node_vars!(fn_low, fn_l, eq, ii, 3, el_x, el_y)
            elseif jj == nd
                set_node_vars!(fn_low, fn_r, eq, ii, 4, el_x, el_y)
            end
        end
    end

    for jj in 1:nd
        for ii in Base.OneTo(nd)
            xx = xf + dx * xg[ii] # face x coordinate picked same as soln pt
            yy = yf + dy * xg[jj] # face x coordinate picked same as soln pt
            X = SVector(xx, yy)
            u_node = get_node_vars(u, eq, ii, jj)
            s_node = calc_source(u_node, X, t, source_terms, eq)
            multiply_add_to_node_vars!(r,
                                       -alpha * dt, # / (dx * dy * wg[ii] * wg[jj]),
                                       s_node, eq, ii, jj)
            multiply_add_to_node_vars!(blend_resl,
                                       -dt, # / (dx * dy * wg[ii] * wg[jj]),
                                       s_node, eq, ii, jj)
        end
    end
    end # timer
end

function blend_flux_face_x_residual!(el_x, el_y, jy, xf, y, u1, ua,
                                     eq::AbstractNonConservativeEquations{2}, dt, grid, op,
                                     scheme, param,
                                     Fn_l, Fn_r, aux, res,
                                     scaling_factor = 1.0)
    blend_face_residual_fo_x!(el_x, el_y, jy, xf, y, u1, ua,
                              eq, dt, grid, op,
                              scheme, param,
                              Fn_l, Fn_r, aux, res,
                              scaling_factor)
end

function blend_face_residual_fo_x!(el_x, el_y, jy, xf, y, u1, ua,
                                   eq::AbstractNonConservativeEquations{2}, dt, grid, op,
                                   scheme, param,
                                   Fn_l, Fn_r, aux, res,
                                   scaling_factor = 1.0)
    @timeit_debug aux.timer "Blending limiter" begin # Check the overhead,
    #! format: noindent
    # it's supposed to be 0.25 microseconds
    @unpack blend = aux
    @unpack bc_x = blend.subroutines
    @unpack alpha = blend.cache
    @unpack dx, dy = grid
    num_flux = scheme.numerical_flux

    @unpack xg, wg = op
    nd = length(xg)

    ul = get_node_vars(u1, eq, nd, jy, el_x - 1, el_y)
    ur = get_node_vars(u1, eq, 1, jy, el_x, el_y)
    fl, fr = flux(xf, y, ul, eq, 1), flux(xf, y, ur, eq, 1)

    X = SVector(xf, y)
    fn = scaling_factor * num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)
    us = 0.5 * (ul + ur) # For "non-conservative numerical flux"
    t = 0.0 # Dummy value
    Bu_l = calc_non_cons_Bu(ul, us, xf, y, t, 1, eq)
    Bu_r = calc_non_cons_Bu(ur, us, xf, y, t, 1, eq)
    fn_l = fn + Bu_l
    fn_r = fn + Bu_r

    Fn_l, Fn_r = get_blended_flux_x(el_x, el_y, jy, eq, dt, grid,
                                    blend, scheme, xf, y, u1, ua, fn_l, fn_r, Fn_l,
                                    Fn_r, op)

    # This subroutine allows user to specify boundary conditions
    # Fn = bc_x(u1, eq, op, xf, y, jy, el_x, el_y, Fn)

    return Fn_l, Fn_r, (1.0 - alpha[el_x - 1, el_y], 1.0 - alpha[el_x, el_y])
    end # timer
end

function blend_flux_face_y_residual!(el_x, el_y, ix, x, yf, u1, ua,
                                     eq::AbstractNonConservativeEquations{2}, dt, grid, op,
                                     scheme, param, Fn_l, Fn_r, aux, res,
                                     scaling_factor = 1.0)
    blend_face_residual_fo_y!(el_x, el_y, ix, x, yf, u1, ua,
                              eq, dt, grid, op,
                              scheme, param,
                              Fn_l, Fn_r, aux, res,
                              scaling_factor)
end

function blend_face_residual_fo_y!(el_x, el_y, ix, x, yf, u1, ua,
                                   eq::AbstractNonConservativeEquations{2}, dt, grid, op,
                                   scheme, param, Fn_l, Fn_r, aux, res,
                                   scaling_factor = 1.0)
    @timeit_debug aux.timer "Blending limiter" begin # Check the overhead,
    #! format: noindent
    # it's supposed to be 0.25 microseconds
    @unpack blend = aux
    @unpack alpha = blend.cache
    num_flux = scheme.numerical_flux
    @unpack dx, dy = grid

    @unpack xg, wg = op
    nd = length(xg)

    ul = get_node_vars(u1, eq, ix, nd, el_x, el_y - 1)
    fl = flux(x, yf, ul, eq, 2)
    ur = get_node_vars(u1, eq, ix, 1, el_x, el_y)
    fr = flux(x, yf, ur, eq, 2)
    X = SVector(x, yf)
    fn = scaling_factor * num_flux(X, ul, ur, fl, fr, ul, ur, eq, 2)
    us = 0.5 * (ul + ur) # For "non-conservative numerical flux"
    t = 0.0 # Dummy value
    Bu_l = calc_non_cons_Bu(ul, us, x, yf, t, 2, eq)
    Bu_r = calc_non_cons_Bu(ur, us, x, yf, t, 2, eq)
    fn_l = fn + Bu_l
    fn_r = fn + Bu_r

    Fn_l, Fn_r = get_blended_flux_y(el_x, el_y, ix, eq, dt, grid, blend,
                                    scheme, x, yf, u1, ua, fn_l, fn_r, Fn_l, Fn_r, op)

    return Fn_l, Fn_r, (1.0 - alpha[el_x, el_y - 1], 1.0 - alpha[el_x, el_y])
    end # timer
end

# We blend the lower order flux with flux at interfaces
function get_blended_flux_x(el_x, el_y, jy, eq::AbstractEquations{2}, dt, grid,
                            blend, scheme, xf, y, u1, ua, fn_l, fn_r, Fn_l, Fn_r, op)
    # if scheme.solver_enum == rkfr
    #     return Fn
    # end

    @unpack alpha, fn_low = blend.cache
    @unpack dx, dy = grid
    @unpack wg = op
    nd = length(wg)
    nx, ny = grid.size

    # Initial trial blended flux
    alp = 0.5 * (alpha[el_x - 1, el_y] + alpha[el_x, el_y])
    Fn_l = (1.0 - alp) * Fn_l + alp * fn_l
    Fn_r = (1.0 - alp) * Fn_r + alp * fn_r

    ua_ll_node = get_node_vars(ua, eq, el_x - 1, el_y)
    λx_ll, _ = blending_flux_factors(eq, ua_ll_node, dx[el_x - 1], dy[el_y])

    ua_rr_node = get_node_vars(ua, eq, el_x, el_y)
    λx_rr, _ = blending_flux_factors(eq, ua_rr_node, dx[el_x], dy[el_y])

    # We see update at solution point in element (el_x-1,el_y)
    u_ll_node = get_node_vars(u1, eq, nd, jy, el_x - 1, el_y)

    # lower order flux on neighbouring subcell face
    fn_inner_ll = get_node_vars(fn_low, eq, jy, 2, el_x - 1, el_y)

    # Test whether lower order update is even admissible
    c_ll = (dt / dx[el_x - 1]) / (wg[nd] * λx_ll) # c is such that u_new = u_prev - c*(Fn-fn)
    low_update_ll = u_ll_node - c_ll * (fn_l - fn_inner_ll)
    test_update_ll = u_ll_node - c_ll * (Fn_l - fn_inner_ll)
    if is_admissible(eq, low_update_ll) == false && el_x > 1
        # @warn "Low x-flux not admissible at " (el_x-1),el_y,xf,y
    end

    if !(is_admissible(eq, test_update_ll))
        @debug "Zhang-Shu fix needed at " (el_x - 1), el_y, xf, y
        Fn_l = zhang_shu_flux_fix(eq, u_ll_node, low_update_ll,
                                  Fn_l, fn_inner_ll, fn_l, c_ll)
    end

    # Now we see the update at solution point in element (el_x,el_y)
    u_rr_node = get_node_vars(u1, eq, 1, jy, el_x, el_y)

    # lower order flux on neighbouring subcell face
    fn_inner_rr = get_node_vars(fn_low, eq, jy, 1, el_x, el_y)

    # Test whether lower order update is even admissible
    c_rr = -(dt / dx[el_x]) / (wg[1] * λx_rr) # c is such that u_new = u_prev - c*(Fn-fn)
    low_update_rr = u_rr_node - c_rr * (fn_r - fn_inner_rr)
    if is_admissible(eq, low_update_rr) == false && el_x < nx + 1
        # @warn "Lower x-flux not admissible at " el_x,el_y,xf,y
    end
    test_update_rr = u_rr_node - c_rr * (Fn_r - fn_inner_rr)

    if !(is_admissible(eq, test_update_rr))
        @debug "Zhang-Shu fix needed at " (el_x - 1), el_y, xf, y
        Fn_r = zhang_shu_flux_fix(eq, u_rr_node, low_update_rr, Fn_r, fn_inner_rr,
                                  fn_r, c_rr)
    end

    return Fn_l, Fn_r
end

function get_blended_flux_y(el_x, el_y, ix, eq::AbstractEquations{2}, dt, grid,
                            blend, scheme, x, yf, u1, ua, fn_l, fn_r, Fn_l, Fn_r, op)
    # if scheme.solver_enum == rkfr # TODO - Don't do this for GL points
    #     return Fn
    # end

    @unpack alpha, fn_low = blend.cache
    @unpack dx, dy = grid
    @unpack wg = op
    nd = length(wg)
    nx, ny = grid.size
    # Initial trial blended flux
    alp = 0.5 * (alpha[el_x, el_y - 1] + alpha[el_x, el_y])
    Fn_l = (1.0 - alp) * Fn_l + alp * fn_l
    Fn_r = (1.0 - alp) * Fn_r + alp * fn_r

    # Candidate in for (el_x, el_y-1)
    ua_ll_node = get_node_vars(ua, eq, el_x, el_y - 1)
    λx_ll, λy_ll = blending_flux_factors(eq, ua_ll_node, dx[el_x], dy[el_y - 1])
    # Candidate in (el_x, el_y)
    ua_rr_node = get_node_vars(ua, eq, el_x, el_y)
    λx_rr, λy_rr = blending_flux_factors(eq, ua_rr_node, dx[el_x], dy[el_y])

    u_ll_node = get_node_vars(u1, eq, ix, nd, el_x, el_y - 1)

    # lower order flux on neighbouring subcell face
    fn_inner_ll = get_node_vars(fn_low, eq, ix, 4, el_x, el_y - 1)

    c_ll = (dt / dy[el_y - 1]) / (wg[nd] * λy_ll)

    # test whether lower order update is even admissible
    low_update_ll = u_ll_node - c_ll * (fn_l - fn_inner_ll)
    test_update_ll = u_ll_node - c_ll * (Fn_l - fn_inner_ll)

    if is_admissible(eq, low_update_ll) == false && el_y > 1
        # @warn "Low y-flux not admissible at " el_x,(el_y-1),x,yf
    end

    if !(is_admissible(eq, test_update_ll))
        @debug "Zhang-Shu fix needed at " el_x, (el_y - 1), xf, y
        Fn_l = zhang_shu_flux_fix(eq, u_ll_node, low_update_ll,
                                  Fn_l, fn_inner_ll, fn_l, c_ll)
    end

    u_rr_node = get_node_vars(u1, eq, ix, 1, el_x, el_y)
    fn_inner_rr = get_node_vars(fn_low, eq, ix, 3, el_x, el_y)
    c_rr = -(dt / dy[el_y]) / (wg[1] * λy_rr)
    low_update_rr = u_rr_node - c_rr * (fn_r - fn_inner_rr)

    if is_admissible(eq, low_update_rr) == false && el_y < ny + 1
        # @warn "Lower y-flux not admissible at " el_x,el_y,x,yf
    end

    test_update_rr = u_rr_node - c_rr * (Fn_r - fn_inner_rr)

    if !(is_admissible(eq, test_update_rr))
        @debug "Zhang-Shu fix needed at " (el_x - 1), el_y, xf, y
        Fn_r = zhang_shu_flux_fix(eq, u_rr_node, low_update_rr, Fn_r, fn_inner_rr,
                                  fn_r, c_rr)
    end

    return Fn_l, Fn_r
end

@inline function trivial_face_residual(i, j, k, x, yf, u1, ua,
                                       eq::AbstractNonConservativeEquations{2},
                                       dt, grid, op, scheme, param, Fn_l, Fn_r, aux, res,
                                       scaling_factor = 1.0)
    return Fn_l, Fn_r, (1.0, 1.0)
end

function compute_non_cons_terms(ul, ur, Ul, Ur, x, y, t, dir,
                                solver, eq::AbstractNonConservativeEquations{2})
    ul_nc, ur_nc = (calc_non_cons_gradient(u, x, y, t, eq) for u in (Ul, Ur))
    u_non_cons_interface = 0.5 * (ul_nc + ur_nc)

    Bul = calc_non_cons_Bu(Ul, u_non_cons_interface, x, y, t, dir, eq)
    Bur = calc_non_cons_Bu(Ur, u_non_cons_interface, x, y, t, dir, eq)

    return Bul, Bur
end

# This will be much for expensive, but seems to not give much more of a benefit
# than the above function. So, we will use the above function for now and
# keep testing.
function compute_non_cons_terms_cHT112(ul, ur, Ul, Ur, x, y, t, dir,
                                       solver::cHT112,
                                       eq::AbstractNonConservativeEquations{2})
    u1l, u1r = ul, ur
    u2l, u2r = 2.0 * Ul - u1l, 2.0 * Ur - u1r

    u1l_nc, u1r_nc = (calc_non_cons_gradient(u, x, y, t, eq) for u in (u1l, u1r))
    u2l_nc, u2r_nc = (calc_non_cons_gradient(u, x, y, t, eq) for u in (u2l, u2r))

    u1_non_cons_interface = 0.5 * (u1l_nc + u1r_nc)
    u2_non_cons_interface = 0.5 * (u2l_nc + u2r_nc)

    Bu1_l = (calc_non_cons_Bu(u1l, u1_non_cons_interface, x, y, t, dir, eq)
             -
             calc_non_cons_Bu(u1l, u1l_nc, x, y, t, dir, eq))

    Bu1_r = (calc_non_cons_Bu(u1r, u1_non_cons_interface, x, y, t, dir, eq)
             -
             calc_non_cons_Bu(u1r, u1r_nc, x, y, t, dir, eq))

    Bu2_l = (calc_non_cons_Bu(u2l, u2_non_cons_interface, x, y, t, dir, eq)
             -
             calc_non_cons_Bu(u2l, u2l_nc, x, y, t, dir, eq))

    Bu2_r = (calc_non_cons_Bu(u2r, u2_non_cons_interface, x, y, t, dir, eq)
             -
             calc_non_cons_Bu(u2r, u2r_nc, x, y, t, dir, eq))

    return 0.5 * (Bu1_l + Bu2_l), 0.5 * (Bu1_r + Bu2_r)
end

#-------------------------------------------------------------------------------
# Add numerical flux to residual
#-------------------------------------------------------------------------------
function compute_face_residual!(eq::AbstractNonConservativeEquations{2}, grid, op,
                                cache, problem,
                                scheme::Scheme{<:cRKSolver}, param, aux, t, dt, u1,
                                Fb, Ub, ua, res, scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack bl, br, xg, wg, degree = op
    nd = degree + 1
    nx, ny = grid.size
    @unpack dx, dy, xf, yf = grid
    @unpack numerical_flux, solver = scheme
    @unpack blend = aux
    @unpack blend_face_residual_x!, blend_face_residual_y!, get_element_alpha = blend.subroutines
    @unpack u1_b, Bb, Fnum = cache

    # Vertical faces, x flux
    @threaded for element in CartesianIndices((1:(nx + 1), 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        # This is face between elements (el_x-1, el_y), (el_x, el_y)
        x = xf[el_x]
        for jy in Base.OneTo(nd)
            y = yf[el_y] + xg[jy] * dy[el_y]
            ul, ur = (get_node_vars(u1_b, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(u1_b, eq, jy, 1, el_x, el_y))
            Fl, Fr = (get_node_vars(Fb, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(Fb, eq, jy, 1, el_x, el_y))
            Ul, Ur = (get_node_vars(Ub, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(Ub, eq, jy, 1, el_x, el_y))
            X = SVector{2}(x, y)

            Bul, Bur = compute_non_cons_terms(ul, ur, Ul, Ur, x, y, t, 1, solver, eq)

            Fn = numerical_flux(X, ul, ur, Fl, Fr, Ul, Ur, eq, 1)

            Fn_l = Fn + Bul
            Fn_r = Fn + Bur
            Fn_l_limited, Fn_r_limited, _ = blend_face_residual_x!(el_x,
                                                                   el_y, jy,
                                                                   x, y, u1,
                                                                   ua,
                                                                   eq, dt,
                                                                   grid, op,
                                                                   scheme,
                                                                   param,
                                                                   Fn_l,
                                                                   Fn_r,
                                                                   aux,
                                                                   res,
                                                                   scaling_factor)

            set_node_vars!(Fnum, Fn_l_limited, eq, jy, 2, el_x - 1, el_y)
            set_node_vars!(Fnum, Fn_r_limited, eq, jy, 1, el_x, el_y)
        end
    end

    # Horizontal faces, y flux
    @threaded for element in CartesianIndices((1:nx, 1:(ny + 1))) # Loop over cells
        el_x, el_y = element[1], element[2]
        # This is the face between elements (el_x,el_y-1) and (el_x,el_y)
        y = yf[el_y]
        for ix in Base.OneTo(nd)
            x = xf[el_x] + xg[ix] * dx[el_x]
            ul, ur = get_node_vars(u1_b, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(u1_b, eq, ix, 3, el_x, el_y)
            Fl, Fr = get_node_vars(Fb, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(Fb, eq, ix, 3, el_x, el_y)
            Ul, Ur = get_node_vars(Ub, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(Ub, eq, ix, 3, el_x, el_y)
            X = SVector{2}(x, y)
            Bul, Bur = compute_non_cons_terms(ul, ur, Ul, Ur, x, y, t, 2, solver, eq)
            Fn = numerical_flux(X, ul, ur, Fl, Fr, Ul, Ur, eq, 2)
            Fn_l = Fn + Bul
            Fn_r = Fn + Bur
            Fn_l_limited, Fn_r_limited, _ = blend_face_residual_y!(el_x,
                                                                   el_y, ix,
                                                                   x, y,
                                                                   u1, ua,
                                                                   eq, dt,
                                                                   grid, op,
                                                                   scheme,
                                                                   param,
                                                                   Fn_l,
                                                                   Fn_r,
                                                                   aux,
                                                                   res,
                                                                   scaling_factor)

            set_node_vars!(Fnum, Fn_l_limited, eq, ix, 4, el_x, el_y - 1)
            set_node_vars!(Fnum, Fn_r_limited, eq, ix, 3, el_x, el_y)
        end
    end

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        alpha = get_element_alpha(blend, el_x, el_y) # TODO - Use a function to get this
        one_m_alp = 1.0 - alpha
        for ix in Base.OneTo(nd)

            # For higher order residual
            for jy in Base.OneTo(nd)
                Fl = get_node_vars(Fnum, eq, jy, 1, el_x, el_y)
                Fr = get_node_vars(Fnum, eq, jy, 2, el_x, el_y)
                Fd = get_node_vars(Fnum, eq, ix, 3, el_x, el_y)
                Fu = get_node_vars(Fnum, eq, ix, 4, el_x, el_y)
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

    add_low_order_face_residual!(get_element_alpha, eq, grid, op, aux, dt, Fnum, res)

    return nothing
    end # timer
end

#-------------------------------------------------------------------------------
# Cell residual functions
#-------------------------------------------------------------------------------
function flux_der!(volume_integral, r1, u_tuples_out, F_G_U_S, A_rk_tuple,
                   b_rk_coeff, u_in, op, local_grid, eq::AbstractEquations{2})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    F, G, U, S = F_G_U_S
    xc, yc, dx, dy, lamx, lamy, dt = local_grid
    nd = length(xg)
    # Solution points
    for j in 1:nd, i in 1:nd
        x = xc - 0.5 * dx + xg[i] * dx
        y = yc - 0.5 * dy + xg[j] * dy
        u_node = get_node_vars(u_in, eq, i, j)
        flux1, flux2 = flux(x, y, u_node, eq)

        # TOTHINK - Should the `integral_contribution` approach be tried here?
        for i_u in eachindex(u_tuples_out)
            u = u_tuples_out[i_u]
            a = -A_rk_tuple[i_u]
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(u, a * lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(u, a * lamy * Dm[jj, j], flux2, eq, i, jj)
            end
        end
        multiply_add_to_node_vars!(F, b_rk_coeff, flux1, eq, i, j)
        multiply_add_to_node_vars!(G, b_rk_coeff, flux2, eq, i, j)
        multiply_add_to_node_vars!(U, b_rk_coeff, u_node, eq, i, j)
    end
end

function flux_der!(volume_integral::MyVolumeIntegralFluxDifferencing,
                   r1, u_tuples_out, F_G_U_S, A_rk_tuple, b_rk_coeff, u_in, op,
                   local_grid, eq::AbstractEquations{2})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    b = b_rk_coeff # b is the coefficient for the Runge-Kutta method
    # @assert false
    F, G, U, S = F_G_U_S
    xc, yc, dx, dy, lamx, lamy, dt = local_grid
    @unpack flux_conservative, flux_non_conservative = volume_integral
    nd = length(xg)
    # Solution points
    for j in 1:nd, i in 1:nd
        x = xc - 0.5 * dx + xg[i] * dx
        y = yc - 0.5 * dy + xg[j] * dy
        u_node = get_node_vars(u_in, eq, i, j)

        flux1, flux2 = flux(x, y, u_node, eq)
        multiply_add_to_node_vars!(F, b_rk_coeff, flux1, eq, i, j)
        multiply_add_to_node_vars!(G, b_rk_coeff, flux2, eq, i, j)

        for ii in Base.OneTo(nd)
            u_node_ii = get_node_vars(u_in, eq, ii, j)
            flux1 = flux_conservative(u_node, u_node_ii, 1, eq)
            # ut              += -lam * D * f for each variable
            # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
            # flux1 = flux(x, y, u_node, eq, 1)
            multiply_add_to_node_vars!(r1, 2.0 * b * lamx * Dm[ii, i], flux1, eq, ii, j)
        end

        for jj in Base.OneTo(nd)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            u_node_jj = get_node_vars(u_in, eq, i, jj)
            flux2 = flux_conservative(u_node, u_node_jj, 2, eq)
            # flux2 = flux(x, y, u_node, eq, 2)
            multiply_add_to_node_vars!(r1, 2.0 * b * lamy * Dm[jj, j], flux2, eq, i, jj)
        end

        multiply_add_to_node_vars!(U, b_rk_coeff, u_node, eq, i, j)

        # TOTHINK - Should the `integral_contribution` approach be tried here?
        for i_u in eachindex(u_tuples_out)
            u = u_tuples_out[i_u]
            a = -A_rk_tuple[i_u]
            for ii in Base.OneTo(nd)
                u_node_ii = get_node_vars(u_in, eq, ii, j)
                flux1 = flux_conservative(u_node, u_node_ii, 1, eq)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(u, a * lamx * 2.0 * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                u_node_jj = get_node_vars(u_in, eq, i, jj)
                flux2 = flux_conservative(u_node, u_node_jj, 2, eq)
                multiply_add_to_node_vars!(u, a * lamy * 2.0 * Dm[jj, j], flux2, eq, i, jj)
            end
        end
    end
end

function noncons_flux_der!(volume_integral, u_tuples_out, res, A_rk_tuple, b_rk_coeff, u_in,
                           op, local_grid, eq::AbstractNonConservativeEquations{2})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    xc, yc, dx, dy, lamx, lamy, t, dt = local_grid
    nd = length(xg)
    # Solution points
    # Compute the contribution of non-conservative equation (u_x, u_y)
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        x_ = xc - 0.5 * dx + xg[j] * dx
        y_ = yc - 0.5 * dy + xg[i] * dy
        u_node = get_node_vars(u_in, eq, i, j)

        integral_contribution = zero(u_node)
        for ii in Base.OneTo(nd) # Computes derivative in reference coordinates
            # TODO - Replace with multiply_non_conservative_node_vars!
            # and then you won't need the `eq_nc` struct.
            u_node_ii = get_node_vars(u_in, eq, ii, j)
            u_non_cons_ii = calc_non_cons_gradient(u_node_ii, x_, y_, t, eq)
            noncons_flux1 = calc_non_cons_Bu(u_node, u_non_cons_ii, x_, y_, t, 1, eq)
            integral_contribution = (integral_contribution +
                                     lamx * Dm[i, ii] * noncons_flux1)
        end

        for jj in Base.OneTo(nd) # Computes derivative in reference coordinates
            u_node_jj = get_node_vars(u_in, eq, i, jj)
            u_non_cons_jj = calc_non_cons_gradient(u_node_jj, x_, y_, t, eq)
            noncons_flux2 = calc_non_cons_Bu(u_node, u_non_cons_jj, x_, y_, t, 2, eq)
            integral_contribution = (integral_contribution +
                                     lamy * Dm[j, jj] * noncons_flux2)
        end

        for i_u in eachindex(u_tuples_out)
            u = u_tuples_out[i_u]
            multiply_add_to_node_vars!(u, -A_rk_tuple[i_u], integral_contribution, eq, i, j)
        end
        multiply_add_to_node_vars!(res, b_rk_coeff, integral_contribution, eq, i, j)
    end
end

function noncons_flux_der!(volume_integral::MyVolumeIntegralFluxDifferencing,
                           u_tuples_out, res, A_rk_tuple, b_rk_coeff, u_in,
                           op, local_grid, eq::AbstractNonConservativeEquations{2})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    xc, yc, dx, dy, lamx, lamy, t, dt = local_grid
    @unpack flux_non_conservative = volume_integral
    nd = length(xg)
    # Solution points
    # Compute the contribution of non-conservative equation (u_x, u_y)
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        x_ = xc - 0.5 * dx + xg[j] * dx
        y_ = yc - 0.5 * dy + xg[i] * dy
        u_node = get_node_vars(u_in, eq, i, j)

        integral_contribution = zero(u_node)
        for ii in Base.OneTo(nd) # Computes derivative in reference coordinates
            # TODO - Replace with multiply_non_conservative_node_vars!
            # and then you won't need the `eq_nc` struct.
            u_node_ii = get_node_vars(u_in, eq, ii, j)
            u_non_cons_ii = calc_non_cons_gradient(u_node_ii, x_, y_, t, eq)
            # noncons_flux1 = calc_non_cons_Bu(u_node, u_non_cons_ii, x_, y_, t, 1, eq)
            noncons_flux1 = flux_non_conservative(u_node, u_node_ii, 1, eq)
            integral_contribution = (integral_contribution +
                                     lamx * Dm[i, ii] * noncons_flux1)
        end

        for jj in Base.OneTo(nd) # Computes derivative in reference coordinates
            u_node_jj = get_node_vars(u_in, eq, i, jj)
            u_non_cons_jj = calc_non_cons_gradient(u_node_jj, x_, y_, t, eq)
            # noncons_flux2 = calc_non_cons_Bu(u_node, u_non_cons_jj, x_, y_, t, 2, eq)
            noncons_flux2 = flux_non_conservative(u_node, u_node_jj, 2, eq)
            integral_contribution = (integral_contribution +
                                     lamy * Dm[j, jj] * noncons_flux2)
        end

        for i_u in eachindex(u_tuples_out)
            u = u_tuples_out[i_u]
            multiply_add_to_node_vars!(u, -A_rk_tuple[i_u], integral_contribution, eq, i, j)
        end
        multiply_add_to_node_vars!(res, b_rk_coeff, integral_contribution, eq, i, j)
    end
end

function source_term_explicit!(u_tuples_out, F_G_U_S, A_rk_tuple, b_rk_coeff, c_rk_coeff,
                               u_in, op, local_grid, source_terms, eq::AbstractEquations{2})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    xc, yc, dx, dy, lamx, lamy, t, dt = local_grid
    nd = length(xg)
    _, _, _, S = F_G_U_S
    # Solution points
    # Compute the contribution of non-conservative equation (u_x, u_y)
    for j in 1:nd, i in 1:nd
        x_ = xc - 0.5 * dx + xg[i] * dx
        y_ = yc - 0.5 * dy + xg[j] * dy
        X = SVector(x_, y_)
        u_node = get_node_vars(u_in, eq, i, j)

        # Source terms
        s_node = calc_source(u_node, X, t + c_rk_coeff * dt, source_terms, eq)
        for i_u in eachindex(u_tuples_out)
            multiply_add_to_node_vars!(u_tuples_out[i_u], A_rk_tuple[i_u] * dt, s_node, eq,
                                       i, j)
        end
        multiply_add_to_node_vars!(S, b_rk_coeff, s_node, eq, i, j)
    end
end

function source_term_implicit!(u_tuples_out, F_G_U_S, A_rk_tuple, b_rk_coeff, c_rk_coeff,
                               u_in, op, local_grid, problem, scheme, implicit_solver,
                               source_terms, aux, eq::AbstractEquations{2})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    xc, yc, dx, dy, lamx, lamy, t, dt = local_grid
    nd = length(xg)
    _, _, _, S = F_G_U_S
    for j in 1:nd, i in 1:nd
        x_ = xc - 0.5 * dx + xg[i] * dx
        y_ = yc - 0.5 * dy + xg[j] * dy
        X = SVector(x_, y_)
        # Source terms
        lhs = get_node_vars(u_tuples_out[1], eq, i, j) # lhs in the implicit source solver
        aux_node = get_cache_node_vars(aux, u_in, problem, scheme, eq, i, j)
        u_node_implicit, s_node = implicit_source_solve(lhs, eq, X, t + c_rk_coeff * dt,
                                                        A_rk_tuple[1] * dt,
                                                        source_terms,
                                                        aux_node, implicit_solver)
        for i_u in eachindex(u_tuples_out)
            multiply_add_to_node_vars!(u_tuples_out[i_u], A_rk_tuple[i_u] * dt, s_node, eq,
                                       i, j)
        end
        multiply_add_to_node_vars!(S, b_rk_coeff, s_node, eq, i, j)
    end
end

function F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                          eq::AbstractEquations{2})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    F, G, U, S = F_G_U_S
    xc, yc, dx, dy, lamx, lamy, t, dt = local_grid
    nd = length(xg)
    # Solution points
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
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

        U_node = scheme.dissipation(u1_, U, eq, i, j)

        # Ub = UT * V
        # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
        multiply_add_to_node_vars!(Ub_, Vl[i], U_node, eq, j, 1)
        multiply_add_to_node_vars!(Ub_, Vr[i], U_node, eq, j, 2)

        # Ub = U * V
        # Ub[i] += ∑_j U[i,j]*V[j]
        multiply_add_to_node_vars!(Ub_, Vl[j], U_node, eq, i, 3)
        multiply_add_to_node_vars!(Ub_, Vr[j], U_node, eq, i, 4)

        S_node = get_node_vars(S, eq, i, j)

        multiply_add_to_node_vars!(r1, -dt, S_node, eq, i, j)
    end
end

function F_G_S_to_res_Ub!(volume_integral::MyVolumeIntegralFluxDifferencing,
                          r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                          eq::AbstractEquations{2})
    @unpack xg, wg, Dm, D1, bV, Vl, Vr = op
    F, G, U, S = F_G_U_S
    xc, yc, dx, dy, lamx, lamy, t, dt = local_grid
    nd = length(xg)
    # Solution points
    for j in Base.OneTo(nd), i in Base.OneTo(nd)
        F_node = get_node_vars(F, eq, i, j)
        G_node = get_node_vars(G, eq, i, j)
        for ii in Base.OneTo(nd)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
            # multiply_add_to_node_vars!(r1, lamx * Dm[ii, i], F_node, eq, ii, j)
            multiply_add_to_node_vars!(r1, lamx * bV[ii, i], F_node, eq, ii, j)
        end
        for jj in Base.OneTo(nd)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            # multiply_add_to_node_vars!(r1, lamy * Dm[jj, j], G_node, eq, i, jj)
            multiply_add_to_node_vars!(r1, lamy * bV[jj, j], G_node, eq, i, jj)
        end

        U_node = scheme.dissipation(u1_, U, eq, i, j)

        # Ub = UT * V
        # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
        multiply_add_to_node_vars!(Ub_, Vl[i], U_node, eq, j, 1)
        multiply_add_to_node_vars!(Ub_, Vr[i], U_node, eq, j, 2)

        # Ub = U * V
        # Ub[i] += ∑_j U[i,j]*V[j]
        multiply_add_to_node_vars!(Ub_, Vl[j], U_node, eq, i, 3)
        multiply_add_to_node_vars!(Ub_, Vr[j], U_node, eq, i, 4)

        S_node = get_node_vars(S, eq, i, j)

        multiply_add_to_node_vars!(r1, -dt, S_node, eq, i, j)
    end
end

function Bb_to_res!(eq::AbstractNonConservativeEquations{2}, local_grid, op, Ub, res)
    @unpack bl, br, xg, wg, degree = op
    nd = degree + 1

    xc, yc, dx, dy, lamx, lamy, t, dt = local_grid

    for ix in Base.OneTo(nd)
        for jy in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[jy] * dx
            xl, xr = (xc - 0.5 * dx, xc + 0.5 * dx)
            y = yc - 0.5 * dy + xg[jy] * dy
            yl, yr = (yc - 0.5 * dy, yc + 0.5 * dy)
            Ul = get_node_vars(Ub, eq, jy, 1)
            Ur = get_node_vars(Ub, eq, jy, 2)
            Ud = get_node_vars(Ub, eq, ix, 3)
            Uu = get_node_vars(Ub, eq, ix, 4)

            Ul_nc = calc_non_cons_gradient(Ul, xl, y, t, eq)
            Ur_nc = calc_non_cons_gradient(Ur, xr, y, t, eq)
            Ud_nc = calc_non_cons_gradient(Ud, x, yl, t, eq)
            Uu_nc = calc_non_cons_gradient(Uu, x, yr, t, eq)

            Bul = calc_non_cons_Bu(Ul, Ul_nc, xl, y, t, 1, eq)
            Bur = calc_non_cons_Bu(Ur, Ur_nc, xr, y, t, 1, eq)
            Bud = calc_non_cons_Bu(Ud, Ud_nc, x, yl, t, 2, eq)
            Buu = calc_non_cons_Bu(Uu, Uu_nc, x, yr, t, 2, eq)

            for n in eachvariable(eq)
                res[n, ix, jy] -= lamy * br[jy] * Buu[n]
                res[n, ix, jy] -= lamy * bl[jy] * Bud[n]
                res[n, ix, jy] -= lamx * br[ix] * Bur[n]
                res[n, ix, jy] -= lamx * bl[ix] * Bul[n]
            end
        end
    end

    return nothing
end

# TODO - This should be merged with the other compute_cell_residual! function and should
# cost nothing extra because of multiple dispatch.
# Maybe that approach could be taken with the higher order cRK methods, and this one
# could be kept as it.
function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cRK22}, aux, t, dt, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack compute_bflux! = scheme.bflux
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    # A struct containing information about the non-conservative part of the equation
    eq_nc = non_conservative_equation(eq)
    nc_var = nvariables(eq_nc)

    tA_rk = ((0.0, 0.0),
             (0.5, MyZero()))
    tb_rk = (0.0, 1.0)
    tc_rk = (0.0, 0.5)

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u2, F, G, U, S = cell_arrays[id]

        u2 .= @view u1[:, :, :, el_x, el_y]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        flux_der!(volume_integral, r1, (u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], u1_,
                  op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u2,), r1, (tA_rk[2][1],), tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_explicit!((u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], tc_rk[1], u1_,
                              op,
                              local_grid,
                              source_terms, eq)
        noncons_flux_der!(volume_integral, (), r1, (tA_rk[2][2],), tb_rk[2], u2, op,
                          local_grid, eq)

        flux_der!(volume_integral, r1, (), F_G_U_S, (tA_rk[2][2],), tb_rk[2], u2, op,
                  local_grid, eq)

        source_term_explicit!((), F_G_U_S, (tA_rk[2][2],), tb_rk[2], tc_rk[2], u2, op,
                              local_grid,
                              source_terms, eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u1_, u2, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cRK33}, aux, t, dt, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack compute_bflux! = scheme.bflux
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    # Written with transpose for ease of readability
    tA_rk = ((0.0, 0.0, 0.0),
             (1.0 / 3.0, 0.0, 0.0),
             (MyZero(), 2.0 / 3.0, MyZero()))
    tb_rk = (0.25, MyZero(), 0.75)
    tc_rk = (0.0, 1.0 / 3.0, 2.0 / 3.0)

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u2, u3, F, G, U, S = cell_arrays[id]

        u2 .= @view u1[:, :, :, el_x, el_y]
        u3 .= @view u1[:, :, :, el_x, el_y]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        # Stage 1
        flux_der!(volume_integral, r1, (u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], u1_,
                  op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u2,), r1, (tA_rk[2][1],), tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_explicit!((u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], tc_rk[1], u1_,
                              op, local_grid, source_terms, eq)

        # Stage 2
        flux_der!(volume_integral, r1, (u3,), F_G_U_S, (tA_rk[3][2],), tb_rk[2], u2, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u3,), r1, (tA_rk[3][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        source_term_explicit!((u3,), F_G_U_S, (tA_rk[3][2],), tb_rk[2], tc_rk[2], u2,
                              op,
                              local_grid,
                              source_terms, eq)

        # Stage 3 (no derivatives)
        flux_der!(volume_integral, r1, (), F_G_U_S, (tA_rk[3][3],), tb_rk[3], u3, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (), r1, (tA_rk[3][3],), tb_rk[3], u3, op,
                          local_grid, eq)
        source_term_explicit!((), F_G_U_S, (tA_rk[3][3],), tb_rk[3], tc_rk[3], u3, op,
                              local_grid,
                              source_terms, eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u1_, u3, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cRK44}, aux, t, dt, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack compute_bflux! = scheme.bflux
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    # Written with transpose for ease of readability
    z0 = MyZero()
    tA_rk = ((z0, z0, z0, z0),
             (0.5, z0, z0, z0),
             (z0, 0.5, z0, z0),
             (z0, z0, 1.0, z0))
    tb_rk = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
    tc_rk = (0.0, 0.5, 0.5, 1.0)

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u2, u3, u4, F, G, U, S = cell_arrays[id]

        u2 .= @view u1[:, :, :, el_x, el_y]
        u3 .= @view u1[:, :, :, el_x, el_y]
        u4 .= @view u1[:, :, :, el_x, el_y]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        # Stage 1
        flux_der!(volume_integral, r1, (u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], u1_,
                  op, local_grid, eq)
        noncons_flux_der!(volume_integral, (u2,), r1, (tA_rk[2][1],), tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_explicit!((u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], tc_rk[1], u1_,
                              op, local_grid, source_terms, eq)

        # Stage 2
        flux_der!(volume_integral, r1, (u3,), F_G_U_S, (tA_rk[3][2],), tb_rk[2], u2, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u3,), r1, (tA_rk[3][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        source_term_explicit!((u3,), F_G_U_S, (tA_rk[3][2],), tb_rk[2], tc_rk[2], u2,
                              op, local_grid, source_terms, eq)

        # Stage 3
        flux_der!(volume_integral, r1, (u4,), F_G_U_S, (tA_rk[4][3],), tb_rk[3], u3, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u4,), r1, (tA_rk[4][3],), tb_rk[3], u3, op,
                          local_grid, eq)
        source_term_explicit!((u4,), F_G_U_S, (tA_rk[4][3],), tb_rk[3], tc_rk[3], u3,
                              op,
                              local_grid, source_terms, eq)

        # Stage 4 (no derivatives)
        flux_der!(volume_integral, r1, (), F_G_U_S, (tA_rk[4][4],), tb_rk[4], u4, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (), r1, (tA_rk[4][4],), tb_rk[4], u4, op,
                          local_grid, eq)
        source_term_explicit!((), F_G_U_S, (tA_rk[4][4],), tb_rk[4], tc_rk[4],
                              u4, op, local_grid, source_terms, eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx, dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u1_, u2, u3, u4, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cHT112}, aux, t, dt, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack compute_bflux! = scheme.bflux
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack implicit_solver = solver
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    tA_rk = ((0.0, 0.0),
             (1.0, MyZero()))
    tb_rk = (0.5, 0.5)
    # tc_rk = (0.0, 1.0)

    A_rk = ((0.0, 0.0),
            (0.5, 0.5))
    b_rk = (0.5, 0.5)
    c_rk = (0.0, 1.0)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u2, F, G, U, S = cell_arrays[id]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        # TODO - FIX THIS HARDCODING!!
        u2 .= @view u1[:, :, :, el_x, el_y]
        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]
        # Stage 1
        flux_der!(volume_integral, r1, (u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], u1_,
                  op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u2,), r1, (tA_rk[2][1],), tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_explicit!((u2,), F_G_U_S, (A_rk[2][1],), b_rk[1], c_rk[1], u1_, op,
                              local_grid,
                              source_terms, eq)
        source_term_implicit!((u2,), F_G_U_S, (A_rk[2][2],), b_rk[2], c_rk[2], u1_, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        noncons_flux_der!(volume_integral, (), r1, (tA_rk[2][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        flux_der!(volume_integral, r1, (), F_G_U_S, (tA_rk[2][2],), tb_rk[2], u2, op,
                  local_grid, eq)
        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

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

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cSSP2IMEX222}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack compute_bflux! = scheme.bflux
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack implicit_solver = solver
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    gamma = 1.0 - 1.0 / sqrt(2.0)

    z0 = MyZero()
    tA_rk = ((z0, z0),
             (1.0, z0))
    tb_rk = (0.5, 0.5)
    # tc_rk = (0.0, 1.0)

    A_rk = ((gamma, z0),
            (1.0 - 2.0 * gamma, gamma))
    b_rk = (0.5, 0.5)
    c_rk = (gamma, 1.0 - gamma)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u1_, u2, F, G, U, S = cell_arrays[id]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        # TODO - FIX THIS HARDCODING!!
        u1_ .= @view u1[:, :, :, el_x, el_y]
        u2 .= @view u1[:, :, :, el_x, el_y]
        u = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        source_term_implicit!((u1_,), F_G_U_S, (A_rk[1][1],), z0, z0, u, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        flux_der!(volume_integral, r1, (u2,), F_G_U_S, (tA_rk[2][1],), tb_rk[1], u1_,
                  op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u2,), r1, (tA_rk[2][1],), tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_explicit!((u2,), F_G_U_S, (A_rk[2][1],), b_rk[1], c_rk[1], u1_, op,
                              local_grid,
                              source_terms, eq)
        source_term_implicit!((u2,), F_G_U_S, (A_rk[2][2],), b_rk[2], c_rk[2], u1_, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        noncons_flux_der!(volume_integral, (), r1, (tA_rk[2][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        flux_der!(volume_integral, r1, (), F_G_U_S, (tA_rk[2][2],), tb_rk[2], u2, op,
                  local_grid, eq)
        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u, u2, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cARS222}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack compute_bflux! = scheme.bflux
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack implicit_solver = solver
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    gamma = 1.0 - 1.0 / sqrt(2.0)
    delta = 1.0 - 1.0 / (2.0 * gamma)

    z0 = MyZero()
    tA_rk = ((z0, z0, z0),
             (gamma, z0, z0),
             (delta, 1.0 - delta, z0))
    tb_rk = (delta, 1.0 - delta, 0.0)
    # tc_rk = (0.0, 1.0)

    A_rk = ((z0, z0, z0),
            (z0, gamma, z0),
            (z0, 1.0 - gamma, gamma))
    b_rk = (0.0, 1.0 - gamma, gamma)
    c_rk = (0.0, gamma, 1.0)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u2, u3, F, G, U, S = cell_arrays[id]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        # TODO - FIX THIS HARDCODING!!
        u1_ = @view u1[:, :, :, el_x, el_y]
        u2 .= @view u1[:, :, :, el_x, el_y]
        u3 .= @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        # u1_ .= u

        flux_der!(volume_integral, r1, (u2, u3), F_G_U_S, (tA_rk[2][1], tA_rk[3][1]),
                  tb_rk[1], u1_, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u2, u3), r1, (tA_rk[2][1], tA_rk[3][1]),
                          tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_implicit!((u2, u3), F_G_U_S, (A_rk[2][2], A_rk[3][2]), b_rk[2],
                              c_rk[2], u1_, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        flux_der!(volume_integral, r1, (u3,), F_G_U_S, (tA_rk[3][2],), tb_rk[2], u2, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u3,), r1, (tA_rk[3][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        source_term_implicit!((u3,), F_G_U_S, (A_rk[3][3],), b_rk[3], c_rk[3], u2, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u, u2, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cBPR343}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack compute_bflux! = scheme.bflux
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack implicit_solver = solver
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    z0 = MyZero()
    tA_rk = ((z0, z0, z0, z0, z0),
             (1.0, z0, z0, z0, z0),
             (4.0 / 9.0, 2.0 / 9.0, z0, z0, z0),
             (0.25, z0, 0.75, z0, z0),
             (0.25, z0, 0.75, z0, z0))
    tb_rk = (0.25, z0, 0.75, z0, z0)
    # tc_rk = (z0, 1.0)

    A_rk = ((z0, z0, z0, z0, z0),
            (0.5, 0.5, z0, z0, z0),
            (5.0 / 18.0, -1.0 / 9.0, 0.5, z0, z0),
            (0.5, z0, z0, 0.5, z0),
            (0.25, z0, 0.75, -0.5, 0.5))
    b_rk = (0.25, z0, 0.75, -0.5, 0.5)
    c_rk = (z0, 1.0, 2.0 / 3.0, 1.0, 1.0)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u2, u3, u4, u5, F, G, U, S = cell_arrays[id]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        # TODO - FIX THIS HARDCODING!!
        u1_ = @view u1[:, :, :, el_x, el_y]
        u2 .= u1_
        u3 .= u1_
        u4 .= u1_
        u5 .= u1_
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        # u1_ .= u

        # Stage 1
        flux_der!(volume_integral, r1, (u2, u3, u4, u5), F_G_U_S,
                  (tA_rk[2][1], tA_rk[3][1], tA_rk[4][1], tA_rk[5][1]),
                  tb_rk[1], u1_, op, local_grid, eq)
        noncons_flux_der!(volume_integral, (u2, u3, u4, u5), r1,
                          (tA_rk[2][1], tA_rk[3][1], tA_rk[4][1], tA_rk[5][1]),
                          tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_explicit!((u2, u3, u4, u5), F_G_U_S,
                              (A_rk[2][1], A_rk[3][1], A_rk[4][1], A_rk[5][1]),
                              b_rk[1], c_rk[1], u1_, op, local_grid,
                              source_terms, eq)
        source_term_implicit!((u2, u3, u4, u5), F_G_U_S,
                              (A_rk[2][2], A_rk[3][2], A_rk[4][2], A_rk[5][2]), b_rk[2],
                              c_rk[2], u1_, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 2
        flux_der!(volume_integral, r1, (u3,), F_G_U_S, (tA_rk[3][2],), tb_rk[2], u2, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u3,), r1, (tA_rk[3][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        source_term_implicit!((u3, u4, u5), F_G_U_S,
                              (A_rk[3][3], A_rk[4][3], A_rk[5][3]), b_rk[3],
                              c_rk[3], u2, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 3
        flux_der!(volume_integral, r1, (u4, u5), F_G_U_S, (tA_rk[4][3], tA_rk[5][3]),
                  tb_rk[3], u3, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u4, u5), r1, (tA_rk[4][3], tA_rk[5][3]),
                          tb_rk[3], u3, op,
                          local_grid, eq)
        source_term_implicit!((u4, u5), F_G_U_S, (A_rk[4][4], A_rk[5][4]), b_rk[4],
                              c_rk[4], u3, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 4
        source_term_implicit!((u5,), F_G_U_S, (A_rk[5][5],), b_rk[5], c_rk[5], u4, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u, u2, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cARS443}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack compute_bflux! = scheme.bflux
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack implicit_solver = solver
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    # ARS(4,4,3) / ARS443 (Ascher–Ruuth–Spiteri)
    z0 = MyZero()

    # explicit (non-stiff) tableau (tA_rk, tb_rk)
    tA_rk = ((z0, z0, z0, z0, z0),                    # stage 1
             (1 / 2, z0, z0, z0, z0),                    # stage 2
             (11 / 18, 1 / 18, z0, z0, z0),                    # stage 3
             (5 / 6, -5 / 6, 1 / 2, z0, z0),                    # stage 4
             (1 / 4, 7 / 4, 3 / 4, -7 / 4, z0))
    tb_rk = (1 / 4, 7 / 4, 3 / 4, -7 / 4, 0.0)

    # implicit (stiff) tableau (A_rk, b_rk, c_rk)
    A_rk = ((z0, z0, z0, z0, z0),      # stage 1
            (z0, 1 / 2, z0, z0, z0),      # stage 2
            (z0, 1 / 6, 1 / 2, z0, z0),      # stage 3
            (z0, -1 / 2, 1 / 2, 1 / 2, z0),      # stage 4
            (z0, 3 / 2, -3 / 2, 1 / 2, 1 / 2))
    b_rk = (0.0, 3 / 2, -3 / 2, 1 / 2, 1 / 2)

    # stage times (common representation; first entry is the zero placeholder)
    c_rk = (z0, 1 / 2, 2 / 3, 1 / 2, 1.0)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u2, u3, u4, u5, F, G, U, S = cell_arrays[id]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        # TODO - FIX THIS HARDCODING!!
        u1_ = @view u1[:, :, :, el_x, el_y]
        u2 .= u1_
        u3 .= u1_
        u4 .= u1_
        u5 .= u1_
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        # u1_ .= u

        # Stage 1
        flux_der!(volume_integral, r1, (u2, u3, u4, u5), F_G_U_S,
                  (tA_rk[2][1], tA_rk[3][1], tA_rk[4][1], tA_rk[5][1]),
                  tb_rk[1], u1_, op, local_grid, eq)
        noncons_flux_der!(volume_integral, (u2, u3, u4, u5), r1,
                          (tA_rk[2][1], tA_rk[3][1], tA_rk[4][1], tA_rk[5][1]),
                          tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_explicit!((u2, u3, u4, u5), F_G_U_S,
                              (A_rk[2][1], A_rk[3][1], A_rk[4][1], A_rk[5][1]),
                              b_rk[1], c_rk[1], u1_, op, local_grid,
                              source_terms, eq)
        source_term_implicit!((u2, u3, u4, u5), F_G_U_S,
                              (A_rk[2][2], A_rk[3][2], A_rk[4][2], A_rk[5][2]), b_rk[2],
                              c_rk[2], u1_, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 2
        flux_der!(volume_integral, r1, (u3,), F_G_U_S, (tA_rk[3][2],), tb_rk[2], u2, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u3,), r1, (tA_rk[3][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        source_term_implicit!((u3, u4, u5), F_G_U_S,
                              (A_rk[3][3], A_rk[4][3], A_rk[5][3]), b_rk[3],
                              c_rk[3], u2, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 3
        flux_der!(volume_integral, r1, (u4, u5), F_G_U_S, (tA_rk[4][3], tA_rk[5][3]),
                  tb_rk[3], u3, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u4, u5), r1, (tA_rk[4][3], tA_rk[5][3]),
                          tb_rk[3], u3, op,
                          local_grid, eq)
        source_term_implicit!((u4, u5), F_G_U_S, (A_rk[4][4], A_rk[5][4]), b_rk[4],
                              c_rk[4], u3, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 4
        source_term_implicit!((u5,), F_G_U_S, (A_rk[5][5],), b_rk[5], c_rk[5], u4, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u, u2, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations, grid, op,
                                    problem, scheme::Scheme{<:cAGSA343}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx, ny = grid.size
    @unpack compute_bflux! = scheme.bflux
    @unpack solver = scheme
    @unpack volume_integral = solver
    @unpack implicit_solver = solver
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend_cell_residual! = aux.blend.subroutines

    @unpack cell_arrays, eval_data, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    a_tilde_21 = (-139833537) / 38613965
    c1 = a11 = 168999711 / 74248304
    gamma = a22 = 202439144 / 118586105
    a_tilde_31 = 85870407 / 49798258
    a_tilde_32 = (-121251843) / 1756367063
    b_tilde_2 = 1 / 6
    b_tilde_3 = 2 / 3
    b_tilde_1 = 1 - b_tilde_2 - b_tilde_3
    a_tilde_41 = b_tilde_1
    a_tilde_42 = b_tilde_2
    a_tilde_43 = b_tilde_3

    a21 = 44004295 / 24775207
    a31 = (-6418119) / 169001713
    a32 = (-748951821) / 1043823139
    a33 = 12015439 / 183058594
    a42 = b2 = 1 / 3
    a43 = b3 = 0
    a41 = b1 = 1 - gamma - b2 - b3
    a44 = gamma

    c2 = a21 + a22
    c3 = a31 + a32 + a33
    c4 = 1.0

    z0 = MyZero()
    tA_rk = ((z0, z0, z0, z0),
             (a_tilde_21, z0, z0, z0),
             (a_tilde_31, a_tilde_32, z0, z0),
             (a_tilde_41, a_tilde_42, a_tilde_43, z0))
    tb_rk = (b_tilde_1, b_tilde_2, b_tilde_3, z0)
    # tc_rk = (z0, 1.0)

    A_rk = ((a11, z0, z0, z0),
            (a21, a22, z0, z0),
            (a31, a32, a33, z0),
            (a41, a42, a43, a44))
    b_rk = (b1, b2, b3, gamma)
    c_rk = (c1, c2, c3, c4)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        id = Threads.threadid()
        u1_, u2, u3, u4, F, G, U, S = cell_arrays[id]
        F_G_U_S = (F, G, U, S)
        refresh!.(F_G_U_S)

        # TODO - FIX THIS HARDCODING!!
        u = @view u1[:, :, :, el_x, el_y]
        u1_ .= u
        u2 .= u1_
        u3 .= u1_
        u4 .= u1_
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        # Stage 1
        source_term_implicit!((u1_, u2, u3, u4), F_G_U_S,
                              (A_rk[1][1], A_rk[2][1], A_rk[3][1], A_rk[4][1]), b_rk[1],
                              c_rk[1],
                              u, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 2
        flux_der!(volume_integral, r1, (u2, u3, u4), F_G_U_S,
                  (tA_rk[2][1], tA_rk[3][1], tA_rk[4][1]),
                  tb_rk[1], u1_, op, local_grid, eq)
        noncons_flux_der!(volume_integral, (u2, u3, u4), r1,
                          (tA_rk[2][1], tA_rk[3][1], tA_rk[4][1]),
                          tb_rk[1], u1_, op,
                          local_grid, eq)
        source_term_implicit!((u2, u3, u4), F_G_U_S,
                              (A_rk[2][2], A_rk[3][2], A_rk[4][2]), b_rk[2],
                              c_rk[2], u1_, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 3
        flux_der!(volume_integral, r1, (u3, u4), F_G_U_S, (tA_rk[3][2], tA_rk[4][2]),
                  tb_rk[2], u2, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u3, u4), r1, (tA_rk[3][2], tA_rk[4][2]),
                          tb_rk[2], u2, op,
                          local_grid,
                          eq)
        source_term_implicit!((u3, u4), F_G_U_S, (A_rk[3][3], A_rk[4][3]), b_rk[3],
                              c_rk[3], u2, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        # Stage 4
        flux_der!(volume_integral, r1, (u4,), F_G_U_S, (tA_rk[4][3],), tb_rk[3], u3, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (u4,), r1, (tA_rk[4][3],), tb_rk[3], u3, op,
                          local_grid, eq)
        source_term_implicit!((u4,), F_G_U_S, (A_rk[4][4]), b_rk[4],
                              c_rk[4], u3, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux, eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        u = @view u1[:, :, :, el_x, el_y]
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u, nothing, res)
        # Interpolate to faces
        @views cell_data = (u, u2, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end
