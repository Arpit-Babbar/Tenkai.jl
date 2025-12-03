import Tenkai: compute_face_residual!, compute_cell_residual_cRK!, get_blended_flux,
               blend_cell_residual_fo!, blend_face_residual_fo!, blend_flux_only,
               blend_flux_face_residual!, trivial_face_residual

using Tenkai: cRKSolver, True, False

function setup_arrays(grid, scheme::Scheme{<:cRKSolver},
                      eq::AbstractNonConservativeEquations{1})
    gArray(nvar, nx) = OffsetArray(zeros(nvar, nx + 2),
                                   OffsetArrays.Origin(1, 0))
    function gArray(nvar, n1, nx)
        OffsetArray(zeros(nvar, n1, nx + 2),
                    OffsetArrays.Origin(1, 1, 0))
    end
    # Allocate memory
    nc_var = nvariables(non_conservative_equation(eq))
    @unpack degree = scheme
    nd = degree + 1
    nx = grid.size
    nvar = nvariables(eq)
    u1 = gArray(nvar, nd, nx)
    ua = gArray(nvar, nx)
    res = gArray(nvar, nd, nx)
    Bb = OffsetArray(zeros(nvar, nc_var, 2, nx + 2), OffsetArrays.Origin(1, 1, 1, 0))
    Fb = gArray(nvar, 2, nx)
    Ub = gArray(nvar, 2, nx)
    u1_b = copy(Ub)
    ub_N = gArray(nvar, 2, nx) # The final stage of cRK before communication

    if degree == 0
        cell_data_size = 7 # TODO - Make this 0
        eval_data_size = 6 # TODO - Make this 0
    elseif degree == 1
        cell_data_size = 8 # TODO - Make this 7
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
    else
        @assert false "Degree not implemented"
    end

    MArr = MArray{Tuple{nvariables(eq), nd}, Float64}
    cell_data = alloc_for_threads(MArr, cell_data_size)

    MArr = MArray{Tuple{nvariables(eq), 1}, Float64}
    eval_data = alloc_for_threads(MArr, eval_data_size)

    cache = (; u1, ua, res, Fb, Ub, Bb, u1_b, ub_N, cell_data, eval_data)
    return cache
end

function update_ghost_values_Bb!(problem, scheme,
                                 eq::AbstractNonConservativeEquations{1},
                                 grid, aux, op, cache, t, dt)
    # TODO - Move this to its right place!!
    @unpack Bb = cache
    nx = size(Bb, 4) - 2
    nvar = size(Bb, 1) # Temporary, should take from eq
    nvar_nc = size(Bb, 2)

    if problem.periodic_x
        # Left ghost cells
        copyto!(Bb, CartesianIndices((1:nvar, 1:nvar_nc, 2:2, 0:0)),
                Bb, CartesianIndices((1:nvar, 1:nvar_nc, 2:2, nx:nx)))

        # Right ghost cells
        copyto!(Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:1, (nx + 1):(nx + 1))),
                Bb, CartesianIndices((1:nvar, 1:nvar_nc, 1:1, 1:1)))

        return nothing
    end
    # Left ghost cells
    for n_nc in 1:nvar_nc, n in 1:nvar
        Bb[n, n_nc, 2, 0] = Bb[n, n_nc, 1, 1]
        Bb[n, n_nc, 1, nx + 1] = Bb[n, n_nc, 2, nx]
    end

    return nothing
end

function compute_non_cons_terms(ul, ur, Ul, Ur, x, t,
                                solver, eq::AbstractNonConservativeEquations{1})
    ul_nc, ur_nc = (calc_non_cons_gradient(u, x, t, eq) for u in (Ul, Ur))
    u_non_cons_interface = 0.5 * (ul_nc + ur_nc)

    Bul = calc_non_cons_Bu(Ul, u_non_cons_interface, x, t, eq)
    Bur = calc_non_cons_Bu(Ur, u_non_cons_interface, x, t, eq)

    return Bul, Bur
end

function flux_der!(volume_integral, r1, u_tuples_out, F_U_S, A_rk_tuple,
                   b_rk_coeff, u_in, op, local_grid, eq::AbstractEquations{1})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    F, U, S = F_U_S
    xc, dx, lamx, t, dt = local_grid
    nd = length(xg)
    # Solution points
    for i in 1:nd
        x = xc - 0.5 * dx + xg[i] * dx
        u_node = get_node_vars(u_in, eq, i)
        flux1 = flux(x, u_node, eq)

        # TOTHINK - Should the `integral_contribution` approach be tried here?
        for i_u in eachindex(u_tuples_out)
            u = u_tuples_out[i_u]
            a = -A_rk_tuple[i_u]
            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii] += -lam * Dm[ii,i] f[i] (sum over i)
                multiply_add_to_node_vars!(u, a * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end
        multiply_add_to_node_vars!(F, b_rk_coeff, flux1, eq, i)
        multiply_add_to_node_vars!(U, b_rk_coeff, u_node, eq, i)
    end
end

function noncons_flux_der!(volume_integral, u_tuples_out, res, A_rk_tuple, b_rk_coeff, u_in,
                           op, local_grid, eq::AbstractNonConservativeEquations{1})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    xc, dx, lamx, t, dt = local_grid
    nd = length(xg)
    # Solution points
    # Compute the contribution of non-conservative equation (u_x, u_y)
    for i in Base.OneTo(nd)
        x_ = xc - 0.5 * dx + xg[i] * dx
        u_node = get_node_vars(u_in, eq, i)

        integral_contribution = zero(u_node)
        for ii in Base.OneTo(nd) # Computes derivative in reference coordinates
            # TODO - Replace with multiply_non_conservative_node_vars!
            # and then you won't need the `eq_nc` struct.
            u_node_ii = get_node_vars(u_in, eq, ii)
            u_non_cons_ii = calc_non_cons_gradient(u_node_ii, x_, t, eq)
            noncons_flux1 = calc_non_cons_Bu(u_node, u_non_cons_ii, x_, t, eq)
            integral_contribution = (integral_contribution +
                                     lamx * Dm[i, ii] * noncons_flux1)
        end

        for i_u in eachindex(u_tuples_out)
            u = u_tuples_out[i_u]
            multiply_add_to_node_vars!(u, -A_rk_tuple[i_u], integral_contribution, eq, i)
        end
        multiply_add_to_node_vars!(res, b_rk_coeff, integral_contribution, eq, i)
    end
end

function source_term_explicit!(u_tuples_out, F_U_S, A_rk_tuple, b_rk_coeff, c_rk_coeff,
                               u_in, op, local_grid, source_terms, eq::AbstractEquations{1})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    xc, dx, lamx, t, dt = local_grid
    nd = length(xg)
    _, _, S = F_U_S
    # Solution points
    # Compute the contribution of non-conservative equation (u_x, u_y)
    for i in 1:nd
        x_ = xc - 0.5 * dx + xg[i] * dx
        X = x_
        u_node = get_node_vars(u_in, eq, i)

        # Source terms
        s_node = calc_source(u_node, X, t + c_rk_coeff * dt, source_terms, eq)
        for i_u in eachindex(u_tuples_out)
            multiply_add_to_node_vars!(u_tuples_out[i_u], A_rk_tuple[i_u] * dt, s_node, eq,
                                       i)
        end
        multiply_add_to_node_vars!(S, b_rk_coeff, s_node, eq, i)
    end
end

function source_term_implicit!(u_tuples_out, F_G_U_S, A_rk_tuple, b_rk_coeff, c_rk_coeff,
                               u_in, op, local_grid, problem, scheme, implicit_solver,
                               source_terms, aux, eq::AbstractEquations{1})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    xc, dx, lamx, t, dt = local_grid
    nd = length(xg)
    _, _, _, S = F_G_U_S
    for i in 1:nd
        x_ = xc - 0.5 * dx + xg[i] * dx
        X = SVector(x_)
        # Source terms
        lhs = get_node_vars(u_tuples_out[1], eq, i) # lhs in the implicit source solver
        # TODO - Improve get_cache_node_vars to make it work here
        # aux_node = get_cache_node_vars(aux, u_in, problem, scheme, eq, i, j)
        u_node = get_node_vars(u_in, eq, i)
        u_node_implicit, s_node = implicit_source_solve(lhs, eq, X, t + c_rk_coeff * dt,
                                                        A_rk_tuple[1] * dt,
                                                        source_terms,
                                                        u_node, implicit_solver)
        for i_u in eachindex(u_tuples_out)
            multiply_add_to_node_vars!(u_tuples_out[i_u], A_rk_tuple[i_u] * dt, s_node, eq,
                                       i)
        end
        multiply_add_to_node_vars!(S, b_rk_coeff, s_node, eq, i)
    end
end

function F_U_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_U_S, op, local_grid, scheme,
                          eq::AbstractEquations{1})
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    F, U, S = F_U_S
    xc, dx, lamx, t, dt = local_grid
    nd = length(xg)
    # Solution points
    for i in Base.OneTo(nd)
        F_node = get_node_vars(F, eq, i)
        for ii in Base.OneTo(nd)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
            multiply_add_to_node_vars!(r1, lamx * D1[ii, i], F_node, eq, ii)
        end

        U_node = scheme.dissipation(u1_, U, eq, i)

        # Ub = UT * V
        # Ub += ∑_i U[i] * V[i] = ∑_i U[i] * V[i]
        multiply_add_to_node_vars!(Ub_, Vl[i], U_node, eq, 1)
        multiply_add_to_node_vars!(Ub_, Vr[i], U_node, eq, 2)

        S_node = get_node_vars(S, eq, i)
        multiply_add_to_node_vars!(r1, -dt, S_node, eq, i)
    end
end

#-------------------------------------------------------------------------------
# Add numerical flux to residual
#-------------------------------------------------------------------------------
function compute_face_residual!(eq::AbstractNonConservativeEquations{1}, grid, op, cache,
                                problem, scheme::Scheme{<:cRKSolver}, param, aux, t, dt,
                                u1, Fb, Ub,
                                ua, res,
                                scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack ub_N = cache
    @unpack xg, wg, bl, br = op
    nd = op.degree + 1
    nx = grid.size
    @unpack dx, xf = grid
    num_flux = scheme.numerical_flux
    @unpack blend = aux

    # Vertical faces, x flux
    for i in 1:(nx + 1)
        # Face between i-1 and i
        Ul = @view Ub[:, 2, i - 1]
        Ur = @view Ub[:, 1, i] # Right face
        x = xf[i]
        @views Fn = num_flux(x, ua[:, i - 1], ua[:, i],
                             Fb[:, 2, i - 1], Fb[:, 1, i],
                             Ub[:, 2, i - 1], Ub[:, 1, i], eq, 1)
        # The left element will use its B(u) and the right element will use its B(u)
        # although the two will use the common 0.5 * (hl + hr). For degreee 0, this
        # does reduce to the first order FVM.
        uN_l, uN_r = get_node_vars(ub_N, eq, 2, i - 1),
                     get_node_vars(ub_N, eq, 1, i)
        Bul, Bur = compute_non_cons_terms(uN_l, uN_r, Ul, Ur, x, t,
                                          scheme.solver, eq)

        # TODO - Bul, Bur are not being limited. Fix this and make it as in the 2-D version!
        Fn_l = Fn + Bul
        Fn_r = Fn + Bur

        (Fn_l, Fn_r), blend_fac = blend.blend_face_residual!(i, x, u1, ua, eq, t, dt,
                                                             grid,
                                                             op, problem,
                                                             scheme, param, Fn_l, Fn_r,
                                                             aux, nothing,
                                                             res, scaling_factor)
        for ix in 1:nd
            for n in 1:nvariables(eq)
                res[n, ix, i - 1] += dt / dx[i - 1] * blend_fac[1] * Fn_l[n] * br[ix]
                res[n, ix, i] += dt / dx[i] * blend_fac[2] * Fn_r[n] * bl[ix]
            end
        end
    end
    return nothing
    end # timer
end # compute_face_residual!

@inline function trivial_face_residual(i, x, u1, ua,
                                       eq::AbstractNonConservativeEquations{1},
                                       t, dt, grid, op, problem, scheme, param, Fn_l, Fn_r,
                                       aux,
                                       lamx, res, scaling_factor = 1)
    return (Fn_l, Fn_r), (1.0, 1.0)
end

# These methods work for other non-conservative equations as well!

import Tenkai: update_ghost_values_cRK!
function update_ghost_values_cRK!(problem, scheme, eq::AbstractNonConservativeEquations,
                                  grid, aux,
                                  op, cache,
                                  t, dt, scaling_factor = 1)
    update_ghost_values_lwfr!(problem, scheme, eq, grid, aux, op, cache, t, dt,
                              scaling_factor)

    update_ghost_values_u1!(eq, problem, grid, op, cache.u1, aux, t)
    update_ghost_values_ub_N!(problem, scheme, eq, grid, aux, op, cache, t, dt)
    update_ghost_values_Bb!(problem, scheme, eq, grid, aux, op, cache, t, dt)
end

function update_ghost_values_cRK!(problem, scheme::Scheme{<:cRK44},
                                  eq::AbstractNonConservativeEquations{1},
                                  grid, aux,
                                  op, cache,
                                  t, dt, scaling_factor = 1)
    update_ghost_values_lwfr!(problem, scheme, eq, grid, aux, op, cache, t, dt,
                              scaling_factor)

    update_ghost_values_u1!(eq, problem, grid, op, cache.u1, aux, t)
    update_ghost_values_ub_N!(problem, scheme, eq, grid, aux, op, cache, t, dt)
    update_ghost_values_Bb!(problem, scheme, eq, grid, aux, op, cache, t, dt)
end

function Bb_to_res!(eq::AbstractNonConservativeEquations{1}, local_grid, op, Ub, res)
    @unpack bl, br, xg, wg, degree = op
    nd = degree + 1

    xc, dx, lamx, t, dt = local_grid

    for ix in Base.OneTo(nd)
        xl, xr = (xc - 0.5 * dx, xc + 0.5 * dx)
        Ul = get_node_vars(Ub, eq, 1)
        Ur = get_node_vars(Ub, eq, 2)

        Ul_nc = calc_non_cons_gradient(Ul, xl, t, eq)
        Ur_nc = calc_non_cons_gradient(Ur, xr, t, eq)

        Bul = calc_non_cons_Bu(Ul, Ul_nc, xl, t, eq)
        Bur = calc_non_cons_Bu(Ur, Ur_nc, xr, t, eq)

        for n in eachvariable(eq)
            res[n, ix] -= lamx * br[ix] * Bur[n]
            res[n, ix] -= lamx * bl[ix] * Bul[n]
        end
    end
    return nothing
end

# 1st order FV cell residual
@inbounds @inline function blend_cell_residual_fo!(cell,
                                                   eq::AbstractNonConservativeEquations{1},
                                                   problem, scheme, aux, lamx,
                                                   t, dt, dx, xf, op, u1, u, ua, f, r,
                                                   scaling_factor = 1.0)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead, it's supposed
    #! format: noindent
    # to be 0.25 microseconds
    @unpack blend = aux
    @unpack source_terms = problem
    @unpack Vl, Vr, xg, wg = op
    fn_low = @view blend.fn_low[:, :, cell]
    num_flux = scheme.numerical_flux
    nd = length(xg)

    # if blend.alpha[cell] < 1e-12
    #     store_low_flux!(u, cell, xf, dx, op, blend, eq, scaling_factor)
    #     return nothing
    # end

    resl = @view blend.resl[:, :, cell]
    nvar = nvariables(eq)
    @unpack xxf, fn = blend
    # Get subcell faces
    xxf[0] = xf
    for ii in Base.OneTo(nd)
        xxf[ii] = xxf[ii - 1] + dx * wg[ii]
    end
    fill!(resl, zero(eltype(resl)))
    # Add first source term contribution

    for j in 2:nd
        xx = xxf[j]
        ul, ur = get_node_vars(u, eq, j - 1), get_node_vars(u, eq, j)
        fl, fr = flux(xx, ul, eq), flux(xx, ur, eq)
        fn = scaling_factor * num_flux(xx, ul, ur, fl, fr, ul, ur, eq, 1)
        us = 0.5 * (ul + ur) # For "non-conservative numerical flux"
        Bu_l = calc_non_cons_Bu(ul, us, xx, t, eq)
        Bu_r = calc_non_cons_Bu(ur, us, xx, t, eq)
        fn_l = fn + Bu_l
        fn_r = fn + Bu_r

        for n in 1:nvar
            resl[n, j - 1] += fn_l[n] / wg[j - 1]
            resl[n, j] -= fn_r[n] / wg[j]
        end
    end
    @views fn_low[:, 1] .= wg[1] * resl[:, 1]
    @views fn_low[:, 2] .= -wg[end] * resl[:, end]

    for j in 1:nd
        u_node = get_node_vars(u, eq, j)
        x = xf + dx * xg[j]
        s_node = calc_source(u_node, x, t, source_terms, eq)
        for n in 1:nvar
            resl[n, j] -= s_node[n] * dx
        end
    end

    axpby!(blend.alpha[cell] * dt / dx, resl, 1.0 - blend.alpha[cell], r)

    resl .*= dt / dx
    end # timer
end

function get_blended_flux(el_x, eq::AbstractEquations{1}, dt, grid,
                          blend, scheme, xf, u1, fn_l, fn_r, Fn_l, Fn_r,
                          lamx, op, alp)
    @unpack wg = op
    @unpack fn_low = blend
    @unpack dx = grid

    nd = length(op.xg)
    Fn_l = (1.0 - alp) * Fn_l + alp * fn_l
    Fn_r = (1.0 - alp) * Fn_r + alp * fn_r

    # Check if the end point of left cell is updated with postiivity
    fn_inner_left_cell = get_node_vars(fn_low, eq, 2, el_x - 1)
    u_node = get_node_vars(u1, eq, nd, el_x - 1)

    c_ll = dt / (dx[el_x - 1] * wg[end]) # c is such that unew = u - c(Fn-fn_inner)

    test_update_ll = u_node - c_ll * (Fn_l - fn_inner_left_cell)
    lower_order_update_ll = u_node - c_ll * (fn_l - fn_inner_left_cell)
    if is_admissible(eq, test_update_ll) == false
        @debug "Using first order flux at" el_x, xf
        Fn_l = zhang_shu_flux_fix(eq, u_node, lower_order_update_ll,
                                  Fn_l, fn_inner_left_cell, fn_l, c_ll)
    end

    # Check if the first point of right cell is updated with postiivity
    fn_inner_right_cell = get_node_vars(fn_low, eq, 1, el_x)
    u_node_ = get_node_vars(u1, eq, 1, el_x)

    c_rr = -(dt / dx[el_x]) / wg[1] # c is such that unew = u - c(Fn-fn_inner)

    test_rr = u_node_ - c_rr * (Fn_r - fn_inner_right_cell)
    lower_order_update_rr = u_node_ - c_rr * (fn_r - fn_inner_right_cell)

    if is_admissible(eq, test_rr) == false
        @debug "Using first order flux at" el_x, xf
        Fn_r = zhang_shu_flux_fix(eq, u_node_, lower_order_update_rr,
                                  Fn_r, fn_inner_right_cell, fn_r, c_rr)
    end
    return Fn_l, Fn_r
end

@inline function blend_flux_face_residual!(i, xf, u1, ua,
                                           eq::AbstractNonConservativeEquations{1},
                                           t, dt, grid, op, problem, scheme, param, Fn_l,
                                           Fn_r,
                                           aux,
                                           lamx, res, scaling_factor = 1.0)
    @unpack blend = aux
    alp = 0.0
    return blend_flux_only(i, op, scheme, blend, grid, xf, u1, eq, dt, alp, Fn_l, Fn_r,
                           lamx,
                           scaling_factor)
end

@inbounds @inline function blend_flux_only(i, op, scheme, blend, grid,
                                           xf, u1, eq::AbstractNonConservativeEquations{1},
                                           dt, alp,
                                           Fn_l, Fn_r, lamx, scaling_factor)
    nd = length(op.xg)
    num_flux = scheme.numerical_flux
    # Test for non-admissibility using a lower order flux candidate
    # which will probably be a FO flux, saved from previous checks.
    @views ul, ur = u1[:, nd, i - 1], u1[:, 1, i]
    fl = flux(xf, ul, eq)
    fr = flux(xf, ur, eq)
    fn = scaling_factor * num_flux(xf, ul, ur, fl, fr, ul, ur, eq, 1)
    us = 0.5 * (ul + ur)
    Bul = calc_non_cons_Bu(ul, us, xf, 0.0, eq)
    Bur = calc_non_cons_Bu(ur, us, xf, 0.0, eq)
    fn_l = fn + Bul
    fn_r = fn + Bur
    # Fn = (1.0 - alp) * Fn + alp * fn
    Fn_l, Fn_r = get_blended_flux(i, eq, dt, grid, blend, scheme, xf, u1, fn_l, fn_r,
                                  Fn_l, Fn_r,
                                  lamx, op, alp)
    return (Fn_l, Fn_r), (1.0, 1.0)
end

@inbounds @inline function blend_face_residual_fo!(i, xf, u1, ua,
                                                   eq::AbstractNonConservativeEquations{1},
                                                   t, dt, grid, op, problem, scheme,
                                                   param,
                                                   Fn_l, Fn_r, aux, lamx, res,
                                                   scaling_factor)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead,
    #! format: noindent
    # it's supposed to be 0.25 microseconds
    @unpack blend = aux
    alpha = blend.alpha # factor of non-smooth part
    num_flux = scheme.numerical_flux
    @unpack dx = grid
    nvar = nvariables(eq)

    @unpack xg, wg = op
    nd = length(xg)
    alp = 0.5 * (alpha[i - 1] + alpha[i])
    # if alp < 1e-12
    #     return blend_flux_only(i, op, scheme, blend, grid, xf, u1, eq, dt, alp, Fn,
    #                            lamx,
    #                            scaling_factor)
    # end

    # Reuse arrays to save memory
    @unpack fl, fr, fn = blend

    # The lower order residual of blending scheme comes from lower order
    # numerical flux at the subcell faces. Here we deal with the residual that
    # occurs from those faces that are common to both the subcell and supercell

    # Low order numerical flux
    ul, ur = @views u1[:, nd, i - 1], u1[:, 1, i]
    fl = flux(xf, ul, eq)
    fr = flux(xf, ur, eq)
    fn = scaling_factor * num_flux(xf, ul, ur, fl, fr, ul, ur, eq, 1)

    us = 0.5 * (ul + ur) # For "non-conservative numerical flux"
    Bu_l = calc_non_cons_Bu(ul, us, xf, t, eq)
    Bu_r = calc_non_cons_Bu(ur, us, xf, t, eq)

    fn_l = fn + Bu_l
    fn_r = fn + Bu_r

    # alp = test_alp(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op, alp)

    # Fn = (1.0 - alp) * Fn + alp * fn
    # # TODO - Do this correctly
    # Fn = get_blended_flux(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op,
    #                       alp)
    Fn_l, Fn_r = get_blended_flux(i, eq, dt, grid,
                                  blend, scheme, xf, u1, fn_l, fn_r, Fn_l, Fn_r,
                                  lamx, op, alp)

    # Blend low and higher order flux
    # for n=1:nvar
    #    Fn[n] = @views (1.0-alp)*Fn[n] + alp*fn[n]
    # end
    # Fn_ = (1.0 - alp) * Fn + alp * fn
    # r = @view res[:, :, i-1]

    # # For the sub-cells which have same interface as super-cells, the same
    # # numflux Fn is used in place of the lower order flux
    # for n=1:nvar
    #    r[n,nd] += alpha[i-1] * dt/dx[i-1] *Fn_[n]/wg[nd] # blend.lamx=dt/dx*alpha
    #    # r[n,nd] += blend.lamx[i-1]*Fn[n]/wg[nd] # blend.lamx=dt/dx*alpha
    # end
    # r = @view res[:, :, i]
    # for n=1:nvar
    #    r[n,1] -= alpha[i] * dt/dx[i] * Fn_[n]/wg[1]     # blend.lamx=dt/dx*lamx
    #    # r[n,1] -= blend.lamx[i]*Fn[n]/wg[1]     # blend.lamx=dt/dx*lamx
    # end

    # # We adjust lamx[i] to limit high order face residual
    # one_m_alpha = (1.0-alpha[i-1], 1.0-alpha[i]) # factor of smooth part
    # return Fn_, one_m_alpha

    # Blend low and higher order flux
    # for n=1:nvar
    #    Fn[n] = (1.0-alp)*Fn[n] + alp*fn[n]
    # end

    # Fn_ = fn

    r = @view res[:, :, i - 1]
    # For the sub-cells which have same interface as super-cells, the same
    # numflux Fn is used in place of the lower order flux
    for n in 1:nvar
        # r[n,nd] += alpha[i-1] * dt/dx[i-1] * Fn_[n]/wg[nd] # alpha[i-1] already in blend.lamx
        r[n, nd] += dt / dx[i - 1] * alpha[i - 1] * Fn_l[n] / wg[nd] # alpha[i-1] already in blend.lamx

        # store for extra limiting (TODO - Multiple dispatch to avoid this unless needed?)
        blend.resl[n, nd, i - 1] += dt / dx[i - 1] * Fn_l[n] / wg[nd]
    end

    r = @view res[:, :, i]
    for n in 1:nvar
        # r[n,1] -= alpha[i] * dt/dx[i] * Fn_[n]/wg[1] # alpha[i-1] already in blend.lamx
        r[n, 1] -= dt / dx[i] * alpha[i] * Fn_r[n] / wg[1] # alpha[i-1] already in blend.lamx

        # store for extra limiting (TODO - Multiple dispatch to avoid this unless needed?)
        blend.resl[n, 1, i] -= dt / dx[i] * Fn_r[n] / wg[1] # store for extra limiting
    end
    # lamx[i] = (1.0-alpha[i])*lamx[i] # factor of smooth part
    # Fn = (1.0 - alpha[i]) * Fn
    # one_m_alpha = (1.0 - alpha[i-1], 1.0 - alpha[i])
    # return Fn_, one_m_alpha
    return (Fn_l, Fn_r), (1.0 - alpha[i - 1], 1.0 - alpha[i])
    end # timer
end

# TODO - This should be merged with the other compute_cell_residual! function and should
# cost nothing extra because of multiple dispatch.
# Maybe that approach could be taken with the higher order cRK methods, and this one
# could be kept as it.
function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cRK22}, aux, t, dt, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, bl, br, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack solver = scheme
    @unpack volume_integral = solver

    # A struct containing information about the non-conservative part of the equation
    eq_nc = non_conservative_equation(eq)

    @unpack cell_data, eval_data, ub_N, ua, u1, res, Fb, Ub, Bb = cache

    F, f, U, u2, u_non_cons_x_, u2_non_cons_x_, S = cell_data[Threads.threadid()]

    u_non_cons_x = @view u_non_cons_x_[1:1, 1:nd]
    u2_non_cons_x = @view u2_non_cons_x_[1:1, 1:nd]

    tA_rk = ((0.0, 0.0),
             (0.5, MyZero()))
    tb_rk = (0.0, 1.0)
    tc_rk = (0.0, 0.5)

    refresh!.((res, Ub, Fb, ub_N, Bb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        local_grid = (xc, dx, lamx, t, dt)
        u2 .= @view u1[:, :, cell]
        refresh!.((u_non_cons_x, u2_non_cons_x))
        u1_ = @view u1[:, :, cell]
        r1 = @view res[:, :, cell]
        Ub_ = @view Ub[:, :, cell]

        F_U_S = (F, U, S)
        refresh!.(F_U_S)

        flux_der!(volume_integral, r1, (u2,), F_U_S, (tA_rk[2][1],), tb_rk[1], u1_,
                  op, local_grid, eq)

        noncons_flux_der!(volume_integral, (u2,), r1, (tA_rk[2][1],), tb_rk[1], u1_, op,
                          local_grid, eq)

        source_term_explicit!((u2,), F_U_S, (tA_rk[2][1],), tb_rk[1], tc_rk[1], u1_,
                              op,
                              local_grid,
                              source_terms, eq)

        noncons_flux_der!(volume_integral, (), r1, (tA_rk[2][2],), tb_rk[2], u2, op,
                          local_grid, eq)

        flux_der!(volume_integral, r1, (), F_U_S, (tA_rk[2][2],), tb_rk[2], u2, op,
                  local_grid, eq)

        source_term_explicit!((), F_U_S, (tA_rk[2][2],), tb_rk[2], tc_rk[2], u2, op,
                              local_grid,
                              source_terms, eq)

        F_U_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_U_S, op, local_grid, scheme,
                         eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u1_, cache.ua, f, r1)

        if bflux_ind == extrapolate
            for i in Base.OneTo(nd)
                Fl_node = get_node_vars(F, eq, i)
                Fr_node = get_node_vars(F, eq, i)
                multiply_add_to_node_vars!(Fb, Vl[i], Fl_node, eq, 1, cell)
                multiply_add_to_node_vars!(Fb, Vr[i], Fr_node, eq, 2, cell)
            end
        else
            u2l, u2r = eval_data[Threads.threadid()]
            refresh!.((u2l, u2r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u2_node = get_node_vars(u2, eq, i)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
            end
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)

            set_node_vars!(Fb, f2l, eq, 1, cell)
            set_node_vars!(Fb, f2r, eq, 2, cell)
        end
    end
end

function compute_cell_residual_cRK!(eq::AbstractNonConservativeEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cRK44}, aux, t, dt, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, bl, br, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation
    @unpack solver = scheme
    @unpack volume_integral = solver

    # A struct containing information about the non-conservative part of the equation
    eq_nc = non_conservative_equation(eq)

    @unpack cell_data, eval_data, ub_N, ua, u1, res, Fb, Ub, Bb = cache

    F, f, U, u2, u3, u4, S = cell_data[Threads.threadid()]

    # Written with transpose for ease of readability
    z0 = MyZero()
    tA_rk = ((z0, z0, z0, z0),
             (0.5, z0, z0, z0),
             (z0, 0.5, z0, z0),
             (z0, z0, 1.0, z0))
    tb_rk = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
    tc_rk = (0.0, 0.5, 0.5, 1.0)

    refresh!.((res, Ub, Fb, ub_N, Bb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        local_grid = (xc, dx, lamx, t, dt)
        u2 .= @view u1[:, :, cell]
        u3 .= @view u1[:, :, cell]
        u4 .= @view u1[:, :, cell]
        u1_ = @view u1[:, :, cell]
        r1 = @view res[:, :, cell]
        Ub_ = @view Ub[:, :, cell]

        F_U_S = (F, U, S)
        refresh!.(F_U_S)

        # Stage 1
        flux_der!(volume_integral, r1, (u2,), F_U_S, (tA_rk[2][1],), tb_rk[1], u1_,
                  op, local_grid, eq)

        noncons_flux_der!(volume_integral, (u2,), r1, (tA_rk[2][1],), tb_rk[1], u1_, op,
                          local_grid, eq)

        source_term_explicit!((u2,), F_U_S, (tA_rk[2][1],), tb_rk[1], tc_rk[1], u1_,
                              op, local_grid, source_terms, eq)

        # Stage 2
        flux_der!(volume_integral, r1, (u3,), F_U_S, (tA_rk[3][2],), tb_rk[2], u2,
                  op, local_grid, eq)
        noncons_flux_der!(volume_integral, (u3,), r1, (tA_rk[3][2],), tb_rk[2], u2, op,
                          local_grid, eq)
        source_term_explicit!((u3,), F_U_S, (tA_rk[3][2],), tb_rk[2], tc_rk[2], u2,
                              op, local_grid, source_terms, eq)

        # Stage 3
        flux_der!(volume_integral, r1, (u4,), F_U_S, (tA_rk[4][3],), tb_rk[3], u3,
                  op, local_grid, eq)
        noncons_flux_der!(volume_integral, (u4,), r1, (tA_rk[4][3],), tb_rk[3], u3, op,
                          local_grid, eq)
        source_term_explicit!((u4,), F_U_S, (tA_rk[4][3],), tb_rk[3], tc_rk[3], u3,
                              op, local_grid, source_terms, eq)

        # Stage 4 (no derivatives)
        flux_der!(volume_integral, r1, (), F_U_S, (tA_rk[4][4],), tb_rk[4], u4, op,
                  local_grid, eq)
        noncons_flux_der!(volume_integral, (), r1, (tA_rk[4][4],), tb_rk[4], u4, op,
                          local_grid, eq)
        source_term_explicit!((), F_U_S, (tA_rk[4][4],), tb_rk[4], tc_rk[4], u4, op,
                              local_grid, source_terms, eq)

        F_U_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_U_S, op, local_grid, scheme, eq)

        Bb_to_res!(eq, local_grid, op, Ub_, r1)

        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u1_, cache.ua, f, r1)

        if bflux_ind == extrapolate
            for i in Base.OneTo(nd)
                Fl_node = get_node_vars(F, eq, i)
                Fr_node = get_node_vars(F, eq, i)
                multiply_add_to_node_vars!(Fb, Vl[i], Fl_node, eq, 1, cell)
                multiply_add_to_node_vars!(Fb, Vr[i], Fr_node, eq, 2, cell)
            end
        else
            ul, ur, u2l, u2r, u3l, u3r, u4l, u4r = eval_data[Threads.threadid()]
            refresh!.((ul, ur, u2l, u2r, u3l, u3r, u4l, u4r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u1, eq, i, cell)
                u2_node = get_node_vars(u2, eq, i)
                u3_node = get_node_vars(u3, eq, i)
                u4_node = get_node_vars(u4, eq, i)
                multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, 1)
                multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, 1)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u4l, Vl[i], u4_node, eq, 1)
                multiply_add_to_node_vars!(u4r, Vr[i], u4_node, eq, 1)
            end
            # IDEA - Try this in TVB limiter as well
            ul_node = get_node_vars(ul, eq, 1)
            ur_node = get_node_vars(ur, eq, 1)
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            u4l_node = get_node_vars(u4l, eq, 1)
            u4r_node = get_node_vars(u4r, eq, 1)
            fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            f4l, f4r = flux(xl, u4l_node, eq), flux(xr, u4r_node, eq)
            multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fl, 1.0 / 3.0, f2l, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 1.0 / 3.0, f3l, 1.0 / 6.0, f4l, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 1.0 / 6.0, fr, 1.0 / 3.0, f2r, eq, 2, cell)
            multiply_add_to_node_vars!(Fb, 1.0 / 3.0, f3r, 1.0 / 6.0, f4r, eq, 2, cell)
        end
    end
end
