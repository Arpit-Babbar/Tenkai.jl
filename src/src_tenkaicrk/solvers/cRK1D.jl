using Tenkai: periodic, dirichlet, neumann, reflect, extrapolate, evaluate,
              update_ghost_values_periodic!,
              update_ghost_values_fn_blend!,
              get_node_vars, set_node_vars!,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, multiply_add_set_node_vars!,
              comp_wise_mutiply_node_vars!, flux, update_ghost_values_lwfr!,
              calc_source, store_low_flux!, blend_flux_only, get_blended_flux,
              get_first_node_vars, get_second_node_vars

import Tenkai: compute_face_residual!, setup_arrays

using SimpleUnPack
using TimerOutputs
using MuladdMacro
using OffsetArrays
using StaticArrays
using LinearAlgebra: norm, axpby!

using Tenkai: @threaded, alloc_for_threads
using Tenkai.Equations: nvariables, eachvariable
using Tenkai: refresh!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#-------------------------------------------------------------------------------
# Ghost values function for the non-conservative part of the equation
#-------------------------------------------------------------------------------
function update_ghost_values_ub_N!(problem, scheme, eq::AbstractEquations{1},
                                   grid, aux, op, cache, t, dt)
    # TODO - Move this to its right place!!
    @unpack ub_N = cache
    nx = size(ub_N, 3) - 2
    nvar = size(ub_N, 1) # Temporary, should take from eq

    if problem.periodic_x
        # Left ghost cells
        copyto!(ub_N, CartesianIndices((1:nvar, 2:2, 0:0)),
                ub_N, CartesianIndices((1:nvar, 2:2, nx:nx)))

        # Right ghost cells
        copyto!(ub_N, CartesianIndices((1:nvar, 1:1, (nx + 1):(nx + 1))),
                ub_N, CartesianIndices((1:nvar, 1:1, 1:1)))

        return nothing
    end
    # Left ghost cells
    for n in 1:nvar
        ub_N[n, 2, 0] = ub_N[n, 1, 1]
        ub_N[n, 1, nx + 1] = ub_N[n, 2, nx]
    end

    return nothing
end

function update_ghost_values_Bb!(problem, scheme, eq::AbstractEquations{1},
                                 grid, aux, op, cache, t, dt)
    return nothing
end

# 1st order FV cell residual
@inbounds @inline function blend_cell_residual_fo_imex!(cell, eq::AbstractEquations{1},
                                                        problem, scheme, aux, lamx,
                                                        t, dt, dx, xf, op, u1, u, ua, f,
                                                        r,
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

    if blend.alpha[cell] < 1e-12
        store_low_flux!(u, cell, xf, dx, op, blend, eq, scaling_factor)
        return nothing
    end

    resl = blend.resl
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
        for n in 1:nvar
            resl[n, j - 1] += fn[n] / wg[j - 1]
            resl[n, j] -= fn[n] / wg[j]
        end
    end
    @views fn_low[:, 1] .= wg[1] * resl[:, 1]
    @views fn_low[:, 2] .= -wg[end] * resl[:, end]

    # for j in (1, nd)
    #     u_node = get_node_vars(u, eq, j)
    #     x = xf + dx * xg[j]
    #     # Replace this with an implicit solve. You can only do it for 1 < j < nd. For
    #     # the other points, the explicit terms are not fully computed yet!
    #     s_node = calc_source(u_node, x, t, source_terms, eq)
    #     for n in 1:nvar
    #         resl[n, j] -= s_node[n] * dx # The dx is put here just so that it is cancelled later
    #     end
    # end

    for j in 2:(nd - 1)
        u_node = get_node_vars(u, eq, j)
        x = xf + dx * xg[j]
        # Replace this with an implicit solve. You can only do it for 1 < j < nd. For
        # the other points, the explicit terms are not fully computed yet!

        res_node = get_node_vars(resl, eq, j)
        lhs = u_node - dt / dx * res_node
        u_new = implicit_source_solve(lhs, eq, x, t, dt,
                                      source_terms, u_node)
        s_node = calc_source(u_new, x, t, source_terms, eq)
        # s_node = calc_source(u_node, x, t, source_terms, eq)
        for n in 1:nvar
            resl[n, j] -= s_node[n] * dx # The dx is put here just so that it is cancelled later
        end
    end

    axpby!(blend.alpha[cell] * dt / dx, resl, 1.0 - blend.alpha[cell], r)
    end # timer
end

@inbounds @inline function blend_face_residual_fo_imex!(el_x, xf, u1, ua,
                                                        eq::AbstractEquations{1},
                                                        t, dt, grid, op, problem,
                                                        scheme, param,
                                                        Fn, aux, lamx, res,
                                                        scaling_factor)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead,
    #! format: noindent
    # it's supposed to be 0.25 microseconds
    @unpack blend = aux
    @unpack source_terms = problem
    alpha = blend.alpha # factor of non-smooth part
    num_flux = scheme.numerical_flux
    @unpack dx = grid
    nvar = nvariables(eq)

    @unpack xg, wg = op
    nd = length(xg)
    alp = 0.5 * (alpha[el_x - 1] + alpha[el_x])
    if alp < 1e-12
        return blend_flux_only(el_x, op, scheme, blend, grid, xf, u1, eq, dt, alp,
                               Fn, lamx,
                               scaling_factor)
    end

    # Reuse arrays to save memory
    @unpack fl, fr, fn, fn_low = blend

    # The lower order residual of blending scheme comes from lower order
    # numerical flux at the subcell faces. Here we deal with the residual that
    # occurs from those faces that are common to both the subcell and supercell

    # Low order numerical flux
    ul, ur = @views u1[:, nd, el_x - 1], u1[:, 1, el_x]
    fl = flux(xf, ul, eq)
    fr = flux(xf, ur, eq)
    fn = scaling_factor * num_flux(xf, ul, ur, fl, fr, ul, ur, eq, 1)

    # alp = test_alp(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op, alp)

    Fn = (1.0 - alp) * Fn + alp * fn

    Fn = get_blended_flux(el_x, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx,
                          op,
                          alp)

    fn_inner_left_cell = get_node_vars(fn_low, eq, 2, el_x - 1)
    u_node = get_node_vars(u1, eq, nd, el_x - 1)

    c_ll = dt / (dx[el_x - 1] * wg[end]) # c is such that unew = u - c(Fn-fn_inner)

    lhs_left_cell = u_node - c_ll * (Fn - fn_inner_left_cell)

    x_left = xf - dx[el_x - 1] + dx[el_x - 1] * xg[end]
    u_new_left_cell = implicit_source_solve(lhs_left_cell, eq, x_left, t, dt,
                                            source_terms, u_node)
    s_node_left = calc_source(u_new_left_cell, x_left, t, source_terms, eq)

    r = @view res[:, :, el_x - 1]
    # For the sub-cells which have same interface as super-cells, the same
    # numflux Fn is used in place of the lower order flux
    for n in 1:nvar
        # r[n,nd] += alpha[i-1] * dt/dx[i-1] * Fn_[n]/wg[nd] # alpha[i-1] already in blend.lamx
        # Add source terms and implicit solve here
        # You will also need fn_low to perform the implicit solve
        r[n, nd] += dt / dx[el_x - 1] * alpha[el_x - 1] * Fn[n] / wg[nd] # alpha[i-1] already in blend.lamx
        r[n, nd] -= dt * alpha[el_x - 1] * s_node_left[n]
    end

    fn_inner_right_cell = get_node_vars(fn_low, eq, 1, el_x)
    u_node_ = get_node_vars(u1, eq, 1, el_x)

    c_rr = -(dt / dx[el_x]) / wg[1] # c is such that unew = u - c(Fn-fn_inner)

    lhs_right_cell = u_node_ - c_rr * (Fn - fn_inner_right_cell)

    x_right = xf + dx[el_x] * xg[1]
    u_new_right_cell = implicit_source_solve(lhs_right_cell, eq, x_right, t, dt,
                                             source_terms,
                                             u_node_)

    s_node_right = calc_source(u_new_right_cell, x_right, t, source_terms, eq)

    r = @view res[:, :, el_x]
    for n in 1:nvar
        # r[n,1] -= alpha[i] * dt/dx[i] * Fn_[n]/wg[1] # alpha[i-1] already in blend.lamx
        # Add source terms and implicit solve here
        r[n, 1] -= dt / dx[el_x] * alpha[el_x] * Fn[n] / wg[1] # alpha[i-1] already in blend.lamx
        r[n, 1] -= dt * alpha[el_x] * s_node_right[n]
    end
    # lamx[i] = (1.0-alpha[i])*lamx[i] # factor of smooth part
    # Fn = (1.0 - alpha[i]) * Fn
    # one_m_alpha = (1.0 - alpha[i-1], 1.0 - alpha[i])
    # return Fn_, one_m_alpha
    return Fn, (1.0 - alpha[el_x - 1], 1.0 - alpha[el_x])
    end # timer
end

import Tenkai: compute_cell_residual_cRK!, update_solution_cRK!
function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cIMEX111}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack ub_N, ua, u1, res, Fb, Ub, = cache
    nx = grid.size

    refresh!(res) # Reset previously used variables to zero
    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        xc = grid.xc[cell]
        u_node = get_node_vars(u1, eq, 1, cell)

        flux1 = flux(xc, u_node, eq)

        # Add source term contribution to ut and some to S
        for face in 1:2
            set_node_vars!(Ub, u_node, eq, face, cell)
            set_node_vars!(Fb, flux1, eq, face, cell)
            set_node_vars!(ub_N, u_node, eq, face, cell)
        end
    end
    return nothing
end

@inbounds @inline function update_solution_cRK!(u1, eq::AbstractEquations{1}, grid,
                                                problem::Problem{<:Any},
                                                scheme::Scheme{<:cIMEX111}, res, aux,
                                                t, dt)
    @timeit aux.timer "Update solution" begin
    #! format: noindent
    @unpack source_terms = problem
    nx = grid.size

    for cell in Base.OneTo(nx)
        xc = grid.xc[cell]

        # To be used as initial guess for the implicit solve
        # The numerical flux has already been added to it. Is that okay?
        u_node = get_node_vars(u1, eq, 1, cell)

        res_node = get_node_vars(res, eq, 1, cell)

        lhs = u_node - res_node # lhs in the implicit source solver

        # lhs = u_node

        aux_node = get_cache_node_vars(aux, u1, problem, scheme, eq, 1, cell)

        # Implicit solver evolution
        u_node_implicit = implicit_source_solve(lhs, eq, xc, t, dt, source_terms,
                                                aux_node)

        s_node_implicit = calc_source(u_node_implicit, xc, t, source_terms, eq)
        multiply_add_to_node_vars!(u1, dt, s_node_implicit, eq, 1, cell)
        multiply_add_to_node_vars!(u1, -1.0, res_node, eq, 1, cell)

        # set_node_vars!(u1, u_node_implicit - res_node + dt*s_node_implicit, eq, 1, cell)
        # @assert maximum(u_node_implicit - lhs - dt*s_node_implicit) < 1e-12 u_node_implicit, lhs + dt*s_node_implicit, u_node_implicit - lhs - dt*s_node_implicit
        # set_node_vars!(u1, lhs + dt*s_node_implicit, eq, 1, cell)
        set_node_vars!(u1, u_node_implicit, eq, 1, cell)
    end

    return nothing
    end # timer
end

function implicit_source_solve(lhs, eq, x, t, coefficient, source_terms, u_node,
                               implicit_solver = newton_solver)
    # TODO - Make sure that the final source computation is used after the implicit solve
    implicit_F(u_new) = u_new - lhs -
                        coefficient * calc_source(u_new, x, t, source_terms, eq)

    u_new = implicit_solver(implicit_F, u_node)
    return u_new
end

function implicit_source_solve(lhs, eq, x, t, coefficient, source_terms::Nothing,
                               u_node)
    return lhs
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cHT112}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    F, f, U, u2, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points

            flux1 = flux(x_, u_node, eq)

            set_node_vars!(F, 0.5 * flux1, eq, i)
            set_node_vars!(U, 0.5 * u_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u2, -1.0 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u2
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            s_node = calc_source(u_node, x_, t, source_terms, eq)
            set_node_vars!(S, 0.5 * s_node, eq, i)
            # Add source term contribution to u2
            multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i)

            lhs = get_node_vars(u2, eq, i) # lhs in the implicit source solver

            # By default, it is just u_node but the user can use it to set something else here.
            aux_node = get_cache_node_vars(aux, u1, problem, scheme, eq, i, cell)

            u2_node_implicit = implicit_source_solve(lhs, eq, x_,
                                                     t + dt, # TOTHINK - Somehow t instead of t + dt
                                                     # gives better accuracy, although it is
                                                     # not supposed to
                                                     0.5 * dt, source_terms,
                                                     aux_node) # aux_node used as initial guess
            set_node_vars!(u2, u2_node_implicit, eq, i)

            s2_node = calc_source(u2_node_implicit, x_, t + dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, s2_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)

            flux1 = flux(x_, u2_node, eq)

            multiply_add_to_node_vars!(F, 0.5, flux1, eq, i)
            multiply_add_to_node_vars!(U, 0.5, u2_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end

            S_node = get_node_vars(S, eq, i)

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
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
            ul, ur, u2l, u2r = eval_data[Threads.threadid()]
            refresh!.((ul, ur, u2l, u2r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u1, eq, i, cell)
                u2_node = get_node_vars(u2, eq, i)
                multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, 1)
                multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, 1)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
            end
            ul_node = get_node_vars(ul, eq, 1)
            ur_node = get_node_vars(ur, eq, 1)
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)
            set_node_vars!(Fb, 0.5 * (fl + f2l), eq, 1, cell)
            set_node_vars!(Fb, 0.5 * (fr + f2r), eq, 2, cell)
        end
    end
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cSSP2IMEX222}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    gamma = 1.0 - 1.0 / sqrt(2.0)

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    F, f, U, u2, u3, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]
        u3 .= @view u1[:, :, cell]

        # Compute u2 and add its contributions to u3
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            lhs = u_node # lhs in the implicit source solver
            u2_node = implicit_source_solve(lhs, eq, x_, t + gamma * dt, gamma * dt,
                                            source_terms, u_node)
            set_node_vars!(u2, u2_node, eq, i)

            flux1 = flux(x_, u2_node, eq)

            set_node_vars!(F, 0.5 * flux1, eq, i)
            set_node_vars!(U, 0.5 * u2_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u3, -lamx * Dm[ii, i], flux1, eq, ii)
            end

            s2_node = calc_source(u2_node, x_, t + gamma * dt, source_terms, eq)
            multiply_add_to_node_vars!(u3, (1.0 - 2.0 * gamma) * dt, s2_node, eq, i)
            set_node_vars!(S, 0.5 * s2_node, eq, i)
        end

        # Add source term contribution to u3
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u3, eq, i) # lhs in the implicit source solver
            u3_node = implicit_source_solve(lhs, eq, x_, t + (1.0 - gamma) * dt,
                                            gamma * dt, source_terms, u2_node)
            set_node_vars!(u3, u3_node, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_to_node_vars!(F, 0.5, flux1, eq, i)
            multiply_add_to_node_vars!(U, 0.5, u3_node, eq, i)
            s3_node = calc_source(u3_node, x_, t + (1.0 - gamma) * dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, s3_node, eq, i)
        end

        for i in Base.OneTo(nd)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
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
            u2l, u2r, u3l, u3r = eval_data[Threads.threadid()]
            refresh!.((u2l, u2r, u3l, u3r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u2_node = get_node_vars(u2, eq, i)
                u3_node = get_node_vars(u3, eq, i)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
            end
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            set_node_vars!(Fb, 0.5 * (f2l + f3l), eq, 1, cell)
            set_node_vars!(Fb, 0.5 * (f2r + f3r), eq, 2, cell)
        end
    end
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cSSP2IMEX332}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    at21 = at31 = at32 = 0.5
    bt1 = bt2 = bt3 = 1.0 / 3.0

    a11 = a22 = 0.25
    a31 = a32 = a33 = 1.0 / 3.0
    c1, c2, c3 = 0.25, 0.25, 1.0
    b1, b2, b3 = bt1, bt2, bt3

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    F, f, U, u2, u3, u4, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]
        u3 .= @view u1[:, :, cell]
        u4 .= @view u1[:, :, cell]

        # Compute u2 and add its contributions to u3, u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            lhs = u_node # lhs in the implicit source solver
            u2_node = implicit_source_solve(lhs, eq, x_, t + a11 * dt, c1 * dt,
                                            source_terms, u_node)
            set_node_vars!(u2, u2_node, eq, i)

            flux1 = flux(x_, u2_node, eq)

            set_node_vars!(F, bt1 * flux1, eq, i)
            set_node_vars!(U, bt1 * u2_node, eq, i)

            s2_node = calc_source(u2_node, x_, t + c1 * dt, source_terms, eq)
            # multiply_add_to_node_vars!(u3, a21 * dt, s2_node, eq, i) # Not needed because a21 = 0
            multiply_add_to_node_vars!(u4, a31 * dt, s2_node, eq, i)
            set_node_vars!(S, b1 * s2_node, eq, i)

            # Flux derivative contribution to u3, u4
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u3, -at21 * lamx * Dm[ii, i], flux1, eq, ii)
                multiply_add_to_node_vars!(u4, -at31 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u3
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u3, eq, i) # lhs in the implicit source solver
            u3_node = implicit_source_solve(lhs, eq, x_, t + c2 * dt,
                                            a22 * dt, source_terms, u2_node)
            set_node_vars!(u3, u3_node, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_to_node_vars!(F, bt2, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt2, u3_node, eq, i)
            s3_node = calc_source(u3_node, x_, t + c2 * dt, source_terms, eq)
            multiply_add_to_node_vars!(S, b2, s3_node, eq, i)
            multiply_add_to_node_vars!(u4, a32 * dt, s3_node, eq, i)
        end

        # Add flux derivative terms to u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u3_node = get_node_vars(u3, eq, i)
            flux1 = flux(x_, u3_node, eq)
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u4, -at32 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u3_node = get_node_vars(u3, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u4, eq, i) # lhs in the implicit source solver
            u4_node = implicit_source_solve(lhs, eq, x_, t + c3 * dt,
                                            a33 * dt, source_terms, u3_node)
            set_node_vars!(u4, u4_node, eq, i)

            flux1 = flux(x_, u4_node, eq)

            multiply_add_to_node_vars!(F, bt3, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt3, u4_node, eq, i)
            s4_node = calc_source(u4_node, x_, t + c3 * dt, source_terms, eq)
            multiply_add_to_node_vars!(S, b3, s4_node, eq, i)

            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
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
            u2l, u2r, u3l, u3r, u4l, u4r = eval_data[Threads.threadid()]
            refresh!.((u2l, u2r, u3l, u3r, u4l, u4r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u2_node = get_node_vars(u2, eq, i)
                u3_node = get_node_vars(u3, eq, i)
                u4_node = get_node_vars(u4, eq, i)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u4l, Vl[i], u4_node, eq, 1)
                multiply_add_to_node_vars!(u4r, Vr[i], u4_node, eq, 1)
            end
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            u4l_node = get_node_vars(u4l, eq, 1)
            u4r_node = get_node_vars(u4r, eq, 1)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            f4l, f4r = flux(xl, u4l_node, eq), flux(xr, u4r_node, eq)
            set_node_vars!(Fb, (f2l + f3l + f4l) / 3.0, eq, 1, cell)
            set_node_vars!(Fb, (f2r + f3r + f4r) / 3.0, eq, 2, cell)
        end
    end
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cSSP2IMEX433}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    alpha = 0.24169426078821
    beta = 0.06042356519705
    eta = 0.12915286960590

    at32 = 1.0
    at42 = at43 = 0.25

    ct3 = 1.0
    ct4 = 0.5

    bt2, bt3, bt4 = 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0

    a11 = a22 = a33 = a44 = alpha
    a21 = -alpha
    a32 = 1.0 - alpha
    a41 = beta
    a42 = eta
    a43 = 0.5 - beta - eta - alpha

    c1 = alpha
    c2 = 0.0
    c3 = 1.0
    c4 = 0.5
    b2, b3, b4 = bt2, bt3, bt4

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    F, f, U, u2, u3, u4, u5, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]
        u3 .= @view u1[:, :, cell]
        u4 .= @view u1[:, :, cell]
        u5 .= @view u1[:, :, cell]

        # Compute u2, u3 and add their contributions to u3, u4, u5
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            lhs = u_node # lhs in the implicit source solver
            u2_node = implicit_source_solve(lhs, eq, x_, t + a11 * dt, c1 * dt,
                                            source_terms, u_node)
            set_node_vars!(u2, u2_node, eq, i)

            flux1 = flux(x_, u2_node, eq)

            s2_node = calc_source(u2_node, x_, t + c1 * dt, source_terms, eq)
            multiply_add_to_node_vars!(u3, a21 * dt, s2_node, eq, i)
            # multiply_add_to_node_vars!(u4, a31 * dt, s2_node, eq, i) # Not needed because a31 = 0
            multiply_add_to_node_vars!(u5, a41 * dt, s2_node, eq, i)

            lhs = get_node_vars(u3, eq, i) # lhs in the implicit source solver
            u3_node = implicit_source_solve(lhs, eq, x_, t + c2 * dt, a22 * dt,
                                            source_terms,
                                            u2_node)

            set_node_vars!(u3, u3_node, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_set_node_vars!(F, bt2, flux1, eq, i)
            multiply_add_set_node_vars!(U, bt2, u3_node, eq, i)
            s3_node = calc_source(u3_node, x_, t + c2 * dt, source_terms, eq)
            multiply_add_set_node_vars!(S, b2, s3_node, eq, i)
            multiply_add_to_node_vars!(u4, a32 * dt, s3_node, eq, i)
            multiply_add_to_node_vars!(u5, a42 * dt, s3_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u4, -at32 * lamx * Dm[ii, i], flux1, eq, ii)
                multiply_add_to_node_vars!(u5, -at42 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            # TODO - Should initial guess be lhs?
            guess_u4 = get_node_vars(u3, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u4, eq, i) # lhs in the implicit source solver
            u4_node = implicit_source_solve(lhs, eq, x_, t + c3 * dt, a33 * dt,
                                            source_terms,
                                            guess_u4)
            set_node_vars!(u4, u4_node, eq, i)

            flux1 = flux(x_, u4_node, eq)

            multiply_add_to_node_vars!(F, bt3, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt3, u4_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u5, -at43 * lamx * Dm[ii, i], flux1, eq, ii)
            end

            s4_node = calc_source(u4_node, x_, t + c3 * dt, source_terms, eq)
            multiply_add_to_node_vars!(u5, a43 * dt, s4_node, eq, i)
            multiply_add_to_node_vars!(S, b3, s4_node, eq, i)
        end

        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_guess = get_node_vars(u4, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u5, eq, i) # lhs in the implicit source solver
            u5_node = implicit_source_solve(lhs, eq, x_, t + c4 * dt,
                                            a44 * dt, source_terms, u_guess)
            set_node_vars!(u5, u5_node, eq, i)

            flux1 = flux(x_, u5_node, eq)

            multiply_add_to_node_vars!(F, bt4, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt4, u5_node, eq, i)
            s5_node = calc_source(u5_node, x_, t + c4 * dt, source_terms, eq)
            multiply_add_to_node_vars!(S, b4, s5_node, eq, i)

            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
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
            u3l, u3r, u4l, u4r, u5l, u5r = eval_data[Threads.threadid()]
            refresh!.((u3l, u3r, u4l, u4r, u5l, u5r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u3_node = get_node_vars(u3, eq, i)
                u4_node = get_node_vars(u4, eq, i)
                u5_node = get_node_vars(u5, eq, i)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u4l, Vl[i], u4_node, eq, 1)
                multiply_add_to_node_vars!(u4r, Vr[i], u4_node, eq, 1)
                multiply_add_to_node_vars!(u5l, Vl[i], u5_node, eq, 1)
                multiply_add_to_node_vars!(u5r, Vr[i], u5_node, eq, 1)
            end
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            u4l_node = get_node_vars(u4l, eq, 1)
            u4r_node = get_node_vars(u4r, eq, 1)
            u5l_node = get_node_vars(u5l, eq, 1)
            u5r_node = get_node_vars(u5r, eq, 1)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            f4l, f4r = flux(xl, u4l_node, eq), flux(xr, u4r_node, eq)
            f5l, f5r = flux(xl, u5l_node, eq), flux(xr, u5r_node, eq)
            set_node_vars!(Fb, bt2 * f3l + bt3 * f4l + bt4 * f5l, eq, 1, cell)
            set_node_vars!(Fb, bt2 * f3r + bt3 * f4r + bt4 * f5r, eq, 2, cell)
        end
    end
end
end # muladd
