using Tenkai: periodic, dirichlet, neumann, reflect, extrapolate, evaluate,
              update_ghost_values_periodic!,
              update_ghost_values_fn_blend!,
              get_node_vars, set_node_vars!,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, multiply_add_set_node_vars!,
              comp_wise_mutiply_node_vars!, flux, update_ghost_values_lwfr!,
              PositivityBlending, NoPositivityBlending, update_solution_lwfr!

using SimpleUnPack
using TimerOutputs
using MuladdMacro
using OffsetArrays
using StaticArrays

using Tenkai: @threaded, alloc_for_threads
using Tenkai.Equations: nvariables, eachvariable
using Tenkai: refresh!

import Tenkai: extrap_bflux!, get_bflux_function, setup_arrays,
               compute_cell_residual_cRK!, update_solution_cRK!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

import Tenkai: setup_arrays

function fo_blend_imex(eq::AbstractEquations{2, <:Any})
    (;
     cell_residual! = blend_cell_residual_fo_imex!,
     face_residual_x! = blend_face_residual_fo_x!,
     face_residual_y! = blend_face_residual_fo_y!,
     name = "fo_imex")
end

function update_ghost_values_ub_N!(problem, scheme, eq::AbstractEquations{2}, grid, aux,
                                   op,
                                   cache, t, dt)
    return nothing # To be implemented
end

function compute_cell_residual_cRK!(eq::AbstractEquations{2}, grid, op,
                                    problem, scheme::Scheme{<:cIMEX111}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell Residual" begin
    #! format: noindent
    nx, ny = grid.size
    refresh!(u) = fill!(u, zero(eltype(u)))

    @unpack u1, res, Fb, Ub, = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        x, y = grid.xc[el_x], grid.yc[el_y]

        id = Threads.threadid()

        u1_ = @view u1[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]
        Fb_ = @view Fb[:, :, :, el_x, el_y]

        u_node = get_node_vars(u1_, eq, 1, 1)
        flux1, flux2 = flux(x, y, u_node, eq)

        for face in 1:2
            set_node_vars!(Ub_, u_node, eq, 1, face)
            set_node_vars!(Fb_, flux1, eq, 1, face)
        end

        for face in 3:4
            set_node_vars!(Ub_, u_node, eq, 1, face)
            set_node_vars!(Fb_, flux2, eq, 1, face)
        end
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractEquations{2}, grid, op,
                                    problem, scheme::Scheme{<:cHT112}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell Residual" begin
    #! format: noindent
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

            set_node_vars!(F, 0.5 * flux1, eq, i, j)
            set_node_vars!(G, 0.5 * flux2, eq, i, j)
            set_node_vars!(U, 0.5 * u_node, eq, i, j)

            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(u2, -lamx * Dm[ii, i], flux1, eq,
                                           ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(u2, -lamy * Dm[jj, j], flux2, eq,
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
            set_node_vars!(S, 0.5 * s_node, eq, i, j)
            multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i, j)

            lhs = get_node_vars(u2, eq, i, j) # lhs in the implicit source solver

            # By default, it is just u_node but the user can use it to set something else here.
            aux_node = get_cache_node_vars(aux, u1_, problem, scheme, eq, i, j)

            u2_node_implicit = implicit_source_solve(lhs, eq, X,
                                                     t + dt, # TOTHINK - Somehow t instead of t + dt
                                                     # gives better accuracy, although it is
                                                     # not supposed to
                                                     0.5 * dt, source_terms,
                                                     aux_node) # aux_node used as initial guess
            set_node_vars!(u2, u2_node_implicit, eq, i, j)

            s2_node = calc_source(u2_node_implicit, X, t + dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.5, s2_node, eq, i, j)
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy

            u2_node = get_node_vars(u2, eq, i, j)

            flux1, flux2 = flux(x, y, u2_node, eq)

            multiply_add_to_node_vars!(F, 0.5, flux1, eq, i, j)
            multiply_add_to_node_vars!(G, 0.5, flux2, eq, i, j)
            multiply_add_to_node_vars!(U, 0.5, u2_node, eq, i, j)

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

            S_node = get_node_vars(S, eq, i, j)
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
                              (A_rk[1][1], A_rk[2][1], A_rk[3][1], A_rk[4][1]),
                              b_rk[1],
                              c_rk[1],
                              u, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux,
                              eq)

        # Stage 2
        flux_der!(volume_integral, r1, (u2, u3, u4), F_G_U_S,
                  (tA_rk[2][1], tA_rk[3][1], tA_rk[4][1]),
                  tb_rk[1], u1_, op, local_grid, eq)
        source_term_implicit!((u2, u3, u4), F_G_U_S,
                              (A_rk[2][2], A_rk[3][2], A_rk[4][2]), b_rk[2],
                              c_rk[2], u1_, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux,
                              eq)

        # Stage 3
        flux_der!(volume_integral, r1, (u3, u4), F_G_U_S,
                  (tA_rk[3][2], tA_rk[4][2]),
                  tb_rk[2], u2, op,
                  local_grid, eq)
        source_term_implicit!((u3, u4), F_G_U_S, (A_rk[3][3], A_rk[4][3]), b_rk[3],
                              c_rk[3], u2, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux,
                              eq)

        # Stage 4
        flux_der!(volume_integral, r1, (u4,), F_G_U_S, (tA_rk[4][3],), tb_rk[3], u3,
                  op,
                  local_grid, eq)
        source_term_implicit!((u4,), F_G_U_S, (A_rk[4][4]), b_rk[4],
                              c_rk[4], u3, op,
                              local_grid,
                              problem, scheme, implicit_solver, source_terms, aux,
                              eq)

        F_G_S_to_res_Ub!(volume_integral, r1, Ub_, u1_, F_G_U_S, op, local_grid,
                         scheme,
                         eq)

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

# TODO - Can we avoid this code repetetition? Example, by using the other version with the
# source terms passed as Nothing.
function blend_cell_residual_fo_imex!(el_x, el_y, eq::AbstractEquations{2}, problem,
                                      scheme,
                                      aux, t, dt, grid, dx, dy, xf, yf, op, u1, u_, f,
                                      res,
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

    u = @view u1[:, :, :, el_x, el_y]
    r = @view res[:, :, :, el_x, el_y]
    blend_resl = @view blend.cache.resl[:, :, :, el_x, el_y]
    blend_resl .= zero(eltype(blend_resl))

    if alpha < 1e-12
        store_low_flux!(u, el_x, el_y, xf, yf, dx, dy, op, blend, eq,
                        scaling_factor)
        return nothing
    end

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
            multiply_add_to_node_vars!(r, # r[ii-1,jj]+=alpha*dt/(dx*wg[ii-1])*fn
                                       alpha * dt / (dx * wg[ii - 1]),
                                       fn, eq, ii - 1, jj)
            multiply_add_to_node_vars!(r, # r[ii,jj]+=alpha*dt/(dx*wg[ii])*fn
                                       -alpha * dt / (dx * wg[ii]),
                                       fn, eq, ii, jj)

            # TODO - Maybe collect only in blend_resl and then copy to r
            # or use multiple dispatch to do this only when needed
            multiply_add_to_node_vars!(blend_resl, # r[ii-1,jj]+=dt/(dx*wg[ii-1])*fn
                                       dt / (dx * wg[ii - 1]),
                                       fn, eq, ii - 1, jj)
            multiply_add_to_node_vars!(blend_resl, # r[ii,jj]+=dt/(dx*wg[ii])*fn
                                       -dt / (dx * wg[ii]),
                                       fn, eq, ii, jj)
            # TOTHINK - Can checking this in every step of the loop be avoided
            if ii == 2
                set_node_vars!(fn_low, fn, eq, jj, 1, el_x, el_y)
            elseif ii == nd
                set_node_vars!(fn_low, fn, eq, jj, 2, el_x, el_y)
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
            multiply_add_to_node_vars!(r, # r[ii,jj-1]+=alpha*dt/(dy*wg[jj-1])*fn
                                       alpha * dt / (dy * wg[jj - 1]),
                                       fn,
                                       eq, ii, jj - 1)
            multiply_add_to_node_vars!(r, # r[ii,jj]+=alpha*dt/(dy*wg[jj])*fn
                                       -alpha * dt / (dy * wg[jj]),
                                       fn,
                                       eq, ii, jj)

            # TODO - Maybe collect only in blend_resl and then copy to r
            # or use multiple dispatch to do this only when needed
            multiply_add_to_node_vars!(blend_resl, # r[ii,jj-1]+=dt/(dy*wg[jj-1])*fn
                                       dt / (dy * wg[jj - 1]),
                                       fn,
                                       eq, ii, jj - 1)

            multiply_add_to_node_vars!(blend_resl, # r[ii,jj]+=dt/(dy*wg[jj])*fn
                                       -dt / (dy * wg[jj]),
                                       fn,
                                       eq, ii, jj)

            # TOTHINK - Can checking this in every step of the loop be avoided
            if jj == 2
                set_node_vars!(fn_low, fn, eq, ii, 3, el_x, el_y)
            elseif jj == nd
                set_node_vars!(fn_low, fn, eq, ii, 4, el_x, el_y)
            end
        end
    end
    end # timer
end

@inbounds @inline function blend_cell_residual_stiff!(blend_cell_residual!::typeof(blend_cell_residual_fo_imex!),
                                                      eq,
                                                      u1, res, grid, problem, op, t, dt,
                                                      aux)
    @timeit aux.timer "Update solution implicit source" begin
    #! format: noindent

    nx, ny = grid.size
    @unpack xg = op
    @unpack source_terms = problem
    @unpack blend = aux
    blend_res = blend.cache.resl
    nd = length(xg)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        alpha = blend.cache.alpha[el_x, el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        dx, dy = grid.dx[el_x], grid.dy[el_y]

        for j in 1:nd, i in 1:nd
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)

            u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
            res_node = get_node_vars(blend_res, eq, i, j, el_x, el_y)
            lhs = u_node - res_node # lhs in the implicit source solver

            u_node_implicit = implicit_source_solve(lhs, eq, X, t, dt,
                                                    source_terms,
                                                    u_node) # u_node used as initial guess

            s_node_implicit = calc_source(u_node_implicit, X, t + dt, source_terms,
                                          eq)

            multiply_add_to_node_vars!(res, -dt * alpha, s_node_implicit, eq, i, j,
                                       el_x, el_y)
        end
    end
    end # timer
end

@inbounds @inline function blend_cell_residual_stiff!(blend_cell_residual!, eq, u1, res,
                                                      grid,
                                                      problem, op, t, dt, aux)
    return nothing
end

@inbounds @inline function update_with_residuals!(positivity_blending::PositivityBlending,
                                                  eq::AbstractEquations{2}, u1,
                                                  res, aux)
end

@inbounds @inline function update_with_residuals!(positivity_blending::NoPositivityBlending,
                                                  eq::AbstractEquations{2}, u1,
                                                  res, aux)
    update_solution_lwfr!(u1, res, aux)
end

@inbounds @inline function update_solution_cRK!(u1, eq::AbstractEquations{2}, grid,
                                                op, problem, scheme, res, aux, t, dt)
    @timeit aux.timer "Update solution" begin
    #! format: noindent

    nx, ny = grid.size
    @unpack blend = aux

    # To check if the blending scheme is IMEX
    @unpack blend_cell_residual! = blend.subroutines

    # Do the source term evolution
    blend_cell_residual_stiff!(blend_cell_residual!, eq, u1, res, grid, problem, op,
                               t, dt,
                               aux)

    @unpack positivity_blending = blend.parameters

    update_with_residuals!(positivity_blending, eq, u1, res, aux)
    end
end

@inbounds @inline function update_solution_cRK!(u1, eq::AbstractEquations{2}, grid,
                                                op, problem::Problem{<:Any},
                                                scheme::Scheme{<:cIMEX111}, res, aux,
                                                t, dt)
    @timeit aux.timer "Update solution" begin
    #! format: noindent
    nx, ny = grid.size

    @unpack source_terms = problem

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        x, y = grid.xc[el_x], grid.yc[el_y]
        X = SVector(x, y)

        id = Threads.threadid()

        u1_ = @view u1[:, :, :, el_x, el_y]

        u_node = get_node_vars(u1_, eq, 1, 1)
        res_node = get_node_vars(res, eq, 1, 1, el_x, el_y)

        lhs = u_node - res_node # lhs in the implicit source solver

        # Implicit solver evolution
        u_node_implicit = implicit_source_solve(lhs, eq, X, t, dt, source_terms,
                                                u_node) # u_node used as initial guess

        s_node_implicit = calc_source(u_node_implicit, X, t + dt, source_terms, eq)
        multiply_add_to_node_vars!(u1, dt, s_node_implicit, eq, 1, 1, el_x, el_y)
        multiply_add_to_node_vars!(u1, -1.0, res_node, eq, 1, 1, el_x, el_y)

        # set_node_vars!(u1, u_node_implicit - res_node + dt*s_node_implicit, eq, 1, cell)
        # @assert maximum(u_node_implicit - lhs - dt*s_node_implicit) < 1e-12 u_node_implicit, lhs + dt*s_node_implicit, u_node_implicit - lhs - dt*s_node_implicit
        # set_node_vars!(u1, lhs + dt*s_node_implicit, eq, 1, cell)
        set_node_vars!(u1, u_node_implicit, eq, 1, 1, el_x, el_y)
    end

    return nothing
    end # timer
end
end # muladd
