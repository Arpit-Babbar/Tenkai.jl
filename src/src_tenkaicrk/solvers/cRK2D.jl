using Tenkai: periodic, dirichlet, neumann, reflect, extrapolate, evaluate,
              update_ghost_values_periodic!,
              update_ghost_values_fn_blend!,
              get_node_vars, set_node_vars!,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, multiply_add_set_node_vars!,
              comp_wise_mutiply_node_vars!, flux, update_ghost_values_lwfr!

using SimpleUnPack
using TimerOutputs
using MuladdMacro
using OffsetArrays
using StaticArrays

using Tenkai: @threaded, alloc_for_threads
using Tenkai.Equations: nvariables, eachvariable
using Tenkai: refresh!

import Tenkai: extrap_bflux!, get_bflux_function, setup_arrays,
               compute_cell_residual_cRK!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

import Tenkai: setup_arrays

function update_ghost_values_ub_N!(problem, scheme, eq::AbstractEquations{2}, grid, aux,
                                   op,
                                   cache, t, dt)
    return nothing # To be implemented
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
end # muladd
