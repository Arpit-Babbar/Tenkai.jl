import ..Tenkai: setup_arrays_rkfr,
                 compute_cell_residual_rkfr!,
                 update_ghost_values_rkfr!,
                 flux

using ..Tenkai: periodic, dirichlet, neumann, reflect,
                evaluate, extrapolate,
                update_ghost_values_periodic!,
                update_ghost_values_fn_blend!,
                get_node_vars, set_node_vars!,
                add_to_node_vars!, subtract_from_node_vars!,
                multiply_add_to_node_vars!, multiply_add_set_node_vars!,
                comp_wise_mutiply_node_vars!

using SimpleUnPack
using TimerOutputs
using Polyester
using MuladdMacro
using OffsetArrays
using StaticArrays

using Tenkai: @threaded
using ..Equations: AbstractEquations, nvariables, eachvariable

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#------------------------------------------------------------------------------
function setup_arrays_rkfr(grid, scheme, eq::AbstractEquations{2})
    function gArray(nvar, nx, ny)
        OffsetArray(zeros(nvar, nx + 2, ny + 2),
                    OffsetArrays.Origin(1, 0, 0))
    end
    function gArray(nvar, n1, n2, nx, ny)
        OffsetArray(zeros(nvar, n1, n2, nx + 2, ny + 2),
                    OffsetArrays.Origin(1, 1, 1, 0, 0))
    end
    # Allocate memory
    @unpack degree = scheme
    nvar = nvariables(eq)
    nd = degree + 1
    nx, ny = grid.size
    u0 = gArray(nvar, nd, nd, nx, ny)
    u1 = gArray(nvar, nd, nd, nx, ny)
    ua = gArray(nvar, nx, ny)
    res = gArray(nvar, nd, nd, nx, ny)
    Fb = gArray(nvar, nd, 4, nx, ny)
    ub = gArray(nvar, nd, 4, nx, ny) # u restricted to boundary

    cache = (; u0, u1, ua, res, Fb, ub)
    return cache
end

#------------------------------------------------------------------------------
function update_ghost_values_rkfr!(problem, scheme, eq::AbstractEquations{2, 1},
                                   grid, aux, op, cache, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, ub = cache
    update_ghost_values_periodic!(eq, problem, Fb, ub)

    @unpack periodic_x, periodic_y = problem
    if periodic_x && periodic_y
        return nothing
    end

    nx, ny = grid.size
    @unpack degree, xg = op
    nd = degree + 1
    nvar = nvariables(eq)
    @unpack dx, dy, xf, yf = grid
    @unpack boundary_value, boundary_condition = problem
    left, right, bottom, top = boundary_condition

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    if left == dirichlet
        @threaded for j in 1:ny
            x1 = xf[1]
            for k in Base.OneTo(nd)
                y1 = yf[j] + xg[k] * dy[j]
                ub_value = boundary_value(x1, y1, t)
                set_node_vars!(ub, ub_value, eq, k, 2, 0, j)
                fb_value = flux(x1, y1, ub_value, eq, 1)
                set_node_vars!(Fb, fb_value, eq, k, 2, 0, j)

                # Purely upwind at boundary
                # set_node_vars!(ub, ub_value, eq, k, 1, 1, j)
                # set_node_vars!(Fb, fb_value, eq, k, 1, 1, j)
            end
        end
    elseif left in [neumann, reflect]
        @threaded for j in 1:ny
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 2, 0, j] = ub[n, k, 1, 1, j]
                    Fb[n, k, 2, 0, j] = Fb[n, k, 1, 1, j]
                end
                if left == reflect
                    ub[2, k, 2, 0, j] *= -1.0
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
        @threaded for j in 1:ny
            x2 = xf[nx + 1]
            for k in 1:nd
                y2 = yf[j] + xg[k] * dy[j]
                ub_value = boundary_value(x2, y2, t)
                fb_value = flux(x2, y2, ub_value, eq, 1)
                for n in 1:nvar
                    ub[n, k, 2, nx, j] = ub[n, k, 1, nx + 1, j] = ub_value[n] # upwind
                    Fb[n, k, 2, nx, j] = Fb[n, k, 1, nx + 1, j] = fb_value[n] # upwind
                end
            end
        end
    elseif right in [neumann, reflect]
        @threaded for j in 1:ny
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 1, nx + 1, j] = ub[n, k, 2, nx, j]
                    Fb[n, k, 1, nx + 1, j] = Fb[n, k, 2, nx, j]
                end
                if right == reflect
                    ub[2, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[1, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[3, k, 1, nx + 1, j] *= -1.0 # ρ*u1*u2
                    Fb[4, k, 1, nx + 1, j] *= -1.0 # (ρ_e + p) * u1
                end
            end
        end
    else
        println("Incorrect bc specified at right.")
        @assert false
    end

    if bottom == dirichlet # in [dirichlet, reflect]
        @threaded for i in 1:nx
            y3 = yf[1]
            for k in Base.OneTo(nd)
                x3 = xf[i] + xg[k] * dx[i]
                ub_value = boundary_value(x3, y3, t)
                fb_value = flux(x3, y3, ub_value, eq, 2)
                for n in 1:nvar
                    ub[n, k, 3, i, 1] = ub[n, k, 4, i, 0] = ub_value[n] # upwind
                    Fb[n, k, 3, i, 1] = Fb[n, k, 4, i, 0] = fb_value[n] # upwind
                end
            end
        end
    elseif bottom in [neumann, reflect]
        @threaded for i in 1:nx
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 4, i, 0] = ub[n, k, 3, i, 1]
                    Fb[n, k, 4, i, 0] = Fb[n, k, 3, i, 1]
                end
                if bottom == reflect
                    ub[3, k, 4, i, 0] *= -1.0
                    Fb[1, k, 4, i, 0] *= -1.0
                    Fb[2, k, 4, i, 0] *= -1.0
                    Fb[4, k, 4, i, 0] *= -1.0
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert typeof(bottom) <: Tuple{Any, Any}
        bc! = bottom[1]
        bc!(grid, eq, op, Fb, ub)
    end

    if top == dirichlet
        @threaded for i in 1:nx
            y4 = yf[ny + 1]
            for k in 1:nd
                x4 = xf[i] + xg[k] * dx[i]
                ub_value = boundary_value(x4, y4, t)
                fb_value = flux(x4, y4, ub_value, eq, 2)
                for n in 1:nvar
                    ub[n, k, 4, i, ny] = ub[n, k, 3, i, ny + 1] = ub_value[n] # upwind
                    Fb[n, k, 4, i, ny] = Fb[n, k, 3, i, ny + 1] = fb_value[n] # upwind
                    # ub[n, k, 3, i, ny+1] = ub_value[n] # upwind
                    # Fb[n, k, 3, i, ny+1] = fb_value[n] # upwind
                end
            end
        end
    elseif top in [neumann, reflect]
        @threaded for i in 1:nx
            for k in 1:nd
                for n in 1:nvar
                    ub[n, k, 3, i, ny + 1] = ub[n, k, 4, i, ny]
                    Fb[n, k, 3, i, ny + 1] = Fb[n, k, 4, i, ny]
                end
                if top == reflect
                    ub[3, k, 3, i, ny + 1] *= -1.0
                    Fb[1, k, 3, i, ny + 1] *= -1.0
                    Fb[2, k, 3, i, ny + 1] *= -1.0
                    Fb[4, k, 3, i, ny + 1] *= -1.0
                end
            end
        end
    else
        @assert periodic_y "Incorrect bc specified at top"
    end

    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    return nothing
    end # timer
end

#------------------------------------------------------------------------------
# res = ∂_x F_h + ∂_y G_h where F_h, G_h are continuous fluxes. We write it as
# res = Dm*F_δ + gL'*(Fn_L-F_L) + gR'*(Fn_R-F_R) + G_δ*DmT + gL'*(Fn_D-F_D) + gR'*(Fn_U-F_U),
# which we rewrite as
# res = D1*F_δ + gL'*Fn_L + gR'*Fn_R + G_δ*D1T + gL'*Fn_D + gR'*Fn_U.
# The D1 part is the cell residual, which we compute here.

function compute_cell_residual_rkfr!(eq::AbstractEquations{2}, grid, op, problem,
                                     scheme,
                                     aux, t, dt, u1, res, Fb, ub, cache)
    @unpack timer = aux
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack xg, D1, Vl, Vr = op
    nx, ny = grid.size
    nd = length(xg)
    nvar = nvariables(eq)
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    @unpack blend_cell_residual! = blend.subroutines
    @unpack source_terms = problem
    refresh!(u) = fill!(u, zero(eltype(u)))

    refresh!.((ub, Fb, res))
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # element loop
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        u = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        ub_ = @view ub[:, :, :, el_x, el_y]
        Fb_ = @view Fb[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd) # solution points loop
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)
            u_node = get_node_vars(u, eq, i, j)

            f_node, g_node = flux(x, y, u_node, eq)
            # @show el_x, i, el_y, j, x, y
            for ii in Base.OneTo(nd)
                # res = D * f for each variable
                # res[ii,j] = ∑_i D[ii,i] * f[i,j] for each variable
                multiply_add_to_node_vars!(r1, lamx * D1[ii, i], f_node, eq, ii, j)
            end

            for jj in Base.OneTo(nd)
                # res = g * D' for each variable
                # res[i,jj] = ∑_j g[i,j] * D1[jj,j] for each variable
                multiply_add_to_node_vars!(r1, lamy * D1[jj, j], g_node, eq, i, jj)
            end

            s_node = calc_source(u_node, X, t, source_terms, eq)
            multiply_add_to_node_vars!(r1, -dt, s_node, eq, i, j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(ub_, Vl[i], u_node, eq, j, 1)
            multiply_add_to_node_vars!(ub_, Vr[i], u_node, eq, j, 2)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(ub_, Vl[j], u_node, eq, i, 3)
            multiply_add_to_node_vars!(ub_, Vr[j], u_node, eq, i, 4)
        end
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u,
                             nothing, res)

        if bflux_ind == extrapolate
            # Very inefficient, not meant to be used.
            for j in Base.OneTo(nd), i in Base.OneTo(nd) # solution points loop
                x = xc - 0.5 * dx + xg[i] * dx
                y = yc - 0.5 * dy + xg[j] * dy
                u_node = get_node_vars(u, eq, i, j)
                f_node, g_node = flux(x, y, u_node, eq)

                # Ub = UT * V
                # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
                multiply_add_to_node_vars!(Fb_, Vl[i], f_node, eq, j, 1)
                multiply_add_to_node_vars!(Fb_, Vr[i], f_node, eq, j, 2)

                # Ub = U * V
                # Ub[i] += ∑_j U[i,j]*V[j]
                multiply_add_to_node_vars!(Fb_, Vl[j], g_node, eq, i, 3)
                multiply_add_to_node_vars!(Fb_, Vr[j], g_node, eq, i, 4)
            end
        else
            xl, xr = grid.xf[el_x], grid.xf[el_x + 1]
            yd, yu = grid.yf[el_y], grid.yf[el_y + 1]
            dx, dy = grid.dx[el_x], grid.dy[el_y]
            for ii in 1:nd
                ubl, ubr = get_node_vars(ub_, eq, ii, 1),
                           get_node_vars(ub_, eq, ii, 2)
                ubd, ubu = get_node_vars(ub_, eq, ii, 3),
                           get_node_vars(ub_, eq, ii, 4)
                x = xc - 0.5 * dx + xg[ii] * dx
                y = yc - 0.5 * dy + xg[ii] * dy
                fbl, fbr = flux(xl, y, ubl, eq, 1), flux(xr, y, ubr, eq, 1)
                fbd, fbu = flux(x, yd, ubd, eq, 2), flux(x, yu, ubu, eq, 2)
                set_node_vars!(Fb_, fbl, eq, ii, 1)
                set_node_vars!(Fb_, fbr, eq, ii, 2)
                set_node_vars!(Fb_, fbd, eq, ii, 3)
                set_node_vars!(Fb_, fbu, eq, ii, 4)
            end
        end
    end
    return nothing
    end # timer
end
end # muladd
