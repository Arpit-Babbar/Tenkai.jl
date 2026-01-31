
import ..Tenkai: setup_arrays_rkfr,
                 compute_cell_residual_rkfr!,
                 update_ghost_values_rkfr!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# TODO - This is not needed either, right?
function setup_arrays_rkfr(grid, scheme, eq::AbstractNonConservativeEquations{2})
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

    uEltype = eltype(u0)
    A3dp1_x = Array{uEltype, 3}
    A3dp1_y = Array{uEltype, 3}

    # TODO - Move to the blend object
    fstar1_L_threaded = A3dp1_x[A3dp1_x(undef, nvar, nd + 1, nd)
                                for _ in 1:Threads.nthreads()]
    fstar1_R_threaded = A3dp1_x[A3dp1_x(undef, nvar, nd + 1, nd)
                                for _ in 1:Threads.nthreads()]
    fstar2_L_threaded = A3dp1_y[A3dp1_y(undef, nvar, nd, nd + 1)
                                for _ in 1:Threads.nthreads()]
    fstar2_R_threaded = A3dp1_y[A3dp1_y(undef, nvar, nd, nd + 1)
                                for _ in 1:Threads.nthreads()]

    fv_cache = (; fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded,
                fstar2_R_threaded)

    cache = (; u0, u1, ua, res, Fb, ub, fv_cache)
    return cache
end

function compute_cell_residual_rkfr!(eq::AbstractNonConservativeEquations{2}, grid, op,
                                     problem,
                                     scheme::Scheme{<:Union{String, RKFR}}, aux, t, dt,
                                     u1, res, Fb,
                                     ub, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack xg, D1, Vl, Vr, Dm = op
    nx, ny = grid.size
    nd = length(xg)
    nvar = nvariables(eq)
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    @unpack blend_cell_residual! = blend.subroutines
    @unpack source_terms = problem
    refresh!(u) = fill!(u, zero(eltype(u)))

    refresh!.((ub, Fb, res))
    @timeit aux.timer "Cell loop" begin
    #! format: noindent
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

            # Conservative part
            for ii in Base.OneTo(nd)
                # res = D * f for each variable
                # res[ii,j] = ∑_i D[ii,i] * f[i,j] for each variable
                multiply_add_to_node_vars!(r1, lamx * D1[ii, i], f_node, eq, ii,
                                           j)
            end

            for jj in Base.OneTo(nd)
                # res = g * D' for each variable
                # res[i,jj] = ∑_j g[i,j] * D1[jj,j] for each variable
                multiply_add_to_node_vars!(r1, lamy * D1[jj, j], g_node, eq, i,
                                           jj)
            end

            s_node = calc_source(u_node, X, t, source_terms, eq)
            multiply_add_to_node_vars!(r1, -dt, s_node, eq, i, j)

            # Non-conservative part
            integral_contribution = zero(u_node)
            for ii in Base.OneTo(nd) # Computes derivative in reference coordinates
                # TODO - Replace with multiply_non_conservative_node_vars!
                # and then you won't need the `eq_nc` struct.
                u_node_ii = get_node_vars(u, eq, ii, j)
                u_non_cons_ii = calc_non_cons_gradient(u_node_ii, x, y, t, eq)
                noncons_flux1 = calc_non_cons_Bu(u_node, u_non_cons_ii, x, y, t,
                                                 1, eq)
                integral_contribution = (integral_contribution +
                                         lamx * Dm[i, ii] * noncons_flux1)
            end

            for jj in Base.OneTo(nd) # Computes derivative in reference coordinates
                u_node_jj = get_node_vars(u, eq, i, jj)
                u_non_cons_jj = calc_non_cons_gradient(u_node_jj, x, y, t, eq)
                noncons_flux2 = calc_non_cons_Bu(u_node, u_non_cons_jj, x, y, t,
                                                 2, eq)
                integral_contribution = (integral_contribution +
                                         lamy * Dm[j, jj] * noncons_flux2)
            end

            multiply_add_to_node_vars!(r1, 1.0, integral_contribution, eq, i,
                                       j)

            # Ub = UT * V
            # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
            multiply_add_to_node_vars!(ub_, Vl[i], u_node, eq, j, 1)
            multiply_add_to_node_vars!(ub_, Vr[i], u_node, eq, j, 2)

            # Ub = U * V
            # Ub[i] += ∑_j U[i,j]*V[j]
            multiply_add_to_node_vars!(ub_, Vl[j], u_node, eq, i, 3)
            multiply_add_to_node_vars!(ub_, Vr[j], u_node, eq, i, 4)
        end

        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        Bb_to_res_cheap!(eq, local_grid, op, ub_, r1)

        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid,
                             dx,
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
    end # timer
    return nothing
    end # timer
end

function tenkai_flux_diff_kernel!(eq::AbstractNonConservativeEquations{2}, volume_flux,
                                  res, u,
                                  lamx, lamy, grid, op, cache, element)
    @unpack xg, Dsplit = op
    nx, ny = grid.size
    nd = length(xg)
    nvar = nvariables(eq)

    trixi_equations = tenkai2trixiequation(eq)

    for j in 1:nd, i in 1:nd
        # u_node = Trixi.get_node_vars(u, equations, dg, i, j, element...)
        u_node = get_node_vars(u, eq, i, j)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nd
            u_node_ii = get_node_vars(u, eq, ii, j)
            flux1 = volume_flux(u_node, u_node_ii, 1, trixi_equations)
            multiply_add_to_node_vars!(res, lamx * Dsplit[i, ii], flux1, eq, i, j)
            multiply_add_to_node_vars!(res, lamx * Dsplit[ii, i], flux1, eq, ii, j)
        end

        # y direction
        for jj in (j + 1):nd
            u_node_jj = get_node_vars(u, eq, i, jj)
            flux2 = volume_flux(u_node, u_node_jj, 2, trixi_equations)
            multiply_add_to_node_vars!(res, lamy * Dsplit[j, jj], flux2, eq, i, j)
            multiply_add_to_node_vars!(res, lamy * Dsplit[jj, j], flux2, eq, i, jj)
        end
    end
end

function tenkai_flux_diff_nc_kernel!(eq::AbstractNonConservativeEquations{2},
                                     nonconservative_flux,
                                     res, u, lamx, lamy, grid, op, cache, element)
    @unpack xg, Dsplit = op
    nx, ny = grid.size
    nd = length(xg)
    nvar = nvariables(eq)

    trixi_equations = tenkai2trixiequation(eq)

    # Calculate the remaining volume terms using the nonsymmetric generalized flux
    for j in 1:nd, i in 1:nd
        u_node = get_node_vars(u, eq, i, j)

        # The diagonal terms are zero since the diagonal of `derivative_split`
        # is zero. We ignore this for now.

        # x direction
        integral_contribution = zero(u_node)
        for ii in 1:nd
            u_node_ii = get_node_vars(u, eq, ii, j)
            noncons_flux1 = nonconservative_flux(u_node, u_node_ii, 1, trixi_equations)
            integral_contribution = integral_contribution +
                                    lamx * Dsplit[i, ii] * noncons_flux1
        end

        # y direction
        for jj in 1:nd
            u_node_jj = get_node_vars(u, eq, i, jj)
            noncons_flux2 = nonconservative_flux(u_node, u_node_jj, 2, trixi_equations)
            integral_contribution = integral_contribution +
                                    lamy * Dsplit[j, jj] * noncons_flux2
        end

        # The factor 0.5 cancels the factor 2 in the flux differencing form
        multiply_add_to_node_vars!(res, 0.5, integral_contribution,
                                   eq, i, j)
    end
end

function compute_cell_residual_rkfr!(eq::AbstractNonConservativeEquations{2}, grid, op,
                                     problem,
                                     scheme::Scheme{<:RKFR{<:VolumeIntegralFluxDifferencing}},
                                     aux, t, dt, u1, res, Fb, ub, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack xg, D1, Vl, Vr, Dm, Dsplit = op
    nx, ny = grid.size
    nd = length(xg)
    nvar = nvariables(eq)
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    @unpack blend_cell_residual! = blend.subroutines
    @unpack source_terms = problem
    @unpack solver = scheme
    symmetric_flux, nonconservative_flux = solver.volume_integral.volume_flux
    refresh!(u) = fill!(u, zero(eltype(u)))

    refresh!.((ub, Fb, res))
    @timeit aux.timer "Cell loop" begin
    #! format: noindent
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # element loop
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy
        u = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        ub_ = @view ub[:, :, :, el_x, el_y]
        Fb_ = @view Fb[:, :, :, el_x, el_y]

        tenkai_flux_diff_kernel!(eq, symmetric_flux, r1, u, lamx, lamy, grid,
                                 op, cache, (el_x, el_y))
        tenkai_flux_diff_nc_kernel!(eq, nonconservative_flux, r1, u, lamx, lamy,
                                    grid, op, cache, (el_x, el_y))

        for j in Base.OneTo(nd), i in Base.OneTo(nd) # solution points loop
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            X = SVector(x, y)
            u_node = get_node_vars(u, eq, i, j)

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

        local_grid = (xc, yc, dx, dy, lamx, lamy, t, dt)

        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid,
                             dx,
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
    end # timer
    return nothing
    end # timer
end

function compute_face_residual!(eq::AbstractNonConservativeEquations{2}, grid, op,
                                cache, problem,
                                scheme::Scheme{<:Union{String, RKFR}}, param, aux, t,
                                dt, u1,
                                Fb, ub, ua, res, scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack bl, br, xg, wg, degree = op
    nd = degree + 1
    nx, ny = grid.size
    @unpack dx, dy, xf, yf = grid
    @unpack numerical_flux, solver = scheme
    @unpack blend = aux
    @unpack blend_face_residual_x!, blend_face_residual_y!, get_element_alpha = blend.subroutines

    # Vertical faces, x flux
    @threaded for element in CartesianIndices((1:(nx + 1), 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        # This is face between elements (el_x-1, el_y), (el_x, el_y)
        x = xf[el_x]
        for jy in Base.OneTo(nd)
            y = yf[el_y] + xg[jy] * dy[el_y]
            ul, ur = (get_node_vars(ub, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(ub, eq, jy, 1, el_x, el_y))
            Fl, Fr = (get_node_vars(Fb, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(Fb, eq, jy, 1, el_x, el_y))
            Ul, Ur = (get_node_vars(ub, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(ub, eq, jy, 1, el_x, el_y))
            X = SVector{2}(x, y)

            Bul, Bur = compute_non_cons_terms(ul, ur, Ul, Ur, x, y, t, 1, solver,
                                              eq)

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

            set_node_vars!(Fb, Fn_l_limited, eq, jy, 2, el_x - 1, el_y)
            set_node_vars!(Fb, Fn_r_limited, eq, jy, 1, el_x, el_y)
        end
    end

    # Horizontal faces, y flux
    @threaded for element in CartesianIndices((1:nx, 1:(ny + 1))) # Loop over cells
        el_x, el_y = element[1], element[2]
        # This is the face between elements (el_x,el_y-1) and (el_x,el_y)
        y = yf[el_y]
        for ix in Base.OneTo(nd)
            x = xf[el_x] + xg[ix] * dx[el_x]
            ul, ur = get_node_vars(ub, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(ub, eq, ix, 3, el_x, el_y)
            Fl, Fr = get_node_vars(Fb, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(Fb, eq, ix, 3, el_x, el_y)
            Ul, Ur = get_node_vars(ub, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(ub, eq, ix, 3, el_x, el_y)
            X = SVector{2}(x, y)
            Bul, Bur = compute_non_cons_terms(ul, ur, Ul, Ur, x, y, t, 2, solver,
                                              eq)
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

            set_node_vars!(Fb, Fn_l_limited, eq, ix, 4, el_x, el_y - 1)
            set_node_vars!(Fb, Fn_r_limited, eq, ix, 3, el_x, el_y)
        end
    end

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        alpha = get_element_alpha(blend, el_x, el_y) # TODO - Use a function to get this
        one_m_alp = 1.0 - alpha
        for ix in Base.OneTo(nd)

            # For higher order residual
            for jy in Base.OneTo(nd)
                Fl = get_node_vars(Fb, eq, jy, 1, el_x, el_y)
                Fr = get_node_vars(Fb, eq, jy, 2, el_x, el_y)
                Fd = get_node_vars(Fb, eq, ix, 3, el_x, el_y)
                Fu = get_node_vars(Fb, eq, ix, 4, el_x, el_y)
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

            # For lower order residual
            Fd = get_node_vars(Fb, eq, ix, 3, el_x, el_y)
            Fu = get_node_vars(Fb, eq, ix, 4, el_x, el_y)

            multiply_add_to_node_vars!(res, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                                       -alpha * dt / (dy[el_y] * wg[1]),
                                       Fd,
                                       eq, ix, 1, el_x, el_y)

            multiply_add_to_node_vars!(res, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                                       alpha * dt / (dy[el_y] * wg[nd]),
                                       Fu,
                                       eq, ix, nd, el_x, el_y)

            Fl = get_node_vars(Fb, eq, ix, 1, el_x, el_y)
            Fr = get_node_vars(Fb, eq, ix, 2, el_x, el_y)

            multiply_add_to_node_vars!(res, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                                       -alpha * dt / (dx[el_x] * wg[1]), Fl,
                                       eq, 1, ix, el_x, el_y)
            multiply_add_to_node_vars!(res, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                                       alpha * dt / (dx[el_x] * wg[nd]), Fr,
                                       eq, nd, ix, el_x, el_y)
        end
    end

    return nothing
    end # timer
end # compute_face_residual!
end # muladd
