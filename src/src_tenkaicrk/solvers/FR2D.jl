using SimpleUnPack
using MuladdMacro
using TimerOutputs
using Printf
using Tenkai.Polyester
using OffsetArrays # OffsetArray, OffsetMatrix, OffsetVector
using Tenkai.ElasticArrays
using StaticArrays
using Tenkai.WriteVTK
using Tenkai.FLoops
using Tenkai.LoopVectorization
using LinearAlgebra: lmul!, mul!

using Tenkai: refresh!

using TimerOutputs
using OffsetArrays

using Tenkai.JSON3

using Tenkai: nodal2modal, Blend2D, initialize_plot, create_aux_cache,
              multiply_dimensionwise!,
              finite_differences, limit_slope, rkfr, blending_flux_factors, is_admissible,
              zhang_shu_flux_fix

import Tenkai: set_initial_condition!, compute_cell_average!,
               compute_face_residual!, compute_error,
               modal_smoothness_indicator_gassner, update_ghost_values_u1!,
               Blend, create_auxiliaries, blend_face_residual_muscl_x!,
               blend_face_residual_muscl_y!, get_blended_flux_x, get_blended_flux_y,
               blend_cell_residual_muscl!, correct_variable!, update_ghost_values_cRK!

import Tenkai.EqEuler2D: save_solution_file

function set_initial_condition!(u, eq::AbstractEquations{2}, grid::StepGrid, op, problem)
    println("Setting initial condition")
    @unpack initial_value = problem
    nx_tuple, ny_tuple = grid.size
    xg = op.xg
    nd = length(xg)
    for element_index in element_iterator(grid)
        element = element_indices(element_index, grid)
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y] # cell size
        xc, yc = grid.xc[el_x], grid.yc[el_y] # cell center
        for j in 1:nd, i in 1:nd
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            iv = initial_value(x, y)
            set_node_vars!(u, iv, eq, i, j, el_x, el_y)
        end
    end
    return nothing
end

function compute_cell_average!(ua, u1, t, eq::AbstractEquations{2}, grid::StepGrid,
                               problem, scheme, aux, op)
    @timeit aux.timer "Cell averaging" begin
    #! format: noindent
    nx_tuple, ny_tuple = grid.size
    @unpack limiter = scheme
    @unpack xc = grid
    @unpack xg, wg, Vl, Vr = op
    @unpack periodic_x, periodic_y = problem
    @unpack boundary_condition, boundary_value = problem
    left, right, bottom, top = boundary_condition
    nd = length(wg)
    fill!(ua, zero(eltype(ua)))
    # Compute cell averages
    @threaded for element_index in element_iterator(grid)
        element = element_indices(element_index, grid)
        el_x, el_y = element[1], element[2]
        u1_ = @view u1[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_node = get_node_vars(u1_, eq, i, j)
            multiply_add_to_node_vars!(ua, wg[i] * wg[j], u_node, eq, el_x, el_y)
            # Maybe put w2d in op and use it here
        end
        # @show element
    end

    # left boundary condition is Dirichlet
    for el_y in (ny_tuple[1] + 1):ny_tuple[2]
        x = grid.xf[1]
        for j in Base.OneTo(nd)
            y = grid.yf[el_y] + grid.dy[el_y] * xg[j]
            uval = boundary_value(x, y, t)
            for i in Base.OneTo(nd)
                multiply_add_to_node_vars!(ua, wg[i] * wg[j], uval, eq, 0,
                                           el_y)
            end
        end
    end

    # right boundary condition is neumann
    for el_y in 1:ny_tuple[2], n in eachvariable(eq)
        el_x = nx_tuple[2] # right most physical element
        ua[n, el_x + 1, el_y] = ua[n, el_x, el_y]
    end

    # Horizontal walls of bottom are reflective
    for el_x in bottom_horizontal_iterator(grid)
        el_y = bottom_physical_element(el_x, grid)
        for n in eachvariable(eq)
            ua[n, el_x, el_y - 1] = ua[n, el_x, el_y]
        end
        ua[3, el_x, el_y - 1] *= -1.0
    end

    # Vertical walls at bottom are reflective
    for el_y in bottom_vertical_iterator(grid)
        el_x = leftmost_physical_element(el_y, grid) # nx_tuple[1]+1
        for n in eachvariable(eq)
            ua[n, el_x - 1, el_y] = ua[n, el_x, el_y]
        end
        ua[2, el_x - 1, el_y] *= -1.0
    end

    # Horizontal walls at top are reflective
    for el_x in 1:nx_tuple[2]
        el_y = ny_tuple[2] # top_physical_element
        for n in eachvariable(eq)
            ua[n, el_x, el_y + 1] = ua[n, el_x, el_y]
        end
        ua[3, el_x, el_y + 1] *= -1.0
    end
    end # timer
    return nothing
end

function compute_face_residual!(eq::AbstractEquations{2}, grid::StepGrid, op,
                                cache, problem, scheme::Scheme{<:cRKSolver},
                                param, aux, t, dt, u1,
                                Fb, Ub, ua, res, scaling_factor = 1)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack bl, br, xg, degree = op
    nd = degree + 1
    @unpack dx, dy, xf, yf = grid
    @unpack numerical_flux = scheme
    @unpack blend = aux
    @unpack blend_face_residual_x!, blend_face_residual_y! = blend.subroutines

    Fk = copy(Fb)
    # Vertical faces, x flux
    @threaded for element_index in face_x_iterator(grid)
        element = face_x_indices(element_index, grid)
        el_x, el_y = element[1], element[2]
        # Face between (i-1,j) and (i,j)
        x = xf[el_x]
        ual, uar = get_node_vars(ua, eq, el_x - 1, el_y),
                   get_node_vars(ua, eq, el_x, el_y)
        for jy in Base.OneTo(nd)
            y = yf[el_y] + xg[jy] * dy[el_y]
            Fl, Fr = (get_node_vars(Fb, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(Fb, eq, jy, 1, el_x, el_y))
            Ul, Ur = (get_node_vars(Ub, eq, jy, 2, el_x - 1, el_y),
                      get_node_vars(Ub, eq, jy, 1, el_x, el_y))
            X = SVector{2}(x, y)
            Fn = numerical_flux(X, ual, uar, Fl, Fr, Ul, Ur, eq, 1)
            Fn, blend_factors = blend_face_residual_x!(el_x, el_y, jy, x, y, u1, ua,
                                                       eq, dt, grid, op, cache,
                                                       scheme, param, Fn, aux,
                                                       res)

            set_node_vars!(Fb, Fn, eq, jy, 2, el_x - 1, el_y)
            set_node_vars!(Fb, Fn, eq, jy, 1, el_x, el_y)
        end
    end

    # Horizontal faces, y flux
    @threaded for element_index in face_y_iterator(grid)
        element = face_y_indices(element_index, grid)
        el_x, el_y = element[1], element[2]
        # Face between (i,j-1) and (i,j)
        y = yf[el_y]
        ual, uar = get_node_vars(ua, eq, el_x, el_y - 1),
                   get_node_vars(ua, eq, el_x, el_y)
        for ix in Base.OneTo(nd)
            x = xf[el_x] + xg[ix] * dx[el_x]
            Fl, Fr = get_node_vars(Fb, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(Fb, eq, ix, 3, el_x, el_y)
            Ul, Ur = get_node_vars(Ub, eq, ix, 4, el_x, el_y - 1),
                     get_node_vars(Ub, eq, ix, 3, el_x, el_y)
            X = SVector{2}(x, y)
            Fn = numerical_flux(X, ual, uar, Fl, Fr, Ul, Ur, eq, 2)
            Fn, blend_factors = blend_face_residual_y!(el_x, el_y, ix, x, y,
                                                       u1, ua, eq, dt, grid, op, cache,
                                                       scheme, param, Fn, aux,
                                                       res)
            set_node_vars!(Fb, Fn, eq, ix, 4, el_x, el_y - 1)
            set_node_vars!(Fb, Fn, eq, ix, 3, el_x, el_y)
        end
    end

    @threaded for element_index in element_iterator(grid)
        element = element_indices(element_index, grid)
        el_x, el_y = element[1], element[2]
        blend_factor = 1.0 - blend.cache.alpha[el_x, el_y] # TODO - Gives error without blending limiter
        for ix in Base.OneTo(nd)
            for jy in Base.OneTo(nd)
                Fl = get_node_vars(Fb, eq, jy, 1, el_x, el_y)
                Fr = get_node_vars(Fb, eq, jy, 2, el_x, el_y)
                Fd = get_node_vars(Fb, eq, ix, 3, el_x, el_y)
                Fu = get_node_vars(Fb, eq, ix, 4, el_x, el_y)
                multiply_add_to_node_vars!(res,
                                           blend_factor * dt / dy[el_y] * br[jy], Fu,
                                           eq,
                                           ix, jy, el_x, el_y)
                multiply_add_to_node_vars!(res,
                                           blend_factor * dt / dy[el_y] * bl[jy], Fd,
                                           eq,
                                           ix, jy, el_x, el_y)

                multiply_add_to_node_vars!(res,
                                           blend_factor * dt / dx[el_x] * br[ix], Fr,
                                           eq,
                                           ix, jy, el_x, el_y)
                multiply_add_to_node_vars!(res,
                                           blend_factor * dt / dx[el_x] * bl[ix], Fl,
                                           eq,
                                           ix, jy, el_x, el_y)
            end
        end
    end
    return nothing
    end # timer
end

function blend_face_residual_muscl_x!(el_x, el_y, jy, xf, y, u1, ua,
                                      eq::AbstractEquations{2, <:Any}, dt, grid::StepGrid,
                                      op, cache, scheme, param, Fn, aux, res)
    @timeit_debug aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack blend = aux
    @unpack alpha = blend.cache
    @unpack dx, dy = grid
    @unpack tvbM = blend.parameters
    @unpack bc_x = blend.subroutines
    nvar = nvariables(eq)
    num_flux = scheme.numerical_flux
    nx_tuple, ny_tuple = grid.size
    @unpack u1x, u1y = cache

    id = Threads.threadid()

    unph_, = blend.cache.unph[id][1]
    unph = @view unph_[:, 1:2, 1, 1] # Load nvar x 2 array to save storage

    dt = blend.cache.dt[1] # For support with DiffEq

    @unpack xg, wg = op
    nd = length(xg)

    # The two solution points neighbouring the super face have a numerical flux
    # which is to be obtained after blending with the time averaged flux.
    # Thus, the evolution for those points has two be done here.

    # Since there are two points, we store relevant arrays in 2-tuples and loop
    # Those solution values, locations and corresponding (subcell) faces
    # are all stored. The first element corresponds to last point of left array
    # and second element corresponds to first element of right array

    # |-----||-----|
    # |     ||     |
    # |     ||     |
    # |-----||-----|

    # We first find u^{n+1/2}_{±} at the face. For ±, there are two
    # relevant iterations of loops respectively. To do everything in one loop, we stack all
    # quantities relevant to the computation at the very beginning

    # Stack all relevant arrays (ul, u, ur) both for
    arrays1_x = (get_node_vars(u1x, eq, nd - 1, jy, el_x - 1, el_y),
                 get_node_vars(u1x, eq, nd, jy, el_x - 1, el_y),
                 get_node_vars(u1x, eq, 1, jy, el_x, el_y))
    arrays2_x = (get_node_vars(u1x, eq, nd, jy, el_x - 1, el_y),
                 get_node_vars(u1x, eq, 1, jy, el_x, el_y),
                 get_node_vars(u1x, eq, 2, jy, el_x, el_y))

    solns_x = (arrays1_x, arrays2_x)

    # Stack x coordinates of solution points (xl, x, xr)
    sol_coords_x = ((xf - dx[el_x - 1] + xg[nd - 1] * dx[el_x - 1], # xl
                     xf - dx[el_x - 1] + xg[nd] * dx[el_x - 1],   # x
                     xf + xg[1] * dx[el_x]),                  # xr

                    # Corresponding to solns2_x
                    (xf - dx[el_x - 1] + xg[nd] * dx[el_x - 1], # xl
                     xf + xg[1] * dx[el_x],                 # x
                     xf + xg[2] * dx[el_x]))

    # For the y-direction values, the indices may go outside of the cell
    # so we have to break into cases.
    if jy == 1
        ud_1 = get_node_vars(u1y, eq, nd, nd, el_x - 1, el_y - 1) # value below u from arrays1
        ud_2 = get_node_vars(u1y, eq, 1, nd, el_x, el_y - 1) # value below u from arrays2
        # Don't use corner values
        if el_x - 1 == nx_tuple[1] && el_y - 1 == ny_tuple[1]
            ud_1 = ud_2
        end

        # Solution point below y
        yd_1 = yd_2 = grid.yf[el_y] - dy[el_y - 1] + xg[nd] * dy[el_y - 1]

        # Face between y and yd
        yfd_1 = yfd_2 = grid.yf[el_y]
    else
        ud_1 = get_node_vars(u1x, eq, nd, jy - 1, el_x - 1, el_y)
        ud_2 = get_node_vars(u1x, eq, 1, jy - 1, el_x, el_y)

        # Solution points
        yd_1 = yd_2 = grid.yf[el_y] + xg[jy - 1] * dy[el_y]

        # Face between y and yd
        yfd_1 = yfd_2 = grid.yf[el_y]
        for jjy in 1:(jy - 1)
            yfd_1 += wg[jjy] * dy[el_y - 1]
        end
        yfd_2 = yfd_1
    end

    if jy == nd
        uu_1 = get_node_vars(u1y, eq, nd, 1, el_x - 1, el_y + 1)
        uu_2 = get_node_vars(u1y, eq, 1, 1, el_x, el_y + 1)

        # Solution points
        yu_1 = yu_2 = grid.yf[el_y + 1] + xg[1] * grid.dy[el_y + 1]

        # Faces
        yfu_1 = yfu_2 = grid.yf[el_y + 1] # + wg[1]*grid.dy[el_y+1]
    else
        uu_1 = get_node_vars(u1x, eq, nd, jy + 1, el_x - 1, el_y)
        uu_2 = get_node_vars(u1x, eq, 1, jy + 1, el_x, el_y)

        # Solution points
        yu_1 = yu_2 = grid.yf[el_y] + xg[jy + 1] * grid.dy[el_y]

        # Faces
        yfu_1 = yfu_2 = grid.yf[el_y]
        for jjy in 1:jy
            yfu_1 = yfu_2 += wg[jjy] * grid.dy[el_y]
        end
    end

    solns_y = ((ud_1, uu_1), (ud_2, uu_2))

    sol_coords_y = ((yd_1, yu_1), (yd_2, yu_2))

    # Stack x coordinates of faces (xfl, xfr)
    face_coords_x = ((xf - wg[nd] * dx[el_x - 1], xf), # (xfl, xfr)
                     (xf, xf + wg[1] * dx[el_x]))   # (xfl, xfr)
    face_coords_y = ((yfd_1, yfu_1), (yfd_2, yfu_2))

    betas = (2.0 - alpha[el_x - 1, el_y], 2.0 - alpha[el_x, el_y])

    if blend.parameters.pure_fv == true
        betas = (2.0, 2.0)
    end

    for i in 1:2 # Loop over cells
        ul, u_, ur = solns_x[i]
        ud, uu = solns_y[i]

        # TOTHINK - Add this feature
        # u_, ul, ur = conservative2recon.((u_,ul,ur))

        xl, x, xr = sol_coords_x[i]
        yd, yu = sol_coords_y[i]
        xfl, xfr = face_coords_x[i]
        yfd, yfu = face_coords_y[i]

        Δx1, Δx2 = x - xl, xr - x
        Δy1, Δy2 = y - yd, yu - y
        back_x, cent_x, fwd_x = finite_differences(Δx1, Δx2, ul, u_, ur)
        back_y, cent_y, fwd_y = finite_differences(Δy1, Δy2, ud, u_, uu)
        beta = betas[i]
        Mdx2 = tvbM * Δx1
        Mdy2 = tvbM * Δy1
        slope_tuple_x = (minmod(cent_x[n], back_x[n], fwd_x[n], beta, Mdx2)
                         for n in eachvariable(eq))
        slope_tuple_y = (minmod(cent_y[n], back_y[n], fwd_y[n], beta, Mdy2)
                         for n in eachvariable(eq))
        slope_x = SVector{nvar}(slope_tuple_x)
        slope_y = SVector{nvar}(slope_tuple_y)

        ufl = u_ + slope_x * (xfl - x)
        ufr = u_ + slope_x * (xfr - x)
        ufd = u_ + slope_y * (yfd - y)
        ufu = u_ + slope_y * (yfu - y)

        u_star_l = u_ + 2.0 * slope_x * (xfl - x)
        u_star_r = u_ + 2.0 * slope_x * (xfr - x)
        u_star_d = u_ + 2.0 * slope_y * (yfd - y)
        u_star_u = u_ + 2.0 * slope_y * (yfu - y)

        ufl, ufr = limit_slope(eq, slope_x, ufl, u_star_l, ufr, u_star_r, u_,
                               xfl - x, xfr - x)

        ufd, ufu = limit_slope(eq, slope_y, ufd, u_star_d, ufu, u_star_u, u_,
                               yfd - y, yfu - y)
        # TOTHINK - Add this feature
        # Convert back to conservative variables for update
        # ufl, ufr = recon2conservative.((ufl,ufr))

        fl = flux(xfl, y, ufl, eq, 1)
        fr = flux(xfr, y, ufr, eq, 1)
        gd = flux(x, yfd, ufd, eq, 2)
        gu = flux(x, yfu, ufu, eq, 2)

        if i == 1
            uf = ufr # The relevant face is on the right
        elseif i == 2
            uf = ufl # The relevant face is on the left
        end

        # Use finite difference method to evolve face values to time level n+1/2
        multiply_add_set_node_vars!(unph, # unph = uf - 0.5*dt*(fr-fl)/(xfr-xfl)
                                    uf,
                                    -0.5 * dt / (xfr - xfl),
                                    fr,
                                    0.5 * dt / (xfr - xfl),
                                    fl,
                                    eq,
                                    i)

        if !(isnearcorners(el_x, el_y, grid))
            # Avoid taking corner point stencils near corners of physical boundaries
            multiply_add_to_node_vars!(unph, # unph += -0.5*dt*(gu-gd)/(yfu-yfd)
                                       -0.5 * dt / (yfu - yfd),
                                       gu,
                                       0.5 * dt / (yfu - yfd),
                                       gd,
                                       eq,
                                       i)
        end
    end
    # Put reflect bc here!
    ul = get_node_vars(unph, eq, 1)
    ur = get_node_vars(unph, eq, 2)
    # if isnearcorners(el_x, el_y, grid)
    #    ur = get_node_vars(u1, eq, 1,  jy, el_x,   el_y)
    #    ul = get_node_vars(u1, eq, nd, jy, el_x-1, el_y)
    #    if el_x == nx_tuple[1]+1 && el_y == ny_tuple[1]
    #       ul = SVector{4}(ur[1],-ur[2],ur[3],ur[4])
    #    end
    # end
    fl, fr = flux(xf, y, ul, eq, 1), flux(xf, y, ur, eq, 1)
    X = SVector(xf, y)

    fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)

    # Repetetition block
    Fn = get_blended_flux_x(el_x, el_y, jy, eq, dt, grid,
                            blend, scheme, xf, y, u1x, ua, fn, Fn, op)

    # This subroutine allows user to specify boundary conditions
    Fn = bc_x(u1, eq, op, xf, y, jy, el_x, el_y, Fn)

    r = @view res[:, :, jy, el_x - 1, el_y]
    multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                               alpha[el_x - 1, el_y] * dt / (dx[el_x - 1] * wg[nd]), Fn,
                               eq, nd)

    r = @view res[:, :, jy, el_x, el_y]
    multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                               -alpha[el_x, el_y] * dt / (dx[el_x] * wg[1]), Fn,
                               eq, 1)

    return Fn, (1.0 - alpha[el_x - 1, el_y], 1.0 - alpha[el_x, el_y])
    end # timer
end

function get_blended_flux_x(el_x, el_y, jy, eq::AbstractEquations{2}, dt, grid::StepGrid,
                            blend, scheme, xf, y, u1x, ua, fn, Fn, op)
    if scheme.solver_enum == rkfr
        return Fn
    end

    @unpack alpha, fn_low = blend.cache
    @unpack dx, dy = grid
    @unpack wg = op
    nd = length(wg)
    nx_tuple, ny_tuple = grid.size

    # Initial trial blended flux
    alp = 0.5 * (alpha[el_x - 1, el_y] + alpha[el_x, el_y])
    Fn = (1.0 - alp) * Fn + alp * fn

    ua_ll_node = get_node_vars(ua, eq, el_x - 1, el_y)
    λx_ll, _ = blending_flux_factors(eq, ua_ll_node, dx[el_x - 1], dy[el_y])

    ua_rr_node = get_node_vars(ua, eq, el_x, el_y)
    λx_rr, _ = blending_flux_factors(eq, ua_rr_node, dx[el_x], dy[el_y])

    # We see update at solution point in element (el_x-1,el_y)
    u_ll_node = get_node_vars(u1x, eq, nd, jy, el_x - 1, el_y)

    # lower order flux on neighbouring subcell face
    fn_inner_ll = get_node_vars(fn_low, eq, jy, 2, el_x - 1, el_y)

    # Test whether lower order update is even admissible
    c_ll = (dt / dx[el_x - 1]) / (wg[nd] * λx_ll) # c is such that u_new = u_prev - c*(Fn-fn)
    low_update_ll = u_ll_node - c_ll * (fn - fn_inner_ll)
    test_update_ll = u_ll_node - c_ll * (Fn - fn_inner_ll)
    if is_admissible(eq, low_update_ll) == false && el_x > 1
        # @warn "Low x-flux not admissible at " (el_x-1),el_y,xf,y
    end

    if !(is_admissible(eq, test_update_ll))
        @debug "Zhang-Shu fix needed at " (el_x - 1), el_y, xf, y
        Fn = zhang_shu_flux_fix(eq, u_ll_node, low_update_ll,
                                Fn, fn_inner_ll, fn, c_ll)
    end

    # Now we see the update at solution point in element (el_x,el_y)
    u_rr_node = get_node_vars(u1x, eq, 1, jy, el_x, el_y)

    # lower order flux on neighbouring subcell face
    fn_inner_rr = get_node_vars(fn_low, eq, jy, 1, el_x, el_y)

    # Test whether lower order update is even admissible
    c_rr = -(dt / dx[el_x]) / (wg[1] * λx_rr) # c is such that u_new = u_prev - c*(Fn-fn)
    low_update_rr = u_rr_node - c_rr * (fn - fn_inner_rr)
    if is_admissible(eq, low_update_rr) == false && el_x < nx_tuple[2] + 1
        # @warn "Lower x-flux not admissible at " el_x,el_y,xf,y
    end
    test_update_rr = u_rr_node - c_rr * (Fn - fn_inner_rr)

    if !(is_admissible(eq, test_update_rr))
        @debug "Zhang-Shu fix needed at " (el_x - 1), el_y, xf, y
        Fn = zhang_shu_flux_fix(eq, u_rr_node, low_update_rr, Fn, fn_inner_rr,
                                fn, c_rr)
    end

    return Fn
end

function blend_face_residual_muscl_y!(el_x, el_y, ix, x, yf, u1, ua,
                                      eq::AbstractEquations{2, <:Any}, dt, grid::StepGrid,
                                      op,
                                      cache, scheme, param, Fn, aux, res)
    @timeit_debug aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack blend = aux
    @unpack alpha = blend.cache
    @unpack tvbM = blend.parameters
    @unpack dx, dy = grid
    num_flux = scheme.numerical_flux
    nvar = nvariables(eq)
    nx_tuple, ny_tuple = grid.size

    id = Threads.threadid()

    dt = blend.cache.dt[1] # For support with DiffEq

    unph_, = blend.cache.unph[id][1]
    unph = @view unph_[:, 1:2, 1, 1] # Load nvar x 2 array

    @unpack xg, wg = op
    @unpack u1x, u1y = cache
    nd = length(xg)

    # The two solution points neighbouring the super face have a numerical flux
    # which is to be obtained after blending with the time averaged flux.
    # Thus, the evolution for those points has two be done here.

    # Since there are two points, we store relevant arrays in 2-tuples and loop
    # Those solution values, locations and corresponding (subcell) faces
    # are all stored. The first element corresponds to last point of left array
    # and second element corresponds to first element of right array

    # |-----||-----|
    # |     ||     |
    # |     ||     |
    # |-----||-----|

    # We first find u^{n+1/2}_{±} at the face. For that, there are two
    # relevant cells. To do everything in one loop, we stack all
    # quantities relevant to the computation at the very beginning

    # Stack all relevant arrays (ul, u, ur)

    arrays1_y = (get_node_vars(u1y, eq, ix, nd - 1, el_x, el_y - 1),
                 get_node_vars(u1y, eq, ix, nd, el_x, el_y - 1),
                 get_node_vars(u1y, eq, ix, 1, el_x, el_y))
    arrays2_y = (get_node_vars(u1y, eq, ix, nd, el_x, el_y - 1),
                 get_node_vars(u1y, eq, ix, 1, el_x, el_y),
                 get_node_vars(u1y, eq, ix, 2, el_x, el_y))

    solns_y = (arrays1_y, arrays2_y)

    sol_coords_y = ((yf - dy[el_y - 1] + xg[nd - 1] * dy[el_y - 1], # yd
                     yf - dy[el_y - 1] + xg[nd] * dy[el_y - 1],   # y
                     yf + xg[1] * dy[el_y]),                 # yu
                    (yf - dy[el_y - 1] + xg[nd] * dy[el_y - 1],   # yd
                     yf + xg[1] * dy[el_y],                   # y
                     yf + xg[2] * dy[el_y]))

    if ix == 1
        ul_1 = get_node_vars(u1x, eq, nd, nd, el_x - 1, el_y - 1)
        ul_2 = get_node_vars(u1x, eq, nd, 1, el_x - 1, el_y)

        if el_x - 1 == nx_tuple[1] && el_y - 1 == ny_tuple[1]
            ul_2 = ul_1
        end

        # Solution points left of x
        xl_1 = xl_2 = grid.xf[el_x] - dx[el_x - 1] + xg[nd] * dx[el_x - 1]

        # Face between xfl and x
        xfl_1 = xfl_2 = grid.xf[el_x]
    else
        ul_1 = get_node_vars(u1y, eq, ix - 1, nd, el_x, el_y - 1)
        ul_2 = get_node_vars(u1y, eq, ix - 1, 1, el_x, el_y)

        # Solution points left of x
        xl_1 = xl_2 = grid.xf[el_x] + dx[el_x] * xg[ix - 1]

        # Face between xl and x
        xfl_1 = xfl_2 = grid.xf[el_x]
        for iix in 1:(ix - 1)
            xfl_2 += dx[el_x] * wg[iix]
        end
        xfl_1 = xfl_2
    end

    if ix == nd
        ur_1 = get_node_vars(u1x, eq, 1, nd, el_x + 1, el_y - 1)
        ur_2 = get_node_vars(u1x, eq, 1, 1, el_x + 1, el_y)

        # Solution points right of x
        xr_1 = xr_2 = grid.xf[el_x + 1] + xg[1] * grid.dx[el_x + 1]

        # Face between x and xr
        xfr_1 = xfr_2 = grid.xf[el_x + 1]
    else
        ur_1 = get_node_vars(u1y, eq, ix + 1, nd, el_x, el_y - 1)
        ur_2 = get_node_vars(u1y, eq, ix + 1, nd, el_x, el_y)

        # Solution points right of x
        xr_1 = xr_2 = grid.xf[el_x] + xg[ix + 1] * grid.dx[el_x]

        # Face between x and xr
        xfr_1 = xfr_2 = grid.xf[el_x]
        for iix in 1:ix
            xfr_2 += wg[iix] * grid.dx[el_x]
        end
        xfr_1 = xfr_2
    end

    solns_x = ((ul_1, ur_1), (ul_2, ur_2))

    # stack x coordinates of solution points (yd, y, yu)
    sol_coords_x = ((xl_1, xr_1), (xl_2, xr_2))

    face_coords_y = ((yf - wg[nd] * dy[el_y - 1], yf), # yfl, yf
                     (yf, yf + wg[1] * dy[el_y]))
    face_coords_x = ((xfl_1, xfr_1), (xfl_2, xfr_2))

    betas = (2.0 - alpha[el_x, el_y - 1], 2.0 - alpha[el_x, el_y])
    if blend.parameters.pure_fv == true
        betas = (2.0, 2.0)
    end

    for i in 1:2 # Loop over the two relevant cells
        ud, u_, uu = solns_y[i]
        ul, ur = solns_x[i]

        # TOTHINK - Add this feature
        # ud, u_, uu = conservative2recon.((ud,u_,uu))

        yd, y, yu = sol_coords_y[i]
        xl, xr = sol_coords_x[i]
        yfd, yfu = face_coords_y[i]
        xfl, xfr = face_coords_x[i]
        Δy1, Δy2 = y - yd, yu - y
        Δx1, Δx2 = x - xl, xr - x
        back_y, cent_y, fwd_y = finite_differences(Δy1, Δy2, ud, u_, uu)
        back_x, cent_x, fwd_x = finite_differences(Δx1, Δx2, ul, u_, ur)
        beta = betas[i]
        Mdy2 = tvbM * Δy1
        slope_tuple_y = (minmod(cent_y[n], back_y[n], fwd_y[n], beta, Mdy2)
                         for n in eachvariable(eq))
        Mdx2 = tvbM * Δx1
        slope_tuple_x = (minmod(cent_x[n], back_x[n], fwd_x[n], beta, Mdx2)
                         for n in eachvariable(eq))
        slope_y = SVector{nvar}(slope_tuple_y)
        slope_x = SVector{nvar}(slope_tuple_x)

        ufd = u_ + slope_y * (yfd - y)
        ufu = u_ + slope_y * (yfu - y)
        ufl = u_ + slope_x * (xfl - x)
        ufr = u_ + slope_x * (xfr - x)

        u_star_d = u_ + 2.0 * slope_y * (yfd - y)
        u_star_u = u_ + 2.0 * slope_y * (yfu - y)
        u_star_l = u_ + 2.0 * slope_x * (xfl - x)
        u_star_r = u_ + 2.0 * slope_x * (xfr - x)

        ufd, ufu = limit_slope(eq, slope_y, ufd, u_star_d, ufu, u_star_u, u_,
                               yfd - y, yfu - y)
        ufl, ufr = limit_slope(eq, slope_x, ufl, u_star_l, ufr, u_star_r, u_,
                               xfl - x, xfr - x)

        # TOTHINK - add this feature
        # Convert back to conservative variables for update
        # ufl, ufr = recon2conservative.((ufl, ufr))

        fl = flux(xfl, y, ufl, eq, 1)
        fr = flux(xfr, y, ufr, eq, 1)
        gd = flux(x, yfd, ufd, eq, 2)
        gu = flux(x, yfu, ufu, eq, 2)

        if i == 1
            uf = ufu # relevant face is the one above
        elseif i == 2
            uf = ufd # relevant face is the one below
        end

        # use finite difference method to evolve face values to time n+1/2
        multiply_add_set_node_vars!(unph, # unph = uf - 0.5*dt*(gu-gd)/(yfu-yfd)
                                    uf,
                                    -0.5 * dt / (yfu - yfd),
                                    gu,
                                    -0.5 * dt / (yfu - yfd),
                                    -gd,
                                    eq,
                                    i)
        if !(isnearcorners(el_x, el_y, grid))
            # Avoid corner stencil near physical corners of the domain
            multiply_add_to_node_vars!(unph, # unph = uf - 0.5*dt*(gu-gd)/(yfu-yfd)
                                       -0.5 * dt / (xfr - xfl),
                                       fr,
                                       0.5 * dt / (xfr - xfl),
                                       fl,
                                       eq,
                                       i)
        end
    end

    # Put reflect bc here!
    ud = get_node_vars(unph, eq, 1)
    uu = get_node_vars(unph, eq, 2)
    gd, gu = flux(x, yf, ud, eq, 2), flux(x, yf, uu, eq, 2)
    X = SVector(x, yf)
    fn = num_flux(X, ud, uu, gd, gu, ud, uu, eq, 2)

    Fn = get_blended_flux_y(el_x, el_y, ix, eq, dt, grid, blend,
                            scheme, x, yf, u1y, ua, fn, Fn, op)

    r = @view res[:, ix, :, el_x, el_y - 1]

    multiply_add_to_node_vars!(r, # r[nd] += alpha*dt/(dy*wg[nd])*Fn
                               alpha[el_x, el_y - 1] * dt / (dy[el_y - 1] * wg[nd]),
                               Fn,
                               eq, nd)

    r = @view res[:, ix, :, el_x, el_y]

    multiply_add_to_node_vars!(r, # r[1] -= alpha*dt/(dy*wg[1])*Fn
                               -alpha[el_x, el_y] * dt / (dy[el_y] * wg[1]),
                               Fn,
                               eq, 1)

    return Fn, (1.0 - alpha[el_x, el_y - 1], 1.0 - alpha[el_x, el_y])
    end # timer
end

function get_blended_flux_y(el_x, el_y, ix, eq::AbstractEquations{2}, dt, grid::StepGrid,
                            blend, scheme, x, yf, u1y, ua, fn, Fn, op)
    if scheme.solver_enum == rkfr
        return Fn
    end

    @unpack alpha, fn_low = blend.cache
    @unpack dx, dy = grid
    @unpack wg = op
    nd = length(wg)
    nx_tuple, ny_tuple = grid.size
    # Initial trial blended flux
    alp = 0.5 * (alpha[el_x, el_y - 1] + alpha[el_x, el_y])
    Fn = (1.0 - alp) * Fn + alp * fn

    # Candidate in for (el_x, el_y-1)
    ua_ll_node = get_node_vars(ua, eq, el_x, el_y - 1)
    λx_ll, λy_ll = blending_flux_factors(eq, ua_ll_node, dx[el_x], dy[el_y - 1])
    # Candidate in (el_x, el_y)
    ua_rr_node = get_node_vars(ua, eq, el_x, el_y)
    λx_rr, λy_rr = blending_flux_factors(eq, ua_rr_node, dx[el_x], dy[el_y])

    u_ll_node = get_node_vars(u1y, eq, ix, nd, el_x, el_y - 1)

    # lower order flux on neighbouring subcell face
    fn_inner_ll = get_node_vars(fn_low, eq, ix, 4, el_x, el_y - 1)

    c_ll = (dt / dy[el_y - 1]) / (wg[nd] * λy_ll)

    # test whether lower order update is even admissible
    low_update_ll = u_ll_node - c_ll * (fn - fn_inner_ll)
    test_update_ll = u_ll_node - c_ll * (Fn - fn_inner_ll)

    if is_admissible(eq, low_update_ll) == false && el_y > 1
        # @warn "Low y-flux not admissible at " el_x,(el_y-1),x,yf
    end

    if !(is_admissible(eq, test_update_ll))
        @debug "Zhang-Shu fix needed at " el_x, (el_y - 1), xf, y
        Fn = zhang_shu_flux_fix(eq, u_ll_node, low_update_ll,
                                Fn, fn_inner_ll, fn, c_ll)
    end

    u_rr_node = get_node_vars(u1y, eq, ix, 1, el_x, el_y)
    fn_inner_rr = get_node_vars(fn_low, eq, ix, 3, el_x, el_y)
    c_rr = -(dt / dy[el_y]) / (wg[1] * λy_rr)
    low_update_rr = u_rr_node - c_rr * (fn - fn_inner_rr)

    if is_admissible(eq, low_update_rr) == false && el_y < ny_tuple[2] + 1
        # @warn "Lower y-flux not admissible at " el_x,el_y,x,yf
    end

    test_update_rr = u_rr_node - c_rr * (Fn - fn_inner_rr)

    if !(is_admissible(eq, test_update_rr))
        @debug "Zhang-Shu fix needed at " (el_x - 1), el_y, xf, y
        Fn = zhang_shu_flux_fix(eq, u_rr_node, low_update_rr, Fn, fn_inner_rr,
                                fn, c_rr)
    end

    return Fn
end

function modal_smoothness_indicator_gassner(eq::AbstractEquations{2}, t, iter,
                                            fcount, dt, grid::StepGrid,
                                            scheme, problem, param, aux, op,
                                            u1, ua)
    @timeit aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack dx, dy = grid
    # nx, ny = grid.size
    nx_tuple, ny_tuple = grid.size
    @unpack nvar = eq
    @unpack xg = op
    nd = length(xg)
    @unpack limiter = scheme
    @unpack blend = aux
    @unpack constant_node_factor = blend.parameters
    @unpack amax = blend.parameters      # maximum factor of the lower order term
    @unpack tolE = blend.parameters      # tolerance for denominator
    @unpack E = blend.cache            # content in high frequency nodes
    @unpack alpha, alpha_temp = blend.cache    # vector containing smoothness indicator values
    @unpack (c, a, amin, a0, a1, smooth_alpha, smooth_factor) = blend.parameters # smoothing coefficients
    @unpack get_indicating_variables! = blend.subroutines
    @unpack cache = blend
    @unpack Pn2m = cache

    @threaded for element_index in element_iterator(grid)
        element = element_indices(element_index, grid)
        el_x, el_y = element[1], element[2]
        un, um, tmp = cache.nodal_modal[Threads.threadid()]
        # Continuous extension to faces
        u = @view u1[:, :, :, el_x, el_y]
        @turbo un .= u

        # Copying is needed because we replace these with variables actually
        # used for indicators like primitives or rho*p, etc.

        # Convert un to ind var, get no. of variables used for indicator
        n_ind_nvar = get_indicating_variables!(un, eq)

        multiply_dimensionwise!(um, Pn2m, un, tmp)

        # ind = zeros(n_ind_nvar)
        ind = 0.0
        # KLUDGE - You are assuming n_ind_var = 1

        for n in 1:n_ind_nvar
            # um[n,1,1] *= constant_node_factor
            # TOTHINK - avoid redundant calculations in total_energy_clip1, 2, etc.?
            total_energy = total_energy_clip1 = total_energy_clip2 = 0.0
            for j in Base.OneTo(nd), i in Base.OneTo(nd) # TOTHINK - Why is @turbo bad here?
                total_energy += um[n, i, j]^2
            end

            for j in Base.OneTo(nd - 1), i in Base.OneTo(nd - 1)
                total_energy_clip1 += um[n, i, j]^2
            end

            for j in Base.OneTo(nd - 2), i in Base.OneTo(nd - 2)
                total_energy_clip2 += um[n, i, j]^2
            end

            total_energy_den = total_energy - um[n, 1, 1]^2 +
                               (constant_node_factor * um[n, 1, 1])^2

            if total_energy > tolE
                ind1 = (total_energy - total_energy_clip1) / total_energy_den
            else
                ind1 = 0.0
            end

            if total_energy_clip1 > tolE
                ind2 = (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
            else
                ind2 = 0.0
            end

            ind = max(ind1, ind2)
        end
        E[el_x, el_y] = maximum(ind) # maximum content among all indicating variables

        T = a * 10^(-c * nd^(0.25))
        # alpha(E=0) = 0.0001
        s = log((1.0 - 0.0001) / 0.0001)  # chosen to that E = 0 => alpha = amin
        alpha[el_x, el_y] = 1.0 / (1.0 + exp((-s / T) * (E[el_x, el_y] - T)))

        if alpha[el_x, el_y] < amin # amin = 0.0001
            alpha[el_x, el_y] = 0.0
        elseif alpha[el_x, el_y] > 1.0 - amin
            alpha[el_x, el_y] = 1.0
        end

        alpha[el_x, el_y] = min(alpha[el_x, el_y], amax)
    end

    top_cell = ny_tuple[2]
    for el_y in 1:top_cell
        el_x = leftmost_physical_element(el_y, grid)
        alpha[el_x - 1, el_y] = alpha[el_x, el_y]
        el_x = nx_tuple[2] # right most physical element
        alpha[el_x + 1, el_y] = alpha[el_x, el_y]
    end

    rightmost_cell = nx_tuple[2]
    for el_x in 1:rightmost_cell
        alpha[el_x, ny_tuple[2] + 1] = alpha[el_x, ny_tuple[2]] # top
        el_y = bottom_physical_element(el_x, grid)
        alpha[el_x, el_y - 1] = alpha[el_x, el_y]
    end

    # Smoothening of alpha
    if smooth_alpha == true
        @turbo alpha_temp .= alpha
        for element_index in element_iterator(grid)
            element = element_indices(element_index, grid)
            el_x, el_y = element[1], element[2]
            alpha[el_x, el_y] = max(smooth_factor * alpha_temp[el_x - 1, el_y],
                                    smooth_factor * alpha_temp[el_x, el_y - 1],
                                    alpha[el_x, el_y],
                                    smooth_factor * alpha_temp[el_x + 1, el_y],
                                    smooth_factor * alpha_temp[el_x, el_y + 1])
        end
    end

    if dt > 0.0
        blend.cache.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
    end

    if t < 1e-6
        alpha .= 1.0
    end

    if limiter.pure_fv == true
        @assert scheme.limiter.name == "blend"
        @turbo alpha .= one(eltype(alpha))
    end

    return nothing
    end # timer
end

# Very sad hack!
function create_auxiliaries(eq, op, grid::StepGrid, problem, scheme, param, cache)
    # Setup plotting
    @unpack u1, ua = cache
    timer = TimerOutput()
    plot_data = initialize_plot(eq, op, grid, problem, scheme, timer, u1, ua)
    # Setup blending limiter
    blend = Tenkai.Blend(eq, op, grid, problem, scheme, param, plot_data)
    hierarchical = Tenkai.Hierarchical(eq, op, grid, problem, scheme, param,
                                       plot_data)
    aux_cache = create_aux_cache(eq, op)
    error_file = open("error.txt", "w")
    aux = (; plot_data, blend,
           hierarchical,
           error_file, timer,
           aux_cache, cache) # named tuple;
    return aux
end

function update_ghost_values_u1!(eq::AbstractEquations{2}, problem, grid::StepGrid, op,
                                 u1, aux, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    nx_tuple, ny_tuple = grid.size
    nd = op.degree + 1
    @unpack xg = op
    nvar = size(u1, 1)

    @unpack cache = aux

    @unpack u1x, u1y = cache

    @turbo u1x .= u1
    @turbo u1y .= u1

    left, right, bottom, top = problem.boundary_condition
    boundary_value = problem.boundary_value
    @unpack dx, dy, xf, yf = grid

    # Left bc is Dirichlet
    x = xf[1]
    for j in (ny_tuple[1] + 1):ny_tuple[2]
        for k in 1:nd
            y = yf[j] + xg[k] * dy[j]
            ub = boundary_value(x, y, t)
            for ix in 1:nd, n in 1:nvar
                u1x[n, ix, k, 0, j] = ub[n]
            end
        end
    end

    # Right bc is outflow
    for el_y in 1:ny_tuple[2], j in 1:nd, i in 1:nd
        for n in eachvariable(eq)
            u1x[n, i, j, nx_tuple[2] + 1, el_y] = u1[n, nd, j, nx_tuple[2], el_y]
        end
    end

    # Horizontal bottom walls are reflective
    for el_x in bottom_horizontal_iterator(grid)
        el_y = bottom_physical_element(el_x, grid)
        for j in 1:nd, i in 1:nd
            for n in 1:nvar
                u1y[n, i, j, el_x, el_y - 1] = u1[n, i, 1, el_x, el_y]
            end
            u1y[3, i, j, el_x, el_y - 1] *= -1.0 # rho * v2
        end
    end

    # Vertical bottom walls are reflective
    for el_y in bottom_vertical_iterator(grid)
        el_x = leftmost_physical_element(el_y, grid)
        for j in 1:nd, i in 1:nd
            for n in 1:nvar
                u1x[n, i, j, el_x - 1, el_y] = u1[n, nd, j, el_x, el_y]
            end
            u1x[2, i, j, el_x - 1, el_y] *= -1.0 # ρ * v1
        end
    end

    # Horizontal walls at top are reflective
    for el_x in 1:nx_tuple[2]
        el_y = ny_tuple[2] # top most physical element
        for j in 1:nd, i in 1:nd
            for n in eachvariable(eq)
                u1y[n, i, j, el_x, el_y + 1] = u1[n, i, nd, el_x, el_y]
            end
            u1y[3, i, j, el_x, el_y + 1] *= -1.0 # ρ * v2
        end
    end
    return nothing
    end # timer
end

function compute_error(problem, grid::StepGrid, eq::AbstractEquations{2}, aux, op, u1, t)
    @timeit aux.timer "Compute error" begin
    #! format: noindent
    @unpack error_file, aux_cache = aux
    @unpack error_cache = aux_cache
    xmin, xmax, ymin, ymax = grid.domain
    @unpack xg = op
    nd = length(xg)

    refresh!(u) = fill!(u, zero(eltype(u)))

    @unpack exact_solution = problem

    @unpack xq, w2d, V, arr_cache = error_cache

    nq = length(xq)

    nx_tuple, ny_tuple = grid.size
    @unpack xc, yc, dx, dy = grid

    l1_error, l2_error, energy = 0.0, 0.0, 0.0
    @inbounds @floop for element in CartesianIndices((1:nx_tuple[2], 1:ny_tuple[2]))
        # for element in CartesianIndices((1:nx, 1:ny))
        el_x, el_y = element[1], element[2]
        ue, un = arr_cache[Threads.threadid()]
        for j in 1:nq, i in 1:nq
            x = xc[el_x] - 0.5 * dx[el_x] + dx[el_x] * xq[i]
            y = yc[el_y] - 0.5 * dy[el_y] + dy[el_y] * xq[j]
            ue_node = exact_solution(x, y, t)
            set_node_vars!(ue, ue_node, eq, i, j)
        end
        u1_ = @view u1[:, :, :, el_x, el_y]
        refresh!(un)
        for j in 1:nd, i in 1:nd
            u_node = get_node_vars(u1_, eq, i, j)
            for jj in 1:nq, ii in 1:nq
                # un = V*u*V', so that
                # un[ii,jj] = ∑_ij V[ii,i]*u[i,j]*V[jj,j]
                multiply_add_to_node_vars!(un, V[ii, i] * V[jj, j], u_node, eq, ii, jj)
            end
        end
        l1 = l2 = e = 0.0
        for j in 1:nq, i in 1:nq
            un_node = get_node_vars(un, eq, i, j)
            ue_node = get_node_vars(ue, eq, i, j) # KLUDGE - allocated ue is not needed
            for n in 1:1 # Only computing error in first conservative variable
                du = abs(un_node[n] - ue_node[n])
                l1 += du * w2d[i, j]
                l2 += du * du * w2d[i, j]
                e += un_node[n]^2 * w2d[i, j]
            end
        end
        l1 *= dx[el_x] * dy[el_y]
        l2 *= dx[el_x] * dy[el_y]
        e *= dx[el_x] * dy[el_y]
        @reduce(l1_error+=l1, l2_error+=l2, energy+=e)
        # l1_error += l1; l2_error += l2; energy += e
    end
    domain_size = (xmax - xmin) * (ymax - ymin)
    l1_error = l1_error / domain_size
    l2_error = sqrt(l2_error / domain_size)
    energy = energy / domain_size
    @printf(error_file, "%.16e %.16e %.16e %.16e\n", t, l1_error[1], l2_error[1],
            energy[1])

    return Dict("l1_error" => l1_error, "l2_error" => l2_error,
                "energy" => energy)
    end # timer
end

function Blend(eq::AbstractEquations{2}, op, grid::StepGrid,
               problem::Problem,
               scheme::Scheme,
               param::Parameters,
               plot_data)
    @unpack limiter = scheme

    if limiter.name != "blend"
        subroutines = (; blend_cell_residual! = trivial_cell_residual,
                       blend_face_residual_x! = trivial_face_residual,
                       blend_face_residual_y! = trivial_face_residual)
        nx_tuple, ny_tuple = grid.size
        cache = (;
                 dt = MVector(1.0e20), # filler
                 alpha = zeros(nx_tuple[2], ny_tuple[2]))
        positivity_blending = NoPositivityBlending()
        parameters = (; positivity_blending)
        # If limiter is not blend, replace blending with 'do nothing functions'
        return (; parameters, subroutines, cache)
    end

    println("Setting up blending limiter...")

    @unpack (blend_type, indicating_variables, reconstruction_variables,
    indicator_model, amax, constant_node_factor,
    smooth_alpha, smooth_factor,
    c, a, amin, tvbM,
    debug_blend, pure_fv, bc_x, positivity_blending) = limiter

    @unpack xc, yc, xf, yf, dx, dy = grid
    nx_tuple, ny_tuple = grid.size
    @unpack degree, xg = op
    # @assert Threads.nthreads() == 1
    @assert indicator_model=="gassner" "Other models not implemented"
    @assert degree > 2 || pure_fv == true
    nd = degree + 1
    @unpack nvar = eq

    E1 = a * 10^(-c * (degree + 3)^0.25)
    E0 = E1 * 1e-2 # E < E0 implies smoothness
    tolE = 1.0e-6  # If denominator < tolE, do purely high order
    a0 = 1.0 / 3.0
    a1 = 1.0 - 2.0 * a0              # smoothing coefficients
    parameters = (; E1, E0, tolE, amax, a0, a1, constant_node_factor,
                  smooth_alpha, smooth_factor,
                  c, a, amin, tvbM,
                  pure_fv, positivity_blending, debug = debug_blend)

    # Big arrays
    E = zeros(nx_tuple[2], ny_tuple[2])
    alpha = OffsetArray(zeros(nx_tuple[2] + 2, ny_tuple[2] + 2), OffsetArrays.Origin(0, 0))
    alpha_temp = similar(alpha)
    fn_low = OffsetArray(zeros(nvar,
                               nd, # Dofs on each face
                               4,  # 4 faces
                               nx_tuple[2] + 2, ny_tuple[2] + 2),
                         OffsetArrays.Origin(1, 1, 1, 0, 0))

    # Small cache of many MMatrix with one copy per thread
    abstract_constructor(tuple_, x, origin) = [OffsetArray(MArray{tuple_, Float64}(x),
                                                           OffsetArrays.Origin(origin))]
    # These square brackets are needed when cache_size = 1. Also, when
    # Cache size is > 1, a [1] is needed in constructor

    constructor = x -> abstract_constructor(Tuple{nvar, nd + 2, nd + 2}, x, (1, 0, 0))
    ue = alloc_for_threads(constructor, 1) # u extended by face extrapolation
    constructor = x -> abstract_constructor(Tuple{nd + 1}, x, (0))[1]
    subcell_faces = alloc_for_threads(constructor, 2) # faces of subcells

    constructor = x -> abstract_constructor(Tuple{nd + 2}, x, (0))[1]
    solution_points = alloc_for_threads(constructor, 2)

    constructor = x -> abstract_constructor(Tuple{nvar, 4, nd + 2, nd + 2}, x,
                                            (1, 1, 0, 0))
    unph = alloc_for_threads(constructor, 1)

    constructor = MArray{Tuple{nvar, nd, nd}, Float64}
    nodal_modal = alloc_for_threads(constructor, 3) # stores un, um and a temp

    constructor = MArray{Tuple{nvar}, Float64}
    slopes = alloc_for_threads(constructor, 2)

    Pn2m = nodal2modal(xg)

    cache = (; alpha, alpha_temp, E, ue,
             subcell_faces,
             solution_points,
             fn_low, dt = MVector(1.0),
             nodal_modal, unph, Pn2m, slopes)

    println("Setting up $blend_type blending limiter with $indicator_model "
            *
            "with $indicating_variables indicating variables")

    @show blend_type
    @unpack cell_residual!, face_residual_x!, face_residual_y! = blend_type
    conservative2recon!, recon2conservative! = reconstruction_variables

    subroutines = (; blend_cell_residual! = cell_residual!,
                   blend_face_residual_x! = face_residual_x!,
                   blend_face_residual_y! = face_residual_y!,
                   get_indicating_variables! = indicating_variables,
                   conservative2recon!, recon2conservative!,
                   bc_x, numflux = scheme.numerical_flux # TODO - This disallows
                   # the user from using a different flux in the blending scheme
                   )

    Blend2D(cache, parameters, subroutines)
end

function blend_cell_residual_muscl!(el_x, el_y, eq::AbstractEquations{2},
                                    scheme, aux, dt, grid::StepGrid, dx, dy, xf, yf, op,
                                    cache, u1, ::Any, f, res)
    @timeit_debug aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack blend = aux
    @unpack tvbM = blend.parameters
    @unpack xg, wg = op
    num_flux = scheme.numerical_flux
    nd = length(xg)
    nvar = nvariables(eq)

    @unpack u1x, u1y = cache
    nx_tuple, ny_tuple = grid.size

    id = Threads.threadid()
    xxf, yyf = blend.cache.subcell_faces[id]
    xe, ye = blend.cache.solution_points[id] # cell points & 2 from neighbours
    ue, = blend.cache.ue[id][1] # solution values in cell + 2 from neighbours
    unph, = blend.cache.unph[id][1] # face values evolved to to time level n+1/2
    dt = blend.cache.dt[1] # For support with DiffEq
    @unpack fn_low = blend.cache
    alpha = blend.cache.alpha[el_x, el_y]

    @unpack conservative2recon!, recon2conservative! = blend.subroutines

    u = @view u1[:, :, :, el_x, el_y]
    ux = @view u1x[:, :, :, el_x, el_y]
    uy = @view u1y[:, :, :, el_y, el_y]
    r = @view res[:, :, :, el_x, el_y]

    # limit the higher order part
    lmul!(1.0 - alpha, r)

    # compute subcell faces
    xxf[0], yyf[0] = xf, yf
    for ii in Base.OneTo(nd)
        xxf[ii] = xxf[ii - 1] + dx * wg[ii]
        yyf[ii] = yyf[ii - 1] + dy * wg[ii]
    end

    # Get solution points
    # xe[0] = xf - dx[el_x-1]*(1.0-xg[nd])
    xe[0] = xf - grid.dx[el_x - 1] * (1.0 - xg[nd])   # Last solution point of left cell
    xe[1:nd] .= xf .+ dx * xg          # Solution points inside the cell
    xe[nd + 1] = xf + grid.dx[el_x] + grid.dx[el_x + 1] * (xg[1]) # First point of right cell

    ye[0] = yf - grid.dy[el_y - 1] * (1.0 - xg[nd]) # Last solution point of lower cell
    ye[1:nd] .= yf .+ dy * xg                         # solution points inside the cell
    ye[nd + 1] = yf + grid.dy[el_y] + grid.dy[el_y + 1] * xg[1]  # First point of upper cell

    # EFFICIENCY - Add @turbo here
    for j in Base.OneTo(nd), i in Base.OneTo(nd), n in eachvariable(eq)
        ue[n, i, j] = u[n, i, j] # values from current cell
    end

    # EFFICIENCY - Add @turbo here
    for k in Base.OneTo(nd), n in eachvariable(eq)
        ue[n, 0, k] = u1x[n, nd, k, el_x - 1, el_y] # values from left neighbour
        ue[n, nd + 1, k] = u1x[n, 1, k, el_x + 1, el_y] # values from right neighbour

        ue[n, k, 0] = u1y[n, k, nd, el_x, el_y - 1] # values from lower neighbour
        ue[n, k, nd + 1] = u1y[n, k, 1, el_x, el_y + 1] # values from upper neighbour
    end

    # Loop over subcells
    for jj in Base.OneTo(nd), ii in Base.OneTo(nd)
        u_ = get_node_vars(ue, eq, ii, jj)
        ul = get_node_vars(ue, eq, ii - 1, jj)
        ur = get_node_vars(ue, eq, ii + 1, jj)
        ud = get_node_vars(ue, eq, ii, jj - 1)
        uu = get_node_vars(ue, eq, ii, jj + 1)

        # TOTHINK - Add this feature
        # u_, ul, ur, ud, uu = conservative2recon.((u_,ul,ur,ud,uu))

        # Compute finite differences
        Δx1, Δx2 = xe[ii] - xe[ii - 1], xe[ii + 1] - xe[ii]
        back_x, cent_x, fwd_x = finite_differences(Δx1, Δx2, ul, u_, ur)
        Δy1, Δy2 = ye[jj] - ye[jj - 1], ye[jj + 1] - ye[jj]
        back_y, cent_y, fwd_y = finite_differences(Δy1, Δy2, ud, u_, uu)

        Mdx2 = tvbM * Δx1
        Mdy2 = tvbM * Δy1

        # Slopes of linear approximation in cell
        # Ideally, I'd name both the tuples as slope_tuple, but that
        # was giving a type instability
        # beta1 = 2.0
        # if blend.parameters.pure_fv == true # KLUDGE - Do this in blend.paramaeters
        beta1, beta2 = 2.0 - alpha, 2.0 - alpha # Unfortunate way to fix type instability
        # else
        #    beta1, beta2 = 2.0 - alpha, 2.0 - alpha
        # end

        slope_tuple_x = (minmod(cent_x[n], back_x[n], fwd_x[n], beta1, Mdx2)
                         for n in eachvariable(eq))

        slope_x = SVector{nvar}(slope_tuple_x)
        # beta2 = 2.00.0
        slope_tuple_y = (minmod(cent_y[n], back_y[n], fwd_y[n], beta2, Mdy2)
                         for n in eachvariable(eq))
        slope_y = SVector{nvar}(slope_tuple_y)
        ufl = u_ + slope_x * (xxf[ii - 1] - xe[ii]) # left  face value u_{i-1/2,j}
        ufr = u_ + slope_x * (xxf[ii] - xe[ii]) # right face value u_{i+1/2,j}

        ufd = u_ + slope_y * (yyf[jj - 1] - ye[jj]) # lower face value u_{i,j-1/2}
        ufu = u_ + slope_y * (yyf[jj] - ye[jj]) # upper face value u_{i,j+1/2}

        # KLUDGE - u_star's are not needed in this function, just create and use them
        # in limit_slope

        u_star_l = u_ + 2.0 * slope_x * (xxf[ii - 1] - xe[ii])
        u_star_r = u_ + 2.0 * slope_x * (xxf[ii] - xe[ii])

        u_star_d = u_ + 2.0 * slope_y * (yyf[jj - 1] - ye[jj])
        u_star_u = u_ + 2.0 * slope_y * (yyf[jj] - ye[jj])

        ufl, ufr = limit_slope(eq, slope_x, ufl, u_star_l, ufr, u_star_r, u_,
                               xxf[ii - 1] - xe[ii], xxf[ii] - xe[ii])
        ufd, ufu = limit_slope(eq, slope_y, ufd, u_star_d, ufu, u_star_u, u_,
                               yyf[jj - 1] - ye[jj], yyf[jj] - ye[jj])

        # TOTHINK - Add this feature
        # Convert back to conservative variables for update
        # ufl, ufr, ufd, ufu = recon2conservative.((ufl,ufr,ufd,ufu))

        fl = flux(xxf[ii - 1], ye[jj], ufl, eq, 1)
        fr = flux(xxf[ii], ye[jj], ufr, eq, 1)

        gd = flux(xe[ii], yyf[jj - 1], ufd, eq, 2)
        gu = flux(xe[ii], yyf[jj], ufu, eq, 2)

        # Use finite difference method to evolve face values to time level n+1/2

        multiply_add_set_node_vars!(unph, # u_{i-1/2+,j}=u_{i-1/2,j}-0.5*dt*(fr-fl)/(xfr-xfl)
                                    ufl,  # u_{i-1/2,j}
                                    -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                    fr,
                                    -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                    -fl,
                                    eq,
                                    1, # Left face
                                    ii, jj)
        multiply_add_set_node_vars!(unph, # u_{i+1/2-,j}=u_{i+1/2,j}-0.5*dt*(fr-fl)/(xfr-xfl)
                                    ufr,  # u_{i+1/2,j}
                                    -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                    fr,
                                    -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                    -fl,
                                    eq,
                                    2, # Right face
                                    ii, jj)
        multiply_add_set_node_vars!(unph, # u_{i,j-1/2+}=u_{i,j-1/2}-0.5*dt*(gu-gd)/(yfu-yfd)
                                    ufd,  # u_{i,j-1/2}
                                    -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                    gu,
                                    -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                    -gd,
                                    eq,
                                    3, # Bottom face
                                    ii, jj)
        multiply_add_set_node_vars!(unph, # u_{i,j+1/2-}=u_{i,j+1/2}-0.5*dt*(gu-gd)/(yfu-yfd)
                                    ufu,  # u_{i,j+1/2}
                                    -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                    gu,
                                    -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                    -gd,
                                    eq,
                                    4, # Top face
                                    ii, jj)

        multiply_add_to_node_vars!(unph, # u_{i-1/2+,j}=u_{i,j-1/2}-0.5*dt*(gu-gd)/(yfu-yfd)
                                   -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                   gu,
                                   -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                   -gd,
                                   eq,
                                   1, # Left face
                                   ii, jj)
        multiply_add_to_node_vars!(unph, # u_{i+1/2+,j}=u_{i+1/2,j}-0.5*dt*(gu-gd)/(yfu-yfd)
                                   -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                   gu,
                                   -0.5 * dt / (yyf[jj] - yyf[jj - 1]),
                                   -gd,
                                   eq,
                                   2, # Right face
                                   ii, jj)
        multiply_add_to_node_vars!(unph, # u_{i-1/2+,j}=u_{i-1/2,j}-0.5*dt*(fr-fl)/(xfr-xfl)
                                   -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                   fr,
                                   -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                   -fl,
                                   eq,
                                   3, # Bottom face
                                   ii, jj)
        multiply_add_to_node_vars!(unph, # u_{i-1/2+,j}=u_{i-1/2,j}-0.5*dt*(fr-fl)/(xfr-xfl)
                                   -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                   fr,
                                   -0.5 * dt / (xxf[ii] - xxf[ii - 1]),
                                   -fl,
                                   eq,
                                   4, # Top face
                                   ii, jj)
    end

    # Now we loop over faces and perform the update

    # loop over vertical inner faces between (ii-1,jj) and (ii,jj)
    for ii in 2:nd # Supercell faces will be done in blend_face_residual
        xx = xxf[ii - 1] # face x coordinate, offset because index starts from 0
        for jj in Base.OneTo(nd)
            yy = yf + dy * xg[jj] # y coordinate same as solution point
            X = SVector(xx, yy)
            ul = get_node_vars(unph, eq, 2, ii - 1, jj)
            ur = get_node_vars(unph, eq, 1, ii, jj)
            fl, fr = flux(xx, yy, ul, eq, 1), flux(xx, yy, ur, eq, 1)
            local fn
            try
                fn = num_flux(X, ul, ur, fl, fr, ul, ur, eq, 1)
            catch e
                isa(e, DomainError)
                @show el_x, el_y
                @show ii, jj
                @show ul, ur
                rethrow(e)
            end
            multiply_add_to_node_vars!(r, # r[ii-1,jj] += dt/(dx*wg[ii-1])*fn
                                       alpha * dt / (dx * wg[ii - 1]),
                                       fn,
                                       eq, ii - 1, jj)
            multiply_add_to_node_vars!(r, # r[ii,jj] += dt/(dx*wg[ii])*fn
                                       -alpha * dt / (dx * wg[ii]),
                                       fn,
                                       eq, ii, jj)
            # TOTHINK - Can checking this at every iteration be avoided?
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
            xx = xf + dx * xg[ii] # face x coordinate picked same as the soln point
            X = SVector(xx, yy)
            ud = get_node_vars(unph, eq, 4, ii, jj - 1)
            uu = get_node_vars(unph, eq, 3, ii, jj)
            gd, gu = flux(xx, yy, ud, eq, 2), flux(xx, yy, uu, eq, 2)
            gn = num_flux(X, ud, uu, gd, gu, ud, uu, eq, 2)
            multiply_add_to_node_vars!(r, # r[ii,jj-1]+=alpha*dt/(dy*wg[jj-1])*gn
                                       alpha * dt / (dy * wg[jj - 1]),
                                       gn,
                                       eq, ii, jj - 1)
            multiply_add_to_node_vars!(r, # r[ii,jj]-=alpha*dt/(dy*wg[jj])*gn
                                       -alpha * dt / (dy * wg[jj]),
                                       gn,
                                       eq, ii, jj)
            # TOTHINK - Can checking this at every iteration be avoided
            if jj == 2
                set_node_vars!(fn_low, gn, eq, ii, 3, el_x, el_y)
            elseif jj == nd
                set_node_vars!(fn_low, gn, eq, ii, 4, el_x, el_y)
            end
        end
    end
    end # timer
end

function correct_variable!(eq::AbstractEquations{2}, variable, op, aux,
                           grid::StepGrid, u1, ua, eps_ = 1e-12)
    @unpack Vl, Vr = op
    @unpack aux_cache = aux
    @unpack bound_limiter_cache = aux_cache
    @unpack xc, yc = grid
    nx_tuple, ny_tuple = grid.size
    nd = op.degree + 1

    var_min_avg = 1e20
    for el_y in 1:ny_tuple[2], el_x in 1:nx_tuple[2]
        ua_ = get_node_vars(ua, eq, el_x, el_y)
        var_min_avg = min(var_min_avg, variable(eq, ua_))
        if var_min_avg < 0.0
            @show variable
            println("Positivity limiter failed in element", el_x, " ", el_y,
                    "with centre ", xc[el_x], ", ", yc[el_y])
            throw(DomainError((var_min_avg)))
        end
    end

    eps = 1e-6

    variable_(u_) = variable(eq, u_)

    refresh!(u) = fill!(u, zero(eltype(u)))
    @threaded for element in CartesianIndices((1:nx_tuple[2], 1:ny_tuple[2]))
        el_x, el_y = element[1], element[2]
        # var_ll, var_rr, var_dd, var_uu = aux_cache.bound_limiter_cache[Threads.threadid()]
        # @views var_ll, var_rr, var_dd, var_uu = (a[1,:] for a in
        #                                           bound_limiter_cache[Threads.threadid()])
        u_ll, u_rr, u_dd, u_uu = aux_cache.bound_limiter_cache[Threads.threadid()]
        u1_ = @view u1[:, :, :, el_x, el_y]

        ua_ = get_node_vars(ua, eq, el_x, el_y)
        var_avg = variable(eq, ua_)

        # Correct density

        # Loop to find minimum over inner points and
        # perform extrapolation to non-corner face points
        # TOTEST - This is not correct. Extrapolating and then computing and var
        # is different from computing var. Thus, admissibility of one need
        # not imply the admissibility of the other. The former seems
        # more right to do
        var_min = 1e20
        refresh!.((u_ll, u_rr, u_dd, u_uu))
        # refresh!.((var_ll, var_rr, var_dd, var_uu))
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_node = get_node_vars(u1_, eq, i, j)
            var = variable(eq, u_node)
            # var_ll[j] += var * Vl[i]
            # var_rr[j] += var * Vr[i]
            # var_dd[i] += var * Vl[j]
            # var_uu[i] += var * Vr[j]
            var_min = min(var, var_min)

            # Perform extrapolations to face
            multiply_add_to_node_vars!(u_ll, Vl[i], u_node, eq, j)
            multiply_add_to_node_vars!(u_rr, Vr[i], u_node, eq, j)
            multiply_add_to_node_vars!(u_dd, Vl[j], u_node, eq, i)
            multiply_add_to_node_vars!(u_uu, Vr[j], u_node, eq, i)
        end

        # Now to get the complete minimum, compute var_min at face values as well
        for k in Base.OneTo(nd)
            u_ll_node = get_node_vars(u_ll, eq, k)
            u_rr_node = get_node_vars(u_rr, eq, k)
            u_dd_node = get_node_vars(u_dd, eq, k)
            u_uu_node = get_node_vars(u_uu, eq, k)
            var_ll, var_rr, var_dd, var_uu = variable_.((u_ll_node, u_rr_node,
                                                         u_dd_node, u_uu_node))
            var_min = min(var_min, var_ll, var_rr, var_dd, var_uu)
            # var_min = min(var_min, var_ll[k], var_rr[k], var_dd[k], var_uu[k])
        end

        # # Now, we get minimum corner points (NEED TO BE DECIDED)
        # uld = ulu = urd = uru = 0.0
        # for j in Base.OneTo(nd)
        #    uld += 0.5 * (Vl[j] * ul[j] + Vl[j] * ud[j])
        #    ulu += 0.5 * (Vr[j] * ul[j] + Vl[j] * uu[j])
        #    urd += 0.5 * (Vl[j] * ur[j] + Vr[j] * ud[j])
        #    uru += 0.5 * (Vr[j] * ur[j] + Vr[j] * uu[j])
        # end
        # ρ_min = min(ρ_min, uld, ulu, urd, uru)

        theta = 1.0
        if var_min < eps
            ratio = abs(eps - var_avg) / (abs(var_min - var_avg) + 1e-13)
            theta = min(ratio, 1.0)
        end

        if theta < 1.0
            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                u_node = get_node_vars(u1_, eq, i, j)
                multiply_add_set_node_vars!(u1_,
                                            theta, u_node,
                                            1.0 - theta, ua_,
                                            eq, i, j)
            end
        end
    end
end

function update_ghost_values_cRK!(problem, scheme, eq::AbstractEquations{2},
                                  grid::StepGrid, aux, op, cache, t, dt, scaling_factor = 1)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, Ub = cache
    update_ghost_values_periodic!(eq, problem, Fb, Ub)

    @unpack periodic_x, periodic_y = problem
    if periodic_x && periodic_y
        return nothing
    end

    nx_tuple, ny_tuple = grid.size
    nvar = nvariables(eq)
    @unpack degree, xg, wg = op
    nd = degree + 1
    @unpack dx, dy, xf, yf = grid
    @unpack boundary_condition, boundary_value = problem
    left, right, bottom, top = boundary_condition

    refresh!(u) = fill!(u, 0.0)

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    # Dirichlet bc at left
    pre_allocated = [(zeros(nvar) for _ in 1:2) for _ in 1:Threads.nthreads()]
    for j in 1:ny_tuple[2]
        x = xf[1]
        for k in 1:nd
            y = yf[j] + xg[k] * dy[j]
            # KLUDGE - Don't allocate so much!
            ub, fb = pre_allocated[Threads.threadid()]
            for l in 1:nd
                tq = t + xg[l] * dt
                ubvalue = boundary_value(x, y, tq)
                fbvalue = flux(x, y, ubvalue, eq, 1)
                for n in 1:nvar
                    ub[n] += ubvalue[n] * wg[l]
                    fb[n] += fbvalue[n] * wg[l]
                end
            end
            for n in 1:nvar
                Ub[n, k, 1, 1, j] = Ub[n, k, 2, 0, j] = ub[n] # upwind
                Fb[n, k, 1, 1, j] = Fb[n, k, 2, 0, j] = fb[n] # upwind
            end
        end
    end

    # Right bc are neumann
    for j in 1:ny_tuple[2]
        el_x = nx_tuple[2] # right most element
        for k in 1:nd, n in eachvariable(eq)
            Ub[n, k, 1, el_x + 1, j] = Ub[n, k, 2, el_x, j]
            Fb[n, k, 1, el_x + 1, j] = Fb[n, k, 2, el_x, j]
        end
    end

    # Horizontal bottom walls are reflective
    for el_x in bottom_horizontal_iterator(grid)
        el_y = bottom_physical_element(el_x, grid)
        for k in 1:nd
            for n in 1:nvar
                Ub[n, k, 4, el_x, el_y - 1] = Ub[n, k, 3, el_x, el_y]
                Fb[n, k, 4, el_x, el_y - 1] = Fb[n, k, 3, el_x, el_y]
            end
            Ub[3, k, 4, el_x, el_y - 1] *= -1.0 # rho * v2
            Fb[1, k, 4, el_x, el_y - 1] *= -1.0 # rho * v2
            Fb[2, k, 4, el_x, el_y - 1] *= -1.0 # rho * v1 * v2
            Fb[4, k, 4, el_x, el_y - 1] *= -1.0 # (rho*e + p) * v1
        end
    end

    # Vertical walls at bottom are reflective
    for el_y in bottom_vertical_iterator(grid)
        el_x = leftmost_physical_element(el_y, grid)
        for k in 1:nd
            for n in 1:nvar
                Ub[n, k, 1, el_x - 1, el_y] = Ub[n, k, 2, el_x, el_y]
                Fb[n, k, 1, el_x - 1, el_y] = Fb[n, k, 2, el_x, el_y]
            end
            Ub[2, k, 1, el_x - 1, el_y] *= -1.0 # ρ*u1
            Fb[1, k, 1, el_x - 1, el_y] *= -1.0 # ρ*u1
            Fb[3, k, 1, el_x - 1, el_y] *= -1.0 # ρ*u1*u2
            Fb[4, k, 1, el_x - 1, el_y] *= -1.0 # (ρ_e + p) * u1
        end
    end

    # Horizontal walls at top are reflective
    for el_x in 1:nx_tuple[2]
        el_y = ny_tuple[2] # top most physical element
        for k in 1:nd
            for n in eachvariable(eq)
                Ub[n, k, 3, el_x, el_y + 1] = Ub[n, k, 4, el_x, el_y]
                Fb[n, k, 3, el_x, el_y + 1] = Fb[n, k, 4, el_x, el_y]
            end
            Ub[3, k, 3, el_x, el_y + 1] *= -1.0 # ρ*v2
            Fb[1, k, 3, el_x, el_y + 1] *= -1.0 # ρ*v1*v2
            Fb[2, k, 3, el_x, el_y + 1] *= -1.0 # ρ*v1*v2
            Fb[4, k, 3, el_x, el_y + 1] *= -1.0 # (E+p)*v2
        end
    end

    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    return nothing
    end # timer
end

import Tenkai: compute_cell_residual_cRK!
function compute_cell_residual_cRK!(eq::AbstractEquations{2}, grid::StepGrid, op,
                                    problem, scheme::Scheme{<:cRK44}, aux, t,
                                    dt, cache)
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
    @unpack eval_data, cell_arrays, ua, u1, res, Fb, Ub = cache

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @threaded for element_index in element_iterator(grid) # Loop over cells
        element = element_indices(element_index, grid)
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx, lamy = dt / dx, dt / dy

        u1_ = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        Ub_ = @view Ub[:, :, :, el_x, el_y]

        id = Threads.threadid()
        u2, u3, u4, F, G, U, S = cell_arrays[id]

        u2 .= u1_
        u3 .= u1_
        u4 .= u1_

        # Solution points
        for j in 1:nd, i in 1:nd
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy
            u_node = get_node_vars(u1_, eq, i, j)
            flux1, flux2 = flux(x, y, u_node, eq)
            set_node_vars!(F, flux1 / 6.0, eq, i, j)
            set_node_vars!(G, flux2 / 6.0, eq, i, j)
            set_node_vars!(U, u_node / 6.0, eq, i, j)
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

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy

            u2_node = get_node_vars(u2, eq, i, j)

            flux1, flux2 = flux(x, y, u2_node, eq)

            multiply_add_to_node_vars!(F, 1.0 / 3.0, flux1, eq, i, j)
            multiply_add_to_node_vars!(G, 1.0 / 3.0, flux2, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 3.0, u2_node, eq, i, j)

            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(u3, -0.5 * lamx * Dm[ii, i], flux1, eq,
                                           ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(u3, -0.5 * lamy * Dm[jj, j], flux2, eq,
                                           i, jj)
            end
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy

            u3_node = get_node_vars(u3, eq, i, j)

            flux1, flux2 = flux(x, y, u3_node, eq)

            multiply_add_to_node_vars!(F, 1.0 / 3.0, flux1, eq, i, j)
            multiply_add_to_node_vars!(G, 1.0 / 3.0, flux2, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 3.0, u3_node, eq, i, j)

            for ii in Base.OneTo(nd)
                # ut              += -lam * D * f for each variable
                # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
                multiply_add_to_node_vars!(u4, -lamx * Dm[ii, i], flux1, eq, ii, j)
            end
            for jj in Base.OneTo(nd)
                # C += -lam*g*Dm' for each variable
                # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
                multiply_add_to_node_vars!(u4, -lamy * Dm[jj, j], flux2, eq, i, jj)
            end
        end

        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[i] * dx
            y = yc - 0.5 * dy + xg[j] * dy

            u4_node = get_node_vars(u4, eq, i, j)

            flux1, flux2 = flux(x, y, u4_node, eq)

            multiply_add_to_node_vars!(F, 1.0 / 6.0, flux1, eq, i, j)
            multiply_add_to_node_vars!(G, 1.0 / 6.0, flux2, eq, i, j)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, u4_node, eq, i, j)

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
        blend_cell_residual!(el_x, el_y, eq, scheme, aux, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, cache, u1, u, nothing,
                             res)
        # Interpolate to faces
        @views cell_data = (u1_, u2, u3, u4, el_x, el_y)
        @views compute_bflux!(eq, scheme, grid, cell_data, eval_data, xg, Vl, Vr,
                              F, G, Fb[:, :, :, el_x, el_y], aux)
    end
    end # timer
end

function save_solution_file(u_, time, dt, iter,
                            mesh::StepGrid,
                            equations, op,
                            element_variables = Dict{Symbol, Any}();
                            system = "")
    # Filename without extension based on current time step
    output_directory = "output"
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%06d.h5", iter))
    else
        filename = joinpath(output_directory, @sprintf("solution_%s_%06d.h5", system, iter))
    end

    solution_variables(u) = con2prim(equations, u) # For broadcasting

    nx_tuple, ny_tuple = mesh.size
    u = @view u_[:, :, :, 1:nx_tuple[2], 1:ny_tuple[2]] # Don't plot ghost cells

    # Convert to different set of variables if requested
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    # OffsetArray(reinterpret(eltype(ua), con2prim_.(reinterpret(SVector{nvariables(equation), eltype(ua)}, ua))))
    u_static_reinter = reinterpret(SVector{nvariables(equations), eltype(u)}, u)
    data = Array(reinterpret(eltype(u), solution_variables.(u_static_reinter)))

    # Find out variable count by looking at output from `solution_variables` function
    n_vars = size(data, 1)

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = 2
        attributes(file)["equations"] = "2D Euler Equations"
        attributes(file)["polydeg"] = op.degree
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nx_tuple[2] * ny_tuple[2]
        attributes(file)["mesh_type"] = "StructuredMesh" # For Trixi2Vtk
        attributes(file)["mesh_file"] = "mesh.h5"
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = iter

        # Store each variable of the solution data
        var_names = ("Density", "Velocity x", "Velocity y", "Pressure", "Reactant mass")
        for v in 1:n_vars
            # Convert to 1D array
            file["variables_$v"] = vec(data[v, .., :])

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = var_names[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            file["element_variables_$v"] = element_variable

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end
