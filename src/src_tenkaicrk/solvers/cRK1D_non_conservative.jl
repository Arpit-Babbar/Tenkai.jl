import Tenkai: compute_face_residual!, compute_cell_residual_cRK!

using Tenkai: cRKSolver

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

        Fn, blend_fac = blend.blend_face_residual!(i, x, u1, ua, eq, t, dt, grid,
                                                   op, problem,
                                                   scheme, param, Fn, aux, nothing,
                                                   res, scaling_factor)

        # TODO - Bul, Bur are not being limited. Fix this and make it as in the 2-D version!
        Fl = Fn + Bul
        Fr = Fn + Bur
        for ix in 1:nd
            for n in 1:nvariables(eq)
                res[n, ix, i - 1] += dt / dx[i - 1] * blend_fac[1] * Fl[n] * br[ix]
                res[n, ix, i] += dt / dx[i] * blend_fac[2] * Fr[n] * bl[ix]
            end
        end
    end
    return nothing
    end # timer
end # compute_face_residual!

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

    # A struct containing information about the non-conservative part of the equation
    eq_nc = non_conservative_equation(eq)

    @unpack cell_data, eval_data, ub_N, ua, u1, res, Fb, Ub, Bb = cache

    F, f, U, u2, u_non_cons_x_, u2_non_cons_x_, S = cell_data[Threads.threadid()]

    u_non_cons_x = @view u_non_cons_x_[1:1, 1:nd]
    u2_non_cons_x = @view u2_non_cons_x_[1:1, 1:nd]

    refresh!.((res, Ub, Fb, ub_N, Bb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]
        refresh!.((u_non_cons_x, u2_non_cons_x))

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u2, -0.5 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Compute the contribution of non-conservative equation
        for ix in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[ix] * dx
            u_node = get_node_vars(u1, eq, ix, cell)
            # Compute flux at all solution points
            u_non_cons = calc_non_cons_gradient(u_node, x_, t, eq) # u_non_cons = SVector(u_node[1])
            for iix in Base.OneTo(nd) # Computes derivative in reference coordinates
                multiply_add_to_node_vars!(u_non_cons_x, Dm[iix, ix], u_non_cons,
                                           eq_nc, iix)
            end
        end

        # Add Bu_x and source terms contribution to u2
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            # cache_node = get_cache_node(cache, i, cell)
            u_node = get_node_vars(u1, eq, i, cell)
            u_nc_x_node = get_node_vars(u_non_cons_x, eq_nc, i)
            Bu_x = calc_non_cons_Bu(u_node, u_nc_x_node, x_, t, eq)
            multiply_add_to_node_vars!(u2, -0.5 * lamx, Bu_x, eq, i)
            u_node = get_node_vars(u1, eq, i, cell)
            s_node = calc_source(u_node, x_, t, source_terms, eq)
            # s_node = calc_source(u_node, cache_node, x_, t, source_terms, eq)
            multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i)
        end

        # Compute the contribution of non-conservative equation
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)
            # Compute flux at all solution points
            u2_non_cons = calc_non_cons_gradient(u2_node, x_, t, eq)
            for ii in Base.OneTo(nd) # Computes derivative in reference coordinates
                multiply_add_to_node_vars!(u2_non_cons_x, Dm[ii, i], u2_non_cons,
                                           eq_nc, ii)
            end
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)

            flux1 = flux(x_, u2_node, eq)
            B_node = calc_non_cons_B(u2_node, x_, t, eq)

            multiply_add_to_node_vars!(Bb, Vl[i], B_node, eq, eq_nc, 1, cell)
            multiply_add_to_node_vars!(Bb, Vr[i], B_node, eq, eq_nc, 2, cell)

            u2_non_cons_x_node = get_node_vars(u2_non_cons_x, eq_nc, i)
            Bu2_x = calc_non_cons_Bu(u2_node, u2_non_cons_x_node, x_, t, eq)
            multiply_add_to_node_vars!(res, lamx, Bu2_x, eq, i, cell)

            set_node_vars!(F, flux1, eq, i)
            set_node_vars!(U, u2_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end

            s2_node = calc_source(u2_node, x_, t + 0.5 * dt, source_terms, eq)

            S_node = s2_node # S array is not needed in this degree N = 2 case

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]

        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_dissipation_node_vars(u, U, eq, i)
            u2_node = get_node_vars(u2, eq, i)
            multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, 1, cell)
            multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, 2, cell)

            multiply_add_to_node_vars!(ub_N, Vl[i], u2_node, eq, 1, cell)
            multiply_add_to_node_vars!(ub_N, Vr[i], u2_node, eq, 2, cell)
        end

        local_grid = (xc, dx, lamx, t, dt)
        Ub_ = @view Ub[:, :, cell]
        Bb_to_res!(eq, local_grid, op, Ub_, r)

        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)

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
