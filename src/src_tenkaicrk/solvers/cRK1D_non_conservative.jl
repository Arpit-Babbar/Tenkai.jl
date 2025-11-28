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
