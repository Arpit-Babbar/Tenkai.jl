import ..Tenkai: setup_arrays_rkfr,
                 compute_cell_residual_rkfr!,
                 update_ghost_values_rkfr!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function setup_arrays_rkfr(grid, scheme, eq::AbstractNonConservativeEquations{1})
    function gArray(nvar, nx)
        OffsetArray(zeros(Float64, nvar, nx + 2),
                    OffsetArrays.Origin(1, 0))
    end
    function gArray(nvar, n1, nx)
        OffsetArray(zeros(Float64, nvar, n1, nx + 2),
                    OffsetArrays.Origin(1, 1, 0))
    end
    # Allocate memory
    @unpack degree = scheme
    nd = degree + 1
    nx = grid.size
    nvar = eq.nvar
    u0 = gArray(nvar, nd, nx) # ghost indices not needed, only for copyto!
    u1 = gArray(nvar, nd, nx) # ghost indices needed for blending limiter
    ua = gArray(nvar, nx)
    res = gArray(nvar, nd, nx)
    Fb = gArray(nvar, 2, nx)
    ub = gArray(nvar, 2, nx)
    ub_N = gArray(nvar, 2, nx) # The final stage of cRK before communication

    cache = (; u0, u1, ua, res, Fb, ub, ub_N)
    return cache
end

function update_ghost_values_rkfr!(problem, scheme,
                                   eq::AbstractNonConservativeEquations{1},
                                   grid, aux, op, cache, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, ub_N = cache
    Ub = cache.ub
    update_ghost_values_periodic!(eq, problem, Fb, Ub)
    dt_dummy = 0.0
    update_ghost_values_u1!(eq, problem, grid, op, cache.u1, aux, t)
    update_ghost_values_ub_N!(problem, scheme, eq, grid, aux, op, cache, t,
                              dt_dummy)

    if problem.periodic_x
        return nothing
    end
    nx = grid.size
    xf = grid.xf
    nvar = eq.nvar
    left, right = problem.boundary_condition
    @unpack boundary_value = problem

    # ub = zeros(nvar)
    # fb = zeros(nvar)
    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    if left == dirichlet
        x = xf[1]
        ub = boundary_value(x, t)
        fb = flux(x, ub, eq)
        for n in 1:nvar
            Ub[n, 1, 1] = Ub[n, 2, 0] = ub[n]    # upwind
            Fb[n, 1, 1] = Fb[n, 2, 0] = fb[n]    # upwind
        end
    elseif left == neumann
        for n in 1:nvar
            Ub[n, 2, 0] = Ub[n, 1, 1]
            ub_N[n, 2, 0] = ub_N[n, 1, 1]
            Fb[n, 2, 0] = Fb[n, 1, 1]
        end
    elseif left == reflect
        # velocity reflected back in opposite direction and density is same
        for n in 1:nvar
            Ub[n, 2, 0] = Ub[n, 1, 1]
            Fb[n, 2, 0] = Fb[n, 1, 1]
        end
        Ub[2, 2, 0] = -Ub[2, 2, 0] # velocity reflected back
        Fb[1, 2, 0], Fb[3, 2, 0] = -Fb[1, 2, 0], -Fb[3, 2, 0] # vel multiple term
    else
        println("Incorrect bc specified at left.")
        @assert false
    end

    if right == dirichlet
        x = xf[nx + 1]
        ub = boundary_value(x, t)
        fb = flux(x, ub, eq)
        for n in 1:nvar
            Ub[n, 2, nx] = Ub[n, 1, nx + 1] = ub[n] # upwind
            Fb[n, 2, nx] = Fb[n, 1, nx + 1] = fb[n] # upwind
        end
    elseif right == neumann
        for n in 1:nvar
            Ub[n, 1, nx + 1] = Ub[n, 2, nx]
            ub_N[n, 1, nx + 1] = ub_N[n, 2, nx]
            Fb[n, 1, nx + 1] = Fb[n, 2, nx]
        end
    elseif right == reflect
        # velocity reflected back in opposite direction and density is same
        for n in 1:nvar
            Ub[n, 1, nx + 1] = Ub[n, 2, nx]
            Fb[n, 1, nx + 1] = Fb[n, 2, nx]
        end
        Ub[2, 1, nx + 1] = -Ub[2, 1, nx + 1] # velocity reflected back
        Fb[1, 1, nx + 1], Fb[3, 1, nx + 1] = (-Fb[1, 1, nx + 1],
                                              -Fb[3, 1, nx + 1]) # vel multiple term
    else
        println("Incorrect bc specified at right.")
        @assert false
    end

    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    return nothing
    end # timer
end

function compute_cell_residual_rkfr!(eq::AbstractNonConservativeEquations{1}, grid, op,
                                     problem,
                                     scheme::Scheme{<:String}, aux, t, dt, u1, res, Fb,
                                     ub, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, D1, Vl, Vr, Dm = op
    @unpack blend = aux
    @unpack ub_N = cache
    nx = grid.size
    nd = length(xg)
    @unpack bflux_ind = scheme.bflux
    refresh!(u) = fill!(u, 0.0)
    eq_nc = non_conservative_equation(eq)

    refresh!.((ub, Fb, res, ub_N))
    nvar = nvariables(eq)
    f = zeros(nvar, nd)
    u_non_cons_x = zeros(1, nd)
    @timeit aux.timer "Cell loop" begin
    #! format: noindent
    @inbounds for cell in 1:nx
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u_non_cons_x .= zero(u_non_cons_x)
        xl, xr = grid.xf[cell], grid.xf[cell + 1]
        for ix in Base.OneTo(nd)
            x = xc - 0.5 * dx + xg[ix] * dx
            u_node = get_node_vars(u1, eq, ix, cell)
            u_non_cons = calc_non_cons_gradient(u_node, x, t, eq)
            for iix in 1:nd
                multiply_add_to_node_vars!(u_non_cons_x, Dm[iix, ix],
                                           u_non_cons,
                                           eq_nc, iix)
            end
        end

        for ix in Base.OneTo(nd)
            # Solution points
            x = xc - 0.5 * dx + xg[ix] * dx
            u_node = get_node_vars(u1, eq, ix, cell)
            u_non_cons_x_node = calc_non_cons_gradient(u_non_cons_x, x, t, eq)
            Bu_x = calc_non_cons_Bu(u_node, u_non_cons_x_node, x, t, eq)
            multiply_add_to_node_vars!(res, lamx, Bu_x, eq, ix, cell)
            # Compute flux at all solution points
            flux1 = flux(x, u_node, eq)
            set_node_vars!(f, flux1, eq, ix)
            # KLUDGE - Remove dx, xf arguments. just pass grid and i
            for iix in 1:nd
                multiply_add_to_node_vars!(res, lamx * D1[iix, ix], flux1, eq,
                                           iix, cell)
            end
            multiply_add_to_node_vars!(ub, Vl[ix], u_node, eq, 1, cell)
            multiply_add_to_node_vars!(ub, Vr[ix], u_node, eq, 2, cell)
            multiply_add_to_node_vars!(ub_N, Vl[ix], u_node, eq, 1, cell)
            multiply_add_to_node_vars!(ub_N, Vr[ix], u_node, eq, 2, cell)

            # Source term contribution
            s_node = calc_source(u_node, x, t, source_terms, eq)
            multiply_add_to_node_vars!(res, -dt, s_node, eq, ix, cell)

            if bflux_ind == extrapolate
                multiply_add_to_node_vars!(Fb, Vl[ix], flux1, eq, 1, cell)
                multiply_add_to_node_vars!(Fb, Vr[ix], flux1, eq, 2, cell)
            else
                ubl, ubr = get_node_vars(ub, eq, 1, cell),
                           get_node_vars(ub, eq, 2, cell)
                fbl, fbr = flux(xl, ubl, eq), flux(xr, ubr, eq)
                set_node_vars!(Fb, fbl, eq, 1, cell)
                set_node_vars!(Fb, fbr, eq, 2, cell)
            end
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]

        ub_ = @view ub[:, :, cell]
        local_grid = (xc, dx, lamx, t, dt)
        Bb_to_res!(eq, local_grid, op, ub_, r)

        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt,
                                   dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
    end
    end # timer
    return nothing
    end # timer
end

function compute_face_residual!(eq::AbstractNonConservativeEquations{1},
                                grid,
                                op, cache,
                                problem, scheme::Scheme{<:String},
                                param, aux, t, dt,
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

        Fl = Fn + Bul
        Fr = Fn + Bur

        (Fl, Fr), blend_fac = blend.blend_face_residual!(i, x, u1, ua, eq, t, dt,
                                                         grid,
                                                         op, problem,
                                                         scheme, param, Fl, Fr, aux,
                                                         nothing,
                                                         res, scaling_factor)
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
end # muladd
