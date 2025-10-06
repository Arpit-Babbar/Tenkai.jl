# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#-------------------------------------------------------------------------------
# Choose cfl based on degree and correction function
#-------------------------------------------------------------------------------
function get_cfl(eq::AbstractEquations{1}, scheme::Scheme{<:cRKSolver}, param)
    @unpack solver, degree, correction_function = scheme
    @unpack cfl_safety_factor, cfl_style = param
    @unpack dissipation = scheme
    @assert (degree >= 0&&degree < 5) "Invalid degree"
    os_vector(v) = OffsetArray(v, OffsetArrays.Origin(0))
    local cfl_radau, cfl_g2
    if dissipation == get_second_node_vars || dissipation isa DCSX
        cfl_radau = os_vector([1.0, 0.333, 0.170, 0.103, 0.069])
        cfl_g2 = os_vector([1.0, 1.000, 0.333, 0.170, 0.103])
    elseif dissipation == get_first_node_vars
        cfl_radau = os_vector([1.0, 0.226, 0.117, 0.072, 0.049])
        cfl_g2 = os_vector([1.0, 0.465, 0.204, 0.116, 0.060])
    end
    # Reduce this cfl by a small amount
    if correction_function == "radau"
        return cfl_safety_factor * cfl_radau[degree]
    elseif correction_function == "g2"
        return cfl_safety_factor * cfl_g2[degree]
    else
        println("get_cfl: unknown correction function")
        @assert false
    end
end

function prolong_solution_to_face_and_ghosts!(u1, cache, eq::AbstractEquations{1}, grid,
                                              op,
                                              problem, scheme, aux, t, dt)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    nx = grid.size
    @unpack u1_b = cache
    refresh!(u1_b)
    @unpack degree, Vl, Vr = op
    nd = degree + 1
    @inbounds for cell in 0:(nx + 1)
        for i in 1:nd
            u_node = get_node_vars(u1, eq, i, cell)
            multiply_add_to_node_vars!(u1_b, Vl[i], u_node, eq, 1, cell)
            multiply_add_to_node_vars!(u1_b, Vr[i], u_node, eq, 2, cell)
        end
    end
    return nothing
    end # timer
end

function update_ghost_values_cRK!(problem, scheme::Scheme{<:cRK44},
                                  eq::AbstractEquations{1},
                                  grid, aux, op, cache, t, dt)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, Ub = cache
    update_ghost_values_periodic!(eq, problem, Fb, Ub)
    update_ghost_values_u1!(eq, problem, grid, op, cache.u1, aux, t)

    if problem.periodic_x
        return nothing
    end

    nx = grid.size
    @unpack degree, xg, wg = op
    nd = degree + 1
    dx, xf = grid.dx, grid.xf
    nvar = nvariables(eq)
    @unpack boundary_value, boundary_condition = problem
    left, right = boundary_condition
    refresh!(u) = fill!(u, 0.0)

    ub, fb = zeros(nvar), zeros(nvar)

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    rk4_time_levels = (0, 0.5, 0.5, 1)
    rk4_coeff = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
    if left == dirichlet
        x = xf[1]
        for l in 1:4 # RK4 stages
            tq = t + dt * rk4_time_levels[l]
            ubvalue = boundary_value(x, tq)
            fbvalue = flux(x, ubvalue, eq)
            for n in 1:nvar
                ub[n] += ubvalue[n] * rk4_coeff[l]
                fb[n] += fbvalue[n] * rk4_coeff[l]
            end
        end
        for n in 1:nvar
            Ub[n, 1, 1] = Ub[n, 2, 0] = ub[n]
            Fb[n, 1, 1] = Fb[n, 2, 0] = fb[n]
        end
    elseif left == neumann
        for n in 1:nvar
            Ub[n, 2, 0] = Ub[n, 1, 1]
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

    refresh!.((ub, fb))
    if right == dirichlet
        x = xf[nx + 1]
        for l in 1:4 # RK4 stages
            tq = t + dt * rk4_time_levels[l]
            ubvalue = boundary_value(x, tq)
            fbvalue = flux(x, ubvalue, eq)
            for n in 1:nvar
                ub[n] += ubvalue[n] * rk4_coeff[l]
                fb[n] += fbvalue[n] * rk4_coeff[l]
            end
        end
        for n in 1:nvar
            Ub[n, 2, nx] = Ub[n, 1, nx + 1] = ub[n]
            Fb[n, 2, nx] = Fb[n, 1, nx + 1] = fb[n]
        end
    elseif right == neumann
        for n in 1:nvar
            Ub[n, 1, nx + 1] = Ub[n, 2, nx]
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
    end # timer
    return nothing
end

function setup_arrays(grid, scheme::Scheme{<:cRKSolver},
                      eq::AbstractEquations{1})
    gArray(nvar, nx) = OffsetArray(zeros(nvar, nx + 2),
                                   OffsetArrays.Origin(1, 0))
    function gArray(nvar, n1, nx)
        OffsetArray(zeros(nvar, n1, nx + 2),
                    OffsetArrays.Origin(1, 1, 0))
    end
    # Allocate memory
    @unpack degree = scheme
    nd = degree + 1
    nx = grid.size
    nvar = nvariables(eq)
    u1 = gArray(nvar, nd, nx)
    ua = gArray(nvar, nx)
    res = gArray(nvar, nd, nx)
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

    cache = (; u1, ua, res, Fb, Ub, u1_b, ub_N, cell_data, eval_data)
    return cache
end

function setup_arrays(grid, scheme::Scheme{<:cRKSolver, <:DCSX},
                      eq::AbstractEquations{1})
    gArray(nvar, nx) = OffsetArray(zeros(nvar, nx + 2),
                                   OffsetArrays.Origin(1, 0))
    function gArray(nvar, n1, nx)
        OffsetArray(zeros(nvar, n1, nx + 2),
                    OffsetArrays.Origin(1, 1, 0))
    end
    # Allocate memory
    @unpack degree = scheme
    nd = degree + 1
    nx = grid.size
    nvar = nvariables(eq)
    u1 = gArray(nvar, nd, nx)
    ua = gArray(nvar, nx)
    res = gArray(nvar, nd, nx)
    Fb = gArray(nvar, 2, nx)
    Ub = gArray(nvar, 2, nx)
    u2b, u3b, u4b = (copy(Ub) for _ in 1:3)
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

    cache = (; u1, ua, res, Fb, Ub, ub_N, u2b, u3b, u4b, cell_data, eval_data)
    return cache
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cRK11}, aux, t, dt, cache)
    @unpack source_terms = problem
    @unpack ub_N, ua, u1, res, Fb, Ub, = cache
    nx = grid.size

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        xc = grid.xc[cell]
        u_node = get_node_vars(u1, eq, 1, cell)
        # TODO - This is unneeded and gives an error
        get_cache_node_vars(aux, problem, scheme, eq, 1, cell)
        s_node = calc_source(u_node, xc, t, source_terms, eq)
        set_node_vars!(res, -dt * s_node, eq, 1, cell)
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

function compute_face_residual!(eq::AbstractEquations{1}, grid, op, cache,
                                problem, scheme::Scheme{<:cRKSolver}, param, aux, t, dt,
                                u1, Fb,
                                Ub, ua, res, scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack xg, wg, bl, br = op
    nd = op.degree + 1
    nx = grid.size
    @unpack dx, xf = grid
    num_flux = scheme.numerical_flux
    @unpack blend = aux
    @unpack u1_b = cache

    # Vertical faces, x flux
    for i in 1:(nx + 1)
        # Face between i-1 and i
        x = xf[i]
        @views Fn = num_flux(x,
                             u1_b[:, 2, i - 1], u1_b[:, 1, i],
                             Fb[:, 2, i - 1], Fb[:, 1, i],
                             Ub[:, 2, i - 1], Ub[:, 1, i], eq, 1)
        Fn, blend_fac = blend.blend_face_residual!(i, x, u1, ua, eq, t, dt, grid,
                                                   op, problem,
                                                   scheme, param, Fn, aux, nothing,
                                                   res, scaling_factor)
        for ix in 1:nd
            for n in 1:nvariables(eq)
                res[n, ix, i - 1] += dt / dx[i - 1] * blend_fac[1] * Fn[n] * br[ix]
                res[n, ix, i] += dt / dx[i] * blend_fac[2] * Fn[n] * bl[ix]
            end
        end
    end
    return nothing
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cRK22}, aux, t, dt, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux
    @unpack bl, br = op
    get_dissipation_node_vars = scheme.dissipation

    @unpack cell_data, eval_data, ub_N, ua, u1, res, Fb, Ub = cache

    F, f, U, u2 = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb, ub_N)) # Reset previously used variables to zero

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

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u2, -0.5 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add Bu_x and source terms contribution to u2
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            # cache_node = get_cache_node(cache, i, cell)
            u_node = get_node_vars(u1, eq, i, cell)
            s_node = calc_source(u_node, x_, t, source_terms, eq)
            # s_node = calc_source(u_node, cache_node, x_, t, source_terms, eq)
            multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)

            flux1 = flux(x_, u2_node, eq)

            set_node_vars!(F, flux1, eq, i)
            set_node_vars!(U, u2_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end

            s2_node = calc_source(u2_node, x_, t + 0.5 * dt, source_terms, eq)

            S_node = s2_node # S array is not needed in this degree N = 1 case

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_dissipation_node_vars(u, U, eq, i)
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

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cRK33}, aux, t, dt, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend = aux

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    F, f, U, u2, u3, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]
        u3 .= @view u1[:, :, cell]

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)

            set_node_vars!(F, 0.25 * flux1, eq, i)
            set_node_vars!(U, 0.25 * u_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u2, -lamx * Dm[ii, i] / 3.0, flux1, eq, ii)
            end
        end

        # Add source term contribution to u2 and some to S
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            s_node = calc_source(u_node, x_, t, source_terms, eq)
            multiply_add_to_node_vars!(u2, dt / 3.0, s_node, eq, i)
            set_node_vars!(S, 0.25 * s_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)

            flux1 = flux(x_, u2_node, eq)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u3, -2.0 * lamx * Dm[ii, i] / 3.0, flux1, eq,
                                           ii)
            end
        end

        # Add source term contribution to u3 and some to S
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)
            s2_node = calc_source(u2_node, x_, t + dt / 3.0, source_terms, eq)
            multiply_add_to_node_vars!(u3, 2.0 * dt / 3.0, s2_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u3_node = get_node_vars(u3, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_to_node_vars!(F, 0.75, flux1, eq, i)
            multiply_add_to_node_vars!(U, 0.75, u3_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end

            s3_node = calc_source(u3_node, x_, t + 2.0 / 3.0 * dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 0.75, s3_node, eq, i)
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end

        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_dissipation_node_vars(u, U, eq, i)
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
            ul, ur, u3l, u3r = eval_data[Threads.threadid()]
            refresh!.((ul, ur, u3l, u3r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u1, eq, i, cell)
                u3_node = get_node_vars(u3, eq, i)
                multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, 1)
                multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, 1)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
            end
            # IDEA - Try this in TVB limiter as well
            ul_node = get_node_vars(ul, eq, 1)
            ur_node = get_node_vars(ur, eq, 1)
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            multiply_add_to_node_vars!(Fb, 0.25, fl, 0.75, f3l, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, 0.25, fr, 0.75, f3r, eq, 2, cell)
        end
    end
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cRK44}, aux, t, dt, cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    get_dissipation_node_vars = scheme.dissipation
    @unpack blend = aux

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

        # Solution points
        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            # Compute flux at all solution points
            flux1 = flux(x_, u_node, eq)

            set_node_vars!(F, flux1 / 6.0, eq, i)
            set_node_vars!(U, u_node / 6.0, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u2, -0.5 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u2 and some to S
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            s_node = calc_source(u_node, x_, t, source_terms, eq)
            multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i)
            set_node_vars!(S, s_node / 6.0, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)

            flux1 = flux(x_, u2_node, eq)

            multiply_add_to_node_vars!(F, 1.0 / 3.0, flux1, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 3.0, u2_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u3, -0.5 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u3 and some to S
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)
            s2_node = calc_source(u2_node, x_, t + 0.5 * dt, source_terms, eq)
            multiply_add_to_node_vars!(u3, 0.5 * dt, s2_node, eq, i)
            multiply_add_to_node_vars!(S, 1.0 / 3.0, s2_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u3_node = get_node_vars(u3, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_to_node_vars!(F, 1.0 / 3.0, flux1, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 3.0, u3_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u4, -lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u4 and some to S
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u3_node = get_node_vars(u3, eq, i)
            s3_node = calc_source(u3_node, x_, t + 0.5 * dt, source_terms, eq)
            multiply_add_to_node_vars!(u4, dt, s3_node, eq, i)
            multiply_add_to_node_vars!(S, 1.0 / 3.0, s3_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u4_node = get_node_vars(u4, eq, i)

            flux1 = flux(x_, u4_node, eq)

            multiply_add_to_node_vars!(F, 1.0 / 6.0, flux1, eq, i)
            multiply_add_to_node_vars!(U, 1.0 / 6.0, u4_node, eq, i)

            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end

            s4_node = calc_source(u4_node, x_, t + dt, source_terms, eq)
            multiply_add_to_node_vars!(S, 1.0 / 6.0, s4_node, eq, i)
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end

        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_dissipation_node_vars(u, U, eq, i)
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
end # muladd
