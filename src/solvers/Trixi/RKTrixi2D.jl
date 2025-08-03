using .EqEuler1D: tenkai2trixiequation

@inline function weak_form_kernel!(du, u,
                                   element, mesh::TreeMesh{2},
                                   nonconservative_terms::False, equations,
                                   dg::DGSEM, cache, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_dhat = dg.basis

    # Calculate volume terms in one element
    for j in eachnode(dg), i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, element...)

        flux1 = Trixi.flux(u_node, 1, equations)
        for ii in eachnode(dg)
            Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], flux1,
                                             equations, dg, ii, j, element...)
        end

        flux2 = Trixi.flux(u_node, 2, equations)
        for jj in eachnode(dg)
            Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], flux2,
                                             equations, dg, i, jj, element...)
        end
    end

    return nothing
end

function tenkai2trixiode(solver::TrixiRKSolver, equation::AbstractEquations{2},
                         problem, scheme, param)
    @unpack grid_size = param
    @assert *(ispow2.(grid_size)...) "Grid size must be a power of 2 for TreeMesh."
    @assert scheme.solution_points=="gll" "Only GLL solution points are supported for Trixi."
    @assert scheme.correction_function=="g2" "Only G2 correction function is supported for Trixi."
    trixi_equations = tenkai2trixiequation(equation)
    initial_condition(x, t, equations) = problem.exact_solution(x..., t)
    dg_solver = Trixi.DGSEM(polydeg = scheme.degree,
                            surface_flux = Trixi.flux_lax_friedrichs,
                            volume_integral = Trixi.VolumeIntegralShockCapturingHG(nothing))
    xmin, xmax, ymin, ymax = problem.domain
    coordinates_min = (xmin, ymin)
    coordinates_max = (xmax, ymax)
    mesh = Trixi.TreeMesh(coordinates_min, coordinates_max,
                          initial_refinement_level = Int(log2(grid_size[1])),
                          n_cells_max = 10000000)
    semi = Trixi.SemidiscretizationHyperbolic(mesh, trixi_equations, initial_condition,
                                              dg_solver)
    tspan = (0.0, problem.final_time)
    ode = Trixi.semidiscretize(semi, tspan)
end

function compute_cell_residual_rkfr!(eq::AbstractEquations{2}, grid, op, problem,
                                     scheme::Scheme{<:TrixiRKSolver},
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

    @unpack trixi_ode = cache
    semi = trixi_ode.p

    refresh!.((ub, Fb, res))
    @threaded for element in CartesianIndices((1:nx, 1:ny)) # element loop
        el_x, el_y = element[1], element[2]
        dx, dy = grid.dx[el_x], grid.dy[el_y]
        xc, yc = grid.xc[el_x], grid.yc[el_y]
        lamx = dt / dx # = dt / dy
        u = @view u1[:, :, :, el_x, el_y]
        r1 = @view res[:, :, :, el_x, el_y]
        ub_ = @view ub[:, :, :, el_x, el_y]
        Fb_ = @view Fb[:, :, :, el_x, el_y]

        weak_form_kernel!(res, u1, (el_x, el_y), semi.mesh,
                          Trixi.have_nonconservative_terms(semi.equations),
                          semi.equations, semi.solver, semi.cache,
                          2.0 * lamx)

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
        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid, dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u,
                             nothing, res)
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
    return nothing
    end # timer
end
