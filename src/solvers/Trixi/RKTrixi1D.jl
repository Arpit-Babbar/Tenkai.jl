# Exactly the same as in Trixi.jl, but kept here because it is not in the API.
# https://github.com/trixi-framework/Trixi.jl/blob/3ce203318eb1c13427d145f8a7609db32481bc9a/src/solvers/dgsem_tree/dg_1d.jl#L146
@inline function weak_form_kernel!(du, u,
                                   element, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                   nonconservative_terms::False, equations,
                                   dg::DGSEM, cache, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_dhat = dg.basis

    for i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, element)

        flux1 = Trixi.flux(u_node, 1, equations)
        for ii in eachnode(dg)
            Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], flux1,
                                             equations, dg, ii, element)
        end
    end

    return nothing
end

function compute_cell_residual_rkfr!(eq::AbstractEquations{1}, grid, op, problem,
                                     scheme::Scheme{<:TrixiRKSolver}, aux, t, dt, u1, res,
                                     Fb, ub, cache)
    @timeit aux.timer "Cell residual" begin
    #! format: noindent
    @unpack xg, D1, Vl, Vr = op
    @unpack blend = aux
    nx = grid.size
    nd = length(xg)
    @unpack bflux_ind = scheme.bflux
    refresh!(u) = fill!(u, 0.0)

    @unpack trixi_ode = cache
    semi = trixi_ode.p

    refresh!.((ub, Fb, res))
    nvar = nvariables(eq)
    f = zeros(nvar, nd)
    @timeit aux.timer "Cell loop" begin
    #! format: noindent
    @inbounds for cell in 1:nx
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        xl, xr = grid.xf[cell], grid.xf[cell + 1]

        weak_form_kernel!(res, u1, cell, semi.mesh,
                          Trixi.have_nonconservative_terms(semi.equations),
                          semi.equations, semi.solver, semi.cache)

        # res .*= -semi.cache.elements.inverse_jacobian[1]

        # display(D1 ./ semi.solver.basis.derivative_dhat)
        # display(semi.solver.basis.derivative_dhat)
        # @assert false

        for ix in Base.OneTo(nd)
            # Solution points
            x = xc - 0.5 * dx + xg[ix] * dx
            u_node = get_node_vars(u1, eq, ix, cell)
            # Compute flux at all solution points
            flux1 = flux(x, u_node, eq)
            set_node_vars!(f, flux1, eq, ix)
            # KLUDGE - Remove dx, xf arguments. just pass grid and i
            # for iix in 1:nd
            #     multiply_add_to_node_vars!(res, lamx * D1[iix, ix], flux1, eq,
            #                                iix, cell)
            # end
            multiply_add_to_node_vars!(ub, Vl[ix], u_node, eq, 1, cell)
            multiply_add_to_node_vars!(ub, Vr[ix], u_node, eq, 2, cell)
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

        res[:, :, cell] .*= 2.0 * lamx
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt,
                                   dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
    end
    end # timer
    return nothing
    end # timer
end
