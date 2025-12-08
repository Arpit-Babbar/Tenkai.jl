using .EqEuler1D: tenkai2trixiequation

function tenkai2trixiode(solver::TrixiRKSolver, equation::AbstractEquations{1},
                         problem, scheme, param)
    @unpack grid_size = param
    @assert *(ispow2.(grid_size)...) "Grid size must be a power of 2 for TreeMesh."
    trixi_equations = tenkai2trixiequation(equation)
    initial_condition(x, t, equations) = problem.exact_solution(x..., t)
    dg_solver = Trixi.DGSEM(polydeg = scheme.degree,
                            surface_flux = Trixi.flux_lax_friedrichs,
                            volume_integral = Trixi.VolumeIntegralShockCapturingHG(nothing))
    mesh = Trixi.TreeMesh(problem.domain[1], problem.domain[2],
                          initial_refinement_level = Int(log2(grid_size)),
                          n_cells_max = 10000000)
    semi = Trixi.SemidiscretizationHyperbolic(mesh, trixi_equations, initial_condition,
                                              dg_solver)
    tspan = (0.0, problem.final_time)
    ode = Trixi.semidiscretize(semi, tspan)
end

function get_element_alpha(blend::NamedTuple, element)
    return 0.0
end

function get_element_alpha(blend::Blend1D, element)
    return blend.alpha[element]
end

# Exactly the same as in Trixi.jl, but kept here because it is not in the API.
# https://github.com/trixi-framework/Trixi.jl/blob/3ce203318eb1c13427d145f8a7609db32481bc9a/src/solvers/dgsem_tree/dg_1d.jl#L146
@inline function weak_form_kernel!(du, u,
                                   element, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                   nonconservative_terms::False, equations,
                                   dg::DGSEM, cache, tenkai_op, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    # @unpack derivative_dhat = dg.basis
    @unpack D1 = tenkai_op

    for i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, element)

        flux1 = Trixi.flux(u_node, 1, equations)
        for ii in eachnode(dg)
            Trixi.multiply_add_to_node_vars!(du, alpha * D1[ii, i], flux1,
                                             equations, dg, ii, element)
        end
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                           nonconservative_terms::False, equations,
                                           volume_flux, dg::DGSEM, cache, tenkai_op,
                                           alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    # @unpack derivative_split = dg.basis
    @unpack Dsplit = tenkai_op
    # Calculate volume integral in one element
    for i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg)
            u_node_ii = Trixi.get_node_vars(u, equations, dg, ii, element)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            Trixi.multiply_add_to_node_vars!(du, alpha * Dsplit[i, ii], flux1,
                                             equations, dg, i, element)
            Trixi.multiply_add_to_node_vars!(du, alpha * Dsplit[ii, i], flux1,
                                             equations, dg, ii, element)
        end
    end
end

@inline function calc_volume_integral_local!(volume_integral::VolumeIntegralWeakForm, du, u,
                                             element,
                                             mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                             nonconservative_terms::False, equations,
                                             dg::DGSEM, cache, tenkai_op, alpha = true)
    weak_form_kernel!(du, u, element, mesh, nonconservative_terms, equations, dg, cache,
                      tenkai_op, alpha)
end

@inline function calc_volume_integral_local!(volume_integral::VolumeIntegralFluxDifferencing,
                                             du, u,
                                             element,
                                             mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                             nonconservative_terms::False, equations,
                                             dg::DGSEM, cache, tenkai_op, alpha = true)
    @unpack volume_flux = volume_integral
    flux_differencing_kernel!(du, u, element, mesh, nonconservative_terms, equations,
                              volume_flux, dg, cache, tenkai_op, alpha)
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
    RealT = eltype(grid.xc)
    f = zeros(RealT, nvar, nd)
    @timeit aux.timer "Cell loop" begin
    #! format: noindent
    @inbounds for cell in 1:nx
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        xl, xr = grid.xf[cell], grid.xf[cell + 1]

        alpha = get_element_alpha(blend, cell)

        trixi_equations = get_trixi_equations(semi, eq)

        calc_volume_integral_local!(scheme.solver.volume_integral, res, u1,
                                    cell, semi.mesh,
                                    Trixi.have_nonconservative_terms(trixi_equations),
                                    trixi_equations, semi.solver, semi.cache, op,
                                    lamx * (1.0 - alpha))

        for ix in Base.OneTo(nd)
            # Solution points
            x = xc - 0.5 * dx + xg[ix] * dx
            u_node = get_node_vars(u1, eq, ix, cell)
            # Compute flux at all solution points

            # TODO - This is double the computation. Once in the kernel, once here.
            flux1 = flux(x, u_node, eq)
            set_node_vars!(f, flux1, eq, ix)
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

        u = @view u1[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt,
                                   dx,
                                   grid.xf[cell], op, u1, u, cache.ua, cache, res)
    end
    end # timer
    return nothing
    end # timer
end

@inline eachinterface1d(dg::DGSEM, cache) = 1:(1 + size(cache.elements.surface_flux_values,
                                                        3))

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{1},
                              nonconservative_terms::False, equations,
                              surface_integral, ub, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack neighbor_ids, orientations = cache.interfaces

    @threaded for interface in eachinterface1d(dg, cache)
        # Get neighboring elements
        # left_id = neighbor_ids[1, interface]
        # right_id = neighbor_ids[2, interface]
        left_id = interface - 1
        right_id = interface

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        # left_direction = 2 * orientations[interface]
        # right_direction = 2 * orientations[interface] - 1
        left_direction = 2
        right_direction = 1

        # Call pointwise Riemann solver
        u_ll, u_rr = Trixi.get_node_vars(ub, equations, dg, left_direction, left_id),
                     Trixi.get_node_vars(ub, equations, dg, right_direction, right_id)
        orientation = 1
        flux_ = surface_flux(u_ll, u_rr, orientation, equations)

        # Copy flux to left and right element storage
        for v in Trixi.eachvariable(equations)
            surface_flux_values[v, left_direction, left_id] = flux_[v]
            surface_flux_values[v, right_direction, right_id] = flux_[v]
        end
    end
end

function compute_face_residual!(eq::AbstractEquations{1}, grid, op, cache,
                                problem, scheme::Scheme{<:TrixiRKSolver}, param, aux, t, dt,
                                u1,
                                Fb, ub, ua, res,
                                scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack xg, wg, bl, br = op
    nd = op.degree + 1
    nx = grid.size
    @unpack dx, xf = grid
    @unpack blend = aux

    @unpack trixi_ode = cache
    semi = trixi_ode.p
    @unpack cache = semi
    surface_flux_values = Fb

    trixi_equations = get_trixi_equations(semi, eq)

    calc_interface_flux!(surface_flux_values, semi.mesh,
                         Trixi.have_nonconservative_terms(trixi_equations),
                         trixi_equations, semi.solver.surface_integral, ub,
                         semi.solver, cache)

    for i in 1:nx
        Fl, Fr = get_node_vars(Fb, eq, 1, i), get_node_vars(Fb, eq, 2, i)
        for ix in 1:nd
            for n in 1:nvariables(eq)
                res[n, ix, i] += dt / dx[i] * br[ix] * Fr[n]
                res[n, ix, i] += dt / dx[i] * bl[ix] * Fl[n]
            end
        end
    end
    return nothing
    end # timer
end

@inline function calcflux_fv!(fstar1_L, fstar1_R, u::AbstractArray{<:Any, 3},
                              mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                              nonconservative_terms::False,
                              equations, volume_flux_fv, dg::DGSEM, element, cache)
    fstar1_L[:, 1] .= zero(eltype(fstar1_L))
    fstar1_L[:, nnodes(dg) + 1] .= zero(eltype(fstar1_L))
    fstar1_R[:, 1] .= zero(eltype(fstar1_R))
    fstar1_R[:, nnodes(dg) + 1] .= zero(eltype(fstar1_R))

    for i in 2:nnodes(dg)
        u_ll = Trixi.get_node_vars(u, equations, dg, i - 1, element)
        u_rr = Trixi.get_node_vars(u, equations, dg, i, element)
        flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
        Trixi.set_node_vars!(fstar1_L, flux, equations, dg, i)
        Trixi.set_node_vars!(fstar1_R, flux, equations, dg, i)
    end

    return nothing
end

@inline function fv_kernel!(du, u,
                            mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                            nonconservative_terms, equations,
                            volume_flux_fv, dg::DGSEM, cache, element, tenkai_op,
                            alpha = true)
    @unpack fstar1_L_threaded, fstar1_R_threaded = cache
    @unpack inverse_weights = dg.basis
    # TODO - This only works for GLL + g2. Because for any other combination,
    # the interface fluxe has to be treated differently.
    @unpack wg_inv = tenkai_op

    # Calculate FV two-point fluxes
    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, u, mesh, nonconservative_terms, equations,
                 volume_flux_fv,
                 dg, element, cache)

    # Calculate FV volume integral contribution
    for i in eachnode(dg)
        for v in Trixi.eachvariable(equations)
            du[v, i, element] += (alpha *
                                  (wg_inv[i] *
                                   (fstar1_L[v, i + 1] - fstar1_R[v, i])))
        end
    end

    return nothing
end

@inbounds @inline function blend_cell_residual_fo!(cell, eq::AbstractEquations{1},
                                                   problem, scheme::Scheme{<:TrixiRKSolver},
                                                   aux, lamx,
                                                   t, dt, dx, xf, op, u1, u, ua, cache, r,
                                                   scaling_factor = 1.0)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead, it's supposed
    #! format: noindent
    # to be 0.25 microseconds
    @unpack blend = aux
    @unpack source_terms = problem
    @unpack Vl, Vr, xg, wg = op

    @unpack trixi_ode = cache
    semi = trixi_ode.p
    @unpack volume_integral = semi.solver

    alpha = get_element_alpha(blend, cell)

    trixi_equations = get_trixi_equations(semi, eq)

    # TODO - Use your own fv_kernel!
    fv_kernel!(r, u1, semi.mesh,
               Trixi.have_nonconservative_terms(trixi_equations),
               trixi_equations,
               volume_integral.volume_flux_fv, semi.solver,
               semi.cache, cell, op,
               lamx * alpha)
    end # timer
end
