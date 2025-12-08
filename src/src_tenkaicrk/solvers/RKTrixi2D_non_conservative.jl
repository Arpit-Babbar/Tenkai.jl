using Tenkai
using Tenkai: TreeMesh, DGSEM, True, False, eachnode, DG, eachinterface_x,
              eachinterface_y, tenkai2trixiequation,
              StructuredMesh, UnstructuredMesh2D, P4estMesh, T8codeMesh, nnodes
import Tenkai: flux_differencing_kernel!, tenkai2trixiode, weak_form_kernel!,
               calc_volume_integral_local!

function tenkai2trixiode(solver::TrixiRKSolver{<:VolumeIntegralFluxDifferencing},
                         equation::AbstractNonConservativeEquations{2},
                         problem, scheme, param;
                         surface_flux = (Trixi.flux_lax_friedrichs,
                                         solver.volume_integral.volume_flux[2]))
    @unpack volume_integral = solver
    @assert volume_integral isa VolumeIntegralFluxDifferencing "Only flux diff for non-cons eqns."
    @assert length(volume_integral.volume_flux)==2 "Conservative & non-conservative fluxes needed."
    @unpack grid_size = param
    @assert *(ispow2.(grid_size)...) "Grid size must be a power of 2 for TreeMesh."
    @assert scheme.solution_points=="gll" "Only GLL solution points are supported for Trixi."
    @assert scheme.correction_function=="g2" "Only G2 correction function is supported for Trixi."
    trixi_equations = tenkai2trixiequation(equation)
    initial_condition(x, t, equations) = problem.exact_solution(x..., t)
    dg_solver = Trixi.DGSEM(polydeg = scheme.degree,
                            surface_flux = surface_flux,
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

@inline function weak_form_kernel!(du, u,
                                   element, mesh::TreeMesh{2},
                                   nonconservative_terms::True, equations,
                                   dg::DGSEM, cache, tenkai_op, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_dhat, derivative_split = dg.basis
    @unpack Dsplit = tenkai_op

    weak_form_kernel!(du, u, element, mesh, False(), equations, dg, cache, tenkai_op, alpha)

    nonconservative_flux = flux_central_nc

    # Calculate volume terms in one element
    for j in eachnode(dg), i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, element...)

        # The diagonal terms are zero since the diagonal of `derivative_split`
        # is zero. We ignore this for now.

        # x direction
        integral_contribution = zero(u_node)
        for ii in eachnode(dg)
            u_node_ii = Trixi.get_node_vars(u, equations, dg, ii, j, element...)
            noncons_flux1 = nonconservative_flux(u_node, u_node_ii, 1, equations)
            integral_contribution = integral_contribution +
                                    Dsplit[i, ii] * noncons_flux1
        end

        # y direction
        for jj in eachnode(dg)
            u_node_jj = Trixi.get_node_vars(u, equations, dg, i, jj, element...)
            noncons_flux2 = nonconservative_flux(u_node, u_node_jj, 2, equations)
            integral_contribution = integral_contribution +
                                    Dsplit[j, jj] * noncons_flux2
        end

        # The factor 0.5 cancels the factor 2 in the flux differencing form
        Trixi.multiply_add_to_node_vars!(du, alpha * 0.5f0, integral_contribution,
                                         equations,
                                         dg, i, j, element...)
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element, mesh::TreeMesh{2},
                                           nonconservative_terms::True, equations,
                                           volume_flux, dg::DGSEM, cache, tenkai_op,
                                           alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_split = dg.basis
    symmetric_flux, nonconservative_flux = volume_flux
    @unpack Dsplit = tenkai_op

    # Apply the symmetric flux as usual
    flux_differencing_kernel!(du, u, element, mesh, False(), equations, symmetric_flux,
                              dg, cache, tenkai_op, alpha)

    # Calculate the remaining volume terms using the nonsymmetric generalized flux
    for j in eachnode(dg), i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, element...)

        # The diagonal terms are zero since the diagonal of `derivative_split`
        # is zero. We ignore this for now.

        # x direction
        integral_contribution = zero(u_node)
        for ii in eachnode(dg)
            u_node_ii = Trixi.get_node_vars(u, equations, dg, ii, j, element...)
            noncons_flux1 = nonconservative_flux(u_node, u_node_ii, 1, equations)
            integral_contribution = integral_contribution +
                                    Dsplit[i, ii] * noncons_flux1
        end

        # y direction
        for jj in eachnode(dg)
            u_node_jj = Trixi.get_node_vars(u, equations, dg, i, jj, element...)
            noncons_flux2 = nonconservative_flux(u_node, u_node_jj, 2, equations)
            integral_contribution = integral_contribution +
                                    Dsplit[j, jj] * noncons_flux2
        end

        # The factor 0.5 cancels the factor 2 in the flux differencing form
        Trixi.multiply_add_to_node_vars!(du, alpha * 0.5f0, integral_contribution,
                                         equations,
                                         dg, i, j, element...)
    end
end

function compute_cell_residual_rkfr!(eq::AbstractNonConservativeEquations{2}, grid, op,
                                     problem,
                                     scheme::Scheme{<:TrixiRKSolver}, aux, t, dt, u1, res,
                                     Fb,
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

    @unpack trixi_ode = cache
    semi = trixi_ode.p

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

        trixi_equations = get_trixi_equations(semi, eq)

        calc_volume_integral_local!(scheme.solver.volume_integral, res, u1,
                                    (el_x, el_y), semi.mesh,
                                    Trixi.have_nonconservative_terms(trixi_equations),
                                    #    Trixi.False(),
                                    trixi_equations, semi.solver, semi.cache, op,
                                    lamx)

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

        blend_cell_residual!(el_x, el_y, eq, problem, scheme, aux, t, dt, grid,
                             dx,
                             dy,
                             grid.xf[el_x], grid.yf[el_y], op, u1, u,
                             cache, res)

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

function get_surface_fluxes(::VolumeIntegralFluxDifferencing, surface_flux)
    return surface_flux[1], surface_flux[2]
end

function get_surface_fluxes(::VolumeIntegralWeakForm, surface_flux)
    return surface_flux, flux_central_nc
end

function calc_interface_flux!(solver::TrixiRKSolver, surface_flux_values,
                              mesh::TreeMesh{2},
                              nonconservative_terms::True, equations,
                              surface_integral, ub, dg::DG, tenkai_grid, cache)
    @unpack volume_integral = solver
    surface_flux, nonconservative_flux = get_surface_fluxes(volume_integral,
                                                            surface_integral.surface_flux)
    @unpack u, neighbor_ids, orientations = cache.interfaces

    @threaded for interface in eachinterface_x(dg, tenkai_grid)
        int_x = interface[1]
        int_y = interface[2]
        # Get neighboring elements
        # left_id = neighbor_ids[1, interface]
        # right_id = neighbor_ids[2, interface]
        left_id = (int_x - 1, int_y)
        right_id = (int_x, int_y)

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        # orientation = 2: left -> 4, right -> 3
        left_direction = 2
        right_direction = 1

        orientation = 1
        for i in eachnode(dg)
            # Call pointwise Riemann solver
            # u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
            u_ll, u_rr = Trixi.get_node_vars(ub, equations, dg, i, left_direction,
                                             left_id...),
                         Trixi.get_node_vars(ub, equations, dg, i, right_direction,
                                             right_id...)
            flux = surface_flux(u_ll, u_rr, 1, equations)

            # Compute both nonconservative fluxes
            noncons_left = nonconservative_flux(u_ll, u_rr, orientation, equations)
            noncons_right = nonconservative_flux(u_rr, u_ll, orientation, equations)

            # Copy flux to left and right element storage
            for v in Trixi.eachvariable(equations)
                # Note the factor 0.5 necessary for the nonconservative fluxes based on
                # the interpretation of global SBP operators coupled discontinuously via
                # central fluxes/SATs
                surface_flux_values[v, i, left_direction, left_id...] = flux[v] +
                                                                        0.5f0 *
                                                                        noncons_left[v]
                surface_flux_values[v, i, right_direction, right_id...] = flux[v] +
                                                                          0.5f0 *
                                                                          noncons_right[v]
            end
        end
    end

    @threaded for interface in eachinterface_y(dg, tenkai_grid)
        # Get neighboring elements
        left_id = (interface[1], interface[2] - 1)
        right_id = (interface[1], interface[2])

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        # orientation = 2: left -> 4, right -> 3
        left_direction = 4
        right_direction = 3

        orientation = 2
        for i in eachnode(dg)
            # Call pointwise Riemann solver
            u_ll, u_rr = Trixi.get_node_vars(ub, equations, dg, i, left_direction,
                                             left_id...),
                         Trixi.get_node_vars(ub, equations, dg, i, right_direction,
                                             right_id...)
            flux = surface_flux(u_ll, u_rr, orientation, equations)

            # Compute both nonconservative fluxes
            noncons_left = nonconservative_flux(u_ll, u_rr, orientation, equations)
            noncons_right = nonconservative_flux(u_rr, u_ll, orientation, equations)

            # Copy flux to left and right element storage
            for v in Trixi.eachvariable(equations)
                # Note the factor 0.5 necessary for the nonconservative fluxes based on
                # the interpretation of global SBP operators coupled discontinuously via
                # central fluxes/SATs
                surface_flux_values[v, i, left_direction, left_id...] = flux[v] +
                                                                        0.5f0 *
                                                                        noncons_left[v]
                surface_flux_values[v, i, right_direction, right_id...] = flux[v] +
                                                                          0.5f0 *
                                                                          noncons_right[v]
            end
        end
    end

    return nothing
end

@inline function fv_kernel!(du, u,
                            mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                        UnstructuredMesh2D, P4estMesh{2},
                                        T8codeMesh{2}},
                            nonconservative_terms, equations,
                            volume_flux_fv, dg::DGSEM, cache, tenkai_cache, tenkai_op,
                            element,
                            alpha = true)
    @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = tenkai_cache.fv_cache
    # @unpack inverse_weights = dg.basis
    @unpack wg_inv = tenkai_op

    # Calculate FV two-point fluxes
    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar2_L = fstar2_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    fstar2_R = fstar2_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
                 nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

    # Calculate FV volume integral contribution
    for j in eachnode(dg), i in eachnode(dg)
        for v in Trixi.eachvariable(equations)
            du[v, i, j, element...] += (alpha *
                                        (wg_inv[i] *
                                         (fstar1_L[v, i + 1, j] - fstar1_R[v, i, j]) +
                                         wg_inv[j] *
                                         (fstar2_L[v, i, j + 1] - fstar2_R[v, i, j])))
        end
    end

    return nothing
end

@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                              u::AbstractArray{<:Any, 5},
                              mesh::TreeMesh{2}, nonconservative_terms::True, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)
    volume_flux = volume_flux_fv
    nonconservative_flux = flux_central_nc

    # Fluxes in x
    fstar1_L[:, 1, :] .= zero(eltype(fstar1_L))
    fstar1_L[:, nnodes(dg) + 1, :] .= zero(eltype(fstar1_L))
    fstar1_R[:, 1, :] .= zero(eltype(fstar1_R))
    fstar1_R[:, nnodes(dg) + 1, :] .= zero(eltype(fstar1_R))

    for j in eachnode(dg), i in 2:nnodes(dg)
        u_ll = Trixi.get_node_vars(u, equations, dg, i - 1, j, element...)
        u_rr = Trixi.get_node_vars(u, equations, dg, i, j, element...)

        # Compute conservative part
        f1 = volume_flux(u_ll, u_rr, 1, equations) # orientation 1: x direction

        # Compute nonconservative part
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        f1_L = f1 + 0.5f0 * nonconservative_flux(u_ll, u_rr, 1, equations)
        f1_R = f1 + 0.5f0 * nonconservative_flux(u_rr, u_ll, 1, equations)

        # Copy to temporary storage
        Trixi.set_node_vars!(fstar1_L, f1_L, equations, dg, i, j)
        Trixi.set_node_vars!(fstar1_R, f1_R, equations, dg, i, j)
    end

    # Fluxes in y
    fstar2_L[:, :, 1] .= zero(eltype(fstar2_L))
    fstar2_L[:, :, nnodes(dg) + 1] .= zero(eltype(fstar2_L))
    fstar2_R[:, :, 1] .= zero(eltype(fstar2_R))
    fstar2_R[:, :, nnodes(dg) + 1] .= zero(eltype(fstar2_R))

    # Compute inner fluxes
    for j in 2:nnodes(dg), i in eachnode(dg)
        u_ll = Trixi.get_node_vars(u, equations, dg, i, j - 1, element...)
        u_rr = Trixi.get_node_vars(u, equations, dg, i, j, element...)

        # Compute conservative part
        f2 = volume_flux(u_ll, u_rr, 2, equations) # orientation 2: y direction

        # Compute nonconservative part
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        f2_L = f2 + 0.5f0 * nonconservative_flux(u_ll, u_rr, 2, equations)
        f2_R = f2 + 0.5f0 * nonconservative_flux(u_rr, u_ll, 2, equations)

        # Copy to temporary storage
        Trixi.set_node_vars!(fstar2_L, f2_L, equations, dg, i, j)
        Trixi.set_node_vars!(fstar2_R, f2_R, equations, dg, i, j)
    end

    return nothing
end

function blend_cell_residual_fo!(el_x, el_y, eq::AbstractNonConservativeEquations{2},
                                 problem,
                                 scheme::Scheme{<:TrixiRKSolver},
                                 aux, t, dt, grid, dx, dy, xf, yf, op, u1, u_, cache, res,
                                 scaling_factor = 1.0)
    @timeit_debug aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack blend = aux

    alpha = blend.cache.alpha[el_x, el_y]

    r = @view res[:, :, :, el_x, el_y]

    @unpack trixi_ode = cache
    semi = trixi_ode.p
    @unpack volume_integral = semi.solver

    if alpha < 1e-12
        return nothing
    end

    lamx = dt / dx

    # limit the higher order part
    lmul!(1.0 - alpha, r)

    trixi_equations = get_trixi_equations(semi, eq)

    fv_kernel!(res, u1, semi.mesh,
               Trixi.have_nonconservative_terms(trixi_equations),
               trixi_equations,
               volume_integral.volume_flux_fv, semi.solver,
               semi.cache, cache, op, (el_x, el_y),
               lamx * alpha)

    return nothing
    end # timer
end

function compute_face_residual!(eq::AbstractNonConservativeEquations{2}, grid, op,
                                cache, problem,
                                scheme::Scheme{<:TrixiRKSolver}, param, aux, t, dt, u1,
                                fb, ub, ua, res, scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack bl, br, xg, wg, degree = op
    nd = degree + 1
    nx, ny = grid.size
    @unpack dx, dy, xf, yf = grid
    @unpack numerical_flux, solver = scheme
    @unpack blend = aux
    @unpack blend_face_residual_x!, blend_face_residual_y!, get_element_alpha = blend.subroutines

    @unpack trixi_ode = cache
    semi = trixi_ode.p
    @unpack cache = semi
    surface_flux_values = fb

    trixi_equations = get_trixi_equations(semi, eq)

    calc_interface_flux!(scheme.solver, surface_flux_values, semi.mesh,
                         Trixi.have_nonconservative_terms(trixi_equations),
                         trixi_equations, semi.solver.surface_integral, ub,
                         semi.solver,
                         grid, cache)

    @threaded for element in CartesianIndices((1:nx, 1:ny)) # Loop over cells
        el_x, el_y = element[1], element[2]
        for ix in Base.OneTo(nd)

            # For higher order residual
            for jy in Base.OneTo(nd)
                Fl = get_node_vars(fb, eq, jy, 1, el_x, el_y)
                Fr = get_node_vars(fb, eq, jy, 2, el_x, el_y)
                Fd = get_node_vars(fb, eq, ix, 3, el_x, el_y)
                Fu = get_node_vars(fb, eq, ix, 4, el_x, el_y)
                for n in eachvariable(eq)
                    res[n, ix, jy, el_x, el_y] += dt / dy[el_y] * br[jy] * Fu[n]
                    res[n, ix, jy, el_x, el_y] += dt / dy[el_y] * bl[jy] * Fd[n]
                    res[n, ix, jy, el_x, el_y] += dt / dx[el_x] * br[ix] * Fr[n]
                    res[n, ix, jy, el_x, el_y] += dt / dx[el_x] * bl[ix] * Fl[n]
                end
            end
        end
    end

    return nothing
    end # timer
end # compute_face_residual!
