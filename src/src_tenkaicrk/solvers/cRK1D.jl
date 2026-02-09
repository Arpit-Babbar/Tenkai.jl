using Tenkai: periodic, dirichlet, neumann, reflect, extrapolate, evaluate,
              update_ghost_values_periodic!,
              update_ghost_values_fn_blend!,
              get_node_vars, set_node_vars!,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, multiply_add_set_node_vars!,
              comp_wise_mutiply_node_vars!, flux, update_ghost_values_lwfr!,
              calc_source, store_low_flux!, blend_flux_only, get_blended_flux,
              get_first_node_vars, get_second_node_vars, trivial_cell_residual,
              sum_node_vars_1d

import Tenkai: compute_face_residual!, setup_arrays

using Tenkai.EqTenMoment2D: det_constraint

using SimpleUnPack
using TimerOutputs
using MuladdMacro
using OffsetArrays
using StaticArrays
using LinearAlgebra: norm, axpby!

using Tenkai: @threaded, alloc_for_threads
using Tenkai.Equations: nvariables, eachvariable
using Tenkai: refresh!

function setup_arrays(grid, scheme::Scheme{<:cSSP2IMEX433},
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

    cell_data_size = 14
    eval_data_size = 16

    MArr = MArray{Tuple{nvariables(eq), nd}, Float64}
    cell_data = alloc_for_threads(MArr, cell_data_size)

    MArr = MArray{Tuple{nvariables(eq), 1}, Float64}
    eval_data = alloc_for_threads(MArr, eval_data_size)

    cache = (; u1, ua, res, Fb, Ub, u1_b, ub_N, cell_data, eval_data)
    return cache
end

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function fo_blend_imex(eq::AbstractEquations{1, <:Any})
    (;
     blend_cell_residual! = blend_cell_residual_fo_imex!,
     blend_face_residual! = blend_face_residual_fo_imex!,
     name = "fo_imex")
end

#-------------------------------------------------------------------------------
# Ghost values function for the non-conservative part of the equation
#-------------------------------------------------------------------------------
function update_ghost_values_ub_N!(problem, scheme, eq::AbstractEquations{1},
                                   grid, aux, op, cache, t, dt)
    # TODO - Move this to its right place!!
    @unpack ub_N = cache
    nx = size(ub_N, 3) - 2
    nvar = size(ub_N, 1) # Temporary, should take from eq

    if problem.periodic_x
        # Left ghost cells
        copyto!(ub_N, CartesianIndices((1:nvar, 2:2, 0:0)),
                ub_N, CartesianIndices((1:nvar, 2:2, nx:nx)))

        # Right ghost cells
        copyto!(ub_N, CartesianIndices((1:nvar, 1:1, (nx + 1):(nx + 1))),
                ub_N, CartesianIndices((1:nvar, 1:1, 1:1)))

        return nothing
    end
    # Left ghost cells
    for n in 1:nvar
        ub_N[n, 2, 0] = ub_N[n, 1, 1]
        ub_N[n, 1, nx + 1] = ub_N[n, 2, nx]
    end

    return nothing
end

function update_ghost_values_Bb!(problem, scheme, eq::AbstractEquations{1},
                                 grid, aux, op, cache, t, dt)
    return nothing
end

# 1st order FV cell residual
@inbounds @inline function blend_cell_residual_fo_imex!(cell, eq::AbstractEquations{1},
                                                        problem, scheme, aux, lamx,
                                                        t, dt, dx, xf, op, u1, u, ua, f,
                                                        r,
                                                        scaling_factor = 1.0)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead, it's supposed
    #! format: noindent
    # to be 0.25 microseconds
    @unpack blend = aux
    @unpack source_terms = problem
    @unpack Vl, Vr, xg, wg = op
    fn_low = @view blend.fn_low[:, :, cell]
    num_flux = scheme.numerical_flux
    nd = length(xg)

    blend_res = @view blend.resl[:, :, cell]
    fill!(blend_res, zero(eltype(blend_res)))

    # if blend.alpha[cell] < 1e-12
    #     store_low_flux!(u, cell, xf, dx, op, blend, eq, scaling_factor)
    #     return nothing
    # end
    nvar = nvariables(eq)
    @unpack xxf, fn = blend
    # Get subcell faces
    xxf[0] = xf
    for ii in Base.OneTo(nd)
        xxf[ii] = xxf[ii - 1] + dx * wg[ii]
    end

    for j in 2:nd
        xx = xxf[j]
        ul, ur = get_node_vars(u, eq, j - 1), get_node_vars(u, eq, j)
        fl, fr = flux(xx, ul, eq), flux(xx, ur, eq)
        fn = scaling_factor * num_flux(xx, ul, ur, fl, fr, ul, ur, eq, 1)

        for n in 1:nvar
            blend_res[n, j - 1] += fn[n] / wg[j - 1]
            blend_res[n, j] -= fn[n] / wg[j]
        end
    end
    @views fn_low[:, 1] .= wg[1] * blend_res[:, 1]
    @views fn_low[:, 2] .= -wg[end] * blend_res[:, end]

    blend_res .*= dt / dx

    axpby!(blend.alpha[cell], blend_res, 1.0 - blend.alpha[cell], r)
    end # timer
end

@inbounds @inline function blend_face_residual_fo_imex!(el_x, xf, u1, ua,
                                                        eq::AbstractEquations{1},
                                                        t, dt, grid, op, problem,
                                                        scheme, param,
                                                        Fn, aux, lamx, res,
                                                        scaling_factor)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead,
    #! format: noindent
    # it's supposed to be 0.25 microseconds
    @unpack blend = aux
    @unpack source_terms = problem
    alpha = blend.alpha # factor of non-smooth part
    num_flux = scheme.numerical_flux
    @unpack dx = grid
    nvar = nvariables(eq)

    @unpack xg, wg = op
    nd = length(xg)
    alp = 0.5 * (alpha[el_x - 1] + alpha[el_x])
    # Reuse arrays to save memory
    @unpack fl, fr, fn, fn_low = blend

    # The lower order residual of blending scheme comes from lower order
    # numerical flux at the subcell faces. Here we deal with the residual that
    # occurs from those faces that are common to both the subcell and supercell

    # Low order numerical flux
    ul, ur = @views u1[:, nd, el_x - 1], u1[:, 1, el_x]
    fl = flux(xf, ul, eq)
    fr = flux(xf, ur, eq)
    fn = scaling_factor * num_flux(xf, ul, ur, fl, fr, ul, ur, eq, 1)

    # alp = test_alp(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op, alp)

    Fn = (1.0 - alp) * Fn + alp * fn

    Fn = get_blended_flux(el_x, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx,
                          op, alp)

    r = @view res[:, :, el_x - 1]
    blend_res = @view blend.resl[:, :, el_x - 1]
    # For the sub-cells which have same interface as super-cells, the same
    # numflux Fn is used in place of the lower order flux
    for n in 1:nvar
        # r[n,nd] += alpha[i-1] * dt/dx[i-1] * Fn_[n]/wg[nd] # alpha[i-1] already in blend.lamx
        # Add source terms and implicit solve here
        # You will also need fn_low to perform the implicit solve
        r[n, nd] += dt / (dx[el_x - 1] * wg[nd]) * alpha[el_x - 1] * Fn[n]
        blend_res[n, nd] += dt / (dx[el_x - 1] * wg[nd]) * Fn[n]
    end

    r = @view res[:, :, el_x]
    blend_res = @view blend.resl[:, :, el_x]
    for n in 1:nvar
        # r[n,1] -= alpha[i] * dt/dx[i] * Fn_[n]/wg[1] # alpha[i-1] already in blend.lamx
        # Add source terms and implicit solve here
        r[n, 1] -= dt / dx[el_x] * alpha[el_x] * Fn[n] / wg[1] # alpha[i-1] already in blend.lamx
        blend_res[n, 1] -= dt / dx[el_x] * Fn[n] / wg[1]
    end

    return Fn, (1.0 - alpha[el_x - 1], 1.0 - alpha[el_x])
    end # timer
end

import Tenkai: compute_cell_residual_cRK!, update_solution_cRK!

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cIMEX111}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack ub_N, ua, u1, res, Fb, Ub, = cache
    nx = grid.size

    refresh!(res) # Reset previously used variables to zero
    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        xc = grid.xc[cell]
        u_node = get_node_vars(u1, eq, 1, cell)

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

@inbounds @inline function blend_cell_residual_stiff!(blend_cell_residual!::typeof(blend_cell_residual_fo_imex!),
                                                      eq::AbstractEquations{1},
                                                      u1, res, grid, problem, op, t, dt,
                                                      aux)
    @timeit aux.timer "Update solution implicit source" begin
    #! format: noindent

    @unpack blend = aux
    @unpack source_terms = problem
    alpha = blend.alpha # factor of non-smooth part
    # num_flux = scheme.numerical_flux
    @unpack dx = grid
    nvar = nvariables(eq)

    @unpack xg, wg = op
    nd = length(xg)

    # Reuse arrays to save memory
    @unpack fl, fr, fn, fn_low = blend

    nx = grid.size
    @unpack xg = op
    @unpack source_terms = problem
    @unpack blend = aux
    blend_res = blend.resl
    nd = length(xg)

    # TODO - Remove scheme argument from get_cache_node_vars
    dummy_scheme = nothing

    for el_x in 1:nx
        alpha = blend.alpha[el_x]
        xc = grid.xc[el_x]
        dx = grid.dx[el_x]
        for i in 1:nd
            x = xc - 0.5 * dx + xg[i] * dx

            u_node = get_node_vars(u1, eq, i, el_x)
            blend_res_node = get_node_vars(blend_res, eq, i, el_x)
            lhs = u_node - blend_res_node # lhs in the implicit source solver

            # u_node used as initial guess
            aux_node = get_cache_node_vars(aux, u1, problem, dummy_scheme, eq, i,
                                           el_x)

            u_node_implicit, s_node_implicit = implicit_source_solve(lhs, eq, x,
                                                                     t + dt,
                                                                     dt,
                                                                     source_terms,
                                                                     aux_node)

            multiply_add_to_node_vars!(res, -dt * alpha, s_node_implicit, eq, i,
                                       el_x)
            multiply_add_to_node_vars!(blend_res, -dt, s_node_implicit, eq, i, el_x)
        end
    end
    @unpack pure_fv = aux.blend.parameters
    if pure_fv # TODO - Is this needed?
        res .= blend_res # Copy the blended residual to the res array
    end
    end # timer
end

@inbounds @inline function blend_cell_residual_stiff!(blend_cell_residual!::typeof(trivial_cell_residual),
                                                      eq, u1, res, grid, problem, op, t,
                                                      dt, aux)
    return nothing
end

@inbounds function update_with_residuals!(positivity_blending::PositivityBlending,
                                          eq::AbstractEquations{1}, grid, op, u1,
                                          res, aux)
    @unpack blend = aux
    res_blend = blend.resl
    @threaded for i in eachindex(res_blend)
        res_blend[i] = u1[i] - res_blend[i]
    end
    update_solution_lwfr!(u1, res, aux) # u1 = u1 - res

    u1_low = res_blend

    @unpack variables = positivity_blending
    apply_positivity_blending!(u1, u1_low, eq, grid, op, aux, variables)
end

function find_val_min(eq::AbstractEquations{1}, u, op, variable)
    nd = op.degree + 1
    val_min = 1e20
    for i in 1:nd
        val = variable(eq, get_node_vars(u, eq, i))
        val_min = min(val_min, val)
    end
    return val_min
end

function find_avg(eq::AbstractEquations{1}, u, op)
    @unpack wg = op
    nd = op.degree + 1
    u_avg = zero(get_node_vars(u, eq, 1))
    for i in 1:nd
        u_avg += wg[i] * get_node_vars(u, eq, i)
    end
    return u_avg
end

function find_theta_positivity_blending(eq::AbstractEquations{1},
                                        u_high_node, u_low_node, val_min, variable)
    val_high = variable(eq, u_high_node)
    if val_high > 0.0
        return 1.0
    end
    val_low = variable(eq, u_low_node)

    eps = 0.1 * val_min

    theta = abs(eps - val_low) / (abs(val_high - val_low) + 1e-13)
    return theta
end

function find_theta_positivity_blending(eq::AbstractEquations{1},
                                        u_high_node, u_low_node, val_min,
                                        variable::typeof(det_constraint))
    val_high = variable(eq, u_high_node)
    if val_high > 0.0
        return 1.0
    end

    # eps = min(0.1 * val_min, 1e-10)
    eps = 0.1 * val_min

    func(theta) = variable(eq, theta * u_high_node + (1 - theta) * u_low_node) - eps

    newton_accuracy = min(eps / 10, 1e-12)
    local niters
    if newton_accuracy < 1e-12
        niters = 100
    else
        niters = 10
    end

    initial_guess = 0.5 # 1 would be better, but it was going above 1 too often
    theta = newton_solver_scalar(func, initial_guess, newton_accuracy, niters)

    @assert theta >= 0.0 && theta <= 1.0

    return theta
end

@inbounds function apply_positivity_blending_to_variable!(u1, u1_low,
                                                          eq::AbstractEquations{1},
                                                          grid, op, aux, variable)
    nx = grid.size
    nd = op.degree + 1
    @unpack Vl, Vr = op
    # If it falls below zero, we will lift it up to 0.1 * min(u1_low[:, :, el_x, el_y])

    refresh!(u) = fill!(u, zero(eltype(u)))
    for el_x in 1:nx
        u1_ = @view u1[:, :, el_x]
        u1_low_ = @view u1_low[:, :, el_x]

        val_min = find_val_min(eq, u1_low_, op, variable)
        # @assert val_min>0.0 "Low order minimum $variable is negative: $val_min in cell $el_x"

        theta = 1.0

        for i in 1:nd
            u_high_node = get_node_vars(u1_, eq, i)

            u_low_node = get_node_vars(u1_low_, eq, i)
            theta_ = find_theta_positivity_blending(eq, u_high_node, u_low_node,
                                                    val_min, variable)
            theta = min(theta, theta_)
        end

        if theta < 1.0
            @. u1_ = theta * u1_ + (1.0 - theta) * u1_low_
        end

        u_ll = sum_node_vars_1d(Vl, u1, eq, 1:nd, el_x) # ul = ∑ Vl*u
        u_rr = sum_node_vars_1d(Vr, u1, eq, 1:nd, el_x) # ur = ∑ Vr*u

        # This requires limiting with the cell average, and we use a new theta
        theta = 1.0
        u_low_avg = find_avg(eq, u1_low_, op)
        v_avg = variable(eq, u_low_avg)

        theta_ll = find_theta_positivity_blending(eq, u_ll, u_low_avg, v_avg, variable)
        theta_rr = find_theta_positivity_blending(eq, u_rr, u_low_avg, v_avg, variable)
        theta = min(theta, theta_ll, theta_rr)

        if theta < 1.0
            for i in 1:nd
                u_high_node = get_node_vars(u1_, eq, i)
                u_node_new = theta * u_high_node + (1.0 - theta) * u_low_avg
                set_node_vars!(u1_, u_node_new, eq, i)
            end
        end
    end
end

@inbounds function apply_positivity_blending!(u1, u1_low, eq::AbstractEquations{1},
                                              grid, op, aux,
                                              variables::NTuple{N, Any}) where {N}
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    apply_positivity_blending_to_variable!(u1, u1_low, eq, grid, op, aux, variable)
    apply_positivity_blending!(u1, u1_low, eq, grid, op, aux, remaining_variables)
end

@inbounds function apply_positivity_blending!(u1, u1_low, eq::AbstractEquations{1},
                                              grid, op, aux, variables::Tuple{})
    return nothing
end

@inbounds @inline function update_with_residuals!(positivity_blending::NoPositivityBlending,
                                                  eq::AbstractEquations{1}, grid, op,
                                                  u1,
                                                  res, aux)
    update_solution_lwfr!(u1, res, aux)
end

# TODO - Unify with the 2D version
@inbounds @inline function update_solution_cRK!(u1, eq::AbstractEquations{1}, grid,
                                                op, problem, scheme, res, aux, t, dt)
    @timeit aux.timer "Update solution" begin
    #! format: noindent
    @unpack blend = aux

    # To check if the blending scheme is IMEX
    @unpack blend_cell_residual! = blend

    # Do the source term evolution
    blend_cell_residual_stiff!(blend_cell_residual!, eq, u1, res, grid, problem, op,
                               t, dt, aux)

    @unpack positivity_blending = blend.parameters

    update_with_residuals!(positivity_blending, eq, grid, op, u1, res, aux)
    end
end

@inbounds @inline function update_solution_cRK!(u1, eq::AbstractEquations{1}, grid,
                                                op, problem::Problem{<:Any},
                                                scheme::Scheme{<:cIMEX111}, res, aux,
                                                t, dt)
    @timeit aux.timer "Update solution" begin
    #! format: noindent
    @unpack source_terms = problem
    nx = grid.size

    for cell in Base.OneTo(nx)
        xc = grid.xc[cell]

        # To be used as initial guess for the implicit solve
        # The numerical flux has already been added to it. Is that okay?
        u_node = get_node_vars(u1, eq, 1, cell)

        res_node = get_node_vars(res, eq, 1, cell)

        lhs = u_node - res_node # lhs in the implicit source solver

        # lhs = u_node

        aux_node = get_cache_node_vars(aux, u1, problem, scheme, eq, 1, cell)

        # Implicit solver evolution
        u_node_implicit, s_node_implicit = implicit_source_solve(lhs, eq, xc, t, dt,
                                                                 source_terms,
                                                                 aux_node)

        multiply_add_to_node_vars!(u1, dt, s_node_implicit, eq, 1, cell)
        multiply_add_to_node_vars!(u1, -1.0, res_node, eq, 1, cell)

        set_node_vars!(u1, u_node_implicit, eq, 1, cell)
    end

    return nothing
    end # timer
end

function implicit_source_solve(lhs, eq, x, t, coefficient, source_terms, u_node,
                               implicit_solver = newton_solver)
    # TODO - Make sure that the final source computation is used after the implicit solve
    implicit_F(u_new) = u_new - lhs -
                        coefficient * calc_source(u_new, x, t, source_terms, eq)

    u_new = implicit_solver(implicit_F, u_node)
    source = calc_source(u_new, x, t, source_terms, eq)
    return u_new, source
end

function implicit_source_solve(lhs, eq, x, t, coefficient, source_terms::Nothing,
                               u_node)
    return lhs, zero(lhs)
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cHT112}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    F, f, U, u2, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

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

            set_node_vars!(F, 0.5 * flux1, eq, i)
            set_node_vars!(U, 0.5 * u_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u2, -1.0 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u2
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            s_node = calc_source(u_node, x_, t, source_terms, eq)
            set_node_vars!(S, 0.5 * s_node, eq, i)
            # Add source term contribution to u2
            multiply_add_to_node_vars!(u2, 0.5 * dt, s_node, eq, i)

            lhs = get_node_vars(u2, eq, i) # lhs in the implicit source solver

            # By default, it is just u_node but the user can use it to set something else here.
            aux_node = get_cache_node_vars(aux, u1, problem, scheme, eq, i, cell)

            u2_node_implicit, s2_node = implicit_source_solve(lhs, eq, x_,
                                                              t + dt, # TOTHINK - Somehow t instead of t + dt
                                                              # gives better accuracy, although it is
                                                              # not supposed to
                                                              0.5 * dt, source_terms,
                                                              aux_node) # aux_node used as initial guess
            set_node_vars!(u2, u2_node_implicit, eq, i)

            multiply_add_to_node_vars!(S, 0.5, s2_node, eq, i)
        end

        for i in Base.OneTo(nd)
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i)

            flux1 = flux(x_, u2_node, eq)

            multiply_add_to_node_vars!(F, 0.5, flux1, eq, i)
            multiply_add_to_node_vars!(U, 0.5, u2_node, eq, i)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end

            S_node = get_node_vars(S, eq, i)

            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]

        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)

        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            ul, ur, u2l, u2r = eval_data[Threads.threadid()]
            refresh!.((ul, ur, u2l, u2r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u1, eq, i, cell)
                u2_node = get_node_vars(u2, eq, i)
                multiply_add_to_node_vars!(ul, Vl[i], u_node, eq, 1)
                multiply_add_to_node_vars!(ur, Vr[i], u_node, eq, 1)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
            end
            ul_node = get_node_vars(ul, eq, 1)
            ur_node = get_node_vars(ur, eq, 1)
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)
            set_node_vars!(Fb, 0.5 * (fl + f2l), eq, 1, cell)
            set_node_vars!(Fb, 0.5 * (fr + f2r), eq, 2, cell)
        end
    end
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cSSP2IMEX222}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell Residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    gamma = 1.0 - 1.0 / sqrt(2.0)

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    F, f, U, u2, u3, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]
        u3 .= @view u1[:, :, cell]

        # Compute u2 and add its contributions to u3
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            aux_node = get_cache_node_vars(aux, u1, problem, scheme, eq, i, cell)
            lhs = u_node # lhs in the implicit source solver
            u2_node, s2_node = implicit_source_solve(lhs, eq, x_, t + gamma * dt,
                                                     gamma * dt,
                                                     source_terms, aux_node)
            set_node_vars!(u2, u2_node, eq, i)

            flux1 = flux(x_, u2_node, eq)

            set_node_vars!(F, 0.5 * flux1, eq, i)
            set_node_vars!(U, 0.5 * u2_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u3, -lamx * Dm[ii, i], flux1, eq, ii)
            end

            multiply_add_to_node_vars!(u3, (1.0 - 2.0 * gamma) * dt, s2_node, eq, i)
            set_node_vars!(S, 0.5 * s2_node, eq, i)
        end

        # Add source term contribution to u3
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u3, eq, i) # lhs in the implicit source solver
            aux_node = get_cache_node_vars(aux, u2, problem, scheme, eq, i, 1)
            u3_node, s3_node = implicit_source_solve(lhs, eq, x_,
                                                     t + (1.0 - gamma) * dt,
                                                     gamma * dt, source_terms,
                                                     aux_node)
            set_node_vars!(u3, u3_node, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_to_node_vars!(F, 0.5, flux1, eq, i)
            multiply_add_to_node_vars!(U, 0.5, u3_node, eq, i)
            multiply_add_to_node_vars!(S, 0.5, s3_node, eq, i)
        end

        for i in Base.OneTo(nd)
            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix,
                                           cell)
            end
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end
        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            u2l, u2r, u3l, u3r = eval_data[Threads.threadid()]
            refresh!.((u2l, u2r, u3l, u3r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u2_node = get_node_vars(u2, eq, i)
                u3_node = get_node_vars(u3, eq, i)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
            end
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            set_node_vars!(Fb, 0.5 * (f2l + f3l), eq, 1, cell)
            set_node_vars!(Fb, 0.5 * (f2r + f3r), eq, 2, cell)
        end
    end
    end # timer
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cSSP2IMEX332}, aux, t, dt,
                                    cache)
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    at21 = at31 = at32 = 0.5
    bt1 = bt2 = bt3 = 1.0 / 3.0

    a11 = a22 = 0.25
    a31 = a32 = a33 = 1.0 / 3.0
    c1, c2, c3 = 0.25, 0.25, 1.0
    b1, b2, b3 = bt1, bt2, bt3

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

        # Compute u2 and add its contributions to u3, u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            lhs = u_node # lhs in the implicit source solver
            aux_node = get_cache_node_vars(aux, u1, problem, scheme, eq, i, cell)
            u2_node, s2_node = implicit_source_solve(lhs, eq, x_, t + c1 * dt, a11 * dt,
                                                     source_terms, aux_node)
            set_node_vars!(u2, u2_node, eq, i)

            flux1 = flux(x_, u2_node, eq)

            set_node_vars!(F, bt1 * flux1, eq, i)
            set_node_vars!(U, bt1 * u2_node, eq, i)

            multiply_add_to_node_vars!(u4, a31 * dt, s2_node, eq, i)
            set_node_vars!(S, b1 * s2_node, eq, i)

            # Flux derivative contribution to u3, u4
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u3, -at21 * lamx * Dm[ii, i], flux1, eq, ii)
                multiply_add_to_node_vars!(u4, -at31 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u3
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u2_node = get_node_vars(u2, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u3, eq, i) # lhs in the implicit source solver
            aux_node = get_cache_node_vars(aux, u2, problem, scheme, eq, i, 1)
            u3_node, s3_node = implicit_source_solve(lhs, eq, x_, t + c2 * dt,
                                                     a22 * dt, source_terms, aux_node)
            set_node_vars!(u3, u3_node, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_to_node_vars!(F, bt2, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt2, u3_node, eq, i)
            multiply_add_to_node_vars!(S, b2, s3_node, eq, i)
            multiply_add_to_node_vars!(u4, a32 * dt, s3_node, eq, i)
        end

        # Add flux derivative terms to u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u3_node = get_node_vars(u3, eq, i)
            flux1 = flux(x_, u3_node, eq)
            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u4, -at32 * lamx * Dm[ii, i], flux1, eq, ii)
            end
        end

        # Add source term contribution to u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u3_node = get_node_vars(u3, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u4, eq, i) # lhs in the implicit source solver
            aux_node = get_cache_node_vars(aux, u3, problem, scheme, eq, i, 1)
            u4_node, s4_node = implicit_source_solve(lhs, eq, x_, t + c3 * dt,
                                                     a33 * dt, source_terms, aux_node)
            set_node_vars!(u4, u4_node, eq, i)

            flux1 = flux(x_, u4_node, eq)

            multiply_add_to_node_vars!(F, bt3, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt3, u4_node, eq, i)
            multiply_add_to_node_vars!(S, b3, s4_node, eq, i)

            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix, cell)
            end
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end

        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            u2l, u2r, u3l, u3r, u4l, u4r = eval_data[Threads.threadid()]
            refresh!.((u2l, u2r, u3l, u3r, u4l, u4r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u2_node = get_node_vars(u2, eq, i)
                u3_node = get_node_vars(u3, eq, i)
                u4_node = get_node_vars(u4, eq, i)
                multiply_add_to_node_vars!(u2l, Vl[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u2r, Vr[i], u2_node, eq, 1)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u4l, Vl[i], u4_node, eq, 1)
                multiply_add_to_node_vars!(u4r, Vr[i], u4_node, eq, 1)
            end
            u2l_node = get_node_vars(u2l, eq, 1)
            u2r_node = get_node_vars(u2r, eq, 1)
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            u4l_node = get_node_vars(u4l, eq, 1)
            u4r_node = get_node_vars(u4r, eq, 1)
            f2l, f2r = flux(xl, u2l_node, eq), flux(xr, u2r_node, eq)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            f4l, f4r = flux(xl, u4l_node, eq), flux(xr, u4r_node, eq)
            set_node_vars!(Fb, (f2l + f3l + f4l) / 3.0, eq, 1, cell)
            set_node_vars!(Fb, (f2r + f3r + f4r) / 3.0, eq, 2, cell)
        end
    end
end

function compute_cell_residual_cRK!(eq::AbstractEquations{1}, grid, op,
                                    problem, scheme::Scheme{<:cSSP2IMEX433}, aux, t, dt,
                                    cache)
    @timeit aux.timer "Cell Residual" begin
    #! format: noindent
    @unpack source_terms = problem
    @unpack xg, wg, Dm, D1, Vl, Vr = op
    nd = length(xg)
    nx = grid.size
    @unpack bflux_ind = scheme.bflux
    @unpack blend = aux

    alpha = 0.24169426078821
    beta = 0.06042356519705
    eta = 0.12915286960590

    at32 = 1.0
    at42 = at43 = 0.25

    ct3 = 1.0
    ct4 = 0.5

    bt2, bt3, bt4 = 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0

    a11 = a22 = a33 = a44 = alpha
    a21 = -alpha
    a32 = 1.0 - alpha
    a41 = beta
    a42 = eta
    a43 = 0.5 - beta - eta - alpha

    c1 = alpha
    c2 = 0.0
    c3 = 1.0
    c4 = 0.5
    b2, b3, b4 = bt2, bt3, bt4

    @unpack cell_data, eval_data, ua, u1, res, Fb, Ub = cache

    # TODO - These u2_sec, u3_sec, u4_sec also needed to be added to other solvers.
    # This is also a rather inefficient way of doing this.
    # TODO (More urgent): Raise an issue about this
    F, f, U, u2, u3, u4, u5, u2_sec, u3_sec, u4_sec, S = cell_data[Threads.threadid()]

    refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

    @inbounds for cell in Base.OneTo(nx) # Loop over cells
        dx = grid.dx[cell]
        xc = grid.xc[cell]
        ignored_cell = UsuallyIgnored(cell)
        lamx = dt / dx
        u2 .= @view u1[:, :, cell]
        u3 .= @view u1[:, :, cell]
        u4 .= @view u1[:, :, cell]
        u5 .= @view u1[:, :, cell]

        # Compute u2, u3 and add their contributions to u3, u4, u5
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_node = get_node_vars(u1, eq, i, cell)
            lhs = u_node # lhs in the implicit source solver
            aux_node = get_cache_node_vars(aux, u1, problem, scheme, eq, i, cell)
            u2_node, s2_node = implicit_source_solve(lhs, eq, x_, t + c1 * dt,
                                                     a11 * dt,
                                                     source_terms, aux_node)
            set_node_vars!(u2, u2_node, eq, i)

            # These will be evolved without source terms for better initial guess
            set_node_vars!(u2_sec, aux_node, eq, i)
            set_node_vars!(u3_sec, aux_node, eq, i)
            set_node_vars!(u4_sec, aux_node, eq, i)

            flux1 = flux(x_, u2_node, eq)

            multiply_add_to_node_vars!(u3, a21 * dt, s2_node, eq, i)
            # multiply_add_to_node_vars!(u4, a31 * dt, s2_node, eq, i) # Not needed because a31 = 0
            multiply_add_to_node_vars!(u5, a41 * dt, s2_node, eq, i)

            lhs = get_node_vars(u3, eq, i) # lhs in the implicit source solver
            # aux_node = get_node_vars(u2_sec, eq, i)
            aux_node = get_cache_node_vars(aux, u2_sec, problem, scheme, eq, ignored_cell, i)
            u3_node, s3_node = implicit_source_solve(lhs, eq, x_, t + c2 * dt,
                                                     a22 * dt,
                                                     source_terms,
                                                     aux_node)

            set_node_vars!(u3, u3_node, eq, i)

            flux1 = flux(x_, u3_node, eq)

            multiply_add_set_node_vars!(F, bt2, flux1, eq, i)
            multiply_add_set_node_vars!(U, bt2, u3_node, eq, i)
            multiply_add_set_node_vars!(S, b2, s3_node, eq, i)
            multiply_add_to_node_vars!(u4, a32 * dt, s3_node, eq, i)
            multiply_add_to_node_vars!(u5, a42 * dt, s3_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u4, -at32 * lamx * Dm[ii, i], flux1, eq,
                                           ii)
                multiply_add_to_node_vars!(u5, -at42 * lamx * Dm[ii, i], flux1, eq,
                                           ii)

                multiply_add_to_node_vars!(u4_sec, -at32 * lamx * Dm[ii, i], flux1,
                                           eq,
                                           ii)
            end
        end

        # Add source term contribution to u4
        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            # TODO - Should initial guess be lhs?
            guess_u4 = get_node_vars(u3, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u4, eq, i) # lhs in the implicit source solver
            # aux_node = get_node_vars(u3_sec, eq, i)
            aux_node = get_cache_node_vars(aux, u3_sec, problem, scheme, eq, ignored_cell, i)
            u4_node, s4_node = implicit_source_solve(lhs, eq, x_, t + c3 * dt,
                                                     a33 * dt,
                                                     source_terms,
                                                     aux_node)
            set_node_vars!(u4, u4_node, eq, i)

            flux1 = flux(x_, u4_node, eq)

            multiply_add_to_node_vars!(F, bt3, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt3, u4_node, eq, i)

            for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
                multiply_add_to_node_vars!(u5, -at43 * lamx * Dm[ii, i], flux1, eq,
                                           ii)
            end

            multiply_add_to_node_vars!(u5, a43 * dt, s4_node, eq, i)
            multiply_add_to_node_vars!(S, b3, s4_node, eq, i)
        end

        for i in 1:nd
            x_ = xc - 0.5 * dx + xg[i] * dx
            u_guess = get_node_vars(u4, eq, i) # Initial guess in the implicit solver
            lhs = get_node_vars(u5, eq, i) # lhs in the implicit source solver
            # aux_node = get_node_vars(u4_sec, eq, i)
            aux_node = get_cache_node_vars(aux, u4_sec, problem, scheme, eq, ignored_cell, i)
            u5_node, s5_node = implicit_source_solve(lhs, eq, x_, t + c4 * dt,
                                                     a44 * dt, source_terms,
                                                     aux_node)
            set_node_vars!(u5, u5_node, eq, i)

            flux1 = flux(x_, u5_node, eq)

            multiply_add_to_node_vars!(F, bt4, flux1, eq, i)
            multiply_add_to_node_vars!(U, bt4, u5_node, eq, i)
            multiply_add_to_node_vars!(S, b4, s5_node, eq, i)

            F_node = get_node_vars(F, eq, i)
            for ix in Base.OneTo(nd)
                multiply_add_to_node_vars!(res, lamx * D1[ix, i], F_node, eq, ix,
                                           cell)
            end
            S_node = get_node_vars(S, eq, i)
            multiply_add_to_node_vars!(res, -dt, S_node, eq, i, cell)
        end

        u = @view u1[:, :, cell]
        r = @view res[:, :, cell]
        blend.blend_cell_residual!(cell, eq, problem, scheme, aux, lamx, t, dt, dx,
                                   grid.xf[cell], op, u1, u, cache.ua, f, r)
        # Interpolate to faces
        for i in Base.OneTo(nd)
            U_node = get_node_vars(U, eq, i)
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
            u3l, u3r, u4l, u4r, u5l, u5r = eval_data[Threads.threadid()]
            refresh!.((u3l, u3r, u4l, u4r, u5l, u5r))
            xl, xr = grid.xf[cell], grid.xf[cell + 1]
            for i in Base.OneTo(nd)
                u3_node = get_node_vars(u3, eq, i)
                u4_node = get_node_vars(u4, eq, i)
                u5_node = get_node_vars(u5, eq, i)
                multiply_add_to_node_vars!(u3l, Vl[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u3r, Vr[i], u3_node, eq, 1)
                multiply_add_to_node_vars!(u4l, Vl[i], u4_node, eq, 1)
                multiply_add_to_node_vars!(u4r, Vr[i], u4_node, eq, 1)
                multiply_add_to_node_vars!(u5l, Vl[i], u5_node, eq, 1)
                multiply_add_to_node_vars!(u5r, Vr[i], u5_node, eq, 1)
            end
            u3l_node = get_node_vars(u3l, eq, 1)
            u3r_node = get_node_vars(u3r, eq, 1)
            u4l_node = get_node_vars(u4l, eq, 1)
            u4r_node = get_node_vars(u4r, eq, 1)
            u5l_node = get_node_vars(u5l, eq, 1)
            u5r_node = get_node_vars(u5r, eq, 1)
            f3l, f3r = flux(xl, u3l_node, eq), flux(xr, u3r_node, eq)
            f4l, f4r = flux(xl, u4l_node, eq), flux(xr, u4r_node, eq)
            f5l, f5r = flux(xl, u5l_node, eq), flux(xr, u5r_node, eq)
            set_node_vars!(Fb, bt2 * f3l + bt3 * f4l + bt4 * f5l, eq, 1, cell)
            set_node_vars!(Fb, bt2 * f3r + bt3 * f4r + bt4 * f5r, eq, 2, cell)
        end
    end
    end # timer
end
end # muladd
