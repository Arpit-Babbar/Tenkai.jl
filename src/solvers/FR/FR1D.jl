module FR1D

using StaticArrays

using ..Tenkai: fr_dir, lwfr_dir, rkfr_dir, eq_dir, src_dir
using ..Tenkai: update_ghost_values_periodic!, flux, con2prim
(using ..FR: Scheme, Problem, Parameters, @threaded, PlotData,
             get_filename, minmod, prim2con!, con2prim!, save_solution,
             finite_differences, zhang_shu_flux_fix)
using ..Equations: AbstractEquations, nvariables, eachvariable
using ..Basis: Vandermonde_lag, weights_and_points, nodal2modal, nodal2modal_krivodonova,
               Vandermonde_leg_krivodonova

(import ..Tenkai: update_ghost_values_periodic!,
                  modal_smoothness_indicator,
                  modal_smoothness_indicator_gassner,
                  update_ghost_values_u1!,
                  update_ghost_values_fn_blend!,
                  set_initial_condition!,
                  compute_cell_average!,
                  get_cfl,
                  compute_time_step,
                  compute_face_residual!,
                  apply_tvb_limiter!,
                  apply_tvb_limiterβ!,
                  setup_limiter_tvb,
                  setup_limiter_tvbβ,
                  apply_hierarchical_limiter!,
                  Hierarchical,
                  apply_bound_limiter!,
                  Blend,
                  fo_blend,
                  mh_blend,
                  limit_slope,
                  no_upwinding_x,
                  is_admissible,
                  set_blend_dt!,
                  compute_error,
                  initialize_plot,
                  write_soln!,
                  create_aux_cache,
                  write_poly,
                  write_soln!,
                  post_process_soln)

(using ..FR: periodic, dirichlet, neumann, reflect,
             get_node_vars, set_node_vars!,
             get_first_node_vars, get_second_node_vars,
             add_to_node_vars!, subtract_from_node_vars!,
             multiply_add_to_node_vars!, multiply_add_set_node_vars!,
             comp_wise_mutiply_node_vars!)

using UnPack
using MuladdMacro
using TimerOutputs
using Printf
using OffsetArrays # OffsetArray, OffsetMatrix, OffsetVector
using ElasticArrays
using Plots
using DelimitedFiles
using LinearAlgebra: dot, mul!, BLAS, axpby!, I
using WriteVTK
using JSON3
using LoopVectorization
import Trixi

# TOTHINK - Consider using named tuples in place of all structs on which dispatch
# isn't done

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#-------------------------------------------------------------------------------
# Set initial condition by interpolation in all real cells
#-------------------------------------------------------------------------------
function set_initial_condition!(u, eq::AbstractEquations{1}, grid, op, problem)
    println("Setting initial condition")
    @unpack initial_value = problem
    nx = grid.size
    @unpack xg = op
    nd = length(xg)
    for i in 1:nx
        dx = grid.dx[i] # cell size
        xc = grid.xc[i] # cell center
        for ii in 1:nd
            x = xc - 0.5 * dx + xg[ii] * dx
            u[:, ii, i] .= initial_value(x)
        end
    end
    return nothing
end

#-------------------------------------------------------------------------------
# Compute cell average in all real cells
#-------------------------------------------------------------------------------
function compute_cell_average!(ua, u1, t, eq::AbstractEquations{1}, grid, problem,
                               scheme, aux, op)
    @timeit aux.timer "Cell averaging" begin
        nx = grid.size
        @unpack wg, Vl, Vr = op
        nvar = nvariables(eq)
        for i in 1:nx
            for n in 1:nvar
                u = @view u1[n, :, i]
                ua[n, i] = dot(wg, u)
            end
        end
        # Update ghost values of ua by periodicity or with face averages
        if problem.periodic_x
            for n in 1:nvar
                ua[n, 0] = ua[n, nx]
                ua[n, nx + 1] = ua[n, 1]
            end
        else
            for n in 1:nvar
                ua[n, 0] = ua[n, 1]
                ua[n, nx + 1] = ua[n, nx]
            end
            left, right = problem.boundary_condition
            if left == reflect
                ua[2, 0] = -ua[2, 0]
            end
            if right == reflect
                ua[2, nx + 1] = -ua[2, nx + 1]
            end
        end
        return nothing
    end # timer
end

#-------------------------------------------------------------------------------
# Choose cfl based on degree and correction function
#-------------------------------------------------------------------------------
function get_cfl(eq::AbstractEquations{1}, scheme, param)
    @unpack solver, degree, correction_function = scheme
    @unpack cfl_safety_factor, cfl_style = param
    @unpack dissipation = scheme
    @assert (degree >= 0&&degree < 5) "Invalid degree"
    os_vector(v) = OffsetArray(v, OffsetArrays.Origin(0))
    if solver == "lwfr" || cfl_style == "lw"
        if dissipation == get_second_node_vars # Diss 2
            cfl_radau = os_vector([1.0, 0.333, 0.170, 0.103, 0.069])
            cfl_g2 = os_vector([1.0, 1.000, 0.333, 0.170, 0.103])
            if solver == "rkfr"
                println("Using LW-D2 CFL with RKFR")
            else
                println("Using LW-D2 CFL with LW-D2")
            end
        elseif dissipation == get_first_node_vars # Diss 1
            cfl_radau = os_vector([1.0, 0.226, 0.117, 0.072, 0.049])
            cfl_g2 = os_vector([1.0, 0.465, 0.204, 0.116, 0.060])
            if solver == "rkfr"
                println("Using LW-D1 CFL with RKFR")
            else
                println("Using LW-D1 CFL with LW-D1")
            end
        end
    elseif solver == "rkfr"
        cfl_radau = os_vector([1.0, 0.333, 0.209, 0.145, 0.110])
        cfl_g2 = os_vector([1.0, 1.0, 0.45, 0.2875, 0.212])
    elseif solver == "mdrk"
        cfl_radau = os_vector([1.0, 0.333, 0.170, 0.107, 0.069])
        cfl_g2 = os_vector([1.0, 1.000, 0.333, 0.224, 0.103])
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

#-------------------------------------------------------------------------------
# Compute dt using cell average
#-------------------------------------------------------------------------------
function compute_time_step(eq::AbstractEquations{1, 1}, grid, aux, op, cfl, u1,
                           ua)
    @timeit aux.timer "Time step computation" begin
        speed = eq.speed
        nx = grid.size
        xc = grid.xc
        dx = grid.dx
        den = 0.0
        for i in 1:nx
            sx = @views speed(xc[i], ua[:, i], eq)
            den = max(den, abs.(sx) / dx[i] + 1.0e-12)
        end
        dt = cfl / den
        return dt
    end # Timer
end

#-------------------------------------------------------------------------------
# Compute flux f at all solution points in one cell
# Not in use, not supported
#-------------------------------------------------------------------------------
@inline function compute_flux!(eq::AbstractEquations{1}, flux, x, u, f)
    nd = length(x)
    for ii in 1:nd
        @views f[:, ii] .= flux(x[ii], u[:, ii], eq) # f[:,ii] = flux(x[ii],u[:,ii])
    end
    return nothing
end

#-------------------------------------------------------------------------------
# Interpolate average flux and solution to the two faces of cell
# Not being used, not supported
#-------------------------------------------------------------------------------
@inline function interpolate_to_face1D!(Vl, Vr, U, Ub)
    nvar, nd = size(U)
    @views mul!(Ub[:, 1], U, Vl)
    @views mul!(Ub[:, 2], U, Vr)
    return nothing
end

#-------------------------------------------------------------------------------
# Add numerical flux to residual
#-------------------------------------------------------------------------------
function compute_face_residual!(eq::AbstractEquations{1}, grid, op, scheme,
                                param, aux, t, dt, u1, Fb, Ub, ua, res)
    @timeit aux.timer "Face residual" begin
        @unpack xg, wg, bl, br = op
        nd = op.degree + 1
        nx = grid.size
        @unpack dx, xf = grid
        num_flux = scheme.numerical_flux
        @unpack blend = aux

        # Vertical faces, x flux
        for i in 1:(nx + 1)
            # Face between i-1 and i
            x = xf[i]
            @views Fn = num_flux(x, ua[:, i - 1], ua[:, i],
                                 Fb[:, 2, i - 1], Fb[:, 1, i],
                                 Ub[:, 2, i - 1], Ub[:, 1, i], eq, 1)
            Fn, blend_fac = blend.blend_face_residual!(i, x, u1, ua, eq,
                                                       dt, grid, op,
                                                       scheme, param,
                                                       Fn, aux, nothing,
                                                       res)
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

#-------------------------------------------------------------------------------
# Fill some data in ghost cells using periodicity
#-------------------------------------------------------------------------------
function update_ghost_values_u1!(eq::AbstractEquations{1}, problem, grid, op, u1, aux,
                                 t)
    nx = grid.size
    nd = op.degree + 1
    nvar = size(u1, 1)
    if problem.periodic_x
        copyto!(u1, CartesianIndices((1:nvar, 1:nd, 0:0)),
                u1, CartesianIndices((1:nvar, 1:nd, nx:nx)))
        copyto!(u1, CartesianIndices((1:nvar, 1:nd, (nx + 1):(nx + 1))),
                u1, CartesianIndices((1:nvar, 1:nd, 1:1)))
        return nothing
    end
    left, right = problem.boundary_condition
    boundary_value = problem.boundary_value
    xf = grid.xf
    if left == dirichlet
        x = xf[1]
        ub = boundary_value(x, t)
        for n in 1:nvar
            u1[n, 1:nd, 0] .= ub[n]
        end
    elseif left == neumann
        for n in 1:nvar
            u1[n, 1:nd, 0] = u1[n, 1:nd, 1]
        end
    elseif left == reflect
        for n in 1:nvar
            u1[n, 1:nd, 0] = u1[n, 1:nd, 1]
        end
        u1[2, 1:nd, 0] = -u1[2, 1:nd, 0]
    else
        println("Incorrect bc specified at left.")
        @assert false
    end

    if right == dirichlet
        x = xf[nx + 1]
        ub = boundary_value(x, t)
        for n in 1:nvar
            u1[n, 1:nd, nx + 1] .= ub[n]
        end
    elseif right == neumann
        for n in 1:nvar
            u1[n, 1:nd, nx + 1] = u1[n, 1:nd, nx]
        end
    elseif right == reflect
        for n in 1:nvar
            u1[n, 1:nd, nx + 1] = u1[n, 1:nd, nx]
        end
        u1[2, 1:nd, nx + 1] = -u1[2, 1:nd, nx + 1]
    else
        println("Incorrect bc specified at right.")
        @assert false
    end
end

function update_ghost_values_periodic!(eq::AbstractEquations{1}, problem, Fb,
                                       Ub)
    nx = size(Fb, 3) - 2
    nvar = size(Fb, 1) # Temporary, should take from eq
    if problem.periodic_x
        # Left ghost cells
        copyto!(Ub, CartesianIndices((1:nvar, 2:2, 0:0)),
                Ub, CartesianIndices((1:nvar, 2:2, nx:nx)))
        copyto!(Fb, CartesianIndices((1:nvar, 2:2, 0:0)),
                Fb, CartesianIndices((1:nvar, 2:2, nx:nx)))

        # Right ghost cells
        copyto!(Ub, CartesianIndices((1:nvar, 1:1, (nx + 1):(nx + 1))),
                Ub, CartesianIndices((1:nvar, 1:1, 1:1)))
        copyto!(Fb, CartesianIndices((1:nvar, 1:1, (nx + 1):(nx + 1))),
                Fb, CartesianIndices((1:nvar, 1:1, 1:1)))
    end
    return nothing
end

function update_ghost_values_fn_blend!(eq::AbstractEquations{1}, problem, grid,
                                       aux)
    @unpack blend = aux
    @unpack fn_low = blend
    nx = grid.size
    nvar = size(fn_low, 1)
    if problem.periodic_x
        copyto!(fn_low, CartesianIndices((1:nvar, 2:2, 0:0)),
                fn_low, CartesianIndices((1:nvar, 2:2, nx:nx)))
        copyto!(fn_low, CartesianIndices((1:nvar, 1:1, (nx + 1):(nx + 1))),
                fn_low, CartesianIndices((1:nvar, 1:1, 1:1)))
        return nothing
    else
        for n in 1:nvar
            fn_low[n, 2, 0] = fn_low[n, 2, 1]
            fn_low[n, 1, nx + 1] = fn_low[n, 1, nx]
        end
    end
end

#-------------------------------------------------------------------------------
# Limiters
#-------------------------------------------------------------------------------
# Correct one variable in bound correction

function correct_variable_bound_limiter!(variable, eq, grid, op, ua, u1)
    @unpack Vl, Vr = op
    nx = grid.size
    nd = op.degree + 1
    eps = 1e-10 # TODO - Get a better one
    for element in 1:nx
        var_ll = var_rr = 0.0
        var_min = 1e20
        for i in Base.OneTo(nd)
            u_node = get_node_vars(u1, eq, i, element)
            var = variable(eq, u_node)
            var_ll += var * Vl[i]
            var_rr += var * Vr[i]
            var_min = min(var_min, var)
        end
        var_min = min(var_min, var_ll, var_rr)
        ua_ = get_node_vars(ua, eq, element)
        var_avg = variable(eq, ua_)
        @assert var_avg>0.0 "Failed at element $element", var_avg
        eps_ = min(eps, 0.1 * var_avg)
        ratio = abs(eps_ - var_avg) / (abs(var_min - var_avg) + 1e-13)
        theta = min(ratio, 1.0) # theta for preserving positivity of density
        if theta < 1.0
            for i in 1:nd
                u_node = get_node_vars(u1, eq, i, element)
                multiply_add_set_node_vars!(u1,
                                            theta, u_node,
                                            1 - theta, ua_,
                                            eq, i, element)
            end
        end
    end
end

# Bounds preserving limiter
function apply_bound_limiter!(eq::AbstractEquations{1, 1}, grid, scheme, param, op,
                              ua, u1, aux)
    if scheme.bound_limit != "yes"
        return nothing
    end
    @timeit aux.timer "Bounds limiter" begin
        m, M = param.bounds[1][1], param.bounds[2][1]
        V = op.Vgll # Vandermonde matrix to convert to gll point values
        nx = grid.size
        nd = op.degree + 1
        ue, up = zeros(nd), zeros(nd) # for gl, gll point values
        eps = 1e-12
        ua_ = @view ua[1, :]
        if (minimum(ua_) < m - eps || maximum(ua_) > M + eps)
            println("Min, max = ", minimum(ua_), " ", maximum(ua_))
            @assert false "Bounds not preserved"
        end
        for i in 1:nx
            ua_[i] = max(ua_[i], m)
            ua_[i] = min(ua_[i], M)
        end
        for i in 1:nx
            # get cell face values
            ue = u1[1, :, i]  # Copy of u1
            mul!(up, V, ue) # Store GLL point values in ue
            Mi, mi = (max(maximum(ue), maximum(up)),
                      min(minimum(ue), minimum(up)))
            ratio1 = abs(M - ua_[i]) / (abs(Mi - ua_[i]) + 1e-13)
            ratio2 = abs(m - ua_[i]) / (abs(mi - ua_[i]) + 1e-13)
            theta = min(ratio1, ratio2, 1.0)
            for k in 1:nd
                up[k] = theta * (ue[k] - ua_[i]) + ua_[i] # up = θ(up-ua)+ua
            end
            u1[1, :, i] .= up
        end
        return nothing
    end # timer
end

function setup_limiter_tvb(eq::AbstractEquations{1}; tvbM = 0.0)
    cache_size = 13
    cache = SVector{cache_size}([MArray{Tuple{nvariables(eq), 1}, Float64}(undef)
                                 for _ in Base.OneTo(cache_size)])
    limiter = (; name = "tvb", tvbM = tvbM, cache)
    return limiter
end

function setup_limiter_tvbβ(eq::AbstractEquations{1}; tvbM = 0.0)
    @assert false "Limiter not implemented"
end

# TVB limiter
function apply_tvb_limiter!(eq::AbstractEquations{1, 1}, problem, scheme, grid,
                            param, op, ua, u1, aux)
    @timeit aux.timer "TVB limiter" begin
        nx = grid.size
        @unpack xg, wg, Vl, Vr = op
        @unpack tvbM = scheme.limiter
        nd = length(wg)
        u1_ = @view u1[1, :, :]
        ua_ = @view ua[1, :]
        # Loop over cells
        for el_x in 1:nx
            # face values
            ul, ur = 0.0, 0.0
            for ii in 1:nd
                ul += u1_[ii, el_x] * Vl[ii]
                ur += u1_[ii, el_x] * Vr[ii]
            end
            # slopes b/w centres and faces
            dul, dur = ua_[el_x] - ul, ur - ua_[el_x]
            # minmod to detect jumps
            Mdx2 = tvbM * grid.dx[el_x]^2
            dulm = minmod(dul, ua_[el_x] - ua_[el_x - 1], ua_[el_x + 1] - ua_[el_x],
                          Mdx2)
            durm = minmod(dur, ua_[el_x] - ua_[el_x - 1], ua_[el_x + 1] - ua_[el_x],
                          Mdx2)
            # limit if jumps are detected
            if abs(dul - dulm) > 1e-06 || abs(dur - durm) > 1e-06
                dux = 0.5 * (dulm + durm)
                for ii in 1:nd
                    u1_[ii, el_x] = ua_[el_x] + 2.0 * (xg[ii] - 0.5) * dux
                end
            end
        end
        return nothing
    end # timer
end

function apply_tvb_limiterβ!(eq::AbstractEquations{1, 1}, problem, scheme, grid,
                             param, op, ua, u1, aux)
    @assert false "Limiter not implemented"
end

function apply_hierarchical_limiter!(eq::AbstractEquations{1}, # 1D equations
                                     problem, scheme, grid,
                                     param, op, ua, u1, aux)
    @unpack hierarchical = aux
    @unpack alpha, modes_cache, conservative2recon!, recon2conservative! = hierarchical
    modes, modes_new = modes_cache

    @unpack degree = op
    nvar = nvariables(eq)
    xg_ = op.xg
    nx = grid.size
    nd = degree + 1

    # Work in [-1,1]
    xg = 2.0 * xg_ .- 1.0

    # Convert to modal basis
    Pn2m = nodal2modal_krivodonova(xg)

    # Modal to nodal
    M2Pn = Vandermonde_leg_krivodonova(degree, xg)

    # Get all the coefficients stored in modes
    for i in 1:nx
        @views Trixi.multiply_dimensionwise!(modes[:, :, i], Pn2m, u1[:, :, i])
        # for n in eachvariable(eq)
        #    @views modes[n,:,i] .= Pn2m * u1[n,:,i]
        # end
    end

    # Convert to reconstruction variables
    # FIXME - Note that this is changing each variable
    # locally, unlike the TVD limiter. This may not work
    # because you'd end up comparing characteristic variables
    # corresponding to different matrices!!

    @unpack periodic_x = problem
    @unpack boundary_condition = problem
    left, right = boundary_condition

    if periodic_x == true
        @views modes[:, :, 0] .= modes[:, :, nx]
        @views modes[:, :, nx + 1] .= modes[:, :, 1]
    else
        @views modes[:, :, 0] .= modes[:, :, 1]
        @views modes[:, :, nx + 1] .= modes[:, :, nx]
    end

    if left == reflect
        modes[2, :, 0] .*= -1.0
    end

    if right == reflect
        modes[2, :, nx + 1] .*= -1.0
    end

    modes_new .= modes

    # Characteristic
    # modes_new = copy(modes)
    # Limit modes as needed
    df, db, Dc = hierarchical.local_cache

    dcn = MVector{nvar}(zeros(nvar))

    for cell in 1:nx
        ua_ = @view ua[:, cell]
        for i in Base.OneTo(nd)
            for n in eachvariable(eq)
                df[n, i] = modes[n, i, cell + 1] - modes[n, i, cell]
                db[n, i] = modes[n, i, cell] - modes[n, i, cell - 1]
                Dc[n, i] = modes[n, i, cell]
            end
            @views conservative2recon!(df[:, i], ua_, eq)
            @views conservative2recon!(db[:, i], ua_, eq)
            @views conservative2recon!(Dc[:, i], ua_, eq)
        end
        to_limit = true
        for i in nd:-1:2 # FIXME: - ix = 1 isn't touched, so modes_new initial is never updated
            if to_limit == true
                for n in eachvariable(eq)
                    dcn[n] = minmod(Dc[n, i], alpha * df[n, i - 1],
                                    alpha * db[n, i - 1], 0.0)
                end
                diff = @views sum(abs.(dcn .- Dc[:, i]))
                if diff < 1e-10
                    to_limit = false # Remaining dofs will not be limited
                end
                recon2conservative!(dcn, ua_, eq)
                modes_new[:, i, cell] .= dcn
            else
                @views modes_new[:, i, cell] .= modes[:, i, cell]
            end
        end
    end

    # Legendre basis to Lagrange
    for i in 1:nx
        @views Trixi.multiply_dimensionwise!(u1[:, :, i], M2Pn, modes_new[:, :, i])
        # for n in eachvariable(eq)
        #    @views u1[n,:,i] .= M2Pn * modes[n,:,i]
        # end
    end
    return nothing
end

function modal_smoothness_indicator(eq::AbstractEquations{1}, t, iter, fcount,
                                    dt, grid, scheme, problem, param, aux, op,
                                    u1, ua)
    @unpack indicator_model = scheme.limiter
    if indicator_model == "gassner"
        modal_smoothness_indicator_gassner(eq, t, iter, fcount, dt, grid, scheme,
                                           problem, param, aux, op, u1, ua)
    elseif indicator_model == "gassner_new"
        modal_smoothness_indicator_gassner_new(eq, t, iter, fcount, dt, grid,
                                               scheme, problem, param, aux, op,
                                               u1, ua)
    elseif indicator_model == "gassner_face"
        modal_smoothness_indicator_gassner_face(eq, t, iter, fcount, dt, grid,
                                                scheme, problem, param, aux, op,
                                                u1, ua)
    else
        modal_smoothness_indicator_new(eq, t, iter, fcount, dt, grid, scheme,
                                       problem, param, aux, op, u1, ua)
    end
end

function modal_smoothness_indicator_new(eq::AbstractEquations{1}, t, iter,
                                        fcount, dt, grid, scheme, problem,
                                        param, aux, op, u1, ua)
    @timeit aux.timer "Blending limiter" begin
        @unpack xc, dx = grid
        nx = grid.size
        @unpack nvar = eq
        @unpack Vl, Vr, xg = op
        nd = length(xg)
        @unpack limiter = scheme
        left_bc, right_bc = problem.boundary_condition
        @unpack blend = aux
        amax = blend.amax      # maximum factor of the lower order term
        @unpack E1, E0 = blend # smoothness and discontinuity thresholds
        tolE = blend.tolE      # tolerance for denominator
        E = blend.E            # content in high frequency nodes
        alpha = blend.alpha    # vector containing smoothness indicator values
        @unpack a0, a1 = blend # smoothing coefficients

        # some strings specifying the kind of blending
        @unpack indicator_model, indicating_variables = limiter

        if limiter.pure_fv == true
            @assert scheme.limiter.name == "blend"
            alpha .= 1.0
        end

        # Extend solution points to include boundary
        yg = zeros(nd + 2)
        yg[1] = 0.0
        yg[2:(nd + 1)] = xg
        yg[nd + 2] = 1.0

        # Get nodal basis from values at extended solution points
        Pn2m = nodal2modal(yg)

        un, um = zeros(nvar, nd + 2), zeros(nvar, nd + 2) # Nodal, modal values in a cell
        um_xg = zeros(nvar, nd)

        for i in 1:nx
            # Continuous extension to faces
            u = @view u1[:, :, i]
            @views copyto!(un, CartesianIndices((1:nvar, 2:(nd + 1))),
                           u, CartesianIndices((1:nvar, 1:nd))) # Copy inner values

            # Copying is needed because we replace these with variables actually
            # used for indicators like primitives or rho*p, etc.

            un[:, 1] .= 0.0
            un[:, nd + 2] .= 0.0
            for ii in 1:nd # get face values as average of the two cells
                for n in 1:nvar
                    un[n, 1] += 0.5 *
                                (u1[n, ii, i - 1] * Vr[ii] + u1[n, ii, i] * Vl[ii])
                    un[n, nd + 2] += 0.5 *
                                     (u1[n, ii, i] * Vr[ii] + u1[n, ii, i + 1] * Vl[ii])
                end
            end

            # Convert un to ind var, get no. of variables used for indicator
            n_ind_nvar = @views blend.get_indicating_variables!(un, eq)

            for n in Base.OneTo(n_ind_nvar)
                um[n, :] = @views Pn2m * un[n, :]
            end

            ind_den, ind_num, ind = (zeros(n_ind_nvar), zeros(n_ind_nvar),
                                     zeros(n_ind_nvar))
            for n in 1:n_ind_nvar
                ind_den[n] = sum(um[n, 2:end] .^ 2)      # energy excluding constant node
                if indicator_model == "model1"
                    ind_num[n] = um[n, end - 1]^2 + um[n, end]^2 # energy in last 2 modes
                elseif indicator_model == "model2"
                    @assert indicating_variables == "conservative"
                    Pn2m_xg = nodal2modal(xg)
                    for n in 1:nvar
                        um_xg[n, :] = @views Pn2m_xg * u[n, :]
                        ind_num[n] = @views (sum((um_xg[n, :] - um[n, 1:(end - 2)]) .^
                                                 2)
                                             + um[n, end]^2 + um[n, end - 1]^2)
                    end
                elseif indicator_model == "model3"
                    ind_num[n] = um[n, end - 2]^2 + um[n, end - 1]^2 + um[n, end]^2 # energy in last 3 modes
                elseif indicator_model == "model5"
                    ## Extended nodes
                    # Pn2m_xg = nodal2modal(xg)
                    # um_xg[n,:] .= @views Pn2m_xg*u[n,:]

                    # Second extended nodes
                    ind_den = @views sum(um[n, 2:end] .^ 2)      # Gassner takes constant node
                    ind_num = um[n, end]^2 # energy in last node
                    if ind_den > tolE
                        ind1 = ind_num / ind_den # content of high frequencies
                    else
                        ind1 = 0.0
                    end

                    # First extended node
                    ind_den = @views sum(um[n, 2:(end - 1)] .^ 2)
                    ind_num = um[n, end - 1]^2 # energy in penultimate node
                    if ind_den > tolE
                        ind2 = ind_num / ind_den # content of high frequencies
                    else
                        ind2 = 0.0
                    end

                    ## In-cell nodes

                    # Second in-cell node
                    ind_den = @views sum(um[n, 2:(end - 2)] .^ 2)      # Gassner takes constant node
                    ind_num = um[n, end - 2]^2 # energy in last node
                    if ind_den > tolE
                        ind3 = ind_num / ind_den # content of high frequencies
                    else
                        ind3 = 0.0
                    end

                    # First in-cell node
                    ind_den = @views sum(um[n, 2:(end - 3)] .^ 2)
                    ind_num = um[n, end - 3]^2 # energy in penultimate node
                    if ind_den > tolE
                        ind4 = ind_num / ind_den # content of high frequencies
                    else
                        ind4 = 0.0
                    end

                    # Content is the maximum from last 2 nodes
                    ind[n] = max(ind1, ind2, ind3, ind4)
                else
                    @assert indicator_model=="draconian" "Incorrect indicator model"
                end
                # KLUDGE - Do this properly!!
                if indicator_model != "model5"
                    ind[n] = 0.0

                    if ind_den[n] > tolE
                        ind[n] = ind_num[n] / ind_den[n] # content of high frequencies
                    end
                end
            end
            E[i] = maximum(ind) # KLUDGE - is the variable 'ind' really needed?
            if indicator_model != "draconian"
                if E[i] < E0
                    alpha[i] = 0.0
                elseif E[i] > E1
                    alpha[i] = amax
                else
                    y = log(E[i] / E0) / log(E1 / E0)
                    z = sin(0.5 * pi * y)^2
                    alpha[i] = amax * z
                end
            end
            if indicator_model == "draconian"
                E1 = [0.00548948, 0.0125556, 0.0873583, 0.0563234]
                E0 = 10^(-2) * E1
                M = length(E1)
                alpha_ = zeros(M)
                for m in 1:M
                    m_ = m - 1
                    for n in 1:n_ind_nvar
                        ind_den = @views sum(um[n, 2:(end - m_)] .^ 2)      # Gassner takes constant node
                        ind_num = um[n, end - m_]^2 # energy in last node
                        if ind_den > tolE
                            ind[n] = ind_num / ind_den # content of high frequencies
                        else
                            ind[n] = 0.0
                        end
                    end
                    E[i] = maximum(ind) # KLUDGE - is the variable 'ind' really needed?
                    if E[i] < E0[m]
                        alpha_[m] = 0.0
                    elseif E[i] > E1[m]
                        alpha_[m] = 1.0
                    else
                        y = log(E[i] / E0[m]) / log(E1[m] / E0[m])
                        z = sin(0.5 * pi * y)^2
                        alpha_[m] = amax * z
                    end
                end
                alpha[i] = maximum(alpha_)
            end
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = amax
        end
        # Smoothening of alpha
        alpha0 = copy(alpha)
        # tmp = alpha[0]
        for i in 1:nx
            # tmp0 = alpha[i]
            if alpha[i] < amax || true
                # alpha[i] = a0*tmp + a1*alpha[i] + a0*alpha[i+1]
                alpha[i] = a0 * alpha0[i - 1] + a1 * alpha[i] + a0 * alpha0[i + 1]
            end
            # tmp = tmp0
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = amax
        end

        if dt > 0.0
            blend.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
        end

        blend.lamx .= alpha .* dt ./ dx

        # KLUDGE - Should this be in apply_limiter! function?
        debug_blend_limiter!(eq, grid, problem, scheme, param, aux, op,
                             dt, t, iter, fcount, ua, u1)
    end # timer
end

function modal_smoothness_indicator_gassner(eq::AbstractEquations{1}, t, iter,
                                            fcount, dt, grid, scheme, problem,
                                            param, aux, op, u1, ua)
    @timeit aux.timer "Blending limiter" begin
        @unpack xc, dx = grid
        nx = grid.size
        nvar = nvariables(eq)
        @unpack Vl, Vr, xg = op
        nd = length(xg)
        @unpack limiter = scheme
        left_bc, right_bc = problem.boundary_condition
        @unpack blend = aux
        amax = blend.amax      # maximum factor of the lower order term
        @unpack (constant_node_factor, constant_node_factor2, c, a, amin) = blend.parameters # Multiply constant node by this factor in indicator
        @unpack E1, E0 = blend # smoothness and discontinuity thresholds
        tolE = blend.tolE      # tolerance for denominator
        E = blend.E            # content in high frequency nodes
        alpha = blend.alpha    # vector containing smoothness indicator values
        @unpack a0, a1 = blend # smoothing coefficients

        # some strings specifying the kind of blending
        @unpack (indicator_model, indicating_variables) = limiter

        # Get nodal basis from values at extended solution points
        Pn2m = nodal2modal(xg)

        un, um = zeros(nvar, nd), zeros(nvar, nd) # Nodal, modal values in a cell

        for i in 1:nx
            # Continuous extension to faces
            u = @view u1[:, :, i]
            @views copyto!(un, CartesianIndices((1:nvar, 1:nd)),
                           u, CartesianIndices((1:nvar, 1:nd))) # Copy inner values

            # Copying is needed because we replace these with variables actually
            # used for indicators like primitives or rho*p, etc.

            # Convert un to ind var, get no. of variables used for indicator
            n_ind_nvar = @views blend.get_indicating_variables!(un, eq)

            for n in 1:n_ind_nvar
                um[n, :] = @views Pn2m * un[n, :]
            end

            ind = zeros(n_ind_nvar)

            for n in 1:n_ind_nvar
                # um[n,1] *= constant_node_factor
                # Last node
                ind_den = @views sum(um[n, 1:end] .^ 2)      # Gassner takes constant node
                ind_den -= um[n, 1]^2 - (constant_node_factor * um[n, 1])^2
                ind_num = um[n, end]^2 # energy in last node
                if ind_den > tolE
                    ind1 = ind_num / ind_den # content of high frequencies
                else
                    ind1 = 0.0
                end

                # Penultimate node
                # um[n,1] /= constant_node_factor
                ind_den = @views sum(um[n, 1:(end - 1)] .^ 2)
                ind_den -= um[n, 1]^2 - (constant_node_factor2 * um[n, 1])^2
                ind_num = um[n, end - 1]^2 # energy in penultimate node
                if ind_den > tolE
                    ind2 = ind_num / ind_den # content of high frequencies
                else
                    ind2 = 0.0
                end

                # Content is the maximum from last 2 nodes
                ind[n] = max(ind1, ind2)
            end
            E[i] = maximum(ind) # maximum content among all indicating variables

            T = a * 10^(-c * nd^(0.25))
            s = log((1.0 - 0.0001) / 0.0001)  # chosen to ensure so that E = 0 => alpha = amin
            alpha[i] = 1.0 / (1.0 + exp((-s / T) * (E[i] - T)))

            if alpha[i] < amin # amin = 0.0001
                alpha[i] = 0.0
            elseif alpha[i] > 1.0 - amin
                alpha[i] = 1.0
            end

            # alpha[i] = min(alpha[i], amax)
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = 1.0
        end

        # Smoothening of alpha
        alpha0 = copy(alpha)
        for i in 1:nx
            alpha[i] = max(0.5 * alpha0[i - 1], alpha[i], 0.5 * alpha0[i + 1])
            alpha[i] = min(alpha[i], amax)
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = amax
        end

        if dt > 0.0
            blend.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
        end

        if limiter.pure_fv == true
            @assert scheme.limiter.name == "blend"
            alpha .= 1.0
        end

        blend.lamx .= alpha .* dt ./ dx

        # KLUDGE - Should this be in apply_limiter! function?
        debug_blend_limiter!(eq, grid, problem, scheme, param, aux, op,
                             dt, t, iter, fcount, ua, u1)
    end # timer
end

function modal_smoothness_indicator_gassner_new(eq::AbstractEquations{1}, t,
                                                iter, fcount, dt, grid,
                                                scheme, problem, param, aux,
                                                op, u1, ua)
    @timeit aux.timer "Blending limiter" begin
        @unpack xc, dx = grid
        nx = grid.size
        @unpack nvar = eq
        @unpack Vl, Vr, xg = op
        nd = length(xg)
        @unpack limiter = scheme
        left_bc, right_bc = problem.boundary_condition
        @unpack blend = aux
        amax = blend.amax      # maximum factor of the lower order term
        @unpack E1, E0 = blend # smoothness and discontinuity thresholds
        tolE = blend.tolE      # tolerance for denominator
        E = blend.E            # content in high frequency nodes
        alpha = blend.alpha    # vector containing smoothness indicator values
        @unpack a0, a1 = blend # smoothing coefficients

        amin = 0.001
        amax = 1.0

        # some strings specifying the kind of blending
        @unpack indicator_model, indicating_variables = limiter

        # Get nodal basis from values at extended solution points
        Pn2m = nodal2modal(xg)

        un, um = zeros(nvar, nd), zeros(nvar, nd) # Nodal, modal values in a cell

        for i in 1:nx
            # Continuous extension to faces
            u = @view u1[:, :, i]
            @views copyto!(un, CartesianIndices((1:nvar, 1:nd)),
                           u, CartesianIndices((1:nvar, 1:nd))) # Copy inner values

            # Copying is needed because we replace these with variables actually
            # used for indicators like primitives or rho*p, etc.

            # Convert un to ind var, get no. of variables used for indicator
            n_ind_nvar = @views blend.get_indicating_variables!(un, eq)

            for n in 1:n_ind_nvar
                um[n, :] = @views Pn2m * un[n, :]
            end

            ind = zeros(n_ind_nvar)
            ind_nd, ind_nd_m_1, ind_nd_m_2 = zeros(n_ind_nvar), zeros(n_ind_nvar),
                                             zeros(n_ind_nvar)
            for n in 1:n_ind_nvar
                @views um[n, 1] *= 0.1 # FIXME - Replace with 0.1*cell_max/global_max

                # Last node
                ind_den = @views sum(um[n, 1:end] .^ 2)      # Gassner takes constant node
                ind_num = um[n, end]^2 # energy in last node
                if ind_den > tolE
                    ind_nd[n] = ind_num / ind_den # content of high frequencies
                else
                    ind_nd[n] = 0.0
                end

                # um[n,1] *= 0.2/0.1

                # Penultimate node
                ind_den = @views sum(um[n, 1:(end - 1)] .^ 2)
                ind_num = um[n, end - 1]^2 # energy in penultimate node
                if ind_den > tolE
                    ind_nd_m_1[n] = ind_num / ind_den # content of high frequencies
                else
                    ind_nd_m_1[n] = 0.0
                end

                ind_den = @views sum(um[n, 1:(end - 2)] .^ 2)
                ind_num = um[n, end - 2]^2 # energy in penultimate node
                if ind_den > tolE
                    ind_nd_m_2[n] = ind_num / ind_den # content of high frequencies
                else
                    ind_nd_m_2[n] = 0.0
                end

                # Content is the maximum from last 2 nodes
                # ind[n] = max(ind1[n], ind2[n])

            end
            # E[i] = maximum(ind) # maximum content among all indicating variables

            a_, c_ = 0.5, 1.8
            E_nd = maximum(ind_nd)
            E_nd_m_1 = maximum(ind_nd_m_1)
            E_nd_m_2 = maximum(ind_nd_m_2)
            c_nd = c_
            a_nd = a_
            T_nd = a_nd * 10^(-c_nd * nd^(0.25))
            a_nd_m_1 = a_
            c_nd_m_1 = c_
            T_nd_m_1 = a_nd_m_1 * 10^(-c_nd_m_1 * (nd - 1)^(0.25))
            a_nd_m_2 = a_
            c_nd_m_2 = c_
            T_nd_m_2 = a_nd_m_2 * 10^(-c_nd_m_2 * (nd - 2)^(0.25))
            s = log((1.0 - 0.0001) / 0.0001)  # chosen to ensure so that E = 0 => alpha = amin
            alpha_nd = 1.0 / (1.0 + exp((-s / T_nd) * (E_nd - T_nd)))
            # alpha_nd = 0.0
            alpha_nd_m_1 = 1.0 / (1.0 + exp((-s / T_nd_m_1) * (E_nd_m_1 - T_nd_m_1)))
            # alpha_nd_m_1 = 0.0
            # alpha_nd_m_2 = 1.0 / (1.0 + exp( (-s/T_nd_m_2) * (E_nd_m_2 - T_nd_m_2) ))
            alpha_nd_m_2 = 0.0
            alpha[i] = max(alpha_nd, alpha_nd_m_1, alpha_nd_m_2)

            if alpha[i] < amin # amin = 0.0001
                alpha[i] = 0.0
            elseif alpha[i] > 1.0 - amin
                alpha[i] = 1.0
            end
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = 1.0
        end

        # Smoothening of alpha
        alpha0 = copy(alpha)
        # tmp = alpha[0]
        for i in 1:nx
            # tmp0 = alpha[i]

            if alpha[i] < amax
                alpha[i] = max(0.5 * alpha0[i - 1], alpha[i], 0.5 * alpha0[i + 1])
            end
            alpha[i] = min(alpha[i], amax)
            # tmp = tmp0
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        # alpha[0] = alpha[nx+1] = 1.0

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = amax
        end

        if dt > 0.0
            blend.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
        end

        if limiter.pure_fv == true
            @assert scheme.limiter.name == "blend"
            alpha .= 1.0
        end

        blend.lamx .= alpha .* dt ./ dx

        # KLUDGE - Should this be in apply_limiter! function?
        debug_blend_limiter!(eq, grid, problem, scheme, param, aux, op,
                             dt, t, iter, fcount, ua, u1)
    end # timer
end

function modal_smoothness_indicator_gassner_face(eq, t, iter, fcount, dt, grid,
                                                 scheme, problem, param, aux,
                                                 op, u1, ua)
    @timeit aux.timer "Blending limiter" begin
        @unpack xc, dx = grid
        nx = grid.size
        @unpack nvar = eq
        @unpack Vl, Vr, xg = op
        nd = length(xg)
        @unpack limiter = scheme
        left_bc, right_bc = problem.boundary_condition
        @unpack blend = aux
        amax = blend.amax      # maximum factor of the lower order term
        @unpack E1, E0 = blend # smoothness and discontinuity thresholds
        tolE = blend.tolE      # tolerance for denominator
        E = blend.E            # content in high frequency nodes
        alpha = blend.alpha    # vector containing smoothness indicator values
        @unpack a0, a1 = blend # smoothing coefficients

        amin = 0.001
        amax = 1.0

        # some strings specifying the kind of blending
        @unpack indicator_model, indicating_variables = limiter

        # Extend solution points to include boundary
        yg = zeros(nd + 2)
        yg[1] = 0.0
        yg[2:(nd + 1)] = xg
        yg[nd + 2] = 1.0

        # Get nodal basis from values at extended solution points
        Pn2m = nodal2modal(yg)

        un, um = zeros(nvar, nd + 2), zeros(nvar, nd + 2) # Nodal, modal values in a cell

        for i in 1:nx
            # Continuous extension to faces
            u = @view u1[:, :, i]
            @views copyto!(un, CartesianIndices((1:nvar, 2:(nd + 1))),
                           u, CartesianIndices((1:nvar, 1:nd))) # Copy inner values

            # Copying is needed because we replace these with variables actually
            # used for indicators like primitives or rho*p, etc.

            un[:, 1] .= 0.0
            un[:, nd + 2] .= 0.0
            for ii in 1:nd # get face values as average of the two cells
                for n in 1:nvar
                    un[n, 1] += 0.5 *
                                (u1[n, ii, i - 1] * Vr[ii] + u1[n, ii, i] * Vl[ii])
                    un[n, nd + 2] += 0.5 *
                                     (u1[n, ii, i] * Vr[ii] + u1[n, ii, i + 1] * Vl[ii])
                end
            end

            # Convert un to ind var, get no. of variables used for indicator
            n_ind_nvar = @views blend.get_indicating_variables!(un)

            for n in 1:n_ind_nvar
                um[n, :] = @views Pn2m * un[n, :]
            end

            ind = zeros(n_ind_nvar)
            ind_nd, ind_nd_m_1, ind_nd_m_2, ind_nd_m_3 = (zeros(n_ind_nvar),
                                                          zeros(n_ind_nvar),
                                                          zeros(n_ind_nvar),
                                                          zeros(n_ind_nvar))
            for n in 1:n_ind_nvar
                # Last node
                ind_den = @views sum(um[n, 1:end] .^ 2)      # Gassner takes constant node
                ind_num = um[n, end]^2 # energy in last node
                if ind_den > tolE
                    ind_nd[n] = ind_num / ind_den # content of high frequencies
                else
                    ind_nd[n] = 0.0
                end

                # Penultimate node
                ind_den = @views sum(um[n, 1:(end - 1)] .^ 2)
                ind_num = um[n, end - 1]^2 # energy in penultimate node
                if ind_den > tolE
                    ind_nd_m_1[n] = ind_num / ind_den # content of high frequencies
                else
                    ind_nd_m_1[n] = 0.0
                end

                ind_den = @views sum(um[n, 1:(end - 2)] .^ 2)
                ind_num = um[n, end - 2]^2 # energy in penultimate node
                if ind_den > tolE
                    ind_nd_m_2[n] = ind_num / ind_den # content of high frequencies
                else
                    ind_nd_m_2[n] = 0.0
                end

                ind_den = @views sum(um[n, 1:(end - 3)] .^ 2)
                ind_num = um[n, end - 2]^2 # energy in penultimate node
                if ind_den > tolE
                    ind_nd_m_3[n] = ind_num / ind_den # content of high frequencies
                else
                    ind_nd_m_3[n] = 0.0
                end
            end
            # E[i] = maximum(ind) # maximum content among all indicating variables

            a_, c_ = 0.5, 1.8
            E_ = maximum(ind_nd)
            E_m_1 = maximum(ind_nd_m_1)
            E_m_2 = maximum(ind_nd_m_2)
            E_m_3 = maximum(ind_nd_m_3)
            c_ = 2.0
            a_ = 1.0
            T_ = a_ * 10^(-c_ * (nd + 2)^(0.25))
            a_m_1 = 1.0
            c_m_1 = 2.2
            T_m_1 = a_m_1 * 10^(-c_m_1 * (nd + 1)^(0.25))
            a_m_2 = 1.0
            c_m_2 = 3.0
            T_m_2 = a_m_2 * 10^(-c_m_2 * nd^(0.25))
            a_m_3 = 1.0
            c_m_3 = 3.0
            T_m_3 = a_m_3 * 10^(-c_m_3 * (nd - 1)^(0.25))
            s = log((1.0 - 0.0001) / 0.0001)  # chosen to ensure so that E = 0 => alpha = amin
            alpha_nd = 1.0 / (1.0 + exp((-s / T_) * (E_ - T_)))
            alpha_nd_m_1 = 1.0 / (1.0 + exp((-s / T_m_1) * (E_m_1 - T_m_1)))
            alpha_nd_m_2 = 1.0 / (1.0 + exp((-s / T_m_2) * (E_m_2 - T_m_2)))
            alpha_nd_m_3 = 1.0 / (1.0 + exp((-s / T_m_3) * (E_m_3 - T_m_3)))
            alpha[i] = max(alpha_nd, alpha_nd_m_1, alpha_nd_m_2, alpha_nd_m_3)

            if alpha[i] < amin # amin = 0.0001
                alpha[i] = 0.0
            elseif alpha[i] > 1.0 - amin
                alpha[i] = 1.0
            end
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = 1.0
        end

        # Smoothening of alpha
        alpha0 = copy(alpha)
        # tmp = alpha[0]
        for i in 1:nx
            # tmp0 = alpha[i]

            if alpha[i] < amax
                alpha[i] = max(0.5 * alpha0[i - 1], alpha[i], 0.5 * alpha0[i + 1])
            end
            alpha[i] = min(alpha[i], amax)
            # tmp = tmp0
        end

        if problem.periodic_x
            alpha[0], alpha[nx + 1] = alpha[nx], alpha[1]
        else
            alpha[0], alpha[nx + 1] = alpha[1], alpha[nx]
        end

        # alpha[0] = alpha[nx+1] = 1.0

        if left_bc == neumann && right_bc == neumann
            # Force first order on boundary for Shu-Osher
            alpha[1] = alpha[nx] = amax
        end

        if dt > 0.0
            blend.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
        end

        if limiter.pure_fv == true
            @assert scheme.limiter.name == "blend"
            alpha .= 1.0
        end

        blend.lamx .= alpha .* dt ./ dx

        # KLUDGE - Should this be in apply_limiter! function?
        debug_blend_limiter!(eq, grid, problem, scheme, param, aux, op,
                             dt, t, iter, fcount, ua, u1)
    end # timer
end

function debug_blend_limiter!(eq::AbstractEquations{1}, grid, problem, scheme,
                              param, aux,
                              op, dt, t, iter, fcount, ua, u1)
    @unpack blend, plot_data = aux
    if blend.debug == false
        return nothing
    end
    @timeit aux.timer "Debug blending limiter" begin
        @unpack limiter = scheme
        @unpack final_time = problem

        space_time_alpha = blend.space_time_alpha
        time_levels = blend.time_levels_anim # all time levels (rename)
        @unpack alpha, E = blend
        @unpack nvar = eq
        nx = grid.size
        nd = op.degree + 1
        @unpack p_ua, p_u1 = plot_data
        for n in 0:(nvar - 1)
            # Loop is needed as a temporary hack to fix plotly bug with offset arrays
            for i in 1:nx
                p_ua[end - n][end][:y][i] = alpha[i]
                p_u1[end - n][i][:y] .= alpha[i] # exact.(p[1][2][:x])
            end
        end
        if nvar == 1
            ua_min, ua_max = minimum(ua), maximum(ua)
            d_ua = ua_max - ua_min
            for i in 1:nx
                p_ua[1][end][:y][i] = ua_min + (alpha[i]) * d_ua
                p_u1[1][i][:y] .= ua_min + (alpha[i]) * d_ua
            end
        end
        if nvariables(eq) > 1 # KLUDGE - Create a p
            ua_min, ua_max = [1e20, 1e20, 1e20], [-1e20, -1e20, -1e20]
            for i in 1:nx
                @views up = con2prim(eq, ua[:, i])
                for n in 1:nvar
                    ua_min[n] = min(ua_min[n], up[n])
                    ua_max[n] = max(ua_max[n], up[n])
                end
            end
            for n in 1:nvar
                if nd == 2
                    d_ua = ua_max[n] - ua_min[n]
                else
                    d_ua = ua_max[n] - ua_min[n]
                end
                for i in 1:nx
                    p_ua[n + 1][end][:y][i] = ua_min[n] + (alpha[i]) * d_ua
                    p_u1[n + 1][i][:y] .= ua_min[n] + (alpha[i]) * d_ua
                end
            end
        end

        # Save energy and alpha data to output
        if save_solution(problem, param, t, iter) == true || t < 1e-12 ||
           final_time - (t + dt) < 1e-10
            alpha_ = @view alpha[1:nx]
            @unpack xc = grid
            ndigits = 3 # KLUDGE - Add option to change
            alpha_filename = get_filename("output/alpha", ndigits, fcount)
            energy_filename = get_filename("output/energy", ndigits, fcount)
            alpha_file = open("$alpha_filename.txt", "w")
            energy_file = open("$energy_filename.txt", "w")
            # Start files by current time
            @printf(alpha_file, "%.16e \n", t)
            @printf(energy_file, "%.16e \n", t)
            # Write alpha, energy to respective files
            writedlm(alpha_file, zip(xc, alpha_), " ")
            writedlm(energy_file, zip(xc, E), " ")
            close(alpha_file)
            close(energy_file)

            if final_time - (t + dt) < 1e-10
                cp("$energy_filename.txt", "output/energy.txt", force = true)
                cp("$alpha_filename.txt", "output/alpha.txt", force = true)
            end
        end

        @views append!(space_time_alpha, alpha[1:nx])
        append!(time_levels, t)
        if final_time - (t + dt) < 1e-12
            writedlm("output/time_levels.txt", time_levels, " ")
            writedlm("output/space_time_alpha.txt", space_time_alpha', " ")
            @assert length(xc) * length(time_levels) == length(space_time_alpha)
            vtk = vtk_grid("output/space_time_alpha.vtr", grid.xc, time_levels)
            vtk["alpha"] = space_time_alpha
            out = vtk_save(vtk)
        end
    end # timer
end

#-------------------------------------------------------------------------------
# Lower order residuals for blending
# Break cell j into nd subcells with sizes wj*dx[j] & compute fvm res in each.
# compute_fv_cell_residual! computes the residual that is from subcell faces
# which are in the interior of big cell.
#-------------------------------------------------------------------------------
# 1st order FV cell residual
@inbounds @inline function blend_cell_residual_fo!(cell, eq::AbstractEquations{1},
                                                   scheme, aux, lamx,
                                                   dt, dx, xf, op, u1, u, ua, f, r)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead, it's supposed
        # to be 0.25 microseconds
        @unpack blend = aux
        # if blend.alpha[i] < 1e-12
        #    return nothing
        # end
        @unpack Vl, Vr, xg, wg = op
        num_flux = scheme.numerical_flux
        nd = length(xg)
        resl = blend.resl
        nvar = nvariables(eq)
        @unpack xxf, fn = blend
        fn_low = @view blend.fn_low[:, :, cell]
        # Get subcell faces
        xxf[0] = xf
        for ii in Base.OneTo(nd)
            xxf[ii] = xxf[ii - 1] + dx * wg[ii]
        end
        fill!(resl, zero(eltype(resl)))
        for j in 2:nd
            xx = xxf[j]
            # @views ul, ur = u[:,j-1], u[:,j]
            # fl, fr = flux(xx, ul, eq), flux(xx, ur, eq)
            fn = @views num_flux(xx, u[:, j - 1], u[:, j],
                                 f[:, j - 1], f[:, j],
                                 u[:, j - 1], u[:, j], eq, 1)
            for n in 1:nvar
                resl[n, j - 1] += fn[n] / wg[j - 1]
                resl[n, j] -= fn[n] / wg[j]
            end
        end
        @views fn_low[:, 1] .= wg[1] * resl[:, 1]
        @views fn_low[:, 2] .= -wg[end] * resl[:, end]
        axpby!(blend.alpha[cell] * dt / dx, resl, 1.0 - blend.alpha[cell], r)
    end # timer
end

@inbounds @inline function blend_face_residual_fo!(i, xf, u1, ua,
                                                   eq::AbstractEquations{1},
                                                   dt, grid, op, scheme, param,
                                                   Fn, aux, lamx, res)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead,
        # it's supposed to be 0.25 microseconds
        @unpack blend = aux
        alpha = blend.alpha # factor of non-smooth part
        alp = 0.5 * (alpha[i - 1] + alpha[i])
        num_flux = scheme.numerical_flux
        @unpack dx = grid
        nvar = nvariables(eq)

        @unpack xg, wg = op
        nd = length(xg)

        # Reuse arrays to save memory
        @unpack fl, fr, fn = blend

        # The lower order residual of blending scheme comes from lower order
        # numerical flux at the subcell faces. Here we deal with the residual that
        # occurs from those faces that are common to both the subcell and supercell

        # Low order numerical flux
        ul, ur = @views u1[:, nd, i - 1], u1[:, 1, i]
        fl = flux(xf, ul, eq)
        fr = flux(xf, ur, eq)
        fn = num_flux(xf, ul, ur, fl, fr, ul, ur, eq, 1)

        # alp = test_alp(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op, alp)

        Fn = (1.0 - alp) * Fn + alp * fn

        Fn = get_blended_flux(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op,
                              alp)

        # Blend low and higher order flux
        # for n=1:nvar
        #    Fn[n] = @views (1.0-alp)*Fn[n] + alp*fn[n]
        # end
        # Fn_ = (1.0 - alp) * Fn + alp * fn
        # r = @view res[:, :, i-1]

        # # For the sub-cells which have same interface as super-cells, the same
        # # numflux Fn is used in place of the lower order flux
        # for n=1:nvar
        #    r[n,nd] += alpha[i-1] * dt/dx[i-1] *Fn_[n]/wg[nd] # blend.lamx=dt/dx*alpha
        #    # r[n,nd] += blend.lamx[i-1]*Fn[n]/wg[nd] # blend.lamx=dt/dx*alpha
        # end
        # r = @view res[:, :, i]
        # for n=1:nvar
        #    r[n,1] -= alpha[i] * dt/dx[i] * Fn_[n]/wg[1]     # blend.lamx=dt/dx*lamx
        #    # r[n,1] -= blend.lamx[i]*Fn[n]/wg[1]     # blend.lamx=dt/dx*lamx
        # end

        # # We adjust lamx[i] to limit high order face residual
        # one_m_alpha = (1.0-alpha[i-1], 1.0-alpha[i]) # factor of smooth part
        # return Fn_, one_m_alpha

        # Blend low and higher order flux
        # for n=1:nvar
        #    Fn[n] = (1.0-alp)*Fn[n] + alp*fn[n]
        # end

        # Fn_ = fn

        r = @view res[:, :, i - 1]
        # For the sub-cells which have same interface as super-cells, the same
        # numflux Fn is used in place of the lower order flux
        for n in 1:nvar
            # r[n,nd] += alpha[i-1] * dt/dx[i-1] * Fn_[n]/wg[nd] # alpha[i-1] already in blend.lamx
            r[n, nd] += dt / dx[i - 1] * alpha[i - 1] * Fn[n] / wg[nd] # alpha[i-1] already in blend.lamx
        end

        r = @view res[:, :, i]
        for n in 1:nvar
            # r[n,1] -= alpha[i] * dt/dx[i] * Fn_[n]/wg[1] # alpha[i-1] already in blend.lamx
            r[n, 1] -= dt / dx[i] * alpha[i] * Fn[n] / wg[1] # alpha[i-1] already in blend.lamx
        end
        # lamx[i] = (1.0-alpha[i])*lamx[i] # factor of smooth part
        # Fn = (1.0 - alpha[i]) * Fn
        # one_m_alpha = (1.0 - alpha[i-1], 1.0 - alpha[i])
        # return Fn_, one_m_alpha
        return Fn, (1.0 - alpha[i - 1], 1.0 - alpha[i])
    end # timer
end

function limit_slope(::AbstractEquations{1, 1}, s, ufl, u_s_l, ufr, u_s_r,
                     ue, xl, xr)
    ufl, ufr
end

# MUSCL-hancock cell residual
@inbounds @inline function blend_cell_residual_muscl!(cell,
                                                      eq::AbstractEquations{1},
                                                      scheme, aux, lamx, dt,
                                                      dx, xf, op, u1, u, ua, f, r)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead, it's supposed
        # to be 0.25 microseconds
        @unpack blend = aux
        # if blend.alpha[cell] < 1e-12
        #    return nothing
        # end
        fn_low = @view blend.fn_low[:, :, cell]
        @unpack Vl, Vr, xg, wg = op
        nd = length(xg)
        num_flux = blend.numflux
        nvar = nvariables(eq)
        @unpack xxf, xe, ufl, ufr = blend # re-use values
        # Get subcell faces
        xxf[0] = xf
        for ii in 1:nd
            xxf[ii] = xxf[ii - 1] + dx * wg[ii]
        end
        # @unpack beta = blend
        beta = 2.0 - blend.alpha[cell]

        # Get solution points
        xe[0] = xf - dx * (1.0 - xg[nd])     # Last solution point of left cell
        @turbo xe[1:nd] .= xf .+ dx * xg        # Solution points on cell
        xe[nd + 1] = xf + dx * (1.0 + xg[1]) # First solution point on right cell

        # Force cell-centred approach
        # TOTHINK - Get concrete evidence that this force cell-centred is bad
        # for ii=1:nd
        #    xe[ii] = 0.5 * (xxf[ii-1] + xxf[ii])
        # end
        # xe[ii] = xxf[0] - 0.5(xxf[nd] - xxf[nd-1])
        # xe[nd+1] = xe[nd]

        # Get solution point values
        ue = blend.ue          # u extended to faces
        unph = blend.unph      # u at time n+1/2
        @views begin
            @turbo ue[:, 1:nd] .= u1[:, :, cell]    # values from current cell
            @turbo ue[:, 0] .= u1[:, nd, cell - 1]    # value from left neighbour cells
            @turbo ue[:, nd + 1] .= u1[:, 1, cell + 1]  # value from right neighbour cells
        end

        # @views ue[:,0] = u1[:,:,i] * Vl    # value from left neighbour cells
        # @views ue[:,nd+1] = u1[:,:,i] * Vr  # value from right neighbour cells

        # Convert to variables used for reconstruction, like prim or char.
        # If we are using conservative variables, this function does nothing

        ua_node = get_node_vars(ua, eq, cell)

        for ix in 0:(nd + 1)
            ue_node = get_node_vars(ue, eq, ix)
            ue_recon = blend.conservative2recon!(ue_node, ua_node, eq)
            set_node_vars!(ue, ue_recon, eq, ix)
            # @views blend.conservative2recon!(ue[:,ix], ua_node, eq)
        end
        # u_s_l, u_s_r = zeros(nvar), zeros(nvar)
        # slope = zeros(nvar)
        # Evolve all cells to time level n+1/2
        for ii in 1:nd # loop over (sub)cells
            # slope of linear approximation in cell
            ul = get_node_vars(ue, eq, ii - 1)
            u_ = get_node_vars(ue, eq, ii)
            ur = get_node_vars(ue, eq, ii + 1)

            h1, h2 = xe[ii] - xe[ii - 1], xe[ii + 1] - xe[ii]
            back, cent, fwd = finite_differences(h1, h2, ul, u_, ur)

            slope_tuple = (minmod(cent[n], back[n], fwd[n], beta, 0.0)
                           for n in eachvariable(eq))

            slope = SVector{nvar}(slope_tuple)

            ufl = u_ + slope * (xxf[ii - 1] - xe[ii]) # left face value u_j^{n,-}
            ufr = u_ + slope * (xxf[ii] - xe[ii]) # right face value u_j^{n,+}
            u_s_l = u_ + slope * 2.0 * (xxf[ii - 1] - xe[ii]) # u_j^{*,-}
            u_s_r = u_ + slope * 2.0 * (xxf[ii] - xe[ii]) # u_j^{*,+}
            # for n=1:nvar
            #    h1, h2 = xe[ii]-xe[ii-1], xe[ii+1] - xe[ii]
            #    a,b,c = -( h2/(h1*(h1+h2)) ), (h2-h1)/(h1*h2), ( h1/(h2*(h1+h2)) )
            #    cent_diff = ( a * ue[n,ii-1] + b * ue[n,ii] + c * ue[n,ii+1]  )
            #    # TOTHINK - Confirm that cent_diff is giving benefit in some case.
            #    # In linear advection, smooth test, it gave no advantage
            #    slope[n] = minmod(
            #                       cent_diff,
            #                       beta*(ue[n,ii]-ue[n,ii-1])/(xe[ii]-xe[ii-1]),
            #                      #  (ue[n,ii+1]-ue[n,ii-1])/(xe[ii+1]-xe[ii-1]),
            #                       beta*(ue[n,ii+1]-ue[n,ii])/(xe[ii+1]-xe[ii]),
            #                       0.0) # parameter M = 0.0
            #    ufl[n] = ue[n,ii] + slope[n]*(xxf[ii-1] - xe[ii]) # left face value u_j^{n,-}
            #    ufr[n] = ue[n,ii] + slope[n]*(xxf[ii]   - xe[ii]) # right face value u_j^{n,+}
            #    u_s_l[n] = ue[n,ii] + slope[n]*2.0*(xxf[ii-1] - xe[ii]) # u_j^{*,-}
            #    u_s_r[n] = ue[n,ii] + slope[n]*2.0*(xxf[ii]   - xe[ii]) # u_j^{*,+}
            # end

            # Convert back to conservative for update
            recon2cons(u) = blend.recon2conservative!(u, ua, eq)
            ufl, ufr, u_s_l, u_s_r = recon2cons.((ufl, ufr, u_s_l, u_s_r))

            ufl, ufr = limit_slope(eq, slope, ufl, u_s_l, ufr, u_s_r, u_,
                                   xxf[ii - 1] - xe[ii], xxf[ii] - xe[ii])
            fl = flux(xxf[ii - 1], ufl, eq)        # f(u_j^{n,-})
            fr = flux(xxf[ii], ufr, eq)          # f(u_j^{n,+})
            # Use finite difference to evolve face values to time level n+1/2
            for n in 1:nvar
                unph[n, 1, ii] = ufl[n] + (0.5 * blend.dt[1]
                                  * (fl[n] - fr[n]) / (xxf[ii] - xxf[ii - 1])) # u_j^{n+1/2,-}
                unph[n, 2, ii] = ufr[n] +
                                 0.5 * blend.dt[1] *
                                 ((fl[n] - fr[n])
                                  /
                                  (xxf[ii] - xxf[ii - 1])) # u_j^{n+1/2,+}
            end
        end
        resl = blend.resl # resl pre-stored.

        fill!(resl, zero(eltype(resl)))

        # Final update using mid-point FVM and mid-point quadrature by a face loop
        # Only residuals of inner face is added, outer face in blend_face_residual!
        for ii in 2:nd
            xx = xxf[ii]
            # @views ul, ur = unph[:,2,ii-1], unph[:,1,ii]
            ul = get_node_vars(unph, eq, 2, ii - 1)
            ur = get_node_vars(unph, eq, 1, ii)
            fl = flux(xx, ul, eq)
            fr = flux(xx, ur, eq)
            # numerical flux between subcell j-1 and j
            fn = num_flux(xx, ul, ur, fl, fr, ul, ur, eq, 1)
            for n in 1:nvar
                resl[n, ii - 1] += fn[n] / wg[ii - 1]
                resl[n, ii] -= fn[n] / wg[ii]
            end
        end

        # Store numerical fluxes for positivity correction of high order numerical flux.
        @views begin
            fn_l = wg[1] * get_node_vars(resl, eq, 1)
            set_node_vars!(fn_low, fn_l, eq, 1)
            # @turbo fn_low[:,1] .=  wg[1]   * resl[:,1] Somehow, this caused allocations
            # @turbo fn_low[:,2] .= -wg[end] * resl[:,nd]
            fn_r = -wg[end] * get_node_vars(resl, eq, nd)
            set_node_vars!(fn_low, fn_r, eq, 2)
        end

        # Here, blend.lamx[i] = dt/dx[i]*alpha[i]. KLUDGE - Can this be avoided?
        @turbo for ix in 1:nd, n in 1:nvar
            r[n, ix] = blend.lamx[cell] * resl[n, ix] +
                       (1.0 - blend.alpha[cell]) * r[n, ix]
        end
        # Somehow, broadcasting or
        # axpby!(blend.lamx[cell], resl, 1.0-blend.alpha[cell], r)
        # caused allocations

    end # timer
end

# Merge this with blend_cell_residual
@inbounds function blend_face_residual_muscl!(i, xf, u1, ua,
                                              eq::AbstractEquations{1},
                                              dt, grid, op, scheme,
                                              param, Fn, aux, lamx,
                                              res)
    @timeit aux.timer "Blending limiter" begin # TOTHINK - Check the overhead
        @unpack blend = aux
        # if blend.alpha[i] < 1e-12 && blend.alpha[i-1] < 1e-12
        #    return Fn, 1.0, 1.0
        # end
        alpha = blend.alpha # factor of non-smooth part
        @unpack xg, wg = op
        nd = length(xg)
        num_flux = blend.numflux
        # num_flux = scheme.numerical_flux
        nvar = nvariables(eq)
        alp = 0.5 * (alpha[i - 1] + alpha[i])

        dx = grid.dx
        # Reuse arrays to save memory
        unph = @view blend.unph[:, :, 1]
        beta = 2.0 - alp

        # The lower order residual of blending scheme comes from lower order
        # numerical flux at the subcell faces. Here we deal with the residual that
        # occurs from those faces that are common to both the subcell and supercell

        # We need the left, right face values at time level n+1/2

        # We begin by computing the left face value u^{n+1/2,-}
        # For that, we need face values of current time level
        # in last subcell of supercell i-1

        xfl, xfr = xf - wg[nd] * dx[i - 1], xf           # left, right subfaces
        xl = grid.xf[i] - dx[i - 1] + xg[nd - 1] * dx[i - 1] # sol point of left subcell
        x = grid.xf[i] - dx[i - 1] + xg[nd] * dx[i - 1]   # sol point of subcell
        xr = grid.xf[i] + xg[1] * dx[i]              # sol point of right subcell

        um1 = get_node_vars(u1, eq, nd - 1, i - 1)
        u_ = get_node_vars(u1, eq, nd, i - 1)
        up1 = get_node_vars(u1, eq, 1, i)

        # Convert to variables used for reconstruction, like prim or char.
        # If we are using conservative variables, this function does nothing

        ual = get_node_vars(ua, eq, i - 1)
        uar = get_node_vars(ua, eq, i)

        con2recon_l(u) = blend.conservative2recon!(u, ual, eq)

        um1, u_, up1 = con2recon_l.((um1, u_, up1))

        h1, h2 = x - xl, xr - x
        back, cent, fwd = finite_differences(h1, h2, um1, u_, up1)

        slope_tuple = (minmod(cent[n], back[n], fwd[n], beta, 0.0)
                       for n in eachvariable(eq))

        slope = SVector{nvar}(slope_tuple)

        # left, right face values at current time level
        ufl, ufr = u_ + slope * (xfl - x), u_ + slope * (xfr - x)
        u_s_l, u_s_r = u_ + 2.0 * slope * (xfl - x), u_ + 2.0 * slope * (xfr - x)

        # for n=1:nvar
        #    h1, h2 = x - xl, xr - x
        #    a,b,c = -( h2/(h1*(h1+h2)) ), (h2-h1)/(h1*h2), ( h1/(h2*(h1+h2)) )
        #    cent_diff = ( a * um1[n] + b * u_[n] + c * up1[n]  )
        #    slope[n] = minmod(
        #                       cent_diff,
        #                       beta * ( u_[n]  - um1[n] ) / (x-xl),  # u-um1
        #                       beta * ( up1[n] - u_[n]  ) / (xr-x),  # up1-u
        #                       0.0 )
        #    # left, right face values at current time level
        #    ufl[n], ufr[n] = u_[n] + slope[n]*(xfl-x), u_[n] + slope[n]*(xfr-x)
        #    u_s_l[n], u_s_r[n] = u_[n] + 2.0*slope[n]*(xfl-x), u_[n] + 2.0*slope[n]*(xfr-x)
        # end

        recon2cons_l(u) = blend.recon2conservative!(u, ual, eq)

        ufl, ufr, u_s_l, u_s_r = recon2cons_l.((ufl, ufr, u_s_l, u_s_r))

        ufl, ufr = limit_slope(eq, slope, ufl, u_s_l, ufr, u_s_r, u_, xfl - x, xfr - x)

        # left, right face fluxes at current time level
        fl = flux(xfl, ufl, eq)
        fr = flux(xfr, ufr, eq)
        for n in 1:nvar
            # perform update to get u^{n+1/2,-} with finite difference method
            unph[n, 1] = ufr[n] + 0.5 * blend.dt[1] * (fl[n] - fr[n]) / (xfr - xfl)
        end

        # We now compute the right face value u^{n+1/2,+}
        # For that, we need face values of current time level
        # in first subcell of supercell i

        xfl, xfr = xf, xf + wg[1] * dx[i]            # left, right subfaces
        xl = grid.xf[i] - dx[i - 1] + xg[nd] * dx[i - 1] # solution point of left subcell
        x = grid.xf[i] + xg[1] * dx[i]              # solution point of subcell
        xr = grid.xf[i] + xg[2] * dx[i]              # solution point of right subcell

        um1 = get_node_vars(u1, eq, nd, i - 1)
        u_ = get_node_vars(u1, eq, 1, i)
        up1 = get_node_vars(u1, eq, 2, i)

        # Convert to variables used for reconstruction, like prim or char.
        # If we are using conservative variables, this function does nothing

        con2recon_r(u) = blend.conservative2recon!(u, uar, eq)

        um1, u_, up1 = con2recon_r.((um1, u_, up1))

        h1, h2 = x - xl, xr - x
        back_, cent_, fwd_ = finite_differences(h1, h2, um1, u_, up1)

        slope_tuple = (minmod(cent_[n], back_[n], fwd_[n], beta, 0.0)
                       for n in eachvariable(eq))

        slope = SVector{nvar}(slope_tuple)

        # left, right face values at current time level
        ufl, ufr = u_ + slope * (xfl - x), u_ + slope * (xfr - x)
        u_s_l, u_s_r = u_ + 2.0 * slope * (xfl - x), u_ + 2.0 * slope * (xfr - x)

        recon2cons_r(u) = blend.recon2conservative!(u, uar, eq)

        ufl, ufr, u_s_l, u_s_r = recon2cons_r.((ufl, ufr, u_s_l, u_s_r))

        ufl_, ufr_ = limit_slope(eq, slope, ufl, u_s_l, ufr, u_s_r, u_, xfl - x,
                                 xfr - x)

        # left, right face fluxes at current time level
        fl_ = flux(xl, ufl_, eq)
        fr_ = flux(xr, ufr_, eq)
        for n in 1:nvar
            # perform update to get u^{n+1/2,+} with finite difference method
            unph[n, 2] = ufl_[n] + 0.5 * blend.dt[1] * (fl_[n] - fr_[n]) / (xfr - xfl)
        end

        # left, right fluxes at i^th face at time level n+1/2
        unph_l = get_node_vars(unph, eq, 1)
        unph_r = get_node_vars(unph, eq, 2)
        @views fl = flux(xf, unph_l, eq)
        @views fr = flux(xf, unph_r, eq)

        fn = num_flux(xf, unph_l, unph_r, fl, fr, unph_l, unph_r, eq, 1)

        # If positivity fails, set
        # it's supposed to be 0.25 microseconds
        # alp = test_alp(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op, alp)

        Fn = (1.0 - alp) * Fn + alp * fn

        Fn = get_blended_flux(i, eq, dt, grid, blend, scheme, xf, u1, fn, Fn, lamx, op,
                              alp)
        # Blend low and higher order flux
        # for n=1:nvar
        #    Fn[n] = (1.0-alp)*Fn[n] + alp*fn[n]
        # end

        # Fn_ = (1.0 - alp) * Fn + alp * fn
        # Fn_ = fn

        r = @view res[:, :, i - 1]
        # For the sub-cells which have same interface as super-cells, the same
        # numflux Fn is used in place of the lower order flux
        for n in eachvariable(eq)
            # r[n,nd] += alpha[i-1] * dt/dx[i-1] * Fn_[n]/wg[nd] # alpha[i-1] already in blend.lamx
            r[n, nd] += dt / dx[i - 1] * alpha[i - 1] * Fn[n] / wg[nd] # alpha[i-1] already in blend.lamx
        end

        r = @view res[:, :, i]
        for n in eachvariable(eq)
            # r[n,1] -= alpha[i] * dt/dx[i] * Fn_[n]/wg[1] # alpha[i-1] already in blend.lamx
            r[n, 1] -= dt / dx[i] * alpha[i] * Fn[n] / wg[1] # alpha[i-1] already in blend.lamx
        end

        return Fn, (1.0 - alpha[i - 1], 1.0 - alpha[i])
    end # limiter
end

@inline function trivial_cell_residual(i, eq::AbstractEquations{1}, num_flux,
                                       aux, lamx, dt, dx, xf, op, u1, u, ua, f, r)
    return nothing
end

function test_alp(i, eq::AbstractEquations{1}, dt, grid,
                  blend, scheme, xf, u1, fn, Fn,
                  lamx, op, alp)
    @unpack nvar = eq
    @unpack wg = op
    @unpack fn_low = blend
    @unpack dx = grid

    nd = length(op.xg)

    # Check if the end point of left cell is updated with postiivity
    fn_ll = get_node_vars(fn_low, eq, 2, i - 1)
    u = get_node_vars(u1, eq, nd, i - 1)
    @views test_ll = u -
                     (dt / dx[i - 1]) / wg[end] * ((1 - alp) * Fn + alp * fn - fn_ll)
    if is_admissible(eq, test_ll) == false
        return 1.0
    end

    # Check if the first point of right cell is updated with postiivity
    fn_rr = get_node_vars(fn_low, eq, 1, i)
    u_ = get_node_vars(u1, eq, 1, i)
    @views test_rr = u_ - (dt / dx[i]) / wg[1] * (fn_rr - ((1 - alp) * Fn + alp * fn))
    if is_admissible(eq, test_rr) == false
        return 1.0
    end
    return alp
end

function get_blended_flux(el_x, eq::AbstractEquations{1}, dt, grid,
                          blend, scheme, xf, u1, fn, Fn,
                          lamx, op, alp)
    @unpack wg = op
    @unpack fn_low = blend
    @unpack dx = grid

    nd = length(op.xg)

    # Check if the end point of left cell is updated with postiivity
    fn_inner_left_cell = get_node_vars(fn_low, eq, 2, el_x - 1)
    u_node = get_node_vars(u1, eq, nd, el_x - 1)

    c_ll = dt / (dx[el_x - 1] * wg[end]) # c is such that unew = u - c(Fn-fn_inner)

    test_update_ll = u_node - c_ll * (Fn - fn_inner_left_cell)
    lower_order_update_ll = u_node - c_ll * (fn - fn_inner_left_cell)
    if is_admissible(eq, test_update_ll) == false
        @debug "Using first order flux at" el_x, xf
        Fn = zhang_shu_flux_fix(eq, u_node, lower_order_update_ll,
                                Fn, fn_inner_left_cell, fn, c_ll)
    end

    # Check if the first point of right cell is updated with postiivity
    fn_inner_right_cell = get_node_vars(fn_low, eq, 1, el_x)
    u_node_ = get_node_vars(u1, eq, 1, el_x)

    c_rr = -(dt / dx[el_x]) / wg[1] # c is such that unew = u - c(Fn-fn_inner)

    test_rr = u_node_ - c_rr * (Fn - fn_inner_right_cell)
    lower_order_update_rr = u_node_ - c_rr * (fn - fn_inner_right_cell)

    if is_admissible(eq, test_rr) == false
        @debug "Using first order flux at" el_x, xf
        Fn = zhang_shu_flux_fix(eq, u_node_, lower_order_update_rr,
                                Fn, fn_inner_right_cell, fn, c_rr)
    end
    return Fn
end

@inline function trivial_face_residual(i, x, u1, ua, eq::AbstractEquations{1},
                                       dt, grid, op, scheme, param, Fn, aux,
                                       lamx, res)
    return Fn, (1.0, 1.0)
end

function is_admissible(::AbstractEquations{1, <:Any}, ::AbstractVector)
    # Check if the invariant domain in preserved. This has to be
    # extended in Equation module
    return true
end

@inbounds @inline function characteristic_reconstruction!(ue, ua,
                                                          eq::AbstractEquations{1})
    @assert false "Not implemented"
end

@inbounds @inline function characteristic2conservative!(ue, ua,
                                                        eq::AbstractEquations{1})
    @assert false "Not implemented"
end

struct Blend1D{F1, F2, F3, F4, F5, F6 <: Function, Parameters}
    alpha::OffsetVector{Float64, Vector{Float64}}
    # alpha_prev::Vector{Float64}
    space_time_alpha::ElasticArray{Float64, 2, 1, Array{Float64, 1}}
    time_levels_anim::ElasticArray{Float64, 1, 0, Array{Float64, 1}}
    xc::Array{Float64, 1}
    lamx::OffsetVector{Float64, Vector{Float64}} # lamx with (1-alpha[i]) factor
    ue::OffsetArray{Float64, 2, Array{Float64, 2}}
    xxf::OffsetArray{Float64, 1, Array{Float64, 1}} # faces of subcells
    xe::OffsetArray{Float64, 1, Array{Float64, 1}}  # super cell sol points + faces
    ufl::Vector{Float64}
    ufr::Vector{Float64}
    fl::Vector{Float64}
    fr::Vector{Float64}
    fn::Vector{Float64}
    unph::Array{Float64, 3} # solution
    resl::Array{Float64, 2} # array for lower order FV residual
    fn_low::OffsetArray{Float64, 3, Array{Float64, 3}}  # super cell sol points + faces
    amax::Float64           # upper bound for lower-order factor
    parameters::Parameters # Multiply constant node by this factor in indicator
    E1::Float64             # chosen so that E > E1 implies non-smooth
    E0::Float64             # chosen so that E < E0 implies smooth
    tolE::Float64           # Tolerance for denominator
    dt::Array{Float64, 1}   # Store MUSCL dt for DiffEq package compatibility
    E::Vector{Float64}
    beta::Float64           # \beta parameter in MC limiter for MUSCL
    debug::Bool
    a0::Float64             # Smoothing factor
    a1::Float64             # Smoothing factor
    idata::Vector{Float64}  # indicator data
    blend_cell_residual!::F1
    blend_face_residual!::F2
    get_indicating_variables!::F3
    conservative2recon!::F4
    recon2conservative!::F5
    numflux::F6
end

@inbounds @inline function no_upwinding_x(u1, eq, op, xf, element, Fn)
    return Fn
end

# Create Blend1D struct
function Blend(eq::AbstractEquations{1}, op, grid,
               problem::Problem,
               scheme::Scheme,
               param::Parameters,
               plot_data, bc_x = no_upwinding_x)
    @unpack xc, xf, dx = grid
    nx = grid.size
    @unpack degree, xg = op
    @unpack limiter = scheme

    if limiter.name != "blend"
        return (
                ; blend_cell_residual! = trivial_cell_residual,
                blend_face_residual! = trivial_face_residual,
                dt = [1e20])
    end
    println("Setting up blending limiter...")

    # KLUDGE - Add strings with names to these parameters like
    # indicating_variables, reconstruction_variables, etc
    @unpack (blend_type, indicating_variables, reconstruction_variables,
    smooth_alpha,
    amax, constant_node_factor, constant_node_factor2, c, a, amin,
    indicator_model, debug_blend, pure_fv,
    numflux) = limiter
    parameters = (; c, a, amin, smooth_alpha, constant_node_factor,
                  constant_node_factor2)
    nd = degree + 1
    nvar = nvariables(eq)
    if numflux === nothing
        numflux = scheme.numerical_flux
    end

    # Choose E1 for which E > E1 implies non-smoothness
    if indicator_model == "model1"
        if degree == 0
            E1 = 0.0
        elseif degree == 1
            E1 = 0.008
        elseif degree == 2
            E1 = 0.009
        elseif degree == 3
            E1 = 0.002
        elseif degree == 4
            E1 = 0.002
        else
            println("Indicator not implemented for degree")
            @assert false
        end
    elseif indicator_model == "model2"
        if degree == 1
            E1 = 0.016
        elseif degree == 2
            E1 = 0.014
        elseif degree == 3
            E1 = 0.004
        elseif degree == 4
            E1 = 0.004
        else
            println("Indicator not implemented for degree")
            @assert false
        end
    elseif indicator_model == "model3"
        if degree == 1
            println("Model 3 doesn't work for N=1")
            @assert false
        elseif degree == 2
            E1 = 0.2593
        elseif degree == 3
            E1 = 0.1062
        elseif degree == 4
            E1 = 0.0573
        else
            println("Indicator not implemented for degree")
            @assert false
        end
    elseif indicator_model == "model5"
        @assert degree>2 "Limiter only for degree > 2"
        E1 = 0.0873583
    elseif indicator_model == "draconian"
        @assert degree == 4
        E1 = -10.0
    elseif indicator_model == "gassner"
        @assert degree > 2||pure_fv == true "Limiter only for degree > 2"
        E1 = 0.0 # Irrelevant
    elseif indicator_model == "gassner_new"
        @assert degree>2 "Limiter only for degree > 2"
        E1 = 0.0 # Irrelevant
    elseif indicator_model == "gassner_face"
        @assert degree>2 "Limiter only for degree > 2"
        E1 = 0.0 # Irrelevant
    else
        println("Indicator not implemented for degree")
        @assert false
    end

    E0 = E1 * 1e-2 # E < E0 implies smoothness
    tolE = 1.0e-6  # If denominator < tolE, do purely high order
    E, alpha = zeros(nx), OffsetArray(zeros(nx + 2), OffsetArrays.Origin(0))
    a0 = 1.0 / 3.0
    a1 = 1.0 - 2.0 * a0              # smoothing coefficients
    idata = zeros(nx + 1)                          # t, alpha[1:nx]
    lamx = OffsetArray(zeros(nx + 2),
                       OffsetArrays.Origin(0))   # alpha[i] * dt/dx[i]
    xxf = OffsetArray(zeros(nd + 1),
                      OffsetArrays.Origin(0))   # faces of subcells
    xe = OffsetArray(zeros(nd + 2),
                     OffsetArrays.Origin(0))   # solution points + faces
    ue = OffsetArray(zeros(nvar, nd + 2),
                     OffsetArrays.Origin(1, 0))   # u extended by face extrapolation
    fn_low = OffsetArray(zeros(nvar, 2, nx + 2),
                         OffsetArrays.Origin(1, 1, 0))
    unph = zeros(nvar, 2, nd)              # u at time n+1/2, for MUSCL
    resl = zeros(nvar, nd)                 # lower order residual
    ufl, ufr, fl, fr, fn = [zeros(nvar) for _ in 1:5]

    @unpack p_ua = plot_data

    for n in 0:(nvar - 1)
        plot!(p_ua[end - n], grid.xc, alpha[1:(end - 1)], seriestype = :scatter,
              markershape = :square, color = :black, markersize = 1)
    end

    # Elastic arrays to create space time diagrams
    space_time_alpha = ElasticArray{Float64}(undef, nx, 0) # alpha values
    time_levels_anim = ElasticArray{Float64}(undef, 0)     # Time  levels

    @unpack blend_cell_residual!, blend_face_residual! = blend_type
    @unpack conservative2recon!, recon2conservative! = reconstruction_variables

    beta_muscl = 1e20 # unused dummy
    Blend1D(alpha, space_time_alpha, time_levels_anim, grid.xc, lamx,
            ue, xxf, xe, ufl, ufr, fl, fr, fn, unph, resl, fn_low, amax,
            parameters, E1, E0,
            tolE,
            zeros(1), E, beta_muscl, debug_blend, a0, a1, idata,
            blend_cell_residual!,
            blend_face_residual!, indicating_variables, conservative2recon!,
            recon2conservative!, numflux)
end

function Hierarchical(eq::AbstractEquations{1}, op, grid,
                      problem, scheme, param, plot_data)
    @unpack limiter = scheme
    if limiter.name != "hierarchical"
        return ()
    end
    nx = grid.size
    @unpack nvar = eq
    @unpack degree = op
    @unpack reconstruction, alpha = limiter
    @unpack conservative2recon!, recon2conservative! = reconstruction
    nd = degree + 1
    modes_cache = (OffsetArray(zeros(nvar, nd, nx + 2),
                               OffsetArrays.Origin(1, 1, 0)) for _ in 1:2)

    local_cache = (MArray{Tuple{nvar, nd}}(zeros(nvar, nd)) for _ in 1:3)

    hierarchical = (; modes_cache, local_cache, alpha, conservative2recon!,
                    recon2conservative!)
    # FIXME - Move nodal2modal, modal2nodal here
    return hierarchical
end

# This is a hack fix to support blending with DiffEquations.jl RK schemes.
# At RK stages, dt is not known but is needed for MUSCL. Thus, we store
# the dt within blend struct.
function set_blend_dt!(eq::AbstractEquations{1}, aux, dt)
    @unpack blend = aux
    blend.dt[1] = dt
end

# Pack blending methods into containers for user API

fo_blend(eq::AbstractEquations{1, <:Any}) = (;
                                             blend_cell_residual! = blend_cell_residual_fo!,
                                             blend_face_residual! = blend_face_residual_fo!,
                                             name = "fo")
mh_blend(eq::AbstractEquations{1, <:Any}) = (;
                                             blend_cell_residual! = blend_cell_residual_muscl!,
                                             blend_face_residual! = blend_face_residual_muscl!,
                                             name = "muscl")

#-------------------------------------------------------------------------------
# Compute error norm
#-------------------------------------------------------------------------------
function compute_error(problem, grid, eq::AbstractEquations{1}, aux, op, u1, t)
    @timeit aux.timer "Error computation" begin
        @unpack error_file = aux
        xmin, xmax = grid.domain
        @unpack xg = op
        nd = length(xg)

        @unpack exact_solution = problem

        nq = nd + 10    # number of quadrature points in each direction
        xq, wq = weights_and_points(nq, "gl")

        V = Vandermonde_lag(xg, xq) # matrix evaluating at `xq`
        # using values at solution points `xg`
        nx = grid.size
        xc = grid.xc
        dx = grid.dx

        l1_error, l2_error, linf_error, energy = 0.0, 0.0, 0.0, 0.0
        for i in 1:nx
            un, ue = zeros(nq), zeros(nq) # exact solution
            x = xc[i] - 0.5 * dx[i] .+ dx[i] * xq
            for i in 1:nq
                ue[i] = exact_solution(x[i], t)[1] # Error only for first variable
            end
            @views mul!(un, V, u1[1, :, i])
            du = abs.(un - ue)
            linf = maximum(du)
            l1 = dx[i] * BLAS.dot(nq, du, 1, wq, 1)
            @. du = du * du
            l2 = dx[i] * BLAS.dot(nq, du, 1, wq, 1)
            @. du = un * un
            e = dx[i] * BLAS.dot(nq, du, 1, wq, 1)
            l1_error += l1
            l2_error += l2
            linf_error = max(linf, linf_error)
            energy += e
        end
        domain_size = (xmax - xmin)
        l1_error = l1_error / domain_size
        l2_error = sqrt(l2_error / domain_size)
        energy = energy / domain_size
        @printf(error_file, "%.16e %.16e %.16e %.16e\n", t, l1_error, l2_error, energy)
        return Dict{String, Float64}("l1_error" => l1_error, "l2_error" => l2_error,
                                     "linf_error" => linf_error, "energy" => energy)
    end # timer
end

#-------------------------------------------------------------------------------
function create_aux_cache(eq::AbstractEquations{1}, op)
    nothing
end

function initialize_plot(eq::AbstractEquations{1, 1}, op, grid, problem, scheme,
                         timer, u1, ua)
    @timeit timer "Write solution" begin
        # Clear and re-create output directory
        rm("output", force = true, recursive = true)
        mkdir("output")

        nx = grid.size
        xf = grid.xf
        xc = grid.xc
        @unpack xg, degree = op
        nd = degree + 1
        @unpack initial_value = problem
        nu = max(nd, 2)
        xu = LinRange(0.0, 1.0, nu)
        Vu = Vandermonde_lag(xg, xu) # To get equispaced point values
        p_ua = plot() # Initialize plot object
        y = initial_value.(xc)
        ymin, ymax = @views minimum(y), maximum(y)
        # Add initial value at cell centres as a curve to p_ua, which write_soln!
        # will later replace with cell average values
        @views plot!(p_ua, xc, y, legend = false,
                     label = "Numerical Solution", title = "Cell averages, t = 0.0",
                     ylim = (ymin - 0.1, ymax + 0.1), linestyle = :dot,
                     color = :blue, markerstrokestyle = :dot, seriestype = :scatter,
                     markershape = :circle, markersize = 2, markerstrokealpha = 0)
        x = LinRange(xf[1], xf[end], 1000)
        plot!(p_ua, x, initial_value.(x), label = "Exact", color = :black) # Placeholder for exact
        xlabel!(p_ua, "x")
        ylabel!(p_ua, "u")

        p_u1 = plot() # Initialize plot object
        # Set up p_u1 to contain polynomial approximation as a different curve
        # for each cell
        x = LinRange(xf[1], xf[2], nu)
        u = @views Vu * u1[1, :, 1]
        plot!(p_u1, x, u, color = :blue, label = "u1")
        for i in 2:nx
            x = LinRange(xf[i], xf[i + 1], nu)
            u = @views Vu * u1[1, :, i]
            @views plot!(p_u1, x, u, color = :blue, label = nothing)
        end
        x = LinRange(xf[1], xf[end], 1000)
        plot!(p_u1, x, initial_value.(x), label = "Exact", color = :black) # Placeholder for exact
        anim_ua, anim_u1 = Animation(), Animation() # Initialize animation objects
        plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1)
        return plot_data
    end # timer
end

function write_soln!(base_name, fcount, iter, time, dt,
                     eq::AbstractEquations{1, 1}, grid,
                     problem, param, op, ua, u1, aux; ndigits = 3)
    @timeit aux.timer "Write solution" begin
        @unpack plot_data = aux
        avg_filename = get_filename("output/avg", ndigits, fcount)
        @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
        xc = grid.xc
        nx = grid.size
        @unpack xg, degree = op
        nd = degree + 1
        @unpack animate, save_time_interval, save_iter_interval = param
        @unpack exact_solution, final_time = problem
        exact(x) = exact_solution(x, time)
        nu = max(2, nd)
        xu = LinRange(0.0, 1.0, nu)
        Vu = Vandermonde_lag(xg, xu)

        # Update exact solution value in plots
        np = length(p_ua[1][2][:x]) # number of points of exact soln plotting
        for i in 1:np
            x = p_ua[1][2][:x][i]
            value = exact(x)
            p_ua[1][2][:y][i] = value
            p_u1[1][nx + 1][:y][i] = value
        end

        # Update ylims
        ylims = @views (minimum(p_ua[1][2][:y]) - 0.1, maximum(p_ua[1][2][:y]) + 0.1)
        ylims!(p_ua, ylims)
        ylims!(p_u1, ylims)

        # Write cell averages
        u = @view ua[1, 1:nx]
        writedlm("$avg_filename.txt", zip(xc, u), " ")
        # update solution by updating the y-series values
        for i in 1:nx
            p_ua[1][1][:y][i] = ua[1, i] # Loop is needed for plotly bug
        end
        t = round(time, digits = 3)
        title!(p_ua, "t = $t, iter=$iter")

        # write equispaced values within each cell
        sol_filename = get_filename("output/sol", ndigits, fcount)
        sol_file = open("$sol_filename.txt", "w")
        x, u = zeros(nu), zeros(nu) # Set up to store equispaced point values
        for i in 1:nx
            @views mul!(u, Vu, u1[1, :, i])
            @. x = grid.xf[i] + grid.dx[i] * xu
            p_u1[1][i][:y] .= u
            for ii in 1:nu
                @printf(sol_file, "%e %e\n", x[ii], u[ii])
            end
        end
        title!(p_u1, "t = $t, iter=$iter")
        close(sol_file)
        println("Wrote $avg_filename.txt $sol_filename.txt.")
        if problem.final_time - time < 1e-10
            cp("$avg_filename.txt", "./output/avg.txt", force = true)
            cp("$sol_filename.txt", "./output/sol.txt", force = true)
            println("Wrote final solution to avg.txt, sol.txt.")
        end
        if animate == true
            if abs(time - final_time) < 1.0e-10
                frame(anim_ua, p_ua)
                frame(anim_u1, p_u1)
            end
            if save_iter_interval > 0
                animate_iter_interval = save_iter_interval
                if mod(iter, animate_iter_interval) == 0
                    frame(anim_ua, p_ua)
                    frame(anim_u1, p_u1)
                end
            elseif save_time_interval > 0
                animate_time_interval = save_time_interval
                k1, k2 = ceil(t / animate_time_interval),
                         floor(t / animate_time_interval)
                if (abs(t - k1 * animate_time_interval) < 1e-10 ||
                    abs(t - k2 * animate_time_interval) < 1e-10)
                    frame(anim_ua, p_ua)
                    frame(anim_u1, p_u1)
                end
            end
        end
        fcount += 1
        return fcount
    end # timer
end

function post_process_soln(eq::AbstractEquations{1, 1}, aux, problem, param)
    @timeit aux.timer "Write solution" begin
        @unpack plot_data, error_file, timer = aux
        @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
        @unpack saveto, animate = param
        println("Post processing solution")
        savefig(p_ua, "output/avg.png")
        savefig(p_u1, "output/sol.png")
        savefig(p_ua, "output/avg.html")
        savefig(p_u1, "output/sol.html")
        if animate == true
            gif(anim_ua, "output/avg.mp4", fps = 5)
            gif(anim_u1, "output/sol.mp4", fps = 5)
        end
        println("Wrote avg, sol in gif,html,png format to output directory.")
        plot(p_ua)
        plot(p_u1)
        close(error_file)

        # TOTHINK - also write a file explaining all the parameters.
        # Ideally write a file out of args_dict

    end # timer

    # Print timer data on screen
    print_timer(aux.timer, sortby = :firstexec)
    print("\n")
    show(aux.timer)
    print("\n")
    timer_file = open("./output/timer.json", "w")
    JSON3.write(timer_file, TimerOutputs.todict(timer))
    close(timer_file)

    if saveto != "none"
        if saveto[end] == "/"
            saveto = saveto[1:(end - 1)]
        end
        mkpath(saveto)
        for file in readdir("./output")
            cp("./error.txt", "$saveto/error.txt", force = true) # KLUDGE/TOTHINK - should this be outside loop?
            cp("./output/$file", "$saveto/$file", force = true)
        end
        println("Saved output files to $saveto")
    end
    return nothing
end
end # @muladd

end # module
