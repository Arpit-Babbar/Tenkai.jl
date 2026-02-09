module EqJinXin1D
#! format: noindent

import GZip
using Tenkai.DelimitedFiles
using Plots
using LinearAlgebra
using Tenkai.SimpleUnPack
using Printf
using TimerOutputs
using StaticArrays
using Tenkai.Polyester
using Tenkai.LoopVectorization
using Tenkai.JSON3
using OffsetArrays

using Tenkai
using Accessors: @set
import Tenkai.EqTenMoment1D
using Tenkai.Basis

import Tenkai: admissibility_tolerance, calc_source

import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
               eigmatrix,
               limit_slope, zhang_shu_flux_fix,
               apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
               write_soln!, compute_time_step, post_process_soln,
               compute_face_residual!, correct_variable_bound_limiter!,
               pre_process_limiter!, modal_smoothness_indicator_gassner, set_node_vars!

import Tenkai.TenkaicRK: implicit_source_solve, get_cache_node_vars

using Tenkai.TenkaicRK: newton_solver

import Tenkai.EqEuler1D: rho_p_indicator!, rusanov, max_abs_eigen_value

using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
              get_node_vars,
              nvariables, eachvariable,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, update_ghost_values_u1!,
              debug_blend_limiter!, UsuallyIgnored

import Tenkai.EqEuler1D: max_abs_eigen_value, get_density, get_pressure

import Tenkai.EqEuler1D: max_abs_eigen_value, get_density, get_pressure

import Tenkai.TenkaicRK: implicit_source_solve

# The original system is u_t + f(u)_x = 0. The Jin-Xin relaxed system has variables (u,v).
# The flux is (v, advection(u)). The source terms are (0, -(v-f(u)) / epsilon).

# TODO - Remove TWO_NVAR so that the get_function equation does not have that complicated call
# Wherever you need i in NVAR+1:TWO_NVAR, make it NVAR+i for i in 1:NVAR.
struct JinXin1D{NDIMS, NVAR, TWO_NVAR, Equations <: AbstractEquations{NDIMS, NVAR},
                Advection,
                AdvectionPlus, AdvectionMinus,
                RealT <: Real} <: AbstractEquations{NDIMS, TWO_NVAR}
    equations::Equations
    advection::Advection
    advection_evolution::Vector{RealT}
    advection_plus::AdvectionPlus
    advection_minus::AdvectionMinus
    epsilon_arr::OffsetVector{RealT, Vector{RealT}}
    thresholds::Tuple{RealT, RealT}
    name::String
    initial_values::Dict{String, Function}
    nvar::Int
    jin_xin_dt_scaling::RealT
    indicator_model::String
    numfluxes::Dict{String, Function}
end

function v_var(u, eq::JinXin1D)
    nvar = nvariables(eq.equations)
    two_nvar = nvariables(eq)
    v = SVector((u[i] for i in (nvar + 1):two_nvar)...)
    return v
end

function u_var(u, eq::JinXin1D)
    nvar = nvariables(eq.equations)
    u_ = SVector((u[i] for i in 1:nvar)...)
    return u_
end

function get_density(eq_jin_xin::JinXin1D, u)
    get_density(eq_jin_xin.equations, u_var(u, eq_jin_xin))
end

function get_pressure(eq_jin_xin::JinXin1D, u)
    get_pressure(eq_jin_xin.equations, u_var(u, eq_jin_xin))
end

function flux(x, u, eq::JinXin1D)
    adv = eq.advection_evolution[1]
    u_var_ = u_var(u, eq)
    v_var_ = v_var(u, eq)
    flux_1 = v_var_
    flux_2 = adv^2 * u_var_
    return SVector(flux_1..., flux_2...)
end

function jin_xin_source(u, epsilon, x, t, eq::JinXin1D)
    equations = eq.equations
    u_var_ = u_var(u, eq)
    v_var_ = v_var(u, eq)
    source_1 = zero(u_var_)
    source_2 = -(v_var_ - flux(x, u_var_, equations)) / epsilon
    return SVector(source_1..., source_2...)
end

function get_cache_node_vars(aux, u, problem, scheme, eq::JinXin1D,
                             ignored_cell::UsuallyIgnored,
                             i)
    cell = ignored_cell.value
    u_node = get_node_vars(u, eq, i)
    epsilon_node = eq.epsilon_arr[cell]
    return (u_node, epsilon_node)
end

function get_cache_node_vars(aux, u1, problem, scheme, eq::JinXin1D, i, cell)
    u_node = get_node_vars(u1, eq, i, cell)
    epsilon_node = eq.epsilon_arr[cell]
    return (u_node, epsilon_node)
end

function implicit_source_solve(lhs, eq::JinXin1D, x, t, coefficient, source_terms, aux_node,
                               implicit_solver = newton_solver)
    (u_node, epsilon_node) = aux_node
    # TODO - Make sure that the final source computation is used after the implicit solve
    implicit_F(u_new) = u_new - lhs -
                        coefficient * jin_xin_source(u_new, epsilon_node, x, t, eq)

    u_new = implicit_solver(implicit_F, u_node) # TODO - replace it with the exact solver
    source = jin_xin_source(u_new, epsilon_node, x, t, eq)
    return u_new, source
end

function set_node_vars!(u, aux_node::Tuple, eq::JinXin1D, i)
    (u_node, _) = aux_node
    set_node_vars!(u, u_node, eq, i)
end

# equation is lhs + coefficient * s(u^{n+1}) = u^{n+1}
function implicit_source_solve(lhs, eq_jin_xin::JinXin1D, x, t, coefficient,
                               source_terms::typeof(jin_xin_source),
                               aux_node, implicit_solver = nothing)
    (u_node, epsilon_node) = aux_node
    equations = eq_jin_xin.equations
    u_var_new = u_var(lhs, eq_jin_xin) # Since there is no source term for this part
    flux_new = flux(x, u_var_new, equations)
    v_lhs = v_var(lhs, eq_jin_xin)
    epsilon = epsilon_node
    v_var_new = (epsilon * v_lhs + coefficient * flux_new) / (epsilon + coefficient)
    sol_new = SVector(u_var_new..., v_var_new...)
    source = jin_xin_source(sol_new, epsilon_node, x, t, eq_jin_xin)
    return sol_new, source
end

struct JinXinICBC{InitialCondition, Equations}
    initial_condition::InitialCondition
    equations::Equations
end

function (jin_xin_ic::JinXinICBC)(x)
    @unpack equations = jin_xin_ic.equations
    u = jin_xin_ic.initial_condition(x)
    v = flux(x, u, equations)
    return SVector(u..., v...)
end

# Use this for exact solution or boundary values
function (jin_xin_bc::JinXinICBC)(x, t)
    @unpack equations = jin_xin_bc.equations
    u = jin_xin_bc.initial_condition(x, t)
    v = flux(x, u, equations)
    return SVector(u..., v...)
end

Tenkai.n_plotted_variables(eq_jin_xin::JinXin1D) = nvariables(eq_jin_xin.equations)

function Tenkai.initialize_plot(eq_jin_xin::JinXin1D, op, grid, problem, scheme, timer, u1_,
                                ua_)
    equations = eq_jin_xin.equations
    nvar = nvariables(equations)
    u1 = @view u1_[1:nvar, :, :]
    ua = @view ua_[1:nvar, :, :]
    # @assert false fieldnames(problem.initial_value)
    problem_correct = @set problem.initial_value = problem.initial_value.initial_condition
    return Tenkai.initialize_plot(equations, op, grid, problem_correct, scheme, timer, u1,
                                  ua)
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq_jin_xin::JinXin1D, grid,
                            problem, param, op, ua_, u1_, aux, ndigits = 3)
    equations = eq_jin_xin.equations
    nvar = nvariables(equations)
    u1 = @view u1_[1:nvar, :, :]
    ua = @view ua_[1:nvar, :, :]
    problem_correct = @set problem.initial_value = problem.initial_value.initial_condition
    problem_correct = @set problem.exact_solution = problem.exact_solution.initial_condition
    return Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq_jin_xin.equations, grid,
                              problem_correct, param, op, ua, u1, aux)
end

function Tenkai.post_process_soln(eq::JinXin1D, aux, problem, param, scheme)
    problem_correct = @set problem.initial_value = problem.initial_value.initial_condition
    post_process_soln(eq.equations, aux, problem_correct, param, scheme)
end

@inbounds @inline function roe(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin1D, dir)
    # TODO - This will not be high order accurate. For Roe's flux, find the dissipation part
    # using ual, uar, Ul, Ur and use Fl, Fr for the central part. See the roe flux in EqEuler1D.jl
    return eq.advection_plus(ual, uar, Ul, eq) + eq.advection_minus(ual, uar, Ur, eq)
end

function max_abs_eigen_value(eq::JinXin1D, u)
    # return sqrt(eq.advection(0.0, 1.0, eq)) # TODO - Pretty ugly hack
    return eq.advection_evolution[1]
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin1D, dir)
    λ = max(max_abs_eigen_value(eq, ual), max_abs_eigen_value(eq, uar)) # local wave speed

    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

@inbounds @inline function rho_p_indicator!(un, eq::JinXin1D)
    return Tenkai.EqEuler1D.rho_p_indicator!(un, eq.equations)
end

function pre_process_limiter!(eq::JinXin1D, t, iter, fcount, dt, grid, problem, scheme,
                              param, aux, op, u1, ua)
    @timeit aux.timer "Limiter" begin
    #! format: noindent
    @timeit aux.timer "Pre process limiter" begin
    #! format: noindent
    @unpack limiter = scheme
    if limiter.name != "blend"
        return nothing
    end
    @unpack indicator_model = limiter
    update_ghost_values_u1!(eq, problem, grid, op, u1, aux, t)
    # This will set epislon to E, and keep alpha to be zero.
    if indicator_model == "gassner"
        modal_smoothness_indicator_gassner(eq, t, iter, fcount, dt, grid, scheme,
                                           problem, param, aux, op, u1, ua)
    else
        @assert false "Indicator model $(indicator_model) not implemented yet"
        # elseif indicator_model == "gassner_new"
        #     @assert false "Not implemented yet"
        #     modal_smoothness_indicator_gassner_new(eq, t, iter, fcount, dt, grid,
        #                                            scheme, problem, param, aux, op,
        #                                            u1, ua)
        # elseif indicator_model == "gassner_face"
        #     modal_smoothness_indicator_gassner_face(eq, t, iter, fcount, dt, grid,
        #                                             scheme, problem, param, aux, op,
        #                                             u1, ua)
        # else
        #     modal_smoothness_indicator_new(eq, t, iter, fcount, dt, grid, scheme,
        #                                    problem, param, aux, op, u1, ua)
    end
    return nothing
    end # timer
    end # timer
end

function modal_smoothness_indicator_gassner(eq::JinXin1D, t, iter,
                                            fcount, dt, grid, scheme, problem,
                                            param, aux, op, u1, ua)
    @timeit aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack xc, dx = grid
    nx = grid.size
    nvar = nvariables(eq)
    @unpack Vl, Vr, xg = op
    nd = length(xg)
    @unpack limiter = scheme
    left_bc, right_bc = problem.boundary_condition
    @unpack blend = aux
    amax = blend.amax      # maximum factor of the lower order term
    @unpack (constant_node_factor, constant_node_factor2, c, a, amin, smoothing_in_time) = blend.parameters # Multiply constant node by this factor in indicator
    @unpack E1, E0 = blend # smoothness and discontinuity thresholds
    tolE = blend.tolE      # tolerance for denominator
    E = blend.E            # content in high frequency nodes
    @unpack alpha, alpha0 = blend    # vector containing smoothness indicator values
    @unpack a0, a1 = blend # smoothing coefficients

    epsilon_min, epsilon_max = eq.thresholds

    # some strings specifying the kind of blending
    @unpack (indicator_model, indicating_variables) = limiter

    RealT = eltype(u1)
    # Get nodal basis from values at extended solution points
    Pn2m = nodal2modal(xg)

    un, um = zeros(RealT, nvar, nd), zeros(RealT, nvar, nd) # Nodal, modal values in a cell

    @unpack epsilon_arr = eq

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

        ind = zeros(RealT, n_ind_nvar)

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

        epsilon_arr[i] = max(epsilon_min, min(epsilon_max, 200000 * E[i]))
    end

    if problem.periodic_x
        epsilon_arr[0], epsilon_arr[nx + 1] = epsilon_arr[nx], epsilon_arr[1]
    else
        epsilon_arr[0], epsilon_arr[nx + 1] = epsilon_arr[1], epsilon_arr[nx]
    end

    if left_bc == neumann && right_bc == neumann
        # Force first order on boundary for Shu-Osher
        epsilon_arr[1] = epsilon_arr[nx] = 1.0
    end

    # smoothing in time
    if smoothing_in_time
        for i in 1:nx
            epsilon_arr[i] = max(0.9 * alpha0[i], 0.5 * alpha0[i - 1],
                                 0.5 * alpha0[i + 1],
                                 epsilon_arr[i])
        end
    end
    # Smoothening of alpha
    alpha0 .= epsilon_arr
    for i in 1:nx
        epsilon_arr[i] = max(0.5 * alpha0[i - 1], epsilon_arr[i], 0.5 * alpha0[i + 1])
        epsilon_arr[i] = min(epsilon_arr[i], amax)
    end

    if dt > 0.0
        blend.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
    end

    blend.lamx .= zero(eltype(blend.lamx))

    alpha .= 0.0
    alpha0 .= 0.0

    # KLUDGE - Should this be in apply_limiter! function?
    debug_blend_limiter!(eq, grid, problem, scheme, param, aux, op,
                         dt, t, iter, fcount, ua, u1)
    end # timer
end

# DOES NOT WORK AS WELL AS RUSANOV. STRANGE!!!
@inbounds @inline function upwind(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin1D, dir)
    return Fl
end

# As bad as the upwind flux
@inbounds @inline function flux_central(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin1D, dir)
    return 0.5 * (Fl + Fr)
end

function iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                          u1, aux, variables::NTuple{N, Any}) where {N}
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    correct_variable_bound_limiter!(variable, eq, grid, op, ua, u1)

    # test_variable_bound_limiter!(variable, eq, grid, op, ua, u1)
    iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                     u1, aux, remaining_variables)
    return nothing
end

function iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                          u1, aux, variables::Tuple{})
    return nothing
end

# TODO - Implement the bound limiter
function apply_bound_limiter!(eq_jin_xin::JinXin1D, grid, scheme, param, op, ua, u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end
    # equations = eq_jin_xin.equations
    # nvar = nvariables(equations)
    # u1 = @view u1_[1:nvar, :, :]
    # ua = @view ua_[1:nvar, :, :]
    variables = (get_density, get_pressure)
    iteratively_apply_bound_limiter!(eq_jin_xin, grid, scheme, param, op, ua, u1, aux,
                                     variables)
end

#-------------------------------------------------------------------------------
# Compute dt using cell average
#-------------------------------------------------------------------------------
function compute_time_step(eq_jin_xin::JinXin1D, problem, grid, aux, op, cfl,
                           u1, ua)
    @timeit aux.timer "Time step computation" begin
    #! format: noindent
    @unpack jin_xin_dt_scaling = eq_jin_xin
    nx = grid.size
    dx = grid.dx
    jin_xin_adv = 0.0
    den = 0.0
    for i in 1:nx
        ua_node = get_node_vars(ua, eq_jin_xin, i)
        sx = max_abs_eigen_value(eq_jin_xin.equations, ua_node)
        jin_xin_adv = max(sx, jin_xin_adv)
        den = max(den, abs.(sx) / dx[i] + 1.0e-12)
    end

    dt = cfl * jin_xin_dt_scaling^2 / den
    eq_jin_xin.advection_evolution[1] = jin_xin_adv / jin_xin_dt_scaling
    return dt, eq_jin_xin
    end # timer
end

function get_equation(equations::AbstractEquations{NDIMS, NVARS},
                      advection, advection_plus, advection_minus,
                      epsilon, nx; indicator_model = "gassner",
                      thresholds = (1e-12, 1e-4),
                      jin_xin_dt_scaling = 1.0) where {NDIMS, NVARS}
    name = "1D shallow water equations"
    numfluxes = Dict("roe" => roe, "rusanov" => rusanov)
    initial_values = Dict()

    RealT = typeof(epsilon)

    advection_evolution = zeros(RealT, 1)

    TWO_NVAR = 2 * NVARS

    epsilon_arr = OffsetArray(zeros(RealT, nx + 2), 0:(nx + 1))
    epsilon_arr .= epsilon

    return JinXin1D{NDIMS, NVARS, TWO_NVAR, typeof(equations), typeof(advection),
                    typeof(advection_plus), typeof(advection_minus), typeof(epsilon)}(equations,
                                                                                      advection,
                                                                                      advection_evolution,
                                                                                      advection_plus,
                                                                                      advection_minus,
                                                                                      epsilon_arr,
                                                                                      thresholds,
                                                                                      name,
                                                                                      initial_values,
                                                                                      TWO_NVAR,
                                                                                      jin_xin_dt_scaling,
                                                                                      indicator_model,
                                                                                      numfluxes)
end

end # module
