module EqJinXin2D
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

import Tenkai: admissibility_tolerance

import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
               eigmatrix,
               limit_slope, zhang_shu_flux_fix,
               apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
               write_soln!, compute_time_step, post_process_soln,
               compute_face_residual!, correct_variable_bound_limiter!,
               pre_process_limiter!, modal_smoothness_indicator_gassner, set_node_vars!,
               calc_source

import Tenkai.TenkaicRK: implicit_source_solve, get_cache_node_vars

using Tenkai.TenkaicRK: newton_solver

import Tenkai.EqEuler1D: rho_p_indicator!, rusanov, max_abs_eigen_value, get_density,
                         get_pressure

using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
              get_node_vars,
              nvariables, eachvariable,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, update_ghost_values_u1!,
              debug_blend_limiter!, UsuallyIgnored

# TODO - Have an abstract Jin-Xin as well.

# The original system is u_t + f(u)_x = 0. The Jin-Xin relaxed system has variables (u,v).
# The flux is (v, advection(u)). The source terms are (0, -(v-f(u)) / epsilon).

# TODO - Remove TWO_NVAR so that the get_function equation does not have that complicated call
# Wherever you need i in NVAR+1:TWO_NVAR, make it NVAR+i for i in 1:NVAR.
struct JinXin2D{NDIMS, NVAR, THREE_NVAR, Equations <: AbstractEquations{NDIMS, NVAR},
                RealT <: Real} <: AbstractEquations{NDIMS, THREE_NVAR}
    equations::Equations
    advection_evolution::Vector{RealT} # Maximum eigen value
    epsilon_arr::OffsetArray{RealT, 2, Array{RealT, 2}}
    thresholds::Tuple{RealT, RealT} # For shock-capturing
    name::String
    jin_xin_dt_scaling::RealT
    indicator_model::String
end

function v_var(u, index, eq::JinXin2D)
    nvar = nvariables(eq.equations)
    v = SVector((u[i + index * nvar] for i in (1):nvar)...)
    return v
end

function u_var(u, eq::JinXin2D)
    nvar = nvariables(eq.equations)
    u_ = SVector((u[i] for i in 1:nvar)...)
    return u_
end

function get_density(eq_jin_xin::JinXin2D, u)
    get_density(eq_jin_xin.equations, u_var(u, eq_jin_xin))
end

function get_pressure(eq_jin_xin::JinXin2D, u)
    get_pressure(eq_jin_xin.equations, u_var(u, eq_jin_xin))
end

function flux(x, y, u, eq::JinXin2D, orientation)
    adv = eq.advection_evolution[orientation]
    u_var_ = u_var(u, eq)
    v_var_ = v_var(u, orientation, eq)
    if orientation == 1
        return SVector(v_var_..., adv * u_var_..., zero(u_var_)...)
    else # orientation == 2
        return SVector(v_var_..., zero(u_var_)..., adv * u_var_...)
    end
end

function flux(x, y, u, eq::JinXin2D)
    return flux(x, y, u, eq::JinXin2D, 1), flux(x, y, u, eq::JinXin2D, 2)
end

function jin_xin_source(u, epsilon, x, t, eq::JinXin2D)
    equations = eq.equations
    u_var_ = u_var(u, eq)
    v1_var = v_var(u, 1, eq)
    v2_var = v_var(u, 2, eq)

    source_1 = zero(u_var_)
    source_2 = -(v1_var - flux(x[1], x[2], u_var_, equations, 1)) / epsilon
    source_3 = -(v2_var - flux(x[1], x[2], u_var_, equations, 2)) / epsilon
    return SVector(source_1..., source_2..., source_3...)
end

function get_cache_node_vars(aux, u1, problem, scheme, eq::JinXin2D, ignored_element::UsuallyIgnored, i, j)
    el_x, el_y = ignored_element.value
    u_node = get_node_vars(u1, eq, i, j)
    epsilon_node = eq.epsilon_arr[el_x, el_y]
    return (u_node, epsilon_node)
end

function set_node_vars!(u, aux_node::Tuple, eq::JinXin2D, indices...)
    (u_node, _) = aux_node
    set_node_vars!(u, u_node, eq, indices...)
end

function calc_source(aux_node, x, t, source_terms::typeof(jin_xin_source), eq::JinXin2D)
    (u_node, epsilon_node) = aux_node
    return source_terms(u_node, epsilon_node, x, t, eq)
end

# equation is lhs + coefficient * s(u^{n+1}) = u^{n+1}
function implicit_source_solve(lhs, eq_jin_xin::JinXin2D, x, t, coefficient,
                               source_terms::typeof(jin_xin_source),
                               aux_node, implicit_solver = nothing)
    (u_node, epsilon_node) = aux_node
    equations = eq_jin_xin.equations
    u_var_new = u_var(lhs, eq_jin_xin) # Since there is no source term for this part
    f_new, g_new = flux(x[1], x[2], u_var_new, equations)
    v1_lhs, v2_lhs = v_var(lhs, 1, eq_jin_xin), v_var(lhs, 2, eq_jin_xin)
    epsilon = epsilon_node
    v1_var_new = (epsilon * v1_lhs + coefficient * f_new) / (epsilon + coefficient)
    v2_var_new = (epsilon * v2_lhs + coefficient * g_new) / (epsilon + coefficient)
    sol_new = SVector(u_var_new..., v1_var_new..., v2_var_new...)
    source = jin_xin_source(sol_new, epsilon_node, x, t, eq_jin_xin)
    @assert false sol_new, source, lhs, u_node
    return sol_new, source
end

struct JinXinICBC{InitialCondition, Equations}
    initial_condition::InitialCondition
    equations::Equations
end

function (jin_xin_ic::JinXinICBC)(x, y)
    @unpack equations = jin_xin_ic.equations
    u = jin_xin_ic.initial_condition(x, y)
    v1 = flux(x, y, u, equations, 1)
    v2 = flux(x, y, u, equations, 2)
    return SVector(u..., v1..., v2...)
end

# Use this for exact solution or boundary values
function (jin_xin_bc::JinXinICBC)(x, y, t)
    @unpack equations = jin_xin_bc.equations
    u = jin_xin_bc.initial_condition(x, y, t)
    v1 = flux(x, y, u, equations, 1)
    v2 = flux(x, y, u, equations, 2)
    return SVector(u..., v1..., v2...)
end

Tenkai.n_plotted_variables(eq_jin_xin::JinXin2D) = nvariables(eq_jin_xin.equations)

function Tenkai.initialize_plot(eq_jin_xin::JinXin2D, op, grid, problem, scheme, timer, u1_,
                                ua_)
    equations = eq_jin_xin.equations
    nvar = nvariables(equations)
    u1 = @view u1_[1:nvar, :, :, :, :]
    ua = @view ua_[1:nvar, :, :, :, :]
    # @assert false fieldnames(problem.initial_value)
    problem_correct = @set problem.initial_value = problem.initial_value.initial_condition
    return Tenkai.initialize_plot(equations, op, grid, problem_correct, scheme, timer, u1,
                                  ua)
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq_jin_xin::JinXin2D, grid,
                            problem, param, op, ua_, u1_, aux, ndigits = 3)
    equations = eq_jin_xin.equations
    nvar = nvariables(equations)
    u1 = @view u1_[1:nvar, :, :, :, :]
    ua = @view ua_[1:nvar, :, :, :, :]
    problem_correct = @set problem.initial_value = problem.initial_value.initial_condition
    problem_correct = @set problem.exact_solution = problem.exact_solution.initial_condition
    return Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq_jin_xin.equations, grid,
                              problem_correct, param, op, ua, u1, aux)
end

function Tenkai.post_process_soln(eq::JinXin2D, aux, problem, param, scheme)
    problem_correct = @set problem.initial_value = problem.initial_value.initial_condition
    post_process_soln(eq.equations, aux, problem_correct, param, scheme)
end

function max_abs_eigen_value(eq::JinXin2D, u, dir)
    # return sqrt(eq.advection(0.0, 1.0, eq)) # TODO - Pretty ugly hack
    return eq.advection_evolution[dir]
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin2D, dir)
    λ = max(max_abs_eigen_value(eq, ual, dir), max_abs_eigen_value(eq, uar, dir)) # local wave speed

    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

@inbounds @inline function rho_p_indicator!(un, eq::JinXin2D)
    return Tenkai.EqEuler2D.rho_p_indicator!(un, eq.equations)
end

function pre_process_limiter!(eq::JinXin2D, t, iter, fcount, dt, grid, problem, scheme,
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

function modal_smoothness_indicator_gassner(eq::JinXin2D, t, iter,
                                            fcount, dt, grid, scheme, problem,
                                            param, aux, op, u1, ua)
    @timeit aux.timer "Blending limiter" begin
    #! format: noindent
    @unpack dx, dy = grid
    nx, ny = grid.size
    nvar = nvariables(eq)
    @unpack xg = op
    nd = length(xg)
    @unpack limiter = scheme
    @unpack blend = aux
    @unpack constant_node_factor = blend.parameters
    @unpack amax = blend.parameters      # maximum factor of the lower order term
    @unpack tolE = blend.parameters      # tolerance for denominator
    @unpack E = blend.cache            # content in high frequency nodes
    @unpack alpha_max, alpha, alpha_temp = blend.cache    # vector containing smoothness indicator values
    @unpack (c, a, amin, a0, a1, smooth_alpha, smooth_factor) = blend.parameters # smoothing coefficients
    @unpack get_indicating_variables! = blend.subroutines
    @unpack cache = blend
    @unpack Pn2m = cache

    epsilon_min, epsilon_max = eq.thresholds

    @threaded for element in CartesianIndices((1:nx, 1:ny))
        el_x, el_y = element[1], element[2]
        un, um, tmp = cache.nodal_modal[Threads.threadid()]
        # Continuous extension to faces
        u = @view u1[:, :, :, el_x, el_y]
        @turbo un .= u

        # Copying is needed because we replace these with variables actually
        # used for indicators like primitives or rho*p, etc.

        # Convert un to ind var, get no. of variables used for indicator
        get_indicating_variables!(un, eq)

        multiply_dimensionwise!(um, Pn2m, un, tmp)

        # ind = zeros(n_ind_nvar)
        ind = 0.0
        # KLUDGE - You are assuming n_ind_var = 1

        # um[n,1,1] *= constant_node_factor
        # TOTHINK - avoid redundant calculations in total_energy_clip1, 2, etc.?
        total_energy = total_energy_clip1 = total_energy_clip2 = 0.0
        for j in Base.OneTo(nd), i in Base.OneTo(nd) # TOTHINK - Why is @turbo bad here?
            total_energy += um[1, i, j]^2
        end

        for j in Base.OneTo(nd - 1), i in Base.OneTo(nd - 1)
            total_energy_clip1 += um[1, i, j]^2
        end

        for j in Base.OneTo(nd - 2), i in Base.OneTo(nd - 2)
            total_energy_clip2 += um[1, i, j]^2
        end

        total_energy_den = total_energy - um[1, 1, 1]^2 +
                           (constant_node_factor * um[1, 1, 1])^2

        if total_energy > tolE
            ind1 = (total_energy - total_energy_clip1) / total_energy_den
        else
            ind1 = 0.0
        end

        if total_energy_clip1 > tolE
            ind2 = (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
        else
            ind2 = 0.0
        end

        ind = max(ind1, ind2)
        E[el_x, el_y] = maximum(ind) # maximum content among all indicating variables
        epsilon_arr[el_x, el_y] = max(epsilon_min,
                                      min(epsilon_max, 200000 * E[el_x, el_y]))
    end

    alpha_temp .= epsilon_arr

    if problem.periodic_x
        @turbo for j in 1:ny
            epsilon_arr[0, j] = epsilon_arr[nx, j]
            epsilon_arr[nx + 1, j] = epsilon_arr[1, j]
        end
    else
        @turbo for j in 1:ny
            epsilon_arr[0, j] = epsilon_arr[1, j]
            epsilon_arr[nx + 1, j] = epsilon_arr[nx, j]
        end
    end

    if problem.periodic_y
        @turbo for i in 1:nx
            epsilon_arr[i, 0] = epsilon_arr[i, ny]
            epsilon_arr[i, ny + 1] = epsilon_arr[i, 1]
        end
    else
        @turbo for i in 1:nx
            epsilon_arr[i, 0] = epsilon_arr[i, 1]
            epsilon_arr[i, ny + 1] = epsilon_arr[i, ny]
        end
    end

    # Smoothening of alpha
    if smooth_alpha == true
        @turbo alpha_temp .= alpha
        for j in 1:ny, i in 1:nx
            epsilon_arr[i, j] = max(smooth_factor * alpha_temp[i - 1, j],
                                    smooth_factor * alpha_temp[i, j - 1],
                                    epsilon_arr[i, j],
                                    smooth_factor * alpha_temp[i + 1, j],
                                    smooth_factor * alpha_temp[i, j + 1])
        end
    end

    if dt > 0.0
        blend.cache.dt[1] = dt # hacky fix for compatibility with OrdinaryDiffEq
    end

    # KLUDGE - Should this be in apply_limiter! function?
    debug_blend_limiter!(eq, grid, problem, scheme, param, aux, op,
                         dt, t, iter, fcount, ua, u1)

    return nothing
    end # timer
end

# DOES NOT WORK AS WELL AS RUSANOV. STRANGE!!!
@inbounds @inline function upwind(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin2D, dir)
    return Fl
end

# As bad as the upwind flux
@inbounds @inline function flux_central(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin2D, dir)
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
function apply_bound_limiter!(eq_jin_xin::JinXin2D, grid, scheme, param, op, ua, u1, aux)
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
function compute_time_step(eq_jin_xin::JinXin2D, problem, grid, aux, op, cfl,
                           u1, ua)
    @timeit aux.timer "Time step computation" begin
    #! format: noindent
    @unpack jin_xin_dt_scaling = eq_jin_xin
    nx, ny = grid.size
    @unpack dx, dy = grid
    jin_xin_adv1 = 0.0
    jin_xin_adv2 = 0.0
    den = 0.0
    for element in CartesianIndices((0:(nx + 1), 0:(ny + 1)))
        el_x, el_y = element[1], element[2]
        ua_node = get_node_vars(ua, eq_jin_xin, el_x, el_y)
        sx, sy = max_abs_eigen_value(eq_jin_xin.equations, ua_node, 1),
                 max_abs_eigen_value(eq_jin_xin.equations, ua_node, 2)
        jin_xin_adv1 = max(sx, jin_xin_adv1)
        jin_xin_adv2 = max(sy, jin_xin_adv2)
        den = max(den, abs(sx) / dx[el_x] + abs(sy) / dy[el_y] + 1.0e-12)
    end

    dt = cfl * jin_xin_dt_scaling^2 / den
    eq_jin_xin.advection_evolution[1] = jin_xin_adv1 / jin_xin_dt_scaling
    eq_jin_xin.advection_evolution[2] = jin_xin_adv2 / jin_xin_dt_scaling
    return dt, eq_jin_xin
    end # timer
end

function get_equation(equations::AbstractEquations{NDIMS, NVARS},
                      epsilon, nx, ny; indicator_model = "gassner",
                      thresholds = (1e-12, 1e-4),
                      jin_xin_dt_scaling = 0.5) where {NDIMS, NVARS}
    name = "2D Jin-Xin equations"

    RealT = typeof(epsilon)

    advection_evolution = zeros(RealT, 2)

    THREE_NVAR = 3 * NVARS

    epsilon_arr = OffsetArray(zeros(RealT, nx + 2, ny + 2), 0:(nx + 1), 0:(ny + 1))
    epsilon_arr .= epsilon

    return JinXin2D{NDIMS, NVARS, THREE_NVAR, typeof(equations), typeof(epsilon)}(equations,
                                                                                  advection_evolution,
                                                                                  epsilon_arr,
                                                                                  thresholds,
                                                                                  name,
                                                                                  jin_xin_dt_scaling,
                                                                                  indicator_model)
end

end # module
