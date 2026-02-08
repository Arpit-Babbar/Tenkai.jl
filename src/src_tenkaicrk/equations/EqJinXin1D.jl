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
               compute_face_residual!, correct_variable_bound_limiter!

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!)

import Tenkai.EqEuler1D: max_abs_eigen_value, get_density, get_pressure

import Tenkai.TenkaicRK: implicit_source_solve

# The original system is u_t + f(u)_x = 0. The Jin-Xin relaxed system has variables (u,v).
# The flux is (v, advection(u)). The source terms are (0, -(v-f(u)) / epsilon).

struct JinXin1D{NDIMS, NVAR, TWO_NVAR, Equations <: AbstractEquations{NDIMS, NVAR},
                Advection,
                AdvectionPlus, AdvectionMinus,
                RealT <: Real} <: AbstractEquations{NDIMS, TWO_NVAR}
    equations::Equations
    advection::Advection
    advection_evolution::Vector{RealT}
    advection_plus::AdvectionPlus
    advection_minus::AdvectionMinus
    epsilon::RealT
    name::String
    initial_values::Dict{String, Function}
    nvar::Int
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

function jin_xin_source(u, x, t, eq::JinXin1D)
    equations = eq.equations
    u_var_ = u_var(u, eq)
    v_var_ = v_var(u, eq)
    source_1 = zero(u_var_)
    source_2 = -(v_var_ - flux(x, u_var_, equations)) / eq.epsilon
    return SVector(source_1..., source_2...)
end

# equation is lhs + coefficient * s(u^{n+1}) = u^{n+1}
function implicit_source_solve(lhs, eq_jin_xin::JinXin1D, x, t, coefficient, source_terms::typeof(jin_xin_source),
                               u_node, implicit_solver = nothing)
    equations = eq_jin_xin.equations
    u_var_new = u_var(lhs, eq_jin_xin) # Since there is no source term for this part
    flux_new = flux(x, u_var_new, equations)
    v_lhs = v_var(lhs, eq_jin_xin)
    epsilon = eq_jin_xin.epsilon
    v_var_new = (epsilon * v_lhs + coefficient * flux_new) / (epsilon + coefficient)
    sol_new = SVector(u_var_new..., v_var_new...)
    source = jin_xin_source(sol_new, x, t, eq_jin_xin)
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
function compute_time_step(eq_jin_xin::JinXin1D, problem, grid, aux, op, cfl, u1, ua)
    @timeit aux.timer "Time step computation" begin
    #! format: noindent
    nx = grid.size
    dx = grid.dx
    jin_xin_adv = 0.0
    den = 0.0
    for i in 1:nx
        sx = max_abs_eigen_value(eq_jin_xin.equations, ua[:, i])
        jin_xin_adv = max(sx, jin_xin_adv)
        den = max(den, abs.(sx) / dx[i] + 1.0e-12)
    end

    # Jin-Xin constant is made larger by this factor
    dt_scaling = 0.5

    dt = cfl * dt_scaling^2 / den
    eq_jin_xin.advection_evolution[1] = jin_xin_adv / dt_scaling
    return dt, eq_jin_xin
    end # Timer
end

function get_equation(equations::AbstractEquations{NDIMS, NVARS},
                      advection, advection_plus, advection_minus,
                      epsilon) where {NDIMS, NVARS}
    name = "1D shallow water equations"
    numfluxes = Dict("roe" => roe, "rusanov" => rusanov)
    initial_values = Dict()

    RealT = Float64

    advection_evolution = zeros(RealT, 1)

    TWO_NVAR = 2 * NVARS

    return JinXin1D{NDIMS, NVARS, TWO_NVAR, typeof(equations), typeof(advection),
                    typeof(advection_plus), typeof(advection_minus), typeof(epsilon)}(equations,
                                                                                      advection,
                                                                                      advection_evolution,
                                                                                      advection_plus,
                                                                                      advection_minus,
                                                                                      epsilon,
                                                                                      name,
                                                                                      initial_values,
                                                                                      TWO_NVAR,
                                                                                      numfluxes)
end

end # module
