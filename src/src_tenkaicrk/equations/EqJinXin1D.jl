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

# The original system is u_t + f(u)_x = 0. The Jin-Xin relaxed system has variables (u,v).
# The flux is (v, advection(u)). The source terms are (0, -(v-f(u)) / epsilon).

struct JinXin1D{NDIMS, NVAR, TWO_NVAR, Equations <: AbstractEquations{NDIMS, NVAR},
                Advection,
                AdvectionPlus, AdvectionMinus,
                RealT <: Real} <: AbstractEquations{NDIMS, TWO_NVAR}
    equations::Equations
    advection::Advection
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

function flux(x, u, eq::JinXin1D)
    u_var_ = u_var(u, eq)
    v_var_ = v_var(u, eq)
    flux_1 = v_var_
    flux_2 = eq.advection(x, u_var_, eq)
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
    nvar = nvariables(eq_jin_xin.equations)

    u1 = @view u1_[1:nvar, :, :]
    ua = @view ua_[1:nvar, :]

    @timeit timer "Write solution" begin
    #! format: noindent
    # Clear and re-create output directory
    rm("output", force = true, recursive = true)
    mkdir("output")

    nx = grid.size
    xf = grid.xf
    xc = grid.xc
    @unpack xg, degree = op
    nd = degree + 1
    initial_value = problem.initial_value
    initial_value_(x) = initial_value(x)[1]
    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu) # To get equispaced point values
    p_ua = plot() # Initialize plot object
    y = initial_value_.(xc)
    ymin, ymax = @views minimum(y), maximum(y)
    # Add initial value at cell centres as a curve to p_ua, which write_soln!
    # will later replace with cell average values
    @views plot!(p_ua, xc, y, legend = false,
                 label = "Numerical Solution", title = "Cell averages, t = 0.0",
                 ylim = (ymin - 0.1, ymax + 0.1), linestyle = :dot,
                 color = :blue, markerstrokestyle = :dot, seriestype = :scatter,
                 markershape = :circle, markersize = 2, markerstrokealpha = 0)
    x = LinRange(xf[1], xf[end], 1000)
    plot!(p_ua, x, initial_value_.(x), label = "Exact", color = :black) # Placeholder for exact
    xlabel!(p_ua, "x")
    ylabel!(p_ua, "u")

    p_u1 = plot() # Initialize plot object
    # Set up p_u1 to contain polynomial approximation as a different curve
    # for each cell
    x = LinRange(xf[1], xf[2], nu)
    u = @views Vu * u1_[1, :, 1]
    plot!(p_u1, x, u, color = :blue, label = "u1")
    for i in 2:nx
        x = LinRange(xf[i], xf[i + 1], nu)
        u = @views Vu * u1_[1, :, i]
        @views plot!(p_u1, x, u, color = :blue, label = nothing)
    end
    x = LinRange(xf[1], xf[end], 1000)
    plot!(p_u1, x, initial_value_.(x), label = "Exact", color = :black) # Placeholder for exact
    anim_ua, anim_u1 = Animation(), Animation() # Initialize animation objects
    plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1)
    return plot_data
    end # timer
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::JinXin1D, grid,
                            problem, param, op, ua_, u1_, aux, ndigits = 3)
    nvar = nvariables(eq.equations)
    u1 = @view u1_[1:nvar, :, :]
    ua = @view ua_[1:nvar, :]

    @timeit aux.timer "Write solution" begin
    #! format: noindent
    @unpack plot_data = aux
    avg_filename = get_filename("output/avg", ndigits, fcount)
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    xc = grid.xc
    nx = grid.size
    @unpack xg, degree = op
    nd = degree + 1
    @unpack animate, save_time_interval, save_iter_interval = param
    @unpack exact_solution, final_time = problem
    exact(x) = exact_solution(x, time)[1]
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

function Tenkai.post_process_soln(eq::JinXin1D, aux, problem, param, scheme)
    post_process_soln(eq.equations, aux, problem, param, scheme)
end

@inbounds @inline function roe(x, ual, uar, Fl, Fr, Ul, Ur, eq::JinXin1D, dir)
    # TODO - This will not be high order accurate. For Roe's flux, find the dissipation part
    # using ual, uar, Ul, Ur and use Fl, Fr for the central part. See the roe flux in EqEuler1D.jl
    return eq.advection_plus(ual, uar, Ul, eq) + eq.advection_minus(ual, uar, Ur, eq)
end

function max_abs_eigen_value(eq::JinXin1D, u)
    return sqrt(eq.advection(0.0, 1.0, eq)) # TODO - Pretty ugly hack
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

# TODO - Implement the bound limiter
apply_bound_limiter!(eq::JinXin1D, grid, scheme, param, op, ua, u1, aux) = nothing

#-------------------------------------------------------------------------------
# Compute dt using cell average
#-------------------------------------------------------------------------------
function compute_time_step(eq::JinXin1D, problem, grid, aux, op, cfl, u1,
                           ua)
    @timeit aux.timer "Time step computation" begin
    #! format: noindent
    nx = grid.size
    xc = grid.xc
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        # sx = @views speed(xc[i], ua[1, i], eq.equations)
        sx = max_abs_eigen_value(eq, ua[:, i])
        den = max(den, abs.(sx) / dx[i] + 1.0e-12)
    end
    dt = cfl / den
    return dt, eq
    end # Timer
end

function get_equation(equations::AbstractEquations{NDIMS, NVARS},
                      advection, advection_plus, advection_minus,
                      epsilon) where {NDIMS, NVARS}
    name = "1D shallow water equations"
    numfluxes = Dict("roe" => roe, "rusanov" => rusanov)
    initial_values = Dict()

    TWO_NVAR = 2 * NVARS

    return JinXin1D{NDIMS, NVARS, TWO_NVAR, typeof(equations), typeof(advection),
                    typeof(advection_plus), typeof(advection_minus), typeof(epsilon)}(equations,
                                                                                      advection,
                                                                                      advection_plus,
                                                                                      advection_minus,
                                                                                      epsilon,
                                                                                      name,
                                                                                      initial_values,
                                                                                      TWO_NVAR,
                                                                                      numfluxes)
end

end # module
