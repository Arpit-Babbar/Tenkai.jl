module EqVarAdv1D
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
using Tenkai.Basis

import Tenkai: admissibility_tolerance

(import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
                eigmatrix,
                limit_slope, zhang_shu_flux_fix,
                apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln,
                correct_variable_bound_limiter!)

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!, update_ghost_values_lwfr!, refresh!,
               calc_source, cRKSolver)

import ..TenkaicRK: calc_non_cons_gradient, calc_non_cons_Bu, non_conservative_equation,
                    update_ghost_values_ub_N!, AbstractNonConservativeEquations,
                    calc_non_cons_B

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This is used my multiply_add_to_node_vars! to find out how many variables have to be
# differentiated, which is just 1 (density) in the case of the shear shallow water equations.
struct VarAdvNonConservative1D <: AbstractNonConservativeEquations{1, 1}
end

# Shear shallow water equations in 1D
struct VarAdv1D{Advection} <: AbstractNonConservativeEquations{1, 2}
    nvar::Int64
    name::String
    adv::Advection
    non_conservative_part::VarAdvNonConservative1D
end

#-------------------------------------------------------------------------------
# PDE Information
#-------------------------------------------------------------------------------
@inbounds @inline function flux(x, u, eq::VarAdv1D)
    return SVector(0.0, 0.0)
end

@inbounds @inline flux(U, eq::VarAdv1D) = flux(1.0, U, eq)

non_conservative_equation(eq::VarAdv1D) = eq.non_conservative_part

# This will compute the term to be differentiated.
calc_non_cons_gradient(u_node, x_, t, eq::VarAdv1D) = u_node

function calc_non_cons_B(u, x_, t, eq::VarAdv1D)
    # HERE NOW!!
    @unpack adv = eq
    v, x = u[1], u[2]
    B = SMatrix{2, 1}(adv(x), 0.0)
    # By_u = SVector(0.0, 0.0, 0.0, 0.0, 0.5 * g * h * v1 * h_nc, g * h * v2 * h_nc)
    return B # + By_u
end

# This will compute the action of B on u_non_cons. The u_non_cons may
# be the derivative or it may not. Both quantities need to be computed.
function calc_non_cons_Bu(u, u_non_cons, x_, t, eq::VarAdv1D)
    @unpack adv = eq
    v = u_non_cons[1]
    x = u[2]
    Bx_u = SVector(adv(x) * v, 0.0)
    # By_u = SVector(0.0, 0.0, 0.0, 0.0, 0.5 * g * h * v1 * h_nc, g * h * v2 * h_nc)
    return Bx_u # + By_u
end

function compute_time_step(eq::VarAdv1D, problem, grid, aux, op, cfl, u1, ua)
    @unpack adv = eq
    nx = grid.size
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        u = get_node_vars(ua, eq, i)
        smax = max_abs_eigen_value(eq, u)
        den = max(den, smax / dx[i])
    end
    dt = cfl / den
    return dt, eq
end

function max_abs_eigen_value(eq::VarAdv1D, u)
    @unpack adv = eq
    x = u[2]
    return abs(adv(x))
end

function max_abs_eigen_value(eq::VarAdv1D, ul, ur)
    return max(max_abs_eigen_value(eq, ul), max_abs_eigen_value(eq, ur))
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::VarAdv1D,
                                   dir)
    λ = max(max_abs_eigen_value(eq, ual), max_abs_eigen_value(eq, uar)) # local wave speed

    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function Tenkai.eigmatrix(eq::VarAdv1D, u)
    @unpack gravity = eq

    @assert false "To be implemented!"

    # Inverse eigenvector-matrix
    L11 = 1.0 - g2 * v^2 / c^2
    L21 = (g2 * v^2 - v * c) / (d * c)

    L12 = g1 * v / c^2
    L22 = (c - g1 * v) / (d * c)

    L = SMatrix{nvariables(eq), nvariables(eq)}(L11, L21,
                                                L12, L22)

    # Eigenvector matrix
    R11 = 1.0
    R21 = v

    R12 = f
    R22 = (v + c) * f

    R = SMatrix{nvariables(eq), nvariables(eq)}(R11, R21,
                                                R12, R22)

    return R, L
end

admissibility_tolerance(eq::VarAdv1D) = 1e-10

function Tenkai.limit_slope(eq::VarAdv1D, s, ufl, u_s_l, ufr, u_s_r, ue, xl,
                            xr)
    @assert false "To be implemented!"
end

function Tenkai.zhang_shu_flux_fix(eq::VarAdv1D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    @assert false "To be implemented!"
end

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

function Tenkai.apply_bound_limiter!(eq::VarAdv1D, grid, scheme, param, op,
                                     ua,
                                     u1, aux)
    u1_ = @view u1[1:1, :, :]
    ua_ = @view ua[1:1, :]
    apply_bound_limiter!(eq.non_conservative_part, grid, scheme, param, op, ua_, u1_,
                         aux)
end

function initialize_plot(eq::VarAdv1D, op, grid, problem, scheme,
                         timer, u1_, ua_)
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
    initial_value_ = problem.initial_value
    initial_value = (x) -> initial_value_(x)[1]
    u1 = @view u1_[1:1, :, :]
    ua = @view ua_[1:1, :]
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
                     eq::VarAdv1D, grid,
                     problem, param, op, ua_, u1_, aux; ndigits = 3)
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
    ua = @view ua_[1:1, :]
    u1 = @view u1_[1:1, :, :]
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

function post_process_soln(eq::VarAdv1D, aux, problem, param, scheme)
    @timeit aux.timer "Write solution" begin
    #! format: noindent
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

function get_equation(adv)
    name = "1D non-conservative variable advection"
    nvar = 2
    non_conservative_part = VarAdvNonConservative1D()

    return VarAdv1D(nvar, name, adv, non_conservative_part)
end
end # muladd
end # module
