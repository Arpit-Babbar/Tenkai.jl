module EqMHD1D

import GZip
using DelimitedFiles
using Plots
using LinearAlgebra
using Printf
using TimerOutputs
using StaticArrays
using Polyester
using LoopVectorization
using JSON3
using SimpleUnPack

using Tenkai
using Tenkai.Basis

import Tenkai: admissibility_tolerance

(import Tenkai: flux, prim2con, con2prim,
                eigmatrix,
                limit_slope, zhang_shu_flux_fix,
                apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln)

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars, sum_node_vars_1d,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!,
               correct_variable_bound_limiter!)

import Trixi

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct MHD1D{TrixiEquations} <: AbstractEquations{1, 8}
    trixi_equations::TrixiEquations
    name::String
end

@inbounds @inline function flux(x, u, eq::MHD1D)
    Trixi.flux(u, 1, eq.trixi_equations)
end

@inbounds @inline flux(u, eq::MHD1D) = flux(1.0, u, eq)

# The matrix fprime(U)
function fprime(eq::MHD1D, x, U)
    @assert false "fprime is not implemented for MHD1D"
end

# function converting primitive variables to PDE variables
function prim2con(eq::MHD1D, prim)
    return Trixi.prim2cons(prim, eq.trixi_equations)
end

function con2prim(eq::MHD1D, con)
    return Trixi.cons2prim(con, eq.trixi_equations)
end

function density(eq::MHD1D, u)
    return Trixi.density(u, eq.trixi_equations)
end

function pressure(eq::MHD1D, u)
    return Trixi.pressure(u, eq.trixi_equations)
end

function velocity(eq::MHD1D, u)
    return Trixi.velocity(u, eq.trixi_equations)
end

function Tenkai.is_admissible(eq::MHD1D, u::AbstractVector)
    return density(eq, u) > 0.0 && pressure(eq, u) > 0.0
end

function Tenkai.eigmatrix(eq::MHD1D, u)
    @assert false "eigmatrix is not implemented for MHD1D"
end

function compute_time_step(eq::MHD1D, problem, grid, aux, op, cfl, u1, ua)
    @unpack source_terms = problem
    nx = grid.size
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        u = get_node_vars(ua, eq, i)
        smax = Trixi.calc_fast_wavespeed(u, 1, eq.trixi_equations)
        den = max(den, smax / dx[i])
    end
    dt = cfl / den
    return dt
end

# Looping over a tuple can be made type stable following this
# https://github.com/trixi-framework/Trixi.jl/blob/0fd86e4bd856d894de6a7514edcb9758bf6f8e1e/src/callbacks_stage/positivity_zhang_shu.jl#L39
function iteratively_apply_bound_limiter!(eq::MHD1D, grid, scheme, param, op, ua,
                                          u1, aux, variables::NTuple{N, Any}) where {N}
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    correct_variable_bound_limiter!(variable, eq, grid, op, ua, u1)

    # test_variable_bound_limiter!(variable, eq, grid, op, ua, u1)
    iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua,
                                     u1, aux, remaining_variables)
    return nothing
end

function iteratively_apply_bound_limiter!(eq::MHD1D, grid, scheme, param, op, ua,
                                          u1, aux, variables::Tuple{})
    return nothing
end

function Tenkai.apply_bound_limiter!(eq::MHD1D, grid, scheme, param, op, ua,
                                     u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end
    variables = (density, pressure)
    iteratively_apply_bound_limiter!(eq, grid, scheme, param, op, ua, u1, aux,
                                     variables)
    return nothing
end

@inbounds @inline function rho_p_indicator!(un, eq::MHD1D)
    for ix in 1:size(un, 2) # loop over dofs and faces
        u_node = get_node_vars(un, eq, ix)
        p = pressure(eq, u_node)
        un[1, ix] *= p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

function Tenkai.zhang_shu_flux_fix(eq::MHD1D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    uhigh = uprev - c * (Fn - fn_inner) # First candidate for high order update
    ρ_low, ρ_high = density(eq, ulow), density(eq, uhigh)
    eps = 0.1 * ρ_low
    ratio = abs(eps - ρ_low) / (abs(ρ_high - ρ_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Second candidate for flux
    end

    uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
    p_low, p_high = pressure(eq, ulow), pressure(eq, uhigh)
    eps = 0.1 * p_low
    ratio = abs(eps - p_low) / (abs(p_high - p_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Final flux
    end

    return Fn
end

function Tenkai.limit_slope(eq::MHD1D, slope, ufl, u_star_ll, ufr, u_star_rr,
                            ue, xl, xr, el_x = nothing, el_y = nothing)

    # The MUSCL-Hancock scheme is guaranteed to be admissibility preserving if
    # slope is chosen so that
    # u_star_l = ue + 2.0*slope*xl, u_star_r = ue+2.0*slope*xr are admissible
    # ue is already admissible and we know we can find sequences of thetas
    # to make theta*u_star_l+(1-theta)*ue is admissible.
    # This is equivalent to replacing u_star_l by
    # u_star_l = ue + 2.0*theta*s*xl.
    # Thus, we simply have to update the slope by multiplying by theta.

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, density, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, pressure, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    ufl = ue + slope * xl
    ufr = ue + slope * xr

    return ufl, ufr, slope
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::MHD1D, dir)
    λ = Trixi.max_abs_speed_naive(ual, uar, 1, eq.trixi_equations)

    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

struct ExactSolutionAlfvenWave{TrixiEquations}
    equations::MHD1D{TrixiEquations}
end

function (exact_solution_alfven_wave::ExactSolutionAlfvenWave)(x, t)
    @unpack equations = exact_solution_alfven_wave
    @unpack trixi_equations = equations

    # Set up the Alfven wave initial condition
    return Trixi.initial_condition_convergence_test(x, t, trixi_equations)
end

function Tenkai.initialize_plot(eq::MHD1D, op, grid, problem, scheme, timer, u1, ua)
    @timeit timer "Write solution" begin
    #! format: noindent
    @timeit timer "Initialize write solution" begin
    #! format: noindent
    # Clear and re-create output directory
    rm("output", force = true, recursive = true)
    mkdir("output")

    xc = grid.xc
    nx = grid.size
    @unpack xg = op
    nd = op.degree + 1
    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    xf = grid.xf
    nvar = nvariables(eq)
    # Create plot objects to be later collected as subplots

    # Creating a subplot for title
    p_title = plot(title = "Cell averages plot, $nx cells, t = 0.0",
                   grid = false, showaxis = false, bottom_margin = 0Plots.px)
    # Initialize subplots for density, velocity and pressure
    p_ua, p_u1 = [plot() for _ in 1:nvar], [plot() for _ in 1:nvar]
    labels = Trixi.varnames(Trixi.cons2prim, eq.trixi_equations)
    y = zeros(nx) # put dummy to fix plotly bug with OffsetArrays
    for n in 1:nvar
        @views plot!(p_ua[n], xc, y, label = "Approximate",
                     linestyle = :dot, seriestype = :scatter,
                     color = :blue, markerstrokestyle = :dot,
                     markershape = :circle, markersize = 2,
                     markerstrokealpha = 0)
        xlabel!(p_ua[n], "x")
        ylabel!(p_ua[n], labels[n])
    end
    l_super = @layout[a{0.01h}; b c d; e f g; h i] # Selecting layout for p_title being title
    p_ua = plot(p_title, p_ua[1], p_ua[2], p_ua[3], p_ua[4], p_ua[5], p_ua[6],
                p_ua[7], p_ua[8],
                layout = l_super, size = (2500, 2000)) # Make subplots

    # Set up p_u1 to contain polynomial approximation as a different curve
    # for each cell
    x = LinRange(xf[1], xf[2], nu)
    up1 = zeros(nvar, nd)
    u = zeros(nu)
    for ii in 1:nd
        @views up1[:, ii] .= con2prim(eq, u1[:, ii, 1]) # store prim form in up1
    end

    for n in 1:nvar
        u = @views Vu * up1[n, :]
        plot!(p_u1[n], x, u, color = :red, legend = false)
        xlabel!(p_u1[n], "x")
        ylabel!(p_u1[n], labels[n])
    end
    for i in 2:nx
        for ii in 1:nd
            @views up1[:, ii] .= con2prim(eq, u1[:, ii, i]) # store prim form in up1
        end
        x = LinRange(xf[i], xf[i + 1], nu)
        for n in 1:nvar
            u = @views Vu * up1[n, :]
            plot!(p_u1[n], x, u, color = :red, label = nothing, legend = false)
        end
    end

    l = @layout[a{0.01h}; b c d; e f g; h i] # Selecting layout for p_title being title
    p_u1 = plot(p_title, p_u1[1], p_u1[2], p_u1[3], p_u1[4], p_u1[5], p_u1[6],
                p_u1[7], p_u1[8],
                layout = l,
                size = (2500, 2000)) # Make subplots

    anim_ua, anim_u1 = Animation(), Animation() # Initialize animation objects
    plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1)
    return plot_data
    end # timer
    end # timer
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::MHD1D, grid,
                            problem, param, op, ua, u1, aux, ndigits = 3)
    @timeit aux.timer "Write solution" begin
    #! format: noindent
    @unpack plot_data = aux
    avg_filename = get_filename("output/avg", ndigits, fcount)
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    @unpack final_time = problem
    xc = grid.xc
    nx = grid.size
    @unpack xg = op
    nd = op.degree + 1
    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    nvar = nvariables(eq)
    @unpack save_time_interval, save_iter_interval, animate = param
    avg_file = open("$avg_filename.txt", "w")
    up_ = zeros(nvar)
    ylims = [[Inf, -Inf] for _ in 1:nvar] # set ylims for plots of all variables
    for i in 1:nx
        @views up_ .= con2prim(eq, ua[:, i]) # store primitive form in up_
        @printf(avg_file, "%e %e %e %e %e %e %e %e %e\n", xc[i], up_[1], up_[2],
                up_[3],
                up_[4], up_[5], up_[6], up_[7], up_[8])
        # TOTHINK - Check efficiency of printf
        for n in 1:nvar
            p_ua[n + 1][1][:y][i] = @views up_[n]    # Update y-series
            ylims[n][1] = min(ylims[n][1], up_[n]) # Compute ymin
            ylims[n][2] = max(ylims[n][2], up_[n]) # Compute ymax
        end
    end
    close(avg_file)
    for n in 1:nvar # set ymin, ymax for ua, u1 plots
        ylims!(p_ua[n + 1], (ylims[n][1] - 0.1, ylims[n][2] + 0.1))
        ylims!(p_u1[n + 1], (ylims[n][1] - 0.1, ylims[n][2] + 0.1))
    end
    t = round(time; digits = 3)
    title!(p_ua[1], "Cell averages plot, $nx cells, t = $t")
    sol_filename = get_filename("output/sol", ndigits, fcount)
    sol_file = open(sol_filename * ".txt", "w")
    up1 = zeros(nvar, nd)

    u = zeros(nvar, nu)
    x = zeros(nu)
    for i in 1:nx
        for ii in 1:nd
            @views up1[:, ii] .= con2prim(eq, u1[:, ii, i]) # store prim form in up1
        end
        @. x = grid.xf[i] + grid.dx[i] * xu
        @views mul!(u, up1, Vu')
        for n in 1:nvar
            p_u1[n + 1][i][:y] = u[n, :]
        end
        for ii in 1:nu
            @printf(sol_file, "%e %e %e %e\n", x[ii], u[1, ii], u[2, ii], u[3, ii])
        end
    end
    close(sol_file)
    title!(p_u1[1], "Numerical Solution, $nx cells, t = $t")
    println("Wrote $sol_filename.txt, $avg_filename.txt")
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
            k1, k2 = ceil(time / animate_time_interval),
                     floor(time / animate_time_interval)
            if (abs(time - k1 * animate_time_interval) < 1e-10 ||
                abs(time - k2 * animate_time_interval) < 1e-10)
                frame(anim_ua, p_ua)
                frame(anim_u1, p_u1)
            end
        end
    end
    fcount += 1
    return fcount
    end # timer
end

function Tenkai.post_process_soln(eq::MHD1D, aux, problem, param, scheme)
    @unpack timer, error_file = aux
    @timeit timer "Write solution" begin
    #! format: noindent
    println("Post processing solution")
    nvar = nvariables(eq)
    @unpack plot_data = aux
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    @unpack animate, saveto = param
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
    if saveto != "none"
        if saveto[end] == "/"
            saveto = saveto[1:(end - 1)]
        end
        mkpath(saveto)
        for file in readdir("./output")
            cp("./output/$file", "$saveto/$file", force = true)
        end
        cp("./error.txt", "$saveto/error.txt", force = true)
        println("Saved output files to $saveto")
    end
    end # timer

    # Print timer data on screen
    print_timer(aux.timer, sortby = :firstexec)
    print("\n")
    show(aux.timer)
    print("\n")
    println("Time outside write_soln = "
            *
            "$(( TimerOutputs.tottime(timer)
                - TimerOutputs.time(timer["Write solution"]) ) * 1e-9)s")
    println("─────────────────────────────────────────────────────────────────────────────────────────")
    timer_file = open("./output/timer.json", "w")
    JSON3.write(timer_file, TimerOutputs.todict(timer))
    close(timer_file)
    return nothing
end

function get_equation(gamma)
    name = "Ideal GLM MHD 1D"
    trixi_equations = Trixi.IdealGlmMhdEquations1D(gamma)
    return MHD1D(trixi_equations, name)
end
end # muladd
end # module EqMHD1D
