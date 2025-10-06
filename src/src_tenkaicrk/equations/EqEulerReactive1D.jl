module EqEulerReactive1D
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
                compute_face_residual!, correct_variable_bound_limiter!)

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!, calc_source, cRKSolver)

using ..TenkaicRK: newton_solver, picard_solver

import ..TenkaicRK: calc_non_cons_gradient, calc_non_cons_Bu, non_conservative_equation, implicit_source_solve

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct EulerReactive1D{RealT} <: AbstractEquations{1, 4}
    gamma::RealT
    q0::RealT
    nvar::Int
    name::String
    initial_values::Dict{String, Function}
    numfluxes::Dict{String, Function}
    varnames::Vector{String}
end

@inbounds @inline function density(eq::EulerReactive1D, u)
    return u[1]
end

@inbounds @inline function pressure(eq::EulerReactive1D, u)
    rho, rho_v1, E, rho_z = u
    v1 = rho_v1 / rho
    return (eq.gamma - 1.0) * (E - 0.5 * rho_v1 * v1 - rho_z * eq.q0)
end

@inbounds @inline function rho_p_indicator!(un, eq::EulerReactive1D)
    for ix in 1:size(un, 2) # loop over dofs and faces
        u_node = get_node_vars(un, eq, ix)
        p = pressure(eq, u_node)
        un[1, ix] *= p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

@inbounds @inline function flux(x, u, eq::EulerReactive1D)
    rho, rho_v1, E, rho_z = u
    v1 = rho_v1 / rho
    p = pressure(eq, u)
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = (E + p) * v1
    f4 = rho_z * v1
    return SVector(f1, f2, f3, f4)
end

@inbounds @inline flux(U, eq::EulerReactive1D) = flux(1.0, U, eq)

# The matrix fprime(U)
function fprime(eq::EulerReactive1D, x, U)
    @assert false "Not implemented"
end

# function converting primitive variables to PDE variables
function prim2con(eq::EulerReactive1D, prim) # primitive, gas constant
    rho, v1, p, z = prim
    E = p / (eq.gamma - 1.0) + 0.5 * rho * v1^2 + rho * z * eq.q0
    return SVector(rho, rho * v1, E, rho * z)
end

function con2prim(eq::EulerReactive1D, con)
    rho, rho_v1, E, rho_z = con
    v1 = rho_v1 / rho
    p = pressure(eq, con)
    return SVector(rho, v1, p, rho_z)
end

function con2prim!(eq::EulerReactive1D, cons, prim)
    prim .= con2prim(eq, cons)
end

function max_abs_eigen_value(eq::EulerReactive1D, u)
    @unpack gamma = eq
    rho = u[1]
    p = pressure(eq, u)
    v = u[2] / rho
    c = sqrt(gamma * p / rho)
    # Source = Wang, Zhang, Shu, Ning (2013) - Robust high order DG for 2D gaseous detonations
    return abs(v) + c
end

function compute_time_step(eq::EulerReactive1D, problem, grid, aux, op, cfl, u1, ua)
    nx = grid.size
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        u = get_node_vars(ua, eq, i)
        smax = max_abs_eigen_value(eq, u)
        den = max(den, smax / dx[i])
    end
    dt = cfl / den
    return dt
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::EulerReactive1D,
                                   dir)
    λ = max(max_abs_eigen_value(eq, ual), max_abs_eigen_value(eq, uar)) # local wave speed

    return 0.5 * (Fl + Fr - λ * (Ur - Ul))

    # return SVector(0.5*(Fl[1]+Fr[1]-λ*(Ur[1]-Ul[1])),
    #         0.5*(Fl[2]+Fr[2]-λ*(Ur[2]-Ul[2])),
    #         0.5*(Fl[3]+Fr[3]-λ*(Ur[3]-Ul[3])),
    #         0.5*(Fl[4]+Fr[4]))
end

function Tenkai.apply_bound_limiter!(eq::EulerReactive1D, grid, scheme, param, op, ua,
                                     u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end
    correct_variable_bound_limiter!(density, eq, grid, op, ua, u1)
    correct_variable_bound_limiter!(pressure, eq, grid, op, ua, u1)
end

function correct_variable_bound_limiter!(variable, eq::EulerReactive1D, grid, op, ua,
                                         u1)
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

function eigmatrix(eq::EulerReactive1D, U)
    Id = SMatrix{4, 4}(1.0, 0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 1.0)
    return Id, Id, Id, Id
end

function Tenkai.apply_tvb_limiter!(eq::EulerReactive1D, problem, scheme, grid, param,
                                   op, ua,
                                   u1, aux)
    @timeit aux.timer "TVB limiter" begin
    #! format: noindent
    nx = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack limiter = scheme
    @unpack tvbM, cache = limiter
    left_bc, right_bc = problem.boundary_condition
    nd = length(wg)
    nvar = nvariables(eq)
    # face values
    (uimh, uiph, Δul, Δur, Δual, Δuar, char_Δul, char_Δur, char_Δual, char_Δuar,
    dulm, durm, du) = cache

    # Loop over cells
    for cell in 1:nx
        ual, ua_, uar = (get_node_vars(ua, eq, cell - 1),
                         get_node_vars(ua, eq, cell),
                         get_node_vars(ua, eq, cell + 1))
        R, L = eigmatrix(eq, ua_)
        fill!(uimh, zero(eltype(uimh)))
        fill!(uiph, zero(eltype(uiph)))
        Mdx2 = tvbM * grid.dx[cell]^2
        if left_bc == neumann && right_bc == neumann && (cell == 1 || cell == nx)
            Mdx2 = 0.0 # Force TVD on boundary for Shu-Osher
        end
        # end # timer
        for ii in 1:nd
            u_ = get_node_vars(u1, eq, ii, cell)
            multiply_add_to_node_vars!(uimh, Vl[ii], u_, eq, 1)
            multiply_add_to_node_vars!(uiph, Vr[ii], u_, eq, 1)
        end
        # Get views of needed cell averages
        # slopes b/w centres and faces

        uimh_ = get_node_vars(uimh, eq, 1)
        uiph_ = get_node_vars(uiph, eq, 1)

        # We will set
        # Δul[n] = ua_[n] - uimh[n]
        # Δur[n] = uiph[n] - ua_[n]
        # Δual[n] = ua_[n] - ual[n]
        # Δuar[n] = uar[n] - ua_[n]

        set_node_vars!(Δul, ua_, eq, 1)
        set_node_vars!(Δur, uiph_, eq, 1)
        set_node_vars!(Δual, ua_, eq, 1)
        set_node_vars!(Δuar, uar, eq, 1)

        subtract_from_node_vars!(Δul, uimh_, eq)
        subtract_from_node_vars!(Δur, ua_, eq)
        subtract_from_node_vars!(Δual, ual, eq)
        subtract_from_node_vars!(Δuar, ua_, eq)

        Δul_ = get_node_vars(Δul, eq, 1)
        Δur_ = get_node_vars(Δur, eq, 1)
        Δual_ = get_node_vars(Δual, eq, 1)
        Δuar_ = get_node_vars(Δuar, eq, 1)
        mul!(char_Δul, L, Δul_)   # char_Δul = L*Δul
        mul!(char_Δur, L, Δur_)   # char_Δur = L*Δur
        mul!(char_Δual, L, Δual_) # char_Δual = L*Δual
        mul!(char_Δuar, L, Δuar_) # char_Δuar = L*Δuar

        char_Δul_ = get_node_vars(char_Δul, eq, 1)
        char_Δur_ = get_node_vars(char_Δur, eq, 1)
        char_Δual_ = get_node_vars(char_Δual, eq, 1)
        char_Δuar_ = get_node_vars(char_Δuar, eq, 1)
        for n in eachvariable(eq)
            dulm[n] = minmod(char_Δul_[n], char_Δual_[n], char_Δuar_[n], Mdx2)
            durm[n] = minmod(char_Δur_[n], char_Δual_[n], char_Δuar_[n], Mdx2)
        end

        # limit if jumps are detected
        dulm_ = get_node_vars(dulm, eq, 1)
        durm_ = get_node_vars(durm, eq, 1)
        jump_l = jump_r = 0.0
        for n in 1:nvar
            jump_l += abs(char_Δul_[n] - dulm_[n])
            jump_r += abs(char_Δur_[n] - durm_[n])
        end
        jump_l /= nvar
        jump_r /= nvar

        if jump_l > 1e-06 || jump_r > 1e-06
            add_to_node_vars!(durm, dulm_, eq, 1) # durm = durm + dulm
            # We want durm = 0.5 * (dul + dur), we adjust 0.5 later
            mul!(du, R, durm)            # du = R * (dulm+durm)
            for ii in Base.OneTo(nd)
                du_ = get_node_vars(du, eq, 1)
                set_node_vars!(u1, ua_ + (xg[ii] - 0.5) * du_, # 2.0 adjusted with 0.5 above
                               eq, ii,
                               cell)
            end
        end
    end
    return nothing
    end # timer
end

function Tenkai.limit_slope(eq::EulerReactive1D, slope, ufl, u_star_ll, ufr, u_star_rr,
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

function Tenkai.zhang_shu_flux_fix(eq::EulerReactive1D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
    p_low, p_high = density(eq, ulow), density(eq, uhigh)
    eps = 0.1 * p_low
    ratio = abs(eps - p_low) / (abs(p_high - p_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Final flux
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

varnames(eq::EulerReactive1D) = eq.varnames
varnames(eq::EulerReactive1D, i::Int) = eq.varnames[i]

function Tenkai.initialize_plot(eq::EulerReactive1D, op, grid, problem, scheme, timer,
                                u1,
                                ua)
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
    p_ua, p_u1 = [plot() for _ in 1:nvar], [plot() for _ in 1:nvar]
    labels = varnames(eq)
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
    l_super = @layout[a{0.01h}; b c d; e] # Selecting layout for p_title being title
    p_ua = plot(p_title, p_ua..., layout = l_super,
                size = (1500, 500)) # Make subplots

    # Set up p_u1 to contain polynomial approximation as a different curve
    # for each cell
    x = LinRange(xf[1], xf[2], nu)
    up1 = zeros(nvar, nd)
    u = zeros(nu)
    for ii in 1:nd
        u_node = get_node_vars(u1, eq, ii, 1)
        up1[:, ii] .= con2prim(eq, u_node)
    end

    for n in 1:nvar
        u = @views Vu * up1[n, :]
        plot!(p_u1[n], x, u, color = :red, legend = false)
        xlabel!(p_u1[n], "x")
        ylabel!(p_u1[n], labels[n])
    end

    for i in 2:nx
        for ii in 1:nd
            u_node = get_node_vars(u1, eq, ii, i)
            up1[:, ii] .= con2prim(eq, u_node)
        end
        x = LinRange(xf[i], xf[i + 1], nu)
        for n in 1:nvar
            u = @views Vu * up1[n, :]
            plot!(p_u1[n], x, u, color = :red, label = nothing, legend = false)
        end
    end

    l = @layout[a{0.01h}; b c d; e f g] # Selecting layout for p_title being title
    p_u1 = plot(p_title, p_u1..., layout = l,
                size = (1700, 500)) # Make subplots

    anim_ua, anim_u1 = Animation(), Animation() # Initialize animation objects
    plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1)
    return plot_data
    end # timer
    end # timer
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::EulerReactive1D,
                            grid,
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
        ua_node = get_node_vars(ua, eq, i)
        up_ .= con2prim(eq, ua_node)
        @printf(avg_file, "%e %e %e %e %e \n", xc[i], up_...)
        # TOTHINK - Check efficiency of printf
        for n in eachvariable(eq)
            p_ua[n + 1][1][:y][i] = @views up_[n]  # Update y-series
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
            u_node = get_node_vars(u1, eq, ii, i)
            up1[:, ii] .= con2prim(eq, u_node)
        end
        @. x = grid.xf[i] + grid.dx[i] * xu
        @views mul!(u, up1, Vu')
        for n in 1:nvar
            p_u1[n + 1][i][:y] = u[n, :]
        end
        for ii in 1:nu
            u_node = get_node_vars(u, eq, ii)
            @printf(sol_file, "%e %e %e %e %e \n", x[ii], u_node...)
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

function exact_solution_data(test_case)
    nothing
end

function post_process_soln(eq::EulerReactive1D, aux, problem, param, scheme)
    @unpack timer, error_file = aux
    @timeit timer "Write solution" begin
    #! format: noindent
    println("Post processing solution")
    @unpack plot_data = aux
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    @unpack animate, saveto = param
    @unpack initial_value = problem

    exact_data = exact_solution_data(initial_value)
    @show exact_data
    if exact_data !== nothing
        for n in eachvariable(eq)
            @views plot!(p_ua[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact", color = :black)
            @views plot!(p_u1[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact", color = :black, legend = true)
            ymin = min(minimum(p_ua[n + 1][1][:y]), minimum(exact_data[:, n + 1]))
            ymax = max(maximum(p_ua[n + 1][1][:y]), maximum(exact_data[:, n + 1]))
            ylims!(p_ua[n + 1], (ymin - 0.1, ymax + 0.1))
            ylims!(p_u1[n + 1], (ymin - 0.1, ymax + 0.1))
        end
    end
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

struct SourceTermReactive{RealT}
    A::RealT
    TA::RealT
end
function (source::SourceTermReactive)(u, x, t, eq::EulerReactive1D)
    @unpack A, TA = source
    rho = u[1]
    R = 287.1
    p = pressure(eq, u)
    T = p / rho
    KT = A * exp(-TA / T)
    z_source = -KT * u[4]
    # if abs(z_source) > 1e-10
    #     @show z_source, KT, A, TA, T, p, rho, u[4]
    #     @show exp(-TA/T), TA, T, p, t
    #     @assert false
    # end
    return SVector(0.0, 0.0, 0.0, z_source)
end


function implicit_source_solve(lhs, eq, x, t, coefficient, source_terms::SourceTermReactive, u_node,
                               implicit_solver = picard_solver)
    # TODO - Make sure that the final source computation is used after the implicit solve
    # function implicit_F(u_new)
    #     # u_new_ = SVector(lhs[1], lhs[2], lhs[3], u_new[4])
    #     z = u_new[4] / u_new[1]
    #     z_ = max(z, 0.0)
    #     z_ = min(z, 1.0)
    #     if !(z ≈ z_)
    #         z_u = u_node[4] / u_node[1]
    #         @show z, z_, z_u
    #     end
    #     u_new_ = SVector(u_new[1], u_new[2], u_new[3], z * u_new[1])
    #     return u_new_ - lhs - coefficient * calc_source(u_new_, x, t, source_terms, eq)
    # end

    # u_new = implicit_solver(implicit_F, u_node)

    # if any(isnan, u_new)
    #     @show u_node, lhs, coefficient, x, t
    # end

    # @assert !any(isnan, u_new) "NaN in implicit source solve"

    # @unpack A, TA = source
    # rho = u[1]
    # p = pressure(eq, u)
    # T = p / rho
    # KT = A * exp(-TA / T)

    # @unpack A, TA = source_terms
    # p = pressure(eq, lhs)
    # T = p / lhs[1]
    # KT = A * exp(-TA / T)
    # u4 = lhs[4] / (1.0 - coefficient * (-KT))
    # # u4 = max(u4, 0.0) # Ensure Z remains positive
    # # u4 = min(u4, 1.0) # Ensure Z remains below one
    # u_new = SVector(lhs[1], lhs[2], lhs[3], u4)

    # # Check whether the u_new satisfies the implicit equation
    # rho = lhs[1]
    # R = 287.1
    # p = pressure(eq, lhs)
    # T = p / rho
    # KT = A * exp(-TA / T)
    # # u4_test = lhs + coefficient * calc_source(u_new, x, t, source_terms, eq)
    # u4_test = lhs + coefficient * (-KT) * u_new[4] * SVector(0.0, 0.0, 0.0, 1.0)
    # if norm(u4_test - u_new) > 1e-8
    #     @show norm(u4_test - u_new)
    #     @show lhs
    #     @show u_new, u4_test
    #     @assert false "Implicit solve failed, norm = $(norm(u4_test - u_new))"
    # end
    # @assert norm(u4_test - u_new) < 1e-8 "Implicit solve failed, norm = $(norm(u4_test - u_new))", u_new, lhs, u4_test

    @unpack A, TA = source_terms
    @unpack gamma = eq
    p = pressure(eq, lhs)
    T = p / lhs[1]
    KT = A * exp(-TA / T)
    u4 = lhs[4] / (1.0 - coefficient * (-KT))
    u4 = max(u4, 0.0) # Ensure Z remains positive
    u4 = min(u4, 1.0) # Ensure Z remains below 1
    u_new = SVector(lhs[1], lhs[2], lhs[3], u4)

    @assert isnan(norm(u_new)) == false "NaN in u_new", pressure(eq, lhs)

    # @show u_new, lhs

    return u_new
end

function compute_face_residual!(eq::EulerReactive1D, grid, op, cache,
                                problem, scheme::Scheme{<:cRKSolver}, param, aux, t, dt,
                                u1, Fb,
                                Ub, ua, res, scaling_factor = 1.0)
    @timeit aux.timer "Face residual" begin
    #! format: noindent
    @unpack xg, wg, bl, br = op
    nd = op.degree + 1
    nx = grid.size
    @unpack dx, xf = grid
    num_flux = scheme.numerical_flux
    @unpack blend = aux
    @unpack u1_b = cache

    # Vertical faces, x flux
    for i in 1:(nx + 1)
        alp = 0.5 * (blend.alpha[i-1] + blend.alpha[i])
        x = xf[i]
        local ul, ur
        if alp ≈ 1.0 # This doesn't matter because it is multiplied by zero later
            # this is just to avoid the positivity error
            # Face between i-1 and i
            ul = get_node_vars(u1, eq, nd, i - 1)  # Right of cell i-1
            ur = get_node_vars(u1, eq, 1, i)       # Left of cell i
        else
            # ul = get_node_vars(u1_b, eq, nd, i - 1)  # Right of cell i-1
            # ur = get_node_vars(u1_b, eq, 1, i)       # Left of cell i
            ul = get_node_vars(ua, eq, i - 1)  # Right of cell i-1
            ur = get_node_vars(ua, eq, i)       # Left of cell i
        end
        @views Fn = num_flux(x, ul, ur, Fb[:, 2, i - 1], Fb[:, 1, i],
                            Ub[:, 2, i - 1], Ub[:, 1, i], eq, 1)

        Fn, blend_fac = blend.blend_face_residual!(i, x, u1, ua, eq, t, dt, grid,
                                                   op, problem,
                                                   scheme, param, Fn, aux, nothing,
                                                   res, scaling_factor)
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

function get_equation(gamma, q0)
    numfluxes = Dict{String, Function}("rusanov" => rusanov)
    initial_values = Dict{String, Function}() # Maybe to be filled someday?
    eq = EulerReactive1D(gamma, q0, 4, "EulerReactive1D",
                         numfluxes, initial_values,
                         ["Density", "Velocity", "Pressure", "Z"])
    return eq
end
end # muladd
end # module
