module EqShallowWater1D
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
                write_soln!, compute_time_step, post_process_soln)

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!)

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Shallow water equations in 1D
struct ShallowWater1D <: AbstractEquations{1, 2}
    gravity::Float64 # Gravity
    nvar::Int64
    name::String
    initial_values::Dict{String, Function}
    numfluxes::Dict{String, Function}
end

#-------------------------------------------------------------------------------
# PDE Information
#-------------------------------------------------------------------------------

@inbounds @inline function flux(x, u, eq::ShallowWater1D)
    @unpack gravity = eq
    h, h_v1 = u
    v1 = h_v1 / h
    # Ignore orientation since it is always "1" in 1D
    f1 = h_v1
    f2 = h_v1 * v1 + 0.5 * gravity * h^2
    return SVector(f1, f2)
end

@inbounds @inline flux(U, eq::ShallowWater1D) = flux(1.0, U, eq)

@inline function waterheight(::ShallowWater1D, u::AbstractArray)
    ρ = u[1]
    return ρ
end

# function converting primitive variables to PDE variables
function prim2con(eq::ShallowWater1D, prim) # primitive, gas constant
    U = SVector(prim[1], prim[1] * prim[2])
    return U
end

# function converting pde variables to primitive variables
@inbounds @inline function con2prim(eq::ShallowWater1D, U)
    primitives = SVector(U[1], U[2] / U[1])
    #                   [h ,   u]
    return primitives
end

function con2prim!(eq::ShallowWater1D, cons, prim)
    prim .= con2prim(eq, cons)
end

function compute_time_step(eq::ShallowWater1D, problem, grid, aux, op, cfl, u1, ua)
    nx = grid.size
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        u = get_node_vars(ua, eq, i)
        h, v = con2prim(eq, u)
        c = sqrt(eq.gravity * h)
        smax = abs(v) + c
        den = max(den, smax / dx[i])
    end
    dt = cfl / den
    return dt, eq
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::ShallowWater1D, dir)
    @unpack gravity = eq
    h_ll, h_v1_ll = ual
    h_rr, h_v1_rr = uar
    v1_ll = h_v1_ll / h_ll
    c_ll = sqrt(gravity * h_ll)
    v1_rr = h_v1_rr / h_rr
    c_rr = sqrt(gravity * h_rr)

    λ = max(abs(v1_ll) + c_ll, abs(v1_rr) + c_rr) # local wave speed
    f1 = 0.5 * (Fl[1] + Fr[1]) - 0.5 * λ * (Ur[1] - Ul[1])
    f2 = 0.5 * (Fl[2] + Fr[2]) - 0.5 * λ * (Ur[2] - Ul[2])
    return SVector(f1, f2)
end

function Tenkai.eigmatrix(eq::ShallowWater1D, u)
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

#-------------------------------------------------------------------------------
# Limiters
#-------------------------------------------------------------------------------
function Tenkai.apply_bound_limiter!(eq::ShallowWater1D, grid, scheme, param, op, ua,
                                     u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end

    @timeit aux.timer "Positivity limiter" begin
    #! format: noindent
    @unpack Vl, Vr = op
    nx = grid.size
    nd = op.degree + 1

    variables = (waterheight,)

    # Looping over tuple of functions like this is only type stable if
    # there are only one/two. For tuples with than two functions, see
    # https://github.com/trixi-framework/Trixi.jl/blob/0fd86e4bd856d894de6a7514edcb9758bf6f8e1e/src/callbacks_stage/positivity_zhang_shu.jl#L39

    # Find a minimum for all variables
    eps = 1e-10
    for variable in variables
        for i in 1:nx
            ua_ = get_node_vars(ua, eq, i)
            var = variable(eq, ua_)
            eps = min(eps, var)
        end
        if eps < 0.0
            println("Fatal: Negative states in cell averages")
            @show variable
            println("       minimum cell average = $eps")
            throw(DomainError(eps, "Positivity limiter failed"))
        end
    end

    for variable in variables
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
            ratio = abs(eps - var_avg) / (abs(var_min - var_avg) + 1e-13)
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
    end # timer
end
function Tenkai.apply_tvb_limiter!(eq::ShallowWater1D, problem, scheme, grid, param, op,
                                   ua,
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

admissibility_tolerance(eq::ShallowWater1D) = 1e-10

function Tenkai.limit_slope(eq::ShallowWater1D, s, ufl, u_s_l, ufr, u_s_r, ue, xl, xr)
    eps = admissibility_tolerance(eq)

    variables = (waterheight,)

    for variable in variables
        var_star_tuple = (variable(eq, u_s_l), variable(eq, u_s_r))
        var_low = variable(eq, ue)

        theta = 1.0
        for var_star in var_star_tuple
            if var_star < eps
                # TOTHINK - Replace eps here by 0.1*var_low
                ratio = abs(0.1 * var_low - var_low) / (abs(var_star - var_low) + 1e-13)
                theta = min(ratio, theta)
            end
        end
        s *= theta
        u_s_l = ue + 2.0 * xl * s
        u_s_r = ue + 2.0 * xr * s
    end

    ufl = ue + xl * s
    ufr = ue + xr * s

    return ufl, ufr
end

function Tenkai.zhang_shu_flux_fix(eq::ShallowWater1D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    uhigh = uprev - c * (Fn - fn_inner) # First candidate for high order update
    h_low, h_high = waterheight(eq, ulow), waterheight(eq, uhigh)
    eps = 0.1 * h_low
    ratio = abs(eps - h_low) / (abs(h_high - h_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        return theta * Fn + (1.0 - theta) * fn # Second candidate for flux
    end
end

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

function Tenkai.initialize_plot(eq::ShallowWater1D, op, grid, problem, scheme, timer,
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
    nvar = eq.nvar
    # Create plot objects to be later collected as subplots

    # Creating a subplot for title
    p_title = plot(title = "Cell averages plot, $nx cells, t = 0.0",
                   grid = false, showaxis = false, bottom_margin = 0Plots.px)
    # Initialize subplots for density, velocity and pressure
    p_ua, p_u1 = [plot() for _ in 1:nvar], [plot() for _ in 1:nvar]
    labels = ["Height", "Velocity"]
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
    l_super = @layout[a{0.01h}; b c] # Selecting layout for p_title being title
    p_ua = plot(p_title, p_ua[1], p_ua[2], layout = l_super,
                size = (1500, 500)) # Make subplots

    # Set up p_u1 to contain polynomial approximation as a different curve
    # for each cell
    x = LinRange(xf[1], xf[2], nu)
    up1 = zeros(nvar, nd)
    u = zeros(nu)
    for ii in 1:nd
        @views con2prim!(eq, u1[:, ii, 1], up1[:, ii]) # store prim form in up1
    end

    for n in 1:nvar
        u = @views Vu * up1[n, :]
        plot!(p_u1[n], x, u, color = :red, legend = false)
        xlabel!(p_u1[n], "x")
        ylabel!(p_u1[n], labels[n])
    end
    for i in 2:nx
        for ii in 1:nd
            @views con2prim!(eq, u1[:, ii, i], up1[:, ii]) # store prim form in up1
        end
        x = LinRange(xf[i], xf[i + 1], nu)
        for n in 1:nvar
            u = @views Vu * up1[n, :]
            plot!(p_u1[n], x, u, color = :red, label = nothing, legend = false)
        end
    end

    l = @layout[a{0.01h}; b c d] # Selecting layout for p_title being title
    p_u1 = plot(p_title, p_u1[1], p_u1[2], layout = l,
                size = (1700, 500)) # Make subplots

    anim_ua, anim_u1 = Animation(), Animation() # Initialize animation objects
    plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1)
    return plot_data
    end # timer
    end # timer
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::ShallowWater1D, grid,
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
    nvar = eq.nvar
    @unpack save_time_interval, save_iter_interval, animate = param
    avg_file = open("$avg_filename.txt", "w")
    up_ = zeros(nvar)
    ylims = [[Inf, -Inf] for _ in 1:nvar] # set ylims for plots of all variables
    for i in 1:nx
        @views con2prim!(eq, ua[:, i], up_) # store primitve form in up_
        @printf(avg_file, "%e %e %e\n", xc[i], up_[1], up_[2])
        # TOTHINK - Check efficiency of printf
        for n in 1:(eq.nvar)
            p_ua[n + 1][1][:y][i] = @views up_[n]    # Update y-series
            ylims[n][1] = min(ylims[n][1], up_[n]) # Compute ymin
            ylims[n][2] = max(ylims[n][2], up_[n]) # Compute ymax
        end
    end
    close(avg_file)
    for n in 1:nvar # set ymin, ymax for ua, u1 plots
        M = maximum(abs, ylims[n])
        ylims!(p_ua[n + 1], (ylims[n][1] - 0.1 * M, ylims[n][2] + 0.1 * M))
        ylims!(p_u1[n + 1], (ylims[n][1] - 0.1 * M, ylims[n][2] + 0.1 * M))
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
            @views con2prim!(eq, u1[:, ii, i], up1[:, ii]) # store prim form in up1
        end
        @. x = grid.xf[i] + grid.dx[i] * xu
        @views mul!(u, up1, Vu')
        for n in 1:nvar
            p_u1[n + 1][i][:y] = u[n, :]
        end
        for ii in 1:nu
            @printf(sol_file, "%e %e %e\n", x[ii], u[1, ii], u[2, ii])
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

function Tenkai.post_process_soln(eq::ShallowWater1D, aux, problem, param, scheme)
    @unpack timer, error_file = aux
    @timeit timer "Write solution" begin
    #! format: noindent
    println("Post processing solution")
    nvar = eq.nvar
    @unpack plot_data = aux
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    @unpack animate, saveto = param
    initial_values = eq.initial_values
    if problem.initial_value in values(initial_values) # Using ready made tests
        initial_value_string, = [a
                                 for (a, b) in initial_values
                                 if
                                 b == problem.initial_value]
        exact_data = exact_solution_data(initial_value_string)

        for n in 1:nvar
            @views plot!(p_ua[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact",
                         color = :black)
            @views plot!(p_u1[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact",
                         color = :black, legend = true)
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

function get_equation(gravity)
    name = "1D shallow water equations"
    numfluxes = Dict("rusanov" => rusanov)
    nvar = 2
    initial_values = Dict()

    return ShallowWater1D(gravity, nvar, name, initial_values, numfluxes)
end
end # muladd
end # module
