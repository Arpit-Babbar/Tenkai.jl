using Tenkai.EqEuler1D: exact_solution_data
import PyPlot as plt
using Plots
using DelimitedFiles
using Plots
using SimpleUnPack: @unpack
using LinearAlgebra
using Printf
# plotlyjs()
gr()
include("mpl.jl")

include("reproduce_base.jl")

markers_array() = ["s", "o", "^", "*", "D", ">", "p"]
colors_array() = ["green", "royalblue", "red", "m", "c", "y", "k"]
function my_save_fig_python(test_case, figure, name;
                            fig_dir = joinpath(rep_dir(), "figures", test_case))
    mkpath(fig_dir)
    figure.savefig(joinpath(fig_dir, name))
    return nothing
end

function my_save_fig_julia(test_case, figure, name)
    fig_dir = joinpath(rep_dir(), "figures", test_case)
    mkpath(fig_dir)
    savefig(figure, joinpath(fig_dir, name))
    return nothing
end

function plot_euler_python(test_case, title; plt_type = "sol",
                           legends = nothing, exact_line_width = 1,
                           soln_line_width = 1, show_tvb = true, xlims = nothing,
                           ylims = nothing,
                           degree = 4, # For different degrees, user has to manually specify
                           xscale = "linear", yscale = "linear")
    plot_names = [test_case * "_blend_FO", test_case * "_blend_MH"]
    if show_tvb
        plot_names = [test_case * "_tvb", plot_names...]
    end
    labels = ["FO", "MH", "TVB"]
    n_plots = length(plot_names)
    markers = markers_array()
    colors = colors_array()
    if legends !== nothing
        @assert length(legends) == n_plots
    end
    exact_data = exact_solution_data(test_case)
    fig_density, ax_density = plt.subplots()
    if xlims !== nothing
        ax_density.set_xlim(xlims)
    end
    if ylims !== nothing
        ax_density.set_ylim(ylims)
    end
    fig_pressure, ax_pressure = plt.subplots()
    ax_density.set_xlabel("x")
    ax_pressure.set_xlabel("x")
    ax_density.set_ylabel("Density")
    ax_pressure.set_ylabel("Pressure")
    ax_density.grid(true, linestyle = "--")
    ax_pressure.grid(true, linestyle = "--")

    # Set scales
    ax_density.set_xscale(xscale)
    ax_density.set_yscale(yscale)
    ax_pressure.set_xscale(xscale)
    ax_pressure.set_yscale(yscale)
    if exact_data !== nothing
        @views ax_density.plot(exact_data[:, 1], exact_data[:, 2], label = "Reference",
                               c = "k", linewidth = exact_line_width)
        @views ax_pressure.plot(exact_data[:, 1], exact_data[:, 4], label = "Reference",
                                c = "k", linewidth = exact_line_width)
    end
    if plt_type == "avg"
        filename = "avg.txt"
        seriestype = :scatter
    elseif plt_type == "cts_avg"
        filename = "avg.txt"
        seriestype = :line
    else
        @assert plt_type in ("sol", "cts_sol")
        filename = "sol.txt"
        seriestype = :line
    end
    for i in 1:n_plots
        # data = datas[i]
        label = labels[i]
        dir = joinpath(data_dir(), plot_names[i])
        if legends !== nothing
            label = legends[i]
        end
        soln_data = readdlm("$dir/$filename")
        if plt_type == "avg"
            @views ax_density.plot(soln_data[:, 1], soln_data[:, 2], markers[i],
                                   fillstyle = "none",
                                   c = colors[i], label = label)
            @views ax_pressure.plot(soln_data[:, 1], soln_data[:, 4], markers[i],
                                    fillstyle = "none",
                                    c = colors[i], label = label)
        elseif plt_type in ("cts_sol", "cts_avg")
            @views ax_density.plot(soln_data[:, 1], soln_data[:, 2], fillstyle = "none",
                                   c = colors[i], label = label,
                                   linewidth = soln_line_width)
            @views ax_pressure.plot(soln_data[:, 1], soln_data[:, 4], fillstyle = "none",
                                    c = colors[i], label = label,
                                    linewidth = soln_line_width)
        else
            nu = max(2, degree + 1)
            nx = Int(size(soln_data, 1) / nu)
            @views ax_density.plot(soln_data[1:nu, 1], soln_data[1:nu, 2],
                                   fillstyle = "none",
                                   color = colors[i], label = label,
                                   linewidth = soln_line_width)
            @views ax_pressure.plot(soln_data[1:nu, 1], soln_data[1:nu, 4],
                                    fillstyle = "none",
                                    color = colors[i], label = label,
                                    linewidth = soln_line_width)
            for ix in 2:nx
                i1 = (ix - 1) * nu + 1
                i2 = ix * nu
                @views ax_density.plot(soln_data[i1:i2, 1], soln_data[i1:i2, 2],
                                       fillstyle = "none",
                                       c = colors[i], linewidth = soln_line_width)
                @views ax_pressure.plot(soln_data[i1:i2, 1], soln_data[i1:i2, 4],
                                        fillstyle = "none",
                                        c = colors[i], linewidth = soln_line_width)
            end
        end
    end
    ax_density.legend()
    ax_pressure.legend()
    ax_density.set_title(title)
    ax_pressure.set_title(title)

    my_save_fig_python(test_case, fig_density, "density.pdf")
    my_save_fig_python(test_case, fig_density, "density.png")
    my_save_fig_python(test_case, fig_pressure, "pressure.pdf")
    my_save_fig_python(test_case, fig_pressure, "pressure.png")

    return (fig_density, ax_density), (fig_pressure, ax_pressure)
end

colors = [:green, :blue, :red, :purple]
function quick_plot!(p, u_x, u_y, seriestype, label; color_index, legend,
                     markershape = :none,
                     markersize = 2.3, markerstrokestyle = :dot, linewidth = 1.0)
    plot!(p, u_x, u_y,
          seriestype = seriestype,
          markerstrokestyle = markerstrokestyle,
          markerscale = 0.01, label = label,
          legend = legend, markersize = markersize,
          color = colors[color_index],
          linewidth = linewidth,
          markerstrokealpha = 0,
          thickness_scaling = 1.0,
          markershape = markershape)
end
markershapes = [:diamond, :square, :circle]

function plot_euler_julia(test_case, title; plt_type = "cts_sol", legends = nothing,
                          show_tvb = true,
                          degree = 4)
    plot_names = [test_case * "_blend_FO", test_case * "_blend_MH"]
    if show_tvb
        plot_names = [test_case * "_tvb", plot_names...]
    end
    limiter_labels = ["FO", "MH", "TVB"]
    n_plots = length(plot_names)
    p_title = plot(title = title,
                   grid = false, showaxis = false, bottom_margin = 0Plots.px)
    nvar = 3
    if legends !== nothing
        @assert length(legends) == n_plots
    end
    labels = ["Density", "Velocity", "Pressure"]
    p = [plot(xlabel = "x", ylabel = labels[n], legend = true) for n in 1:nvar]
    exact_data = exact_solution_data(test_case)
    if exact_data !== nothing
        for n in 1:nvar
            @views plot!(p[n], exact_data[:, 1], exact_data[:, n + 1], label = "Exact",
                         color = :black, markeralpha = 0, linewidth = 3.0)
        end
    end
    if plt_type == "avg"
        filename = "avg.txt"
        seriestype = :scatter
    elseif plt_type == "cts_avg"
        filename = "avg.txt"
        seriestype = :line
    else
        @assert plt_type in ("sol", "cts_sol")
        filename = "sol.txt"
        seriestype = :line
    end
    for i in 1:n_plots
        dir = joinpath(data_dir(), plot_names[i])
        label = limiter_labels[i]
        soln_data = readdlm("$dir/$filename")
        if plt_type == "avg"
            for n in 1:nvar
                @views quick_plot!(p[n], soln_data[:, 1], soln_data[:, n + 1], seriestype,
                                   label, color_index = i, legend = true,
                                   markerstrokestyle = :none,
                                   markershape = markershapes[i], markersize = 2.5,
                                   linewidth = 2)
            end
        elseif plt_type in ("cts_sol", "cts_avg")
            for n in 1:nvar
                @views quick_plot!(p[n], soln_data[:, 1], soln_data[:, n + 1], seriestype,
                                   label, color_index = i, legend = true,
                                   markerstrokestyle = :none,
                                   markershape = :none, markersize = 0.0, linewidth = 2)
            end
        else
            nu = max(2, degree + 1)
            nx = Int(size(soln_data, 1) / nu)
            for n in 1:nvar
                @views quick_plot!(p[n], soln_data[1:nu, 1], soln_data[1:nu, n + 1],
                                   seriestype,
                                   markersize = 0.0,
                                   label, markershape = :none, color_index = i,
                                   legend = true)
                for ix in 2:nx
                    i1 = (ix - 1) * nu + 1
                    i2 = ix * nu
                    @views quick_plot!(p[n], soln_data[i1:i2, 1], soln_data[i1:i2, n + 1],
                                       seriestype,
                                       markersize = 0.0,
                                       label, markershape = :none, color_index = i,
                                       legend = false)
                end
                plot!(p[n], legend = true)
            end
        end
    end
    l = @layout[a{0.01h}; b; c; d] # Selecting layout for p_title being title
    p_super = plot(p_title, p[1], p[2], p[3], layout = l,
                   size = (1020, 1200)) # Make subplots

    # Remove repetetive labels
    if exact_data !== nothing # HACKY FIX FOR TEST CASES WITHOUT EXACT SOLUTION
        p_super[3][1][:label] = p_super[2][1][:label] = ""
        for i in 1:n_plots
            p_super[2][i + 1][:label] = ""
            p_super[3][i + 1][:label] = ""
        end
    end

    # fig_dir = joinpath(rep_dir(), "figures", test_case)
    # mkpath(fig_dir)
    # savefig(p_super, joinpath(fig_dir, test_case*".html"))

    my_save_fig_julia(test_case, p_super, test_case * ".html")

    return p_super, p[1], p[2], p[3]
end

function plot2d(sol, testname, filename)
    # @assert Plots.backend() == Plots.GRBackend() # Contour plots take too long with PlotlyJS backend
    grid = sol["grid"]
    op = sol["op"]
    u1 = sol["u"]

    degree = op.degree

    @unpack xf, yf, dx, dy = grid
    nx, ny = grid.size
    @unpack degree, xg = op
    nd = degree + 1

    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    Mx, My = nx * nu, ny * nu
    grid_x = zeros(Mx)
    grid_y = zeros(My)
    for i in 1:nx
        i_min = (i - 1) * nu + 1
        i_max = i_min + nu - 1
        # grid_x[i_min:i_max] .= LinRange(xf[i], xf[i+1], nu)
        grid_x[i_min:i_max] .= xf[i] .+ dx[i] * xg
    end

    for j in 1:ny
        j_min = (j - 1) * nu + 1
        j_max = j_min + nu - 1
        # grid_y[j_min:j_max] .= LinRange(yf[j], yf[j+1], nu)
        grid_y[j_min:j_max] .= yf[j] .+ dy[j] * xg
    end

    u_density = zeros(Mx, My)
    u = zeros(nu)
    for j in 1:ny
        for i in 1:nx
            # KLUDGE - Don't do this, use all values in the cell
            # to get values in the equispaced thing
            for jy in 1:nd
                i_min = (i - 1) * nu + 1
                i_max = i_min + nu - 1
                # u_ = @view u1[1, :, jy, i, j]
                # mul!(u, Vu, u_)
                j_index = (j - 1) * nu + jy
                u_density[i_min:i_max, j_index] .= @view u1[1, :, jy, i, j]
            end
        end
    end

    p = contour(grid_x, grid_y, u_density', color = :black, levels = 10, cbar = false)
    p_fill = contourf(grid_x, grid_y, u_density')

    fig_dir = joinpath(rep_dir(), testname)
    mkpath(fig_dir)

    savefig(p, joinpath(fig_dir, filename * ".png"))
    savefig(p, joinpath(fig_dir, filename * ".pdf"))

    savefig(p_fill, joinpath(fig_dir, filename * "_colour.png"))
    savefig(p_fill, joinpath(fig_dir, filename * "_colour.pdf"))

    return plot(p)
end

function format_with_powers(y, _)
    if y > 1e-4
        y = Int64(y)
        return "\$$y\$"
    else
        return @sprintf "%.4E" y
    end
end

function set_ticks!(ax, log_sub, ticks_formatter; dim = 2)
    # Remove scientific notation and set xticks
    # https://stackoverflow.com/a/49306588/3904031

    function anonymous_formatter(y, _)
        if y > 1e-4
            # y_ = parse(Int64, y)
            y_ = Int64(y)
            if dim == 2
                return "\$$y_^2\$"
            else
                return "\$$y_\$"
            end
        else
            return @sprintf "%.4E" y
        end
    end

    formatter = plt.matplotlib.ticker.FuncFormatter(ticks_formatter)
    # (y, _) -> format"{:.4g}".format(int(y)) ) # format"{:.4g}".format(int(y)))
    # https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-plt.matplotlib
    x_major = plt.matplotlib.ticker.LogLocator(base = 2.0, subs = (log_sub,),
                                               numticks = 20) # ticklabels at 2^i*log_sub
    x_minor = plt.matplotlib.ticker.LogLocator(base = 2.0,
                                               subs = LinRange(1.0, 9.0, 9) * 0.1,
                                               numticks = 10)
    #  Used to manipulate tick labels. See help(plt.matplotlib.ticker.LogLocator) for details)
    ax.xaxis.set_major_formatter(anonymous_formatter)
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_locator(x_major)
    ax.xaxis.set_minor_locator(x_minor)
    ax.tick_params(axis = "both", which = "major")
    ax.tick_params(axis = "both", which = "minor")
end

function add_theo_factors!(ax, ncells, error, degree, i,
                           theo_factor_even, theo_factor_odd)
    if degree isa Int64
        d = degree
    else
        d = parse(Int64, degree)
    end
    min_y = minimum(error[1:(end - 1)])
    @show error, min_y
    xaxis = ncells[(end - 1):end]
    slope = d + 1
    if iseven(slope)
        theo_factor = theo_factor_even
    else
        theo_factor = theo_factor_odd
    end
    y0 = theo_factor * min_y
    y = (1.0 ./ xaxis) .^ slope * y0 * xaxis[1]^slope
    markers = ["s", "o", "*", "^"]
    # if i == 1
    ax.loglog(xaxis, y, label = "\$ O(M^{-$(d + 1)})\$", linestyle = "--",
              marker = markers[i], c = "grey",
              fillstyle = "none")
    # else
    # ax.loglog(xaxis,y, linestyle = "--", c = "grey")
    # end
end

function error_label(error_norm)
    if error_norm in ["l2", "L2"]
        return "\$L^2\$ error"
    elseif error_norm in ["l1", "L1"]
        return "\$L^1\$ error"
    end
    @assert false
end

function plot_python_ncells_vs_y(; legend = nothing,
                                 bflux = "",
                                 degrees = ["3", "4"], show = "yes",
                                 theo_factor_even = 0.8, theo_factor_odd = 0.8,
                                 title = nothing, log_sub = "3.75",
                                 saveto_dir = nothing,
                                 dir = joinpath(data_dir(), "isentropic"),
                                 error_norm = "l2",
                                 limiter = "blend",
                                 ticks_formatter = format_with_powers,
                                 figsize = (6, 7))
    # @assert error_type in ["l2","L2"] "Only L2 error for now"
    fig_error, ax_error = plt.subplots(figsize = figsize)
    colors = ["orange", "royalblue", "green", "m", "c", "y", "k"]
    markers = ["s", "o", "*"]
    for (i, degree) in enumerate(degrees)
        d = parse(Int64, degree)
        error = readdlm(joinpath(dir, "$(error_norm)_$(degree)_$(limiter).txt"))
        marker = markers[i]
        ax_error.loglog(error[:, 1], error[:, 2], marker = marker, c = colors[d], mec = "k",
                        fillstyle = "none", label = "\$ N = $d \$")
        add_theo_factors!(ax_error, error[:, 1], error[:,2], degree, i,
                          theo_factor_even, theo_factor_odd)
    end
    ax_error.set_xlabel("Number of elements \$ (M^2) \$")
    ax_error.set_ylabel(error_label(error_norm))

    set_ticks!(ax_error, log_sub, ticks_formatter)

    ax_error.grid(true, linestyle = "--")

    if title !== nothing
        ax_error.set_title(title)
    end
    ax_error.legend()

    if saveto_dir !== nothing
        mkpath(saveto_dir)
        fig_error.savefig("$saveto_dir/isentropic_convergence_$(limiter).pdf")
        fig_error.savefig("$saveto_dir/isentropic_convergence_$(limiter).png")
    end

    return fig_error
end

function plot_python_ndofs_vs_y(; legend = nothing,
                                bflux = "",
                                degrees = ["3", "4"], show = "yes",
                                theo_factor_even = 0.8, theo_factor_odd = 0.8,
                                title = nothing, log_sub = "3.75",
                                saveto_dir = nothing,
                                dir = joinpath(data_dir(), "isentropic"),
                                error_norm = "l2",
                                limiter = "blend",
                                ticks_formatter = format_with_powers,
                                figsize = (6, 7))
    # @assert error_type in ["l2","L2"] "Only L2 error for now"
    fig_error, ax_error = plt.subplots(figsize = figsize)
    colors = ["orange", "royalblue", "green", "m", "c", "y", "k"]
    markers = ["s", "o", "*"]
    for (i, degree) in enumerate(degrees)
        d = parse(Int64, degree)
        error = readdlm(joinpath(dir, "$(error_norm)_$(degree)_$(limiter).txt"))
        marker = markers[i]
        ax_error.loglog(error[:, 1] * (d + 1), error[:, 2], marker = marker, c = colors[d],
                        mec = "k",
                        fillstyle = "none", label = "\$ N = $d \$")
        add_theo_factors!(ax_error, (d + 1) * error[:, 1], error[:,2], degree, i,
                          theo_factor_even, theo_factor_odd)
    end
    ax_error.set_xlabel("Degrees of freedom \$ (M^2) \$")
    ax_error.set_ylabel(error_label(error_norm))

    set_ticks!(ax_error, log_sub, ticks_formatter)

    ax_error.grid(true, linestyle = "--")

    if title !== nothing
        ax_error.set_title(title)
    end
    ax_error.legend()

    if saveto_dir !== nothing
        mkpath(saveto_dir)
        fig_error.savefig("$saveto_dir/isentropic_convergence_ndofs_$(limiter).pdf")
        fig_error.savefig("$saveto_dir/isentropic_convergence_ndofs_$(limiter).png")
    end

    return fig_error
end

fig_size_() = (6.4, 4.8) # Default size actually
# plot_python_ncells_vs_y(degrees = ["3", "4"], saveto_dir = ".", log_sub = 2.5,
#                         error_norm = "l2", limiter = "blend", figsize = fig_size_())
# plot_python_ncells_vs_y(degrees = ["3", "4"], saveto_dir = ".", log_sub = 2.5,
#                         error_norm = "l2", limiter = "no_limiter", figsize = fig_size_())

# plot_python_ndofs_vs_y(degrees = ["3", "4"], saveto_dir = ".", log_sub = 2.5,
#                        error_norm = "l2", limiter = "blend", figsize = fig_size_())
# plot_python_ndofs_vs_y(degrees = ["3", "4"], saveto_dir = ".", log_sub = 2.5,
#                        error_norm = "l2", limiter = "no_limiter", figsize = fig_size_())
