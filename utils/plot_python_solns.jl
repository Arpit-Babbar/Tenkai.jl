using Tenkai
using Tenkai: repro_dir
include("$repro_dir/plotting.jl")
using DelimitedFiles
using Tenkai.EqLinAdv1D: mult1d
using GZip
using Tenkai.EqEuler1D: exact_solution_data
import PyPlot as plt
using Plots
using DelimitedFiles
using Plots
using SimpleUnPack: @unpack
using LinearAlgebra
using Printf

markers_array_new() = ["s", "o", "^", "*", "D", ">", "p"]
colors_array_new() = ["red", "royalblue", "green", "m", "c", "y", "k"]
function plot_solns(files::Vector{String}, labels::Vector{String};
                    test_case = nothing, title = "",
                    xlims = nothing, ylims = nothing,
                    xscale = "linear", yscale = "linear",
                    exact_line_width = 1, soln_line_width = 1,
                    degree = 3,
                    plt_type = "sol", outdir = nothing)
    @assert length(files) == length(labels)
    markers = markers_array_new()
    colors = colors_array_new()
    fig_density, ax_density = plt.subplots()
    exact_data = exact_solution_data(test_case)
    fig_pressure, ax_pressure = plt.subplots()
    if xlims !== nothing
        ax_density.set_xlim(xlims)
        ax_pressure.set_xlim(xlims)
    end
    if ylims !== nothing
        ax_density.set_ylim(ylims)
    end
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

    n_plots = length(files)
    for i in 1:n_plots
        # data = datas[i]
        label = labels[i]
        soln_data = readdlm(files[i])
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

    if outdir === nothing
        outdir = joinpath(Tenkai.base_dir, "figures", test_case)
    end
    my_save_fig_python(test_case, fig_density, "density.pdf", fig_dir = outdir)
    my_save_fig_python(test_case, fig_density, "density.png", fig_dir = outdir)
    my_save_fig_python(test_case, fig_pressure, "pressure.pdf", fig_dir = outdir)
    my_save_fig_python(test_case, fig_pressure, "pressure.png", fig_dir = outdir)
    plt.close()
end

function plot_solns_scalar(files::Vector{String}, labels::Vector{String};
                           test_case = nothing, title = "",
                           xlims = nothing, ylims = nothing,
                           exact_data = exact_solution_data(test_case),
                           xscale = "linear", yscale = "linear",
                           exact_line_width = 1, soln_line_width = 1,
                           degree = 3,
                           plt_type = "sol", outdir = nothing,
                           exact_label = "Reference")
    @assert length(files) == length(labels)
    markers = markers_array_new()
    colors = colors_array_new()
    fig, ax = plt.subplots()
    if xlims !== nothing
        ax.set_xlim(xlims)
    end
    if ylims !== nothing
        ax.set_ylim(ylims)
    end
    ax.set_xlabel("x")
    ax.set_ylabel("\$ u \$")
    ax.grid(true, linestyle = "--")

    # Set scales
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if exact_data !== nothing
        @views ax.plot(exact_data[:, 1], exact_data[:, 2], label = exact_label,
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

    n_plots = length(files)
    for i in 1:n_plots
        # data = datas[i]
        label = labels[i]
        soln_data = readdlm(files[i])
        if plt_type == "avg"
            @views ax.plot(soln_data[:, 1], soln_data[:, 2], markers[i],
                           fillstyle = "none",
                           c = colors[i], label = label)
        elseif plt_type in ("cts_sol", "cts_avg")
            @views ax.plot(soln_data[:, 1], soln_data[:, 2], fillstyle = "none",
                           c = colors[i], label = label, linewidth = soln_line_width)
        else
            nu = max(2, degree + 1)
            nx = Int(size(soln_data, 1) / nu)
            @views ax.plot(soln_data[1:nu, 1], soln_data[1:nu, 2],
                           fillstyle = "none",
                           color = colors[i], label = label,
                           linewidth = soln_line_width)
            for ix in 2:nx
                i1 = (ix - 1) * nu + 1
                i2 = ix * nu
                @views ax.plot(soln_data[i1:i2, 1], soln_data[i1:i2, 2],
                               fillstyle = "none",
                               c = colors[i], linewidth = soln_line_width)
            end
        end
    end
    ax.legend()
    ax.set_title(title)

    if outdir === nothing
        outdir = joinpath(Tenkai.base_dir, "figures", test_case)
    end
    my_save_fig_python(test_case, fig, "soln.pdf", fig_dir = outdir)
    my_save_fig_python(test_case, fig, "soln.png", fig_dir = outdir)
    plt.close()
    return fig
end

using Tenkai
Eq = Tenkai.EqBurg1D
exact_solution = Eq.exact_solution_burger_sin
test_case = "burg1d"
function files_labels(t)
    (["mdrk_results/output_burg1d_lwfr_t$t/sol.txt",
         "mdrk_results/output_burg1d_lwfr_t$t/sol.txt"],
     ["LWFR", "MDRK"])
end
function my_exact_data(t)
    data = deepcopy(readdlm(files_labels(t)[1][1]))
    exact(x) = exact_solution(x, t)
    @views data[:, 2] .= exact.(data[:, 1])
    return data
end

t = 2.0
function plot_burg_soln(t, exact_data = my_exact_data(t), exact_label = "Exact")
    files, labels = files_labels(t)
    outdir = joinpath(Tenkai.base_dir, "figures", "burg1d_t$t")
    plot_solns_scalar(files, labels;
                      test_case = "burg1d", title = "",
                      exact_data = exact_data,
                      xlims = nothing, ylims = nothing,
                      xscale = "linear", yscale = "linear",
                      exact_line_width = 1, soln_line_width = 1,
                      degree = 3,
                      plt_type = "sol", outdir = outdir, exact_label = exact_label)
end

# plot_burg_soln(2.0)
# plot_burg_soln(4.5)
# exact_data_8 = readdlm("mdrk_results/output_burg1d_reference/avg.txt")
# plot_burg_soln(8.0, exact_data_8, "Reference")
