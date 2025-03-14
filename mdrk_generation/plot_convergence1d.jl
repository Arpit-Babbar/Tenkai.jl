using Tenkai
using Tenkai: utils_dir

include("$utils_dir/plot_python_solns.jl")

# TODO - Move this function to utils
function plot_python_ndofs_vs_y(files::Vector{String}, labels::Vector{String};
                                saveto, degree = 3,
                                theo_factor_even = 0.8, theo_factor_odd = 0.8,
                                title = nothing, log_sub = "2.5",
                                error_norm = "l2",
                                ticks_formatter = format_with_powers,
                                figsize = (6.4, 4.8))
    # @assert error_type in ["l2","L2"] "Only L2 error for now"
    fig_error, ax_error = plt.subplots(figsize = figsize)
    colors = ["orange", "royalblue", "green", "m", "c", "y", "k"]
    markers = ["D", "o", "*", "^"]
    @assert length(files) == length(labels)
    n_plots = length(files)

    for i in 1:n_plots
        data = readdlm(files[i])
        marker = markers[i]
        ax_error.loglog(data[:, 1] * (degree + 1), data[:, 2], marker = marker,
                        c = colors[i],
                        mec = "k", fillstyle = "none", label = labels[i])
    end

    data = readdlm(files[1])
    marker = markers[1]
    add_theo_factors!(ax_error, (degree + 1) * data[:, 1], data[:, 2], degree, 1,
                      theo_factor_even, theo_factor_odd)
    ax_error.set_xlabel("Degrees of freedom")
    ax_error.set_ylabel(error_label(error_norm))

    set_ticks!(ax_error, log_sub, ticks_formatter; dim = 1)

    ax_error.grid(true, linestyle = "--")

    if title !== nothing
        ax_error.set_title(title)
    end
    ax_error.legend()

    fig_error.savefig("$saveto.pdf")
    fig_error.savefig("$saveto.png")

    return fig_error
end

mdrk_data_dir = joinpath(base_dir, "mdrk_results")
figures_dir = joinpath(base_dir, "figures")

burg_figures_dir = joinpath(figures_dir, "burg1d", "convergence")
burg_dir = joinpath(mdrk_data_dir, "burg1d")
mkpath(burg_figures_dir)
function files_burg_bflux(corr, points)
    [ # Comparing bfluxes
     joinpath(burg_dir, "rkfr3_EA_D2_$(corr)_$(points).txt"),
     joinpath(burg_dir, "mdrk3_EA_D2_$(corr)_$(points).txt"),
     # joinpath(burg_dir, "rkfr3_AE_D2_$(corr)_$(points).txt"),
     joinpath(burg_dir, "mdrk3_AE_D2_$(corr)_$(points).txt")]
end

function files_burg_diss(corr, points)
    [ # Comparing dissipation
     joinpath(burg_dir, "rkfr3_EA_D2_$(corr)_$(points).txt"),
     joinpath(burg_dir, "mdrk3_EA_D2_$(corr)_$(points).txt"),
     # joinpath(burg_dir, "rkfr3_EA_D1_$(corr)_$(points).txt"),
     joinpath(burg_dir, "mdrk3_EA_D1_$(corr)_$(points).txt")]
end
files_gl_bflux_compared = files_burg_bflux("radau", "gl")
files_gll_bflux_compared = files_burg_bflux("g2", "gll")
files_gl_diss_compared = files_burg_diss("radau", "gl")
files_gll_diss_compared = files_burg_diss("g2", "gll")
labels_bflux_compared = ["RK", "MDRK-EA",
    #  "RK",
    "MDRK-AE"]
labels_diss_compared = ["RK", "MDRK-D2",
    # "RK-D1",
    "MDRK-D1"]
plot_python_ndofs_vs_y(files_gl_bflux_compared, labels_bflux_compared, title = "GL, Radau",
                       saveto = joinpath(burg_figures_dir, "bflux_compared_gl"))
plot_python_ndofs_vs_y(files_gll_bflux_compared, labels_bflux_compared,
                       title = "GLL, \$ g_2 \$",
                       saveto = joinpath(burg_figures_dir, "bflux_compared_gll"))
plot_python_ndofs_vs_y(files_gl_diss_compared, labels_diss_compared, title = "GL, Radau",
                       saveto = joinpath(burg_figures_dir, "diss_compared_gl"))
plot_python_ndofs_vs_y(files_gll_diss_compared, labels_diss_compared,
                       title = "GLL, \$ g_2 \$",
                       saveto = joinpath(burg_figures_dir, "diss_compared_gll"))

linadv_dir = joinpath(mdrk_data_dir, "linadv1d")
linadv1d_figures_dir = joinpath(figures_dir, "linadv1d", "convergence")
mkpath(linadv1d_figures_dir)
function files_linadv_diss(corr, points)
    [joinpath(linadv_dir, "rkfr3_EA_D2_$(corr)_$(points).txt"),
     joinpath(linadv_dir, "mdrk3_AE_D2_$(corr)_$(points).txt"),
     #    joinpath(linadv_dir, "lwfr3_AE_D1_$(corr)_$(points).txt"),
     joinpath(linadv_dir, "mdrk3_AE_D1_$(corr)_$(points).txt")]
end
files_gl = files_linadv_diss("radau", "gl")
files_gll = files_linadv_diss("g2", "gll")
files_gl_gll = vcat(files_gl, files_gll)

labels = ["RK", "MDRK-D2",
    #   "LW-D1",
    "MDRK-D1"]
plot_python_ndofs_vs_y(files_gl, labels, title = "GL, Radau",
                       saveto = joinpath(linadv1d_figures_dir, "lw_mdrk_gl"),
                       theo_factor_even = 0.6)

plot_python_ndofs_vs_y(files_gll, labels, title = "GLL, \$g_2\$",
                       saveto = joinpath(linadv1d_figures_dir, "lw_mdrk_gll"),
                       theo_factor_even = 0.6)

or2_dir = joinpath(mdrk_data_dir, "or2")
or2_figures_dir = joinpath(figures_dir, "or2", "convergence")
mkpath(or2_figures_dir)
function files_varadv_diss(bflux)
    [joinpath(or2_dir, "rkfr3_EA_D2_radau_gl.txt"),
     joinpath(or2_dir, "mdrk3_$(bflux)_D2_radau_gl.txt"),
     # joinpath(or2_dir, "lwfr3_$(bflux)_D1_radau_gl.txt"),
     joinpath(or2_dir, "mdrk3_$(bflux)_D1_radau_gl.txt")
     ]
end
files_AE = files_varadv_diss("AE")

labels_AE = ["RK", "MDRK-D2-AE", "MDRK-D1-AE"]

files_EA = files_varadv_diss("EA")
labels_EA = ["RK", "MDRK-D2-EA", "MDRK-D1-EA"]
plot_python_ndofs_vs_y(files_AE, labels_AE, title = "GL, Radau",
                       saveto = joinpath(or2_figures_dir, "lw_mdrk_AE"),
                       theo_factor_even = 0.6)

plot_python_ndofs_vs_y(files_EA, labels_EA, title = "GL, Radau",
                       saveto = joinpath(or2_figures_dir, "lw_mdrk_EA"),
                       theo_factor_even = 0.6)

files = [joinpath(or2_dir, "mdrk3_EA_D2_radau_gl.txt"),
    joinpath(or2_dir, "mdrk3_AE_D2_radau_gl.txt")]
labels = ["EA", "AE"]
plot_python_ndofs_vs_y(files, labels, title = "MDRK, GL, Radau, D2",
                       saveto = joinpath(or2_figures_dir, "mdrk_bflux_compared"),
                       theo_factor_even = 0.6)
