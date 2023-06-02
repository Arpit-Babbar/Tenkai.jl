rcParams = plt.PyDict(plt.matplotlib["rcParams"]) # Otherwise, the dictionary will be copied
# as is default in PyClaw. See
# https://github.com/JuliaPy/PyPlot.jl/issues/417#issuecomment-452382111
rcParams["font.size"] = 14
rcParams["font.family"] = "serif"
rcParams["figure.autolayout"] = true
rcParams["lines.linewidth"] = 2
rcParams["lines.markersize"] = 6
rcParams["axes.titlesize"] = 14
rcParams["axes.labelsize"] = 14
ticksize = "medium"
rcParams["xtick.labelsize"] = ticksize
rcParams["ytick.labelsize"] = ticksize
# rcParams["tex.usetex"] = true # This will use LaTeX fonts (slow)