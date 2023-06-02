test_name = "blast"

include("reproduce_base.jl")
include("plotting.jl")

# Python plots use matplotlib to create png, pdf files which look exactly like in the paper
# Julia plots use PlotlyJS to create html plots which are interactive
# You can either view the html file or use the display command
# display(p_super) shows all variables (density, velocity_pressure)
# display(p_density) shows only density (similar for pressure, velocity)

test_case = "blast"
title = uppercasefirst(test_case)

plot_euler_python(test_case,title, exact_line_width = 3, xlims = (0.55, 0.9) )

p_super, p_density, p_density, p_density = plot_euler_julia(test_case, title)

test_case = "shuosher"
title = "Shu-Osher"

plot_euler_python(test_case, title, exact_line_width = 3, xlims = (0.4, 2.4), ylims = (2.6,4.8) )

p_super, p_density, p_density, p_density = plot_euler_julia(test_case, title)

test_case = "sedov1d"
title = "Sedov's blast test"

plot_euler_python(test_case, title, exact_line_width = 1, show_tvb = false)

p_super, p_density, p_density, p_density = plot_euler_julia(test_case, title, show_tvb = false)

test_case = "double_rarefaction"
title = "Double rarefaction Riemann problem"

plot_euler_python(test_case, title, exact_line_width = 1, show_tvb = false)

p_super, p_density, p_density, p_density = plot_euler_julia(test_case, title, show_tvb = false)
p_super

test_case = "leblanc"
title = "Leblanc's Riemann problem"

density, pressure = plot_euler_python(test_case, title, exact_line_width = 1, show_tvb = false, yscale = "log")
p_super, p_density, p_velocity, p_pressure = plot_euler_julia(test_case, title, show_tvb = false)
