using Plots
plotlyjs()
using DelimitedFiles
using Tenkai.EqLinAdv1D: mult1d
using GZip

p = plot(size = (900, 500));
testcase = "mult1d"
# gassner_soln = readdlm("output_$(testcase)_gassner/sol.txt")
time_smooth = readdlm("output_$(testcase)_09_semisuper_time_smooth/sol.txt")
# no_smooth = readdlm("output_$(testcase)_mlp_nosmooth/sol.txt")
# mlp_soln = readdlm("output_$(testcase)_mlp/sol.txt")
no_time_smooth = readdlm("output_$(testcase)_no_time_smooth/sol.txt")

# exact_file = GZip.open(Tenkai.data_dir * "/$(testcase).dat.gz")
# exact_data = readdlm(exact_file)
@views plot!(p, no_time_smooth[:,1], mult1d.(no_time_smooth[:,1]), label = "Exact", color = "black");
# plot!(p, exact_data[:, 1], exact_data[:, 2], label = "Exact", color = :black)
@views plot!(p, no_time_smooth[:, 1], no_time_smooth[:, 2], label = "No time smoothing", color = "red");

@views plot!(p, time_smooth[:, 1], time_smooth[:, 2], label = "Time smoothing", color = "blue");
# @views plot!(p, no_smooth[:,1], no_smooth[:,2], label = "No smooth", color = "blue");
# savefig("$(testcase)_compare_quadrature.html")

display(p)
