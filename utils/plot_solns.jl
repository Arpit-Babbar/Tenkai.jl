using Plots
plotlyjs()
using DelimitedFiles
using Tenkai.EqLinAdv1D: mult1d
using GZip

p = plot(size = (900, 500));
testcase = "blast"
# gassner_soln = readdlm("output_$(testcase)_gassner/sol.txt")
mdrk = readdlm("output_$(testcase)_mdrk/sol.txt")
# no_smooth = readdlm("output_$(testcase)_mlp_nosmooth/sol.txt")
# mlp_soln = readdlm("output_$(testcase)_mlp/sol.txt")
lwfr = readdlm("output_$(testcase)_lwfr/sol.txt")

exact_file = GZip.open(Tenkai.data_dir * "/$(testcase).dat.gz")
exact_data = readdlm(exact_file)
# @views plot!(p, mlp_soln[:,1], mult1d.(mlp_soln[:,1]), label = "Exact", color = "black");
plot!(p, exact_data[:, 1], exact_data[:, 2], label = "Exact", color = :black)
@views plot!(p, lwfr[:, 1], lwfr[:, 2], label = "LWFR", color = "red");

@views plot!(p, mdrk[:, 1], mdrk[:, 2], label = "MDRK", color = "blue");
# @views plot!(p, no_smooth[:,1], no_smooth[:,2], label = "No smooth", color = "blue");
# savefig("$(testcase)_compare_quadrature.html")

display(p)
