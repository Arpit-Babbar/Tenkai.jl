using Tenkai
using Tenkai: utils_dir

include("$utils_dir/plot_python_solns.jl")

f_mdrk(testcase) = "mdrk_results/output_$(testcase)_mdrk/sol.txt"
f_lwfr(testcase) = "mdrk_results/output_$(testcase)_lwfr/sol.txt"
plot_test(test) = plot_solns([f_lwfr(test), f_mdrk(test)], ["LWFR", "MDRK"], test_case = test)

test = "blast"
plot_solns([f_lwfr(test), f_mdrk(test)], ["LWFR", "MDRK"], test_case = test, xlims = (0.55,0.9))

test = "shuosher"
plot_test(test)
outdir = joinpath(Tenkai.base_dir, "figures", test*"_zoomed")
plot_solns([f_lwfr(test), f_mdrk(test)], ["LWFR", "MDRK"], test_case = test, xlims = (0.4, 2.4),
            ylims = (2.6, 4.8), outdir = outdir)

plot_test("sedov1d")
plot_test("double_rarefaction")
test = "leblanc"
plot_solns([f_lwfr(test), f_mdrk(test)], ["LWFR", "MDRK"], test_case = test, xlims = (-5.0,10.0),
            yscale = "log")

