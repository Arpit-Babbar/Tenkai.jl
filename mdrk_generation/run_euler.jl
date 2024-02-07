using Tenkai
using Tenkai: base_dir
using Trixi: trixi_include

mdrk_data_dir = joinpath(base_dir, "mdrk_results")
my_trixi_include(test, solver) = trixi_include(
    "$(@__DIR__)/run_files/run_$test.jl", solver = solver,
    saveto = joinpath(mdrk_data_dir, "output_$(test)_$(solver)"),
    degree = 3)

for test in ("blast", "double_rarefaction", "leblanc", "sedov1d", "shuosher")
    for solver in ("mdrk", "lwfr")
        my_trixi_include(test, solver)
    end
end


