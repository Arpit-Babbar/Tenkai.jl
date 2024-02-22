using Tenkai
using Tenkai: base_dir
using Trixi: trixi_include

mdrk_data_dir = joinpath(base_dir, "mdrk_results")
# Keywords technique from
# https://discourse.julialang.org/t/how-to-use-kwargs-to-avoid-passing-around-keyword-arguments/84856
function my_trixi_include(test, solver, blend_type = :MH; kwargs...)
    trixi_include(
    "$(@__DIR__)/run_files/run_$test.jl", solver = solver,
    blend_type = blend_type,
    saveto = joinpath(mdrk_data_dir, "output_$(test)_$(solver)_$(blend_type)"),
    degree = 3;
    kwargs...)
end

# for test in ("blast", "double_rarefaction", "leblanc", "sedov1d", "shuosher")
#     for solver in ("mdrk", "lwfr")
#         my_trixi_include(test, solver)
#     end
# end

# New results

for test in (
             "titarev_toro",
             )
    for blend_type in (:MH, :FO)
        if blend_type == :FO
            cfl_safety_factor = 0.5
        else
            cfl_safety_factor = 0.95
        end
        my_trixi_include(test, "mdrk", blend_type, cfl_safety_factor = cfl_safety_factor)
    end
end

for test in (
             "larger_density",
             )
    for blend_type in (:MH, :FO)
        if blend_type == :FO
            cfl_safety_factor = 0.5
        else
            cfl_safety_factor = 0.95
        end
        my_trixi_include(test, "mdrk", blend_type, cfl_safety_factor = cfl_safety_factor)
    end
end
