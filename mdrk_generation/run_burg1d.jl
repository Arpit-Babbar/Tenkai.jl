using Tenkai
using Tenkai: base_dir
using TrixiBase: trixi_include
using Tenkai: fo_blend
Eq = Tenkai.EqBurg1D
equation = Eq.get_equation()

mdrk_data_dir = joinpath(base_dir, "mdrk_results")
function burg_trixi_include(solver, t)
    trixi_include("$(@__DIR__)/run_files/run_burg1d.jl", solver = solver,
                  saveto = joinpath(mdrk_data_dir, "output_burg1d_$(solver)_t$t"),
                  degree = 3, final_time = t, limiter = :blend_limiter)
end

for final_time in (2.0, 4.5, 8.0)
    for solver in ("mdrk", "lwfr", "rkfr")
        burg_trixi_include(solver, final_time)
    end
end

# Generate reference solution
trixi_include("$(@__DIR__)/run_files/run_burg1d.jl", solver = "lwfr",
              saveto = joinpath(mdrk_data_dir, "output_burg1d_reference"),
              nx = 3500,
              degree = 3, final_time = 8.0, limiter = :blend_limiter, pure_fv = true,
              blend_type = fo_blend(equation))
