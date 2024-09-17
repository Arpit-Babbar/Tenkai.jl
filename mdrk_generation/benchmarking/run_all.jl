using TrixiBase: trixi_include

using TimerOutputs

using DelimitedFiles

mkpath("Results")

### Rayleigh Taylor

trixi_include(joinpath(@__DIR__, "run_rayleight_taylor.jl"), nx = 5, solver = "rkfr", blend_type = :muscl,
              cfl = 0.98 * 0.215, degree = 3, final_time = 0.1)
sol = trixi_include(joinpath(@__DIR__, "run_rayleight_taylor.jl"), nx = 64, solver = "rkfr", blend_type = :muscl,
                    cfl = 0.98 * 0.215, degree = 3, final_time = 2.5)
rk_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/rayleigh_taylor_rk.txt", [rk_time])

trixi_include(joinpath(@__DIR__, "run_rayleight_taylor.jl"), nx = 5, solver = "mdrk", blend_type = :mh,
                       cfl = 0.0, degree = 3, final_time = 0.1)
# Run again after compilation!
sol = trixi_include(joinpath(@__DIR__, "run_rayleight_taylor.jl"), nx = 64, solver = "mdrk", blend_type = :mh,
cfl = 0.0, degree = 3, final_time = 2.5)
lw_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/rayleigh_taylor_lw.txt", [lw_time])

### Double Mach Reflection

trixi_include(joinpath(@__DIR__, "run_double_mach_reflection.jl"), ny = 5 , solver = "rkfr",blend_type = :muscl,
                       cfl = 0.98 * 0.215, degree = 3, final_time = 0.02)
# Run again after compilation!
sol = trixi_include(joinpath(@__DIR__, "run_double_mach_reflection.jl"), ny = 150 , solver = "rkfr",blend_type = :muscl,
                    cfl = 0.98 * 0.215, degree = 3, final_time = 0.2)
dmr_rk_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/dmr_rk.txt", [dmr_rk_time])

trixi_include(joinpath(@__DIR__, "run_double_mach_reflection.jl"), ny = 5, solver = "mdrk", blend_type = :mh,
                       cfl = 0.0, degree = 3, final_time = 0.02)
# Run again after compilation!
sol = trixi_include(joinpath(@__DIR__, "run_double_mach_reflection.jl"), ny = 150, solver = "mdrk", blend_type = :mh,
                    cfl = 0.0, degree = 3, final_time = 0.2)
dmr_lw_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/dmr_lw.txt", [dmr_lw_time])

### Rotational critical

trixi_include(joinpath(@__DIR__, "run_hurricane_v0critical.jl"), ny = 5, nx = 5, solver = "rkfr",blend_type = :muscl,
                       cfl = 0.98 * 0.215, degree = 3, final_time = 0.0045)
# Run again after compilation!
sol = trixi_include(joinpath(@__DIR__, "run_hurricane_v0critical.jl"), ny = 200, nx = 200, solver = "rkfr",blend_type = :muscl,
                    cfl = 0.98 * 0.215, degree = 3, final_time = 0.045)
rotational_rk_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/hurricane_rk.txt", [rotational_rk_time])

trixi_include(joinpath(@__DIR__, "run_hurricane_v0critical.jl"), ny = 5, nx = 5, solver = "mdrk", blend_type = :mh,
                       cfl = 0.0, degree = 3, final_time = 0.0045)
# Run again after compilation!
sol = trixi_include(joinpath(@__DIR__, "run_hurricane_v0critical.jl"), ny = 200, nx = 200, solver = "mdrk", blend_type = :mh,
                    cfl = 0.0, degree = 3, final_time = 0.045)
rotational_lw_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/hurricane_lw.txt", [rotational_lw_time])

### RP2D

trixi_include(joinpath(@__DIR__, "run_rp2d_12.jl"), ny = 5, nx = 5, solver = "rkfr",blend_type = :muscl,
                       cfl = 0.98 * 0.215, degree = 3, final_time = 0.0045)
# Run again after compilation!
sol = trixi_include(joinpath(@__DIR__, "run_rp2d_12.jl"), ny = 512, nx = 512, solver = "rkfr",blend_type = :muscl,
                    cfl = 0.98 * 0.215, degree = 3, final_time = 0.25)
rp2d_rk_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/rp2_rk.txt", [rp2d_rk_time])

trixi_include(joinpath(@__DIR__, "run_rp2d_12.jl"), ny = 5, nx = 5, solver = "mdrk", blend_type = :mh,
                       cfl = 0.0, degree = 3, final_time = 0.0045)
# Run again after compilation!
sol = trixi_include(joinpath(@__DIR__, "run_rp2d_12.jl"), ny = 512, nx = 512, solver = "mdrk", blend_type = :mh,
                    cfl = 0.0, degree = 3, final_time = 0.25)
rp2d_lw_time = TimerOutputs.tottime(sol["aux"].timer) * 1e-9

writedlm("Results/rp2d_lw.txt", [rp2d_lw_time])
