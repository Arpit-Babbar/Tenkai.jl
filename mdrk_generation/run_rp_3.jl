using Tenkai
using Tenkai: base_dir
using Trixi: trixi_include

for p0 in ["a", "b"]
   filename = "$(@__DIR__)/run_files/run_rp2d_pan2017_3p0_$(p0).jl"
   trixi_include(filename)
end