using Tenkai
using Tenkai: base_dir
using Trixi: trixi_include

for p0 in ["1","03", "05", "075"]
   filename = "$(@__DIR__)/run_files/run_rp2d_pan2017_2p0_$(p0).jl"
   trixi_include(filename)
end