using Tenkai
using Tenkai: base_dir
using Trixi: trixi_include

for p0 in ["01", "015", "025", "05", "1"]
   filename = "$(@__DIR__)/run_files/run_rp2d_pan2017_2p0_$(p0).jl"
   trixi_include(filename)
end