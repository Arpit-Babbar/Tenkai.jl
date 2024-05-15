using Tenkai
using Tenkai: base_dir
using Trixi: trixi_include

for p0 in ["02","03","05", "075", "1"]
   filename = "$(@__DIR__)/run_files/run_rp2d_pan2017_1p0_$(p0).jl"
   trixi_include(filename)
end
