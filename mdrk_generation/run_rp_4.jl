using Tenkai
using Tenkai: base_dir
using TrixiBase: trixi_include

filename = "$(@__DIR__)/run_files/run_rp2d_pan2017_4.jl"
trixi_include(filename)
