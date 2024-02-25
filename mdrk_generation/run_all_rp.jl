using Tenkai
using Tenkai: base_dir
using Trixi: trixi_include

trixi_include("$(@__DIR__)/run_rp_1.jl")
trixi_include("$(@__DIR__)/run_rp_2.jl")
# trixi_include("$(@__DIR__)/run_rp_3.jl")
# trixi_include("$(@__DIR__)/run_rp_4.jl")
