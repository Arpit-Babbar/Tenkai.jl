using SSFR
using Trixi: trixi_include

include("reproduce_base.jl") # defines rep_dir, out_dir, etc.
starting_dir = pwd()
cd(rep_dir())
mkpath(data_dir())

try
   limiter_name = :blend
   for blend_order in (:FO, :MH)
      trixi_include("run_blast.jl", limiter = limiter_name, blend_type = blend_order,
                  saveto = joinpath(data_dir(),"blast_$(limiter_name)_$(blend_order)"))
   end
   limiter_name = :tvb
   trixi_include("run_blast.jl", limiter = limiter_name, saveto = joinpath(data_dir(),"blast_$limiter_name"))

   limiter_name = :blend
   test_case = "shuosher"
   for blend_order in (:FO, :MH)
      trixi_include("run_$(test_case).jl", limiter = limiter_name, blend_type = blend_order,
                  saveto = joinpath(data_dir(),"$(test_case)_$(limiter_name)_$(blend_order)"))
   end
   limiter_name = :tvb
   trixi_include("run_$(test_case).jl", limiter = limiter_name, saveto = joinpath(data_dir(),"$(test_case)_$limiter_name"))

   limiter_name = :blend
   test_case = "sedov1d"
   for blend_order in (:FO, :MH)
      trixi_include("run_$(test_case).jl", limiter = limiter_name, blend_type = blend_order,
                  saveto = joinpath(data_dir(),"$(test_case)_$(limiter_name)_$(blend_order)"))
   end

   limiter_name = :blend
   test_case = "leblanc"
   for blend_order in (:FO,)# :MH)
      if blend_order == :FO
         cfl_safety_factor = 0.9
      else
         cfl_safety_factor = 0.98
      end
      trixi_include("run_$(test_case).jl", limiter = limiter_name, blend_type = blend_order,
                  saveto = joinpath(data_dir(),"$(test_case)_$(limiter_name)_$(blend_order)"),
                  cfl_safety_factor = cfl_safety_factor)
   end
catch e
   println("Error, occured. Returning to starting directory before rethrow")
   cd(starting_dir)
   rethrow(e)
end

cd(starting_dir) # Return to the directory you started with
