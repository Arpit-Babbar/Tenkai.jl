module Utils

using Plots
using UnPack
using DelimitedFiles
using Measures
using NaturalSort
import ..SSFR.EqLinAdv1D as EqLinAdv1D
import ..SSFR.EqEuler1D as EqEuler1D
import ..SSFR.EqBurg1D as EqBurg1D
import ..SSFR: data_dir
using PlutoUI
using JSON3

colors = [:red, :blue, :green, :purple]
quick_plot!(p, u_x, u_y,
            seriestype, label;
            color_index, legend) =  plot!(p, u_x, u_y,
                                          seriestype = seriestype,
                                          markerstrokestyle = :dot,
                                          markerscale = 0.01, label = label,
                                          legend = legend, markersize = 1.5,
                                          color = colors[color_index],
                                          markerstrokealpha = 0,
                                          thickness_scaling = 1.0)
tvb_keys = ["tvbM"]
blend_keys = ["indicator_model", "blend_type", "predictive_blend",
              "positivity_fix", "pure_fv", "debug_blend", "beta_muscl",
              "indicating_variables", "reconstruction_variables",
              "content_tolerance_type","face_fix"]
limiter_keys = vcat(tvb_keys, blend_keys, ["limiter_name"])

function convert_named_tuple_to_dict(named_tuple)
   Dict(string.(keys(named_tuple)) .=> values(named_tuple))
end

# setup Empty limiter info
function setup_limiter_none(;bound_limit)
   limiter_info_nt = (; limiter_name="none", bound_limit = bound_limit)
   limiter_info = convert_named_tuple_to_dict(limiter_info_nt)
   return limiter_info
end

# Setup TVB limiter info
function setup_limiter_tvb(; limiter_name, tvbM, bound_limit)
   @assert limiter_name == "tvb"
   limiter_info_nt = (; limiter_name, tvbM, bound_limit)
   limiter_info = convert_named_tuple_to_dict(limiter_info_nt)
   return limiter_info
end

function setup_limiter_krivodonova(; limiter_name, bound_limit, alpha="1.0")
   @assert limiter_name == "krivodonova"
   limiter_info_nt = (; limiter_name, bound_limit, alpha)
   limiter_info = convert_named_tuple_to_dict(limiter_info_nt)
   return limiter_info
end

# Setup TVB limiter info
function setup_limiter_blend(; limiter_name, indicator_model, blend_type,
                               predictive_blend, indicating_variables,
                               reconstruction_variables,
                               positivity_fix, pure_fv, debug_blend,
                               beta_muscl, bound_limit, content_tolerance_type,
                               face_fix)
   @assert limiter_name == "blend"
   limiter_info_nt = (; limiter_name, indicator_model, blend_type,
                        predictive_blend, positivity_fix, pure_fv,
                        indicating_variables, reconstruction_variables,
                        debug_blend, beta_muscl, bound_limit,
                        content_tolerance_type, face_fix)
   limiter_info = convert_named_tuple_to_dict(limiter_info_nt)
   return limiter_info
end

function setup_run(; base_dir, solver, equation, initial_value,
                     degree, grid_size, solution_points,
                     correction_function, numerical_flux,
                     bflux, cfl_safety_factor, limiter_info,
                     dimension, stamp_threads,
                     cfl_style = "optimal",
                     dissipation = "2",
                     final_time = nothing, size_type = "ncell",
                     threads = "")
   @assert cfl_style in ["optimal", "lw"]
   @assert dissipation in ["1","2"]
   @assert equation in ["la","burg","bucklev","euler","euler2d"]
   if equation == "euler"
      @assert initial_value in ["sod","lax","toro5","blast","shuosher"]
   elseif equation == "la"
      @assert initial_value in keys(EqLinAdv1D.initial_values_la)
   end
   @assert size_type in ["ncell", "ndofs"]

   args = []

   run_dict = Dict(
                     "la" => "const_linadv1d",
                     "burg" => "burg1d",
                     "bucklev" => "bucklev",
                     "sod" => "sod",
                     "lax" => "lax1d",
                     "toro5" => "toro5",
                     "blast" => "blast",
                     "shuosher" => "shuosher",
                     "isentropic" => "isentropic",
                     "isentropic_dumbser" => "isentropic_dumbser",
                     "double_mach_reflection" => "double_mach_reflection"
                  )
   if equation == "euler"
      run_file = run_dict[initial_value]
   elseif equation == "euler2d"
      run_file = run_dict[initial_value]
   elseif equation == "la"
      @assert final_time !== nothing
      if initial_value in keys(EqLinAdv1D.initial_values_la)
         run_file = "const_linadv1d"
      else
         @assert false "Incorrect initial value"
      end
   elseif equation in ["burg1d", "burg"]
      if initial_value == "hat"
         run_file = "burg1d_hat"
      end
   else
      run_file = run_dict[equation]
   end
   dir = "$base_dir/$equation/$initial_value/"
   empty!(args)

   # TODO - A better design would be to write everything into a dictionary
   # and then call a separate function which will create the filename

   @unpack bound_limit, limiter_name = limiter_info
   if limiter_name == "none"
      limiter = ""
      append!(
              args,
              ["--limiter", "none",
              "--bound_limit", bound_limit]
             )
   elseif limiter_name == "tvb"
      @assert "tvbM" in keys(limiter_info)
      @unpack bound_limit, tvbM = limiter_info
      limiter = "_tvb$(tvbM)_bl$(bound_limit)"
      append!(args,
              ["--limiter", "tvb",
               "--tvbM", "$tvbM"])
   elseif limiter_name == "krivodonova"
      @assert "alpha" in keys(limiter_info)
      @unpack bound_limit, alpha = limiter_info
      limiter = "_alpha$(alpha)_bl$(bound_limit)"
      append!(args,
              ["--limiter", "krivodonova"
            #   ,"--alpha", "$alpha"]
              ])
   elseif limiter_name == "blend"
      @assert blend_keys ⊆ keys(limiter_info)

      # TODO - Replace the whole @unpack to append process with a single macro
      @unpack (indicator_model, blend_type, predictive_blend, positivity_fix,
               indicating_variables, reconstruction_variables, content_tolerance_type,
               bound_limit, pure_fv, debug_blend, beta_muscl, face_fix) = limiter_info
      append!(args,
              [
                "--limiter", "blend",
                "--indicator_model", indicator_model,
                "--blend_type",  blend_type,
                "--predictive_blend", "$predictive_blend",
                "--positivity_fix", "$positivity_fix",
                "--indicating_variables", indicating_variables,
                "--reconstruction_variables", reconstruction_variables,
                "--content_tolerance_type", content_tolerance_type,
                "--animate", "false",
                "--pure_fv", "$pure_fv",
                "--debug_blend", "$debug_blend",
                "--beta_muscl", "$beta_muscl",
		          "--face_fix", "$face_fix"]
             )
      limiter = ("_$(indicator_model)_$(blend_type)_pb$(predictive_blend)"
                 *"_rv$(reconstruction_variables)_iv$(indicating_variables)"
                 *"_iv$(content_tolerance_type)"
                 *"_pf$(positivity_fix)_pfv$(pure_fv)"
                 *"_db$(debug_blend)"
                 *"_$(beta_muscl)_bl$(bound_limit)_fffalse") # TODO - Add face_fix here!!
   end

   if size_type == "ncell"
      if dimension == 1
         grid_size_string = grid_size
      elseif dimension == 2
         grid_size_string = "$(grid_size[1])X$(grid_size[2])"
      else
         @assert false "Dimension not implemented"
      end
   elseif size_type == "ndofs"
      nd = parse(Int64, degree) + 1
      if dimension == 1
         grid_size_string = "$(grid_size)ndofs"
         ndofs = parse(Int64, grid_size)
         grid_size = ceil(Int64, ndofs/nd)
         grid_size = "$grid_size"
      elseif dimension == 2
         grid_size_string = "$(grid_size[1])X$(grid_size[2])ndofs"
         ndofs_x = parse(Int64, grid_size[1])
         ndofs_y = parse(Int64, grid_size[2])
         grid_size_x = ceil(Int64, ndofs_x/nd)
         grid_size_y = ceil(Int64, ndofs_y/nd)
         grid_size[1] = "$grid_size_x"
         grid_size[2] = "$grid_size_y"
      else
         @assert false "Dimension not implemented"
      end
   end

   append!(args,
            [
               "--initial_value", initial_value,
               "--degree", degree,
               "--solver", solver,
               "--solution_points", solution_points,
               "--correction_function", correction_function,
               "--bflux", bflux,
               "--cfl_safety_factor", cfl_safety_factor,
               "--bound_limit", bound_limit,
               "--numerical_flux", numerical_flux,
               "--cfl_style", cfl_style,
               "--dissipation", dissipation
            ]
           )
   if dimension == 1
      append!(args, ["--grid_size", "$grid_size"])
   elseif dimension == 2
      append!(args, ["--grid_size", "$(grid_size[1])", "$(grid_size[2])"])
   else
      @assert false "Dimension not implemented"
   end
   if bound_limit == true
      bound_limit = "_bound_limited"
   else
      bound_limit = ""
   end
   dir *= ("$(solver)_$(equation)_$(degree)_$(grid_size_string)_$(solution_points)"
            *"_$(bflux)_$(cfl_safety_factor)"
            *"_$(correction_function)_$(numerical_flux)$(bound_limit)")
   if cfl_style == "lw" && solver == "rkfr"
      dir *= "cfl_style_$cfl_style"
   end

   if dissipation == "1" && (cfl_style == "lw" || solver == "lwfr")
      dir *= "_diss$dissipation"
   end

   if final_time !== nothing
      append!(args, ["--final_time","$final_time"])
      dir *= "t$final_time"
   end
   dir *= "$limiter"

   if stamp_threads == true
      nthreads = Threads.nthreads()
      dir *= "_threads$nthreads"
   end

   if threads != ""
      dir *= "_threads$threads"
   end

   # Put all data in a named tuple and then in a dictionary
   data_nt = (; dir, equation, initial_value, solver, degree, grid_size,
                size_type, solution_points, correction_function,
                numerical_flux, bflux, cfl_safety_factor, final_time,
                args, bound_limit, limiter_info )
   data = convert_named_tuple_to_dict(data_nt)
   data["run_file"] = run_file

   if final_time !== nothing
      data["final_time"] = final_time
   end

   append!(args, ["--saveto", dir])
   return data
end

function local_ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
		Expr(:toplevel,
				:(eval(x) = $(Expr(:core, :eval))($name, x)),
				:(include(x) = $(Expr(:top, :include))($name, x)),
				:(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
				:(include($path))))
	m
end

function pretty_legend(s)
   if s == "tvb"
      return "TVB"
   elseif s == "hll"
      return "HLL"
   elseif s == "hllc"
      return "HLLC"
   elseif s == "muscl"
      return "MH"
   elseif s == "fo"
      return "FO"
   elseif s == "lwfr"
      return "LWFR"
   elseif s == "rkfr"
      return "RKFR"
   elseif s == "gl"
      return "GL"
   elseif s == "gll"
      return "GLL"
   elseif typeof(s) == Float64
      return Int(s)
   elseif typeof(s) == String
      return titlecase(s)
   else
      return s
   end
end

function get_legend(data, info2show)
   legend = ""
   @unpack limiter_info = data
   for info in info2show
      # Remove this append!
      append!(limiter_keys, ["content_tolerance_type"])
      append!(limiter_keys, ["face_fix"])
      @assert info ∈ keys(data) || info ∈ limiter_keys "Invalid legend requiremenet"
   end
   for key in keys(data)
      if key in info2show
         if legend != ""
            legend *= "-"
         end
         legend *= pretty_legend(data[key])
      end
   end
   @unpack limiter_info = data

   for key in keys(limiter_info)
      if key in info2show
         if legend != ""
            legend *= "-"
         end
         legend *= pretty_legend(limiter_info[key])
      end
   end
   return legend
end

function get_error(data, error_type)
   @assert error_type in ["l1", "l2", "L1", "L2"]
   error_file = readdlm(data["dir"]*"/error.txt")
   if error_type in ["l1", "L1"]
      return error_file[end,2]
   else
      return error_file[end,3]
   end
end

function get_total_time(data)
   dict_file    = open(data["dir"]*"/timer.json")
   dict         = JSON3.read(dict_file)
   total_time   = dict[:total_time_ns]
   output_time  = dict[:inner_timers]["Write solution"][:time_ns]
   compute_time = (total_time - output_time) * 1e-9
   return compute_time
end

function plot_euler( datas, title; plt_type = "avg", info2show = "", legends = nothing)
   p_title = plot(title = title,
                  grid = false, showaxis = false, bottom_margin = 0Plots.px)
   nvar = 3
   n_plots = length(datas)
   if legends !== nothing
      @assert length(legends) == n_plots
   end
   labels = ["Density", "Velocity", "Pressure"]
   p = [ plot(xlabel = "x", ylabel = labels[n], legend = true) for n=1:nvar ]
   data_ = datas[1]
   @unpack initial_value = data_
   exact_data = EqEuler1D.exact_solution_data(initial_value)
   for n=1:nvar
      @views plot!(p[n], exact_data[:,1], exact_data[:,n+1], label="Exact",
                   color = :black, markeralpha = 0)
   end
   if plt_type == "avg"
      filename = "avg.txt"
      seriestype = :scatter
   else
      @assert plt_type in ("sol", "cts_sol")
      filename = "sol.txt"
      seriestype = :line
   end
   for i=1:n_plots
      data = datas[i]
      @unpack dir, degree  = data
      if info2show != ""
         label = get_legend(data, info2show)
      end
      if legends !== nothing
         label = legends[i]
      end
      soln_data = readdlm("$dir/$filename")
      if isfile("$(data["dir"])/broken.txt") == true # File is broken
         @assert false "Test case $(data["dir"]) is broken"
      end
      if plt_type in ("avg", "cts_sol")
         for n=1:nvar
            @views quick_plot!(p[n], soln_data[:,1], soln_data[:,n+1], seriestype,
                               label, color_index = i, legend = true)
         end
      else
         degree = parse(Int64, degree)
         nu = max(2, degree+1)
         nx = Int(size(soln_data,1)/nu)
         for n=1:nvar
            @views quick_plot!(p[n], soln_data[1:nu,1], soln_data[1:nu,n+1], seriestype,
                               label, color_index = i, legend = true)
            for ix=2:nx
               i1 = (ix-1)*nu+1
               i2 = ix*nu
               @views quick_plot!(p[n], soln_data[i1:i2,1], soln_data[i1:i2,n+1], seriestype,
                                 label, color_index = i, legend = false)
            end
            plot!(p[n], legend=true)
         end
      end
   end
   l = @layout[ a{0.01h}; b ; c ; d] # Selecting layout for p_title being title
   p = plot(p_title, p[1], p[2], p[3], layout = l,
            size = (1020,1200)) # Make subplots

   # Remove repetetive labels
   p[3][1][:label] = p[2][1][:label] = ""
   for i=1:n_plots
      p[2][i+1][:label] = ""
      p[3][i+1][:label] = ""
   end
   return p
end

function animate_alpha(variable, data)
   variable_dict = Dict("density" => 1, "1" => 1, 1 => 1,
                        "velocity" => 2, "2" => 2, 2 => 2,
                        "pressure" => 3, "3" => 3, 3 => 3)
   n = variable_dict[variable] + 1
   title_dict = Dict(2 => "Density",
                     3 => "Velocity",
                     4 => "Pressure")
   @unpack dir = data
   alpha_s = filter(startswith("$dir/alpha0"), sort(readdir(dir, join=true)))
   avg_s = filter(x -> ( startswith("$dir/sol0")(x) && endswith(".txt")(x) ),
                  sort(readdir(dir, join=true)))

   @assert length(alpha_s) == length(avg_s[2:end]) "$(length(alpha_s)),$(length(avg_s[2:end]))"
   time_levels = length(alpha_s)
   anim = Animation()
   alpha_io = open(alpha_s[1], "r")
   time = parse(Float64, readline(alpha_io) ); time = round(time, digits = 4)
   alpha = readdlm(alpha_io); close(alpha_io)
   avg = readdlm(avg_s[1])
   min_avg, max_avg = (minimum(avg[:,n]), maximum(avg[:,n]))
   @views ylims = (min_avg - 0.1, max_avg + 0.1)
   @views @. alpha[:,2] = min_avg + alpha[:,2] * (max_avg-min_avg)
   p = plot(xlabel = "x", ylabel="$(title_dict[n])", legend = true,
            ylims = ylims, title = "$(title_dict[n]) evolution, t=$time")
   @views quick_plot!(p, avg[:,1], avg[:, n], :line, "u", color_index=1, legend=true)
   @views plot!(p, alpha[:,1], alpha[:,2], label = "alpha",
                seriestype = :scatter,
                markershape = :rect, color = :black, markersize = 0.8)
   frame(anim, p)
   for i=2:time_levels
      avg = readdlm(avg_s[i])
      alpha_io = open(alpha_s[i], "r")
      time = parse(Float64, readline(alpha_io) ); time = round(time, digits = 4)
      alpha = readdlm(alpha_io); close(alpha_io)
      @views min_avg, max_avg = (minimum(avg[:,n]), maximum(avg[:,n]))
      @views @. alpha[:,2] = min_avg + alpha[:,2] * (max_avg-min_avg)
      @views ylims = (min_avg - 0.1, max_avg + 0.1)
      @views p[1][1][:y] .= avg[:,n]
      @views p[1][2][:y] .= alpha[:,2]

      ylims!(p, ylims)
      title!(p,  "$(title_dict[n]) evolution, t=$time")
      frame(anim, p)
   end
   return p, anim
end

function plot_space_time_alpha(data; figure_size = (1000,800))
   dir = data["dir"]
   # return readdir(dir)
   @views space = readdlm("$dir/avg.txt")[:,1]
   time = readdlm("$dir/time_levels.txt")[:,1]
   alpha = readdlm("$dir/space_time_alpha.txt")
   # @views p = contour(space_time[:,1],space_time[:,2], alpha, fill=true, grid=true)
   p = plot(xlabel = "x", ylabel = "time", title = "Space time alpha diagram",
             size = figure_size)
   @views p = heatmap!(p, space, time, alpha, fill=true, grid=true,
                      c = cgrad(:hot, rev=true))
                     #  c = :Reds_9)
                     #   c = cgrad(:gist_rainbow, rev=true))
   return p
end

function exact_scalar(data_, final_time)
   @unpack dir, equation, initial_value = data_
   soln_data = readdlm("$dir/sol.txt")
   x = @view soln_data[:,1]
   if equation == "la"
      iv_dict = EqLinAdv1D.initial_values_la
      @assert initial_value ∈ keys(iv_dict)
      ic, exact_solution = iv_dict[initial_value]
      soln_data[:,2] = exact_solution.(x, final_time)
   elseif equation in ["burg", "burg1d"]
      iv_dict = EqBurg1D.initial_values_burg
      @assert initial_value ∈ keys(iv_dict)
      ic, exact_solution = iv_dict[initial_value]
      soln_data[:,2] = exact_solution.(x, final_time)
   else
      @assert false "Exact solution not implemented"
   end
   return soln_data
end

function plot_scalar( datas, title; plt_type = "avg",
                      info2show = "", final_time, fig_size = (1000,800))
   n_curves = length(datas)
   p = plot(xlabel = "x", ylabel="u", legend = true, size=fig_size)

   # Get exact solution data
   data_ = datas[1]
   exact_data = exact_scalar(data_, final_time)

   @views plot!(p, exact_data[:,1], exact_data[:,2], label="Exact",
                color = :black)
   if plt_type == "avg"
      filename = "avg.txt"
      seriestype = :scatter
   else
      @assert plt_type == "sol"
      filename = "sol.txt"
      seriestype = :line
   end
   for i=1:n_curves
      data = datas[i]
      @unpack dir = data
      if info2show != ""
         label = get_legend(data, info2show)
      end
      soln_data = readdlm("$dir/$filename")
      if plt_type == "avg"
         @views quick_plot!(p, soln_data[:,1], soln_data[:,2], seriestype,
                            label, color_index=i, legend=true)
      else
         @unpack degree = data
         degree = parse(Int64, degree)
         nu = max(2, degree+1)
         nx = Int(size(soln_data,1)/nu)
         @views quick_plot!(p, soln_data[1:nu,1], soln_data[1:nu,2], seriestype,
                            label, color_index=i, legend=true)
         for ix=2:nx
            i1 = (ix-1)*nu+1
            i2 = ix*nu
            @views quick_plot!(p, soln_data[i1:i2,1], soln_data[i1:i2,2], seriestype,
                               nothing, color_index=i, legend=false)
         end
      end
   end
   plot!(p, legend = true, title = title)

   return p
end

function join_plots(p_ua1, p_ua2, label1, label2, title, nvar)
   # p_ua1, p_ua2 are two solution plots, each of which have the approximate
   # cell average  solution and exact solution. We will keep the two
   # approximate solutions in one figure while keeping the exact solution
   for n=1:nvar
      # Add exact solution to p_ua1 as a new figure
      plot!(p_ua1[n+1], p_ua2[n+1][1][:x], p_ua2[n+1][1][:y], linestyle = :dot,
            color = :red, seriestype = :scatter,
            markershape = :circle, markersize = 2, markerstrokealpha = 0)

      # Set legends to none, we'd only show them once throughout subplot
      p_ua1[n+1][1][:label] = ""
      p_ua1[n+1][2][:label] = ""
      p_ua1[n+1][3][:label] = ""
   end

   # Set legends
   title!(p_ua1[1], title)
   p_ua1[2][1][:label] = label1
   p_ua1[2][2][:label] = "Reference"
   p_ua1[2][3][:label] = label2
   return deepcopy(p_ua1)
end

function join_plots(p_ua1, p_ua2, p_ua3, label1, label2, label3, title, nvar)
   for n=1:nvar
      # Add exact solution to p_ua1 as a new figure
      plot!(p_ua1[n+1], p_ua2[n+1][1][:x], p_ua2[n+1][1][:y], linestyle = :dot,
            color = :red, seriestype = :scatter,
            markershape = :circle, markersize = 2, markerstrokealpha = 0)
      plot!(p_ua1[n+1], p_ua3[n+1][1][:x], p_ua3[n+1][1][:y], linestyle = :dot,
            color = :red, seriestype = :scatter,
            markershape = :circle, markersize = 2, markerstrokealpha = 0)
      # Set legends to none, we'd only show them once throughout subplot
      title!(p_ua1[1], title)
      p_ua1[n+1][1][:label] = ""
      p_ua1[n+1][2][:label] = ""
      p_ua1[n+1][3][:label] = ""
      p_ua1[n+1][4][:label] = ""
   end

   # Set legends
   p_ua1[2][1][:label] = label1
   p_ua1[2][2][:label] = "Reference"
   p_ua1[2][3][:label] = label2
   p_ua1[2][4][:label] = label3
   return deepcopy(p_ua1)
end

function join_plots_scalar(p_ua1, p_ua2, label1, label2, title)
   # p_ua1, p_ua2 are two solution plots, each of which have the approximate
   # cell average  solution and exact solution. We will keep the two
   # approximate solutions in one figure while keeping the exact solution
      # Add exact solution to p_ua1 as a new figure
   plot!(p_ua1[1], p_ua2[1][1][:x], p_ua2[1][1][:y], linestyle = :dot,
         color = :red, seriestype = :scatter,
         markershape = :circle, markersize = 2, markerstrokealpha = 0)

   # Set legends to none, we'd only show them once throughout subplot
   p_ua1[1][1][:label] = ""
   p_ua1[1][2][:label] = ""
   p_ua1[1][3][:label] = ""

   # Set legends
   p_ua1[1][1][:label] = label1
   p_ua1[1][2][:label] = "Reference"
   p_ua1[1][3][:label] = label2
   title!(p_ua1, title)
   return deepcopy(p_ua1)
end

end # module
