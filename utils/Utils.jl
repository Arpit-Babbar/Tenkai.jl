module Utils

using Plots
using UnPack
using DelimitedFiles
using Measures
using NaturalSort
using PlutoUI
using JSON3

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
