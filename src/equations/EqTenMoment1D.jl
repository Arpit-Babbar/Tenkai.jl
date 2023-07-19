module EqTenMoment1D

using TimerOutputs
using StaticArrays
using LinearAlgebra
using UnPack
using Plots
using Printf
using JSON3

using Tenkai
using Tenkai.Basis

( # Methods to be extended
import Tenkai: flux, prim2con, con2prim, limit_slope, zhang_shu_flux_fix,
               apply_bound_limiter!, initialize_plot,
               write_soln!, compute_time_step, post_process_soln
)

using Tenkai: eachvariable, PlotData, get_filename

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# The conservative variables are
# ρ, ρv1, ρv2, E11, E12, E22

struct TenMoment1D <: AbstractEquations{1,6}
   varnames::Vector{String}
   name::String
end

# The primitive variables are
# ρ, v1, v2, P11, P12, P22.
# Tensor P is defined so that tensor E = 0.5 * (P + ρ*v⊗v)
function con2prim(eq::TenMoment1D, u)
   ρ   = u[1]
   v1  = u[2] / ρ
   v2  = u[3] / ρ
   P11 = 2.0 * u[4] - ρ * v1 * v1
   P12 = 2.0 * u[5] - ρ * v1 * v2
   P22 = 2.0 * u[6] - ρ * v2 * v2
   return SVector(ρ, v1, v2, P11, P12, P22)
end

function prim2con(eq::TenMoment1D, prim)
   ρ, v1, v2, P11, P12, P22 = prim

   ρv1 = ρ*v1
   ρv2 = ρ*v2
   E11 = 0.5 * (P11 + ρ*v1*v1)
   E12 = 0.5 * (P12 + ρ*v1*v2)
   E22 = 0.5 * (P22 + ρ*v2*v2)

   return SVector(ρ, ρv1, ρv2, E11, E12, E22)
end

# The flux is given by
# (ρ v1, P11 + ρ v1^2, P12 + ρ v1*v2, (E + P) ⊗ v)
@inbounds @inline function flux(x, u, eq::TenMoment1D)
   r, v1, v2, P11, P12, P22 = con2prim(eq, u)

   f1 = r * v1
   f2 = P11 + r * v1 * v1
   f3 = P12 + r * v1 * v2
   f4 = (u[4] + P11) * v1
   f5 = u[5] * v1 + 0.5 * (P11 * v2 + P12 * v1)
   f6 = u[6] * v1 + P12 * v2

   return SVector(f1, f2, f3, f4, f5, f6)
end

# Compute flux directly from the primitive variables
@inbounds @inline function prim2flux(x, prim, eq::TenMoment1D)
   r, v1, v2, P11, P12, P22 = prim

   E11 = 0.5 * (P11 + r * v1 * v1)
   E12 = 0.5 * (P12 + r * v1 * v2)
   E22 = 0.5 * (P22 + r * v2 * v2)

   f1 = r * v1
   f2 = P11 + r * v1 * v1
   f3 = P12 + r * v1 * v2
   f4 = (E11 + P11) * v1
   f5 = E12 * v1 + 0.5 * (P11 * v2 + P12 * v1)
   f6 = E22 * v1 + P12 * v2

   return SVector(f1, f2, f3, f4, f5, f6)
end

@inbounds @inline function hll_speeds_min_max(eq::TenMoment1D, ul, ur)
   # Get conservative variables
   rl, v1l, v2l, P11l, P12l, P22l = con2prim(eq, ul)
   rr, v1r, v2r, P11r, P12r, P22r = con2prim(eq, ur)

   T11l = P11l / rl
   min_l = v1l - sqrt(3.0*T11l)
   max_l = v1l + sqrt(3.0*T11l)

   T11r = P11r / rr
   min_r = v1r - sqrt(3.0*T11r)
   max_r = v1r + sqrt(3.0*T11r)

   sl = min(min_l, min_r)
   sr = max(max_l, max_r)

   # Roe average state
   r   = 0.5(rl + rr)
   srl = sqrt(rl)
   srr = sqrt(rr)
   f1  = 1.0 / (srl + srr)
   v1  = f1 * (srl * v1l + srr * v1r)
   T11 = f1 * (srl * T11l + srr * T11r) + (srl * srr * (v1r - v1l)^2) / (3.0 * (srl + srr)^2)

   sl = min(sl, v1 - sqrt(3.0*T11))
   sr = max(sr, v1 + sqrt(3.0*T11))

   return sl, sr
end

@inbounds @inline function hll(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir, wave_speeds::Function)
   sl, sr = wave_speeds(eq, ual, uar)
   if sl > 0.0
      return Fl
   elseif sr < 0.0
      return Fr
   else
      return (sr*Fl - sl*Fr + sl*sr*(Ur - Ul))/(sr - sl)
   end
end

@inbounds @inline hll(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir) = hll(
   x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir, hll_speeds_min_max)

function max_abs_eigen_value(eq::TenMoment1D, u)
   ρ   = u[1]
   v1  = u[2] / ρ
   P11 = 2.0 * u[4] - ρ * v1 * v1
   T11 = P11 / ρ
   return abs(v1) + sqrt(3.0*T11)
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::TenMoment1D, dir)
   λ = max(max_abs_eigen_value(eq, ual), max_abs_eigen_value(eq, uar)) # local wave speed
   return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function compute_time_step(eq::TenMoment1D, grid, aux, op, cfl, u1, ua)
   nx = grid.size
   dx = grid.dx
   den = 0.0
   for i=1:nx
      u = get_node_vars(ua, eq, i)
      smax = max_abs_eigen_value(eq, u)
      den = max(den, smax/dx[i])
   end
   dt = cfl / den
   return dt
end

#-------------------------------------------------------------------------------
# Limiters
#-------------------------------------------------------------------------------
function Tenkai.apply_bound_limiter!(eq::TenMoment1D, grid, scheme, param, op, ua,
                                     u1, aux)
   if scheme.bound_limit == "no"
      return nothing
   end
end

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

varnames(eq::TenMoment1D) = eq.varnames
varnames(eq::TenMoment1D, i::Int) = eq.varnames[i]

function Tenkai.initialize_plot(eq::TenMoment1D, op, grid, problem, scheme, timer, u1,
                              ua)
   @timeit timer "Write solution" begin
   @timeit timer "Initialize write solution" begin
   # Clear and re-create output directory
   rm("output", force=true, recursive=true)
   mkdir("output")

   xc = grid.xc
   nx = grid.size
   @unpack xg = op
   nd = op.degree + 1
   nu = max(nd, 2)
   xu = LinRange(0.0, 1.0, nu)
   Vu = Vandermonde_lag(xg, xu)
   xf = grid.xf
   nvar = nvariables(eq)
   # Create plot objects to be later collected as subplots

   # Creating a subplot for title
   p_title = plot(title = "Cell averages plot, $nx cells, t = 0.0",
                  grid = false, showaxis = false, bottom_margin = 0Plots.px);
   p_ua, p_u1 = [plot() for _=1:nvar], [plot() for _=1:nvar];
   labels = varnames(eq)
   y = zeros(nx) # put dummy to fix plotly bug with OffsetArrays
   for n=1:nvar
      @views plot!(p_ua[n], xc, y, label = "Approximate",
                   linestyle = :dot, seriestype = :scatter,
                   color = :blue, markerstrokestyle = :dot,
                   markershape = :circle, markersize = 2, markerstrokealpha = 0);
      xlabel!(p_ua[n], "x"); ylabel!(p_ua[n], labels[n])
   end
   l_super = @layout[ a{0.01h}; b c d; e f g] # Selecting layout for p_title being title
   p_ua = plot(p_title, p_ua..., layout = l_super,
               size = (1500,500)); # Make subplots

   # Set up p_u1 to contain polynomial approximation as a different curve
   # for each cell
   x   = LinRange(xf[1], xf[2], nu)
   up1 = zeros(nvar, nd)
   u   = zeros(nu)
   for ii=1:nd
      u_node = get_node_vars(u1, eq, ii, 1)
      up1[:,ii] .= con2prim(eq, u_node)
   end

   for n=1:nvar
      u = @views Vu * up1[n,:]
      plot!(p_u1[n], x, u, color = :red, legend=false);
      xlabel!(p_u1[n], "x"); ylabel!(p_u1[n], labels[n]);
   end

   for i=2:nx
      for ii=1:nd
         u_node = get_node_vars(u1, eq, ii, i)
         up1[:,ii] .= con2prim(eq, u_node)
      end
      x = LinRange(xf[i], xf[i+1], nu)
      for n=1:nvar
         u = @views Vu * up1[n,:]
         plot!(p_u1[n], x, u, color = :red, label=nothing, legend=false)
      end
   end

   l = @layout[ a{0.01h}; b c d; e f g] # Selecting layout for p_title being title
   p_u1 = plot(p_title, p_u1..., layout = l,
               size = (1700,500)); # Make subplots

   anim_ua, anim_u1 = Animation(), Animation(); # Initialize animation objects
   plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1);
   return plot_data
   end # timer
   end # timer
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::TenMoment1D, grid,
                            problem, param, op, ua, u1, aux, ndigits=3)
   @timeit aux.timer "Write solution" begin
   @unpack plot_data = aux
   avg_filename = get_filename("output/avg", ndigits, fcount)
   @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
   @unpack final_time = problem
   xc = grid.xc
   nx = grid.size
   @unpack xg = op
   nd = op.degree + 1
   nu = max(nd, 2)
   xu = LinRange(0.0, 1.0, nu)
   Vu = Vandermonde_lag(xg, xu)
   nvar = nvariables(eq)
   @unpack save_time_interval, save_iter_interval, animate = param
   avg_file = open("$avg_filename.txt", "w")
   up_ = zeros(nvar)
   ylims = [[Inf,-Inf] for _=1:nvar] # set ylims for plots of all variables
   for i=1:nx
      ua_node = get_node_vars(ua, eq, i)
      up_ .= con2prim(eq, ua_node)
      @printf(avg_file, "%e %e %e %e %e %e %e \n", xc[i], up_...)
      # TOTHINK - Check efficiency of printf
      for n in eachvariable(eq)
         p_ua[n+1][1][:y][i] = @views up_[n]    # Update y-series
         ylims[n][1] = min(ylims[n][1], up_[n]) # Compute ymin
         ylims[n][2] = max(ylims[n][2], up_[n]) # Compute ymax
      end
   end
   close(avg_file)
   for n=1:nvar # set ymin, ymax for ua, u1 plots
      ylims!(p_ua[n+1],(ylims[n][1]-0.1,ylims[n][2]+0.1))
      ylims!(p_u1[n+1],(ylims[n][1]-0.1,ylims[n][2]+0.1))
   end
   t = round(time; digits=3)
   title!(p_ua[1], "Cell averages plot, $nx cells, t = $t")
   sol_filename = get_filename("output/sol", ndigits, fcount)
   sol_file = open(sol_filename*".txt", "w")
   up1 = zeros(nvar,nd)

   u = zeros(nvar,nu)
   x = zeros(nu)
   for i=1:nx
      for ii=1:nd
         u_node = get_node_vars(u1, eq, ii, i)
         up1[:, ii] .= con2prim(eq, u_node)
      end
      @. x = grid.xf[i] + grid.dx[i]*xu
      @views mul!(u, up1, Vu')
      for n=1:nvar
         p_u1[n+1][i][:y] = u[n,:]
      end
      for ii=1:nu
         u_node = get_node_vars(u, eq, ii)
         @printf(sol_file, "%e %e %e %e %e %e %e \n", x[ii], u_node...)
      end
   end
   close(sol_file)
   title!(p_u1[1], "Numerical Solution, $nx cells, t = $t")
   println("Wrote $sol_filename.txt, $avg_filename.txt")
   if problem.final_time - time < 1e-10
      cp("$avg_filename.txt","./output/avg.txt", force=true)
      cp("$sol_filename.txt","./output/sol.txt", force=true)
      println("Wrote final solution to avg.txt, sol.txt.")
   end
   if animate == true
      if abs(time - final_time) < 1.0e-10
         frame(anim_ua, p_ua)
         frame(anim_u1, p_u1)
      end
      if save_iter_interval > 0
         animate_iter_interval = save_iter_interval
         if mod(iter, animate_iter_interval) == 0
            frame(anim_ua, p_ua)
            frame(anim_u1, p_u1)
         end
      elseif save_time_interval > 0
         animate_time_interval = save_time_interval
         k1, k2 = ceil(time/animate_time_interval), floor(time/animate_time_interval)
         if (abs(time-k1*animate_time_interval) < 1e-10 ||
            abs(time-k2*animate_time_interval) < 1e-10)
            frame(anim_ua, p_ua)
            frame(anim_u1, p_u1)
         end
      end
   end
   fcount += 1
   return fcount
   end # timer
end

function post_process_soln(eq::TenMoment1D, aux, problem, param)
   @unpack timer, error_file = aux
   @timeit timer "Write solution" begin
   println("Post processing solution")
   @unpack plot_data = aux
   @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
   @unpack animate, saveto = param
   savefig(p_ua, "output/avg.png")
   savefig(p_u1, "output/sol.png")
   savefig(p_ua, "output/avg.html")
   savefig(p_u1, "output/sol.html")
   if animate == true
      gif(anim_ua, "output/avg.mp4", fps = 5)
      gif(anim_u1, "output/sol.mp4", fps = 5)
   end
   println("Wrote avg, sol in gif,html,png format to output directory.")
   plot(p_ua); plot(p_u1);

   close(error_file)
   if saveto != "none"
      if saveto[end] == "/"
         saveto = saveto[1:end-1]
      end
      mkpath(saveto)
      for file in readdir("./output")
         cp("./output/$file", "$saveto/$file", force=true)
      end
      cp("./error.txt", "$saveto/error.txt", force=true)
      println("Saved output files to $saveto")
   end

   end # timer

   # Print timer data on screen
   print_timer(aux.timer, sortby = :firstexec); print("\n")
   show(aux.timer); print("\n")
   println("Time outside write_soln = "
            * "$(( TimerOutputs.tottime(timer)
                  - TimerOutputs.time(timer["Write solution"]) ) * 1e-9)s")
   println("─────────────────────────────────────────────────────────────────────────────────────────")
   timer_file = open("./output/timer.json", "w")
   JSON3.write(timer_file, TimerOutputs.todict(timer))
   close(timer_file)
   return nothing
end


get_equation() = TenMoment1D(["rho", "v1", "v2", "P11", "P12", "P12"], "Ten moment problem")

end # @muladd

end # module