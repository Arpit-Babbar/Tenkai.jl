using StaticArrays
using Tenkai.TenkaicRK
using Tenkai.TenkaicRK.Tenkai

# Submodules
Eq = TenkaicRK.EqShearShallowWater1D
EqTenMom = Tenkai.EqTenMoment1D

xmin, xmax = 0.0, 1.0
boundary_condition = (neumann, neumann)

dummy_bv(x, t) = 0.0

function initial_value_rp(x)
    EqTenMom.rp(x,
                (0.02, 0.0, 0.0, 0.02 * 1e-4, 0.0, 0.02 * 1e-4),
                (0.01, 0.0, 0.0, 0.01 * 1e-4, 0.0, 0.01 * 1e-4),
                0.5)
end

initial_value = initial_value_rp

boundary_value = dummy_bv
exact_solution_rp(x, t) = initial_value_rp(x)
exact_solution = exact_solution_rp

degree = 1
solver = cRK22()
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no"
bflux = evaluate
final_time = 0.5 # Choose 0.125 for sod, two_shock; 0.15 for two_rare_iv; 0.05 for two_rare_vacuum_iv

nx = ceil(Int64, 500)
cfl = 0.0
bounds = ([-Inf], [Inf]) # Not used in Euler
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 * final_time
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0
cfl_safety_factor = 0.5

#------------------------------------------------------------------------------
grid_size = nx
gravity = 9.81
eq = Eq.get_equation(gravity)
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                  final_time, exact_solution)
# limiter = setup_limiter_none()
limiter = setup_limiter_blend(blend_type = mh_blend(eq),
                              # indicating_variables = Eq.rho_p_indicator!,
                              indicating_variables = Eq.conservative_indicator!,
                              reconstruction_variables = conservative_reconstruction,
                              indicator_model = "gassner"
                              # pure_fv = true
                              )
limiter = setup_limiter_tvb(eq; tvbM = tvbM, beta = 0.5)
# limiter = setup_limiter_none()
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(grid_size, cfl, bounds, save_iter_interval, save_time_interval,
                   compute_error_interval, animate = animate,
                   cfl_safety_factor = cfl_safety_factor)
#------------------------------------------------------------------------------
sol = Tenkai.solve(eq, problem, scheme, param);

println(sol["errors"])

return sol;

p_ua = deepcopy(sol["plot_data"].p_ua)
using DelimitedFiles
exact_dambreak = readdlm("exact_dambreak.txt", skipstart = 1)
plot!(p_ua[2], exact_dambreak[:, 1] .+ 0.5, exact_dambreak[:, 2], label = "Exact", lw = 2)
plot!(p_ua[3], exact_dambreak[:, 1] .+ 0.5, exact_dambreak[:, 3], label = "Exact", lw = 2)
plot!(p_ua[4], exact_dambreak[:, 1] .+ 0.5, exact_dambreak[:, 4], label = "Exact", lw = 2)
plot!(p_ua[5], exact_dambreak[:, 1] .+ 0.5, exact_dambreak[:, 5] .* exact_dambreak[:, 2],
      label = "Exact", lw = 2)

ylims!(p_ua[2], (0.01, 0.02))
ylims!(p_ua[3], minimum(exact_dambreak[:, 3]) - 0.01, maximum(exact_dambreak[:, 3]) + 0.2)
ylims!(p_ua[4], minimum(exact_dambreak[:, 4]) - 0.01, maximum(exact_dambreak[:, 4]) + 0.2)
ylims!(p_ua[5], minimum(exact_dambreak[:, 5]) - 0.0001,
       maximum(exact_dambreak[:, 5]) + 0.0001)
p_ua
