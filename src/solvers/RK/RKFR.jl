using Tenkai: set_initial_condition!,
              compute_cell_average!,
              compute_face_residual!,
              set_blend_dt!,
              write_soln!,
              compute_error,
              post_process_soln

using MuladdMacro
using LoopVectorization
using SimpleUnPack
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqTsit5
using DiffEqCallbacks: StepsizeLimiter
using Printf
using LinearAlgebra: axpy!, axpby!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#------------------------------------------------------------------------------
# Dimension independent methods in FR used here
#------------------------------------------------------------------------------
(using Tenkai: apply_limiter!, compute_time_step, adjust_time_step,
               pre_process_limiter!, get_cfl, save_solution)

#------------------------------------------------------------------------------
# Methods to be defined in RKFR1D, RKFR2D
#------------------------------------------------------------------------------
(import Tenkai: setup_arrays_rkfr)
compute_cell_residual_rkfr!() = nothing
update_ghost_values_rkfr!() = nothing

#------------------------------------------------------------------------------
function compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, t,
                                dt, iter, fcount, cache, u1, Fb, ub, ua, res)
    pre_process_limiter!(eq, t, iter, fcount, dt, grid, problem, scheme, param,
                         aux, op, u1, ua)
    compute_cell_residual_rkfr!(eq, grid, op, problem, scheme, aux, t, dt, u1,
                                res, Fb, ub, cache)
    update_ghost_values_rkfr!(problem, scheme, eq, grid, aux, op, cache, t)
    compute_face_residual!(eq, grid, op, cache, problem, scheme, param, aux, t, dt, u1,
                           Fb, ub, ua, res)
    return nothing
end

#------------------------------------------------------------------------------
# For use with DifferentialEquations
#------------------------------------------------------------------------------
function compute_residual_rkfr!(du, u, p, t)
    eq, problem, scheme, param, cfl, grid, aux, op, cache, Fb, ub, ua, res = p
    dt = -1.0
    iter, fcount = 0, 0 # Dummy fillers, RK doesn't support alpha output
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, t, dt,
                           iter, fcount, cache, u, Fb, ub, ua, res)
    @turbo du .= res
    return nothing
end

# DiffEq callback function for time step. We also save solution in this.
# Makes use of two global variables: iter, fcount
function dtFE(u, p, t)
    eq, problem, scheme, param, cfl, grid, aux, op, cache, Fb, ub, ua, res = p

    @unpack compute_error_interval = param

    dt = compute_time_step(eq, problem, grid, aux, op, cfl, u, ua)
    dt = adjust_time_step(problem, param, t, dt, aux)
    set_blend_dt!(eq, aux, dt) # Set dt used by MUSCL-Hancock
    @printf("iter,dt,t   = %5d %12.4e %12.4e \n", iter, dt, t)
    if save_solution(problem, param, t, iter)
        global fcount = write_soln!("sol", fcount, iter, t, dt, eq, grid, problem,
                                    param, op, ua, u, aux)
    end
    if (compute_error_interval > 0 &&
        mod(iter, compute_error_interval) == 0)
        error_norm = compute_error(problem, grid, eq, aux, op, u, t)
    end
    global iter += 1
    return dt
end

# DiffEq limiter functions
# KLUDGE - Pass dt through integrator
function stage_limiter!(u, integrator, p, t)
    eq, problem, scheme, param, cfl, grid, aux, op, cache, Fb, ub, ua, res = p
    compute_cell_average!(ua, u, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u)
end
step_limiter!(u, integrator, p, t) = nothing

#------------------------------------------------------------------------------
# 1st order, 1-stage RK (Forward Euler)
#------------------------------------------------------------------------------
function apply_rk11!(eq, problem, param, grid, op, scheme, aux, t, dt, cache,
                     u0, u1, Fb, ub, ua, res)
    # Stage 1
    ts = t
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-1.0, res, u1)         # u1 = u1 - res
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    return nothing
end

function apply_rk11_muscl!(eq, problem, param, grid, op, scheme, aux, t, dt,
                           cache, u0, u1, Fb, ub, ua, res)
    r1 = @view res[:, 1, :]
    # Stage 1
    update_ghost_values_u1!(eq, problem, grid, op, u1, t)

    @unpack nvar = eq
    @unpack numerical_flux = scheme
    nx = grid.size
    dx = grid.dx[2] # assume uniform grid
    xf = grid.xf
    u = @view u1[:, 1, :]
    unph = ub
    for i in 0:(nx + 1)
        @views eq.con2prim!(u[:, i])
    end

    ufl, ufr = zeros(nvar), zeros(nvar)
    fl, fr, fn = zeros(nvar), zeros(nvar), zeros(nvar)
    for i in 1:nx
        for n in 1:nvar
            # s = minmod( (u[n,i] - u[n,i-1]  ) / dx,
            #             (u[n,i+1] - u[n,i-1]) / (2.0*dx),
            #             (u[n,i+1] - u[n,i]  ) / dx,
            #             0.0 )
            s = (u1[n, 1, i + 1] - u1[n, 1, i - 1]) / (2.0 * dx)
            ufl[n] = u[n, i] - 0.5 * dx * s
            ufr[n] = u[n, i] + 0.5 * dx * s
        end
        eq.prim2con!(ufl)
        eq.prim2con!(ufr)
        eq.flux!(xf[i], ufl, eq, fl)
        eq.flux!(xf[i + 1], ufr, eq, fr)

        for n in 1:nvar
            unph[n, 1, i] = ufl[n] + 0.5 * (dt / dx) * (fl[n] - fr[n])
            unph[n, 2, i] = ufr[n] + 0.5 * (dt / dx) * (fl[n] - fr[n])
        end
    end

    for i in 0:(nx + 1)
        @views eq.prim2con!(u1[:, 1, i])
    end

    update_ghost_values_rkfr!(problem, eq, grid, aux, op, cache, t)

    @assert abs(unph[1, 2, 0] - unph[1, 2, nx]) < 1e-12 # Check if ghost values are update correctly

    fill!(r1, zero(eltype(r1)))
    for i in 1:(nx + 1)
        x = grid.xf[i]
        @views ul, ur = unph[:, 2, i - 1], unph[:, 1, i]
        eq.flux!(x, ul, eq, fl)
        eq.flux!(x, ur, eq, fr)
        numerical_flux(x, ul, ur, fl, fr, ul, ur, eq, 1, fn)
        for n in 1:nvar
            r1[n, i - 1] += (dt / dx) * fn[n]
            r1[n, i] -= (dt / dx) * fn[n]
        end
    end
    for i in 1:nx
        for n in 1:nvar
            u[n, i] -= r1[n, i]
        end
    end
    return nothing
end

#------------------------------------------------------------------------------
# 2nd order, 2-stage SSPRK
#------------------------------------------------------------------------------
function apply_ssprk22!(eq, problem, param, grid, op, scheme, aux,
                        t, dt, cache, u0, u1, Fb, ub, ua, res)
    # Stage 1
    ts = t
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-1.0, res, u1)         # u1 = u1 - res
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 2
    ts = t + dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-1.0, res, u1)         # u1 = u1 - res
    axpby!(0.5, u0, 0.5, u1)    # u1 = u0 + u1
    return nothing
end

#------------------------------------------------------------------------------
# 3rd order, 3-stage SSPRK
#------------------------------------------------------------------------------
function apply_ssprk33!(eq, problem, param, grid, op, scheme, aux,
                        t, dt, cache, u0, u1, Fb, ub, ua, res)
    # Stage 1
    ts = t
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-1.0, res, u1)                     # u1 = u1 - res
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 2
    ts = t + dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-1.0, res, u1)                     # u1 = u1 - res
    axpby!(0.75, u0, 0.25, u1)              # u1 = (3/4)u0 + (1/4)u1
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 3
    ts = t + 0.5 * dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-1.0, res, u1)                     # u1 = u1 - res
    axpby!(1.0 / 3.0, u0, 2.0 / 3.0, u1)        # u1 = (1/3)u0 + (2/3)u1
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    return nothing
end

#------------------------------------------------------------------------------
# z = a*x + y
#------------------------------------------------------------------------------
function axpyz!(a, x, y, z)
    @tturbo for i in eachindex(z)
        z[i] = a * x[i] + y[i]
    end
    return nothing
end

#--------------------------------------------------------
# Four stage, third order SSPRK
#------------------------------------------------------
function apply_ssprk43!(eq, problem, param, grid, op, scheme, aux,
                        t, dt, cache, u0, u1, Fb, ub, ua, res)
    # Stage 1
    ts = t
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-0.5, res, u1)                     # u1 = u1 - res
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 2
    ts = t + 0.5 * dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-0.5, res, u1)                     # u1 = u1 - res
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 3
    ts = t + dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-0.5, res, u1)                     # u1 = u1 - res
    axpby!(2 / 3, u0, 1 / 3, u1)                 # u1 = 2/3*u0 + 1/3*u1
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 4
    ts = t + 0.5 * dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpy!(-0.5, res, u1)                    # u1 = u1 - res
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    return nothing
end

#------------------------------------------------------------------------------
# Classical RK4
#------------------------------------------------------------------------------
function apply_rk4!(eq, problem, param, grid, op, scheme, aux, t,
                    dt, cache, u0, u1, Fb, ub, ua, res)
    utmp = copy(u0)
    # Stage 1
    ts = t
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpyz!(-0.5, res, u0, u1)       # u1   = u0 - 0.5*r1
    axpy!(-1.0 / 6.0, res, utmp)      # utmp = utmp - (1/6)*r1
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 2
    ts = t + 0.5 * dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpyz!(-0.5, res, u0, u1)       # u1   = u0 - 0.5*r1
    axpy!(-1.0 / 3.0, res, utmp)      # utmp = utmp - (1/3)*r1
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 3
    ts = t + 0.5 * dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpyz!(-1.0, res, u0, u1)       # u1   = u0 - r1
    axpy!(-1.0 / 3.0, res, utmp)      # utmp = utmp - (1/3)*r1
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    # Stage 4
    ts = t + dt
    compute_residual_rkfr!(eq, problem, grid, op, scheme, param, aux, ts, dt,
                           iter, fcount, cache, u1, Fb, ub, ua, res)
    axpyz!(-1.0 / 6.0, res, utmp, u1) # u1   = utmp - (1/6)*r1
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
    return nothing
end

#------------------------------------------------------------------------------
# Select time scheme
#------------------------------------------------------------------------------

function get_time_scheme(degree, param)
    time_schemes = Dict("Tsit5" => Tsit5, "SSPRK54" => SSPRK54,
                        "RK11" => apply_rk11!,
                        "SSPRK22" => apply_ssprk22!,
                        "SSPRK33" => apply_ssprk33!,
                        "SSPRK43" => apply_ssprk43!,
                        "RK4" => apply_rk4!)
    @unpack time_scheme = param
    if time_scheme in keys(time_schemes)
        return time_scheme, time_schemes[time_scheme]
    else
        @assert time_scheme == "by degree"
        if degree == 0
            time_scheme == "RK11"
            return time_scheme, apply_rk11!
        elseif degree == 1
            time_scheme = "SSPRK22"
            return time_scheme, apply_ssprk22!
        elseif degree == 2
            time_scheme = "SSPRK33"
            return time_scheme, apply_ssprk33!
        elseif degree == 3
            time_scheme = "SSPRK54"
            return time_scheme, SSPRK54
        elseif degree == 4
            time_scheme = "SSPRK54"
            return time_scheme, SSPRK54
        else
            @assert false "Degree not implemeneted!!"
        end
    end
end

#------------------------------------------------------------------------------
function solve_rkfr(eq, problem, scheme, param, grid, op, aux, cache)
    println("Solving ", eq.name, " using RKFR")

    @unpack cfl, grid_size, compute_error_interval = param
    @unpack solution_points, degree, correction_function = scheme

    @unpack u0, u1, ua, res, Fb, ub = cache

    # Set initial condition
    set_initial_condition!(u1, eq, grid, op, problem)

    # Compute cell average for initial condition
    compute_cell_average!(ua, u1, 0.0, eq, grid, problem, scheme, aux, op)

    # Apply limiter to handle discontinuities of the initial solution
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)

    # Initialize counters
    t = 0.0
    fcount, iter = 0, 0
    @unpack final_time = problem

    # Choose CFL number
    if cfl > 0.0
        @printf("CFL: specified value = %f\n", cfl)
    else
        cfl = get_cfl(eq, scheme, param)
        @printf("CFL: based on stability = %f\n", cfl)
    end

    time_scheme, update_solution_rkfr! = get_time_scheme(degree, param)

    # Fifth order: use DifferentialEquations
    if time_scheme in ["Tsit5", "SSPRK54"]
        global fcount, iter = 0, 0
        println("Using DifferentialEquations")
        p = (eq, problem, scheme, param, cfl, grid, aux, op, cache, Fb, ub, ua,
             res)
        copyto!(u0, u1)
        tspan = (0.0, final_time)
        odeprob = ODEProblem(compute_residual_rkfr!, u0, tspan, p)
        dt = compute_time_step(eq, problem, grid, aux, op, cfl, u1, ua)
        callback_dt = StepsizeLimiter(dtFE, safety_factor = 1.0, max_step = true)
        callback = (callback_dt)
        # Try adding another function layer?
        sol = OrdinaryDiffEqSSPRK.solve(odeprob,
                                        update_solution_rkfr!(stage_limiter!,
                                                              step_limiter!),
                                        dt = dt, adaptive = false,
                                        callback = callback,
                                        saveat = final_time, dense = false,
                                        save_start = false,
                                        save_everystep = false)
        copyto!(u1, sol[1])
        t = sol.t[1]
        error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
        post_process_soln(eq, aux, problem, param, scheme)
        return Dict("u" => u1, "ua" => ua, "errors" => error_norm, "aux" => aux,
                    "plot_data" => aux.plot_data, "grid" => grid,
                    "op" => op, "scheme" => scheme)
    end

    # Save initial solution to file
    fcount = write_soln!("sol", fcount, iter, t, 0.0, eq, grid, problem, param, op,
                         ua, u1, aux)

    # Compute initial error norm
    error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
    println("Starting time stepping")
    while t < final_time
        dt = compute_time_step(eq, problem, grid, aux, op, cfl, u1, ua)
        dt = adjust_time_step(problem, param, t, dt, aux)
        copyto!(u0, u1) # u0 = u1
        update_solution_rkfr!(eq, problem, param, grid, op, scheme, aux, t, dt, cache,
                              u0, u1, Fb, ub, ua, res)
        compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
        t += dt
        iter += 1
        @printf("iter,dt,t = %5d %12.4e %12.4e\n", iter, dt, t)
        if save_solution(problem, param, t, iter)
            fcount = write_soln!("sol", fcount, iter, t, dt, eq, grid, problem, param,
                                 op,
                                 ua, u1, aux)
        end
        if (compute_error_interval > 0 &&
            mod(iter, compute_error_interval) == 0)
            error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
        end
    end
    error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
    post_process_soln(eq, aux, problem, param, scheme)
    return Dict("u" => u1, "ua" => ua, "errors" => error_norm, "aux" => aux,
                "plot_data" => aux.plot_data, "grid" => grid,
                "op" => op, "scheme" => scheme)
end
end # muladd
