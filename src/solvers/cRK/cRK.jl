abstract type cRKSolver end
abstract type AbstractDissipation end # Don't think it'll be used beyond D-CSX

solver2enum(solver::cRKSolver) = ssfr # solver type enum

struct VolumeIntegralWeak end

struct cRK64{VolumeIntegral} <: cRKSolver
    volume_integral::VolumeIntegral
end # TODO - Implement
struct cRK44{VolumeIntegral} <: cRKSolver
    volume_integral::VolumeIntegral
end
struct cRK33{VolumeIntegral} <: cRKSolver
    volume_integral::VolumeIntegral
end
struct cRK22{VolumeIntegral} <: cRKSolver
    volume_integral::VolumeIntegral
end
struct cRK11{VolumeIntegral} <: cRKSolver
    volume_integral::VolumeIntegral
end

function cRK64(; volume_integral = VolumeIntegralWeak())
    return cRK64{typeof(volume_integral)}(volume_integral)
end

function cRK44(; volume_integral = VolumeIntegralWeak())
    return cRK44{typeof(volume_integral)}(volume_integral)
end

function cRK33(; volume_integral = VolumeIntegralWeak())
    return cRK33{typeof(volume_integral)}(volume_integral)
end

function cRK22(; volume_integral = VolumeIntegralWeak())
    return cRK22{typeof(volume_integral)}(volume_integral)
end

function cRK11(; volume_integral = VolumeIntegralWeak())
    return cRK11{typeof(volume_integral)}(volume_integral)
end

struct DCSX <: AbstractDissipation end

function update_ghost_values_cRK!(problem, scheme, eq, grid, aux, op, cache, t, dt)
    update_ghost_values_lwfr!(problem, scheme, eq, grid, aux, op, cache, t, dt)
    update_ghost_values_u1!(eq, problem, grid, op, cache.u1, aux, t)
end

function initialize_solution!(eq, grid, op, problem, scheme, param, aux, cache)
    @unpack u1, ua = cache
    # Set initial condition
    set_initial_condition!(u1, eq, grid, op, problem)

    # Compute cell average for initial condition
    compute_cell_average!(ua, u1, 0.0, eq, grid, problem, scheme, aux, op)

    # Apply limiter to handle discontinuities of the initial solution
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
end

function evolve_solution!(eq, grid, op, problem, scheme, param, aux, iter, t, dt, fcount,
                          cache)
    @unpack ua, u1, res, Fb, Ub = cache

    pre_process_limiter!(eq, t, iter, fcount, dt, grid, problem, scheme,
                         param, aux, op, u1, ua)
    compute_cell_residual_cRK!(eq, grid, op, problem, scheme, aux, t, dt, cache)
    update_ghost_values_cRK!(problem, scheme, eq, grid, aux, op, cache, t, dt)
    prolong_solution_to_face_and_ghosts!(u1, cache, eq, grid, op, problem, scheme, aux, t,
                                         dt)
    compute_face_residual!(eq, grid, op, cache, problem, scheme, param, aux, t, dt, u1,
                           Fb, Ub, ua, res)
    update_solution_cRK!(u1, eq, grid, op, problem, scheme, res, aux, t, dt) # u1 = u1 - res
    compute_cell_average!(ua, u1, t, eq, grid, problem, scheme, aux, op)
    apply_limiter!(eq, problem, grid, scheme, param, op, aux, ua, u1)
end

#-------------------------------------------------------------------------------
# Apply cRK scheme and solve the problem
# N = degree of solution space
#-------------------------------------------------------------------------------
function solve_ssfr(eq, problem, scheme::Scheme{<:cRKSolver}, param, grid, op, aux, cache)
    println("Solving ", eq.name, " using a cRK scheme")

    @unpack final_time = problem
    @unpack grid_size, cfl, compute_error_interval = param

    # Allocate memory
    @unpack u1, ua, res, Fb, Ub = cache

    # Compute initial condition, apply limiter, and compute cell average
    initialize_solution!(eq, grid, op, problem, scheme, param, aux, cache)

    # Initialize counters
    local iter, t, fcount = 0, 0.0, 0

    # Save initial solution to file
    fcount = write_soln!("sol", fcount, iter, t, 0.0, eq, grid, problem, param, op,
                         ua, u1, aux)

    # Choose CFL number
    if cfl > 0.0
        @printf("CFL: specified value = %f\n", cfl)
    else
        cfl = get_cfl(eq, scheme, param)
        @printf("CFL: based on stability = %f\n", cfl)
    end

    # Compute initial error norm
    error_norm = compute_error(problem, grid, eq, aux, op, u1, t)

    local dt
    println("Starting time stepping")
    while t < final_time
        dt = compute_time_step(eq, problem, grid, aux, op, cfl, u1, ua)
        dt = adjust_time_step(problem, param, t, dt, aux)

        evolve_solution!(eq, grid, op, problem, scheme, param, aux, iter, t, dt, fcount,
                         cache)

        t += dt
        iter += 1
        @printf("iter,dt,t = %5d %12.4e %12.4e\n", iter, dt, t)
        if save_solution(problem, param, t, iter)
            fcount = write_soln!("sol", fcount, iter, t, dt, eq, grid, problem, param,
                                 op, ua, u1, aux)
        end
        if (compute_error_interval > 0 && mod(iter, compute_error_interval) == 0)
            error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
        end
    end
    error_norm = compute_error(problem, grid, eq, aux, op, u1, t)
    post_process_soln(eq, aux, problem, param, scheme)

    # KLUDGE - Move to post_process_solution
    if scheme.limiter.name == "blend"
        modal_smoothness_indicator(eq, t, iter, fcount, dt, grid, scheme,
                                   problem, param, aux, op, u1, ua)
    end

    return Dict("u" => u1, "ua" => ua, "errors" => error_norm,
                "plot_data" => aux.plot_data, "grid" => grid,
                "op" => op, "scheme" => scheme,
                "aux" => aux)
end

@inbounds @inline update_solution_cRK!(u1, eq, grid, op, problem, scheme, res, aux, t, dt) = update_solution_lwfr!(u1,
                                                                                                               res,
                                                                                                               aux)
