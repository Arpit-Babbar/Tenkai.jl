using Trixi
using Trixi: @trixi_timeit
using OrdinaryDiffEq

# Taken from https://github.com/trixi-framework/Trixi.jl/blob/main/src/callbacks_step/stepsize.jl

mutable struct StepsizeCallbackM2000{RealT}
    cfl_number::RealT
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:StepsizeCallbackM2000})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    @unpack cfl_number = stepsize_callback
    print(io, "StepsizeCallbackM2000(cfl_number=", cfl_number, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallbackM2000})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = [
            "CFL number" => stepsize_callback.cfl_number
        ]
        Trixi.summary_box(io, "StepsizeCallbackM2000", setup)
    end
end

function StepsizeCallbackM2000(; cfl::Real = 1.0)
    stepsize_callback = StepsizeCallbackM2000(cfl)

    DiscreteCallback(stepsize_callback, stepsize_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: StepsizeCallbackM2000}
    cb.affect!(integrator)
end

# this method is called to determine whether the callback should be activated
function (stepsize_callback::StepsizeCallbackM2000)(u, t, integrator)
    return true
end

function m2000_boundary_dt(equations, dg, cache, cfl_number)
    rho_b, vx_b, vy_b, pres_b = (5.0, 800.0, 0.0, 0.4127)
    u_boundary = Trixi.prim2cons((rho_b, vx_b, vy_b, pres_b), equations)
    λ1, λ2 = Trixi.max_abs_speeds(u_boundary, equations)
    inv_jacobian = cache.elements.inverse_jacobian[1] # Assuming uniform grid
    scaled_speed = sqrt(inv_jacobian) * (λ1 + λ2) # Equivalent to abs(sx)/dx[el_x] + abs(sy)/dy[el_y]
    return 2 * cfl_number / (nnodes(dg) * scaled_speed)
end

# This method is called as callback during the time integration.
@inline function (stepsize_callback::StepsizeCallbackM2000)(integrator)
    # TODO: Taal decide, shall we set the time step even if the integrator is adaptive?
    if !integrator.opts.adaptive
        t = integrator.t
        u_ode = integrator.u
        semi = integrator.p
        mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
        @unpack cfl_number = stepsize_callback
        u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)

        dt = @trixi_timeit Trixi.timer() "calculate dt" cfl_number*Trixi.max_dt(u, t, mesh,
                                                                                Trixi.have_constant_speed(equations),
                                                                                equations,
                                                                                solver,
                                                                                cache)

        # Also include boundary info
        dt = min(dt, m2000_boundary_dt(equations, solver, cache, cfl_number))

        set_proposed_dt!(integrator, dt)
        integrator.opts.dtmax = dt
        integrator.dtcache = dt
    end

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

# Time integration methods from the DiffEq ecosystem without adaptive time stepping on their own
# such as `CarpenterKennedy2N54` require passing `dt=...` in `solve(ode, ...)`. Since we don't have
# an integrator at this stage but only the ODE, this method will be used there. It's called in
# many examples in `solve(ode, ..., dt=stepsize_callback(ode), ...)`.
function (cb::DiscreteCallback{Condition, Affect!})(ode::ODEProblem) where {Condition,
                                                                            Affect! <:
                                                                            StepsizeCallbackM2000
                                                                            }
    stepsize_callback = cb.affect!
    @unpack cfl_number = stepsize_callback
    u_ode = ode.u0
    t = first(ode.tspan)
    semi = ode.p
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)

    dt_ = cfl_number *
          Trixi.max_dt(u, t, mesh, Trixi.have_constant_speed(equations), equations, solver,
                       cache)
    dt = min(dt_, m2000_boundary_dt(equations, solver, cache, cfl_number))

    return dt
end
