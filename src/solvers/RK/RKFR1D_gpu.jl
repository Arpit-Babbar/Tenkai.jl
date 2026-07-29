using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using GPUArraysCore: AbstractGPUArray

# GPU kernels for the RK 1D solver, `limiter = "none"`, `bound_limit = "no"`
# only (see docs/GPU.md). Each pairs with the CPU function of the same
# purpose in FR1D.jl/RKFR1D.jl, which stays untouched and is still what
# `backend = CPU()` runs.

@kernel function cell_average_kernel!(ua, @Const(u1), eq, wg)
    cell = @index(Global)
    nd = length(wg)
    acc = wg[1] * get_node_vars(u1, eq, 1, cell)
    for ix in 2:nd
        acc = acc + wg[ix] * get_node_vars(u1, eq, ix, cell)
    end
    set_node_vars!(ua, acc, eq, cell)
end

# Ghost values of `ua`: single thread, O(1) work.
@kernel function cell_average_ghost_kernel!(ua, eq, nx, periodic_x, left_reflect,
                                            right_reflect)
    if periodic_x
        set_node_vars!(ua, get_node_vars(ua, eq, nx), eq, 0)
        set_node_vars!(ua, get_node_vars(ua, eq, 1), eq, nx + 1)
    else
        left = get_node_vars(ua, eq, 1)
        right = get_node_vars(ua, eq, nx)
        if left_reflect
            left = SVector(left[1], -left[2], left[3])
        end
        if right_reflect
            right = SVector(right[1], -right[2], right[3])
        end
        set_node_vars!(ua, left, eq, 0)
        set_node_vars!(ua, right, eq, nx + 1)
    end
end

# Gather: thread owns output node (iix, cell), sums the D1 contraction over
# ix -- avoids the scatter race a direct port of the CPU loop would have.
@kernel function cell_residual_kernel!(res, @Const(u1), eq, D1, xg, xc, dx, dt)
    iix, cell = @index(Global, NTuple)
    nd = length(xg)
    RealT = eltype(xg)
    half = RealT(0.5)
    lamx = dt / dx[cell]
    acc = zero(SVector{nvariables(eq), RealT})
    for ix in 1:nd
        x = xc[cell] - half * dx[cell] + xg[ix] * dx[cell]
        u_node = get_node_vars(u1, eq, ix, cell)
        acc = acc + (lamx * D1[iix, ix]) * flux(x, u_node, eq)
    end
    multiply_add_to_node_vars!(res, one(RealT), acc, eq, iix, cell)
end

# ub/Fb boundary extrapolation, `bflux = evaluate` or `extrapolate`. Thread
# owns `cell`: the sum over `ix` needs to fully complete before Fb can be
# computed from it (evaluate mode), so this can't be split further without
# recomputing work -- unlike cell_residual_kernel! above.
@kernel function boundary_extrapolate_kernel!(ub, Fb, @Const(u1), eq, Vl, Vr, xg, xc, dx,
                                              ::Val{Bflux}) where {Bflux}
    cell = @index(Global)
    nd = length(xg)
    RealT = eltype(xg)
    half = RealT(0.5)
    ubl = zero(SVector{nvariables(eq), RealT})
    ubr = zero(SVector{nvariables(eq), RealT})
    fbl = zero(SVector{nvariables(eq), RealT})
    fbr = zero(SVector{nvariables(eq), RealT})
    for ix in 1:nd
        x = xc[cell] - half * dx[cell] + xg[ix] * dx[cell]
        u_node = get_node_vars(u1, eq, ix, cell)
        ubl = ubl + Vl[ix] * u_node
        ubr = ubr + Vr[ix] * u_node
        if !Bflux # extrapolate: L2 projection of the pointwise fluxes
            f = flux(x, u_node, eq)
            fbl = fbl + Vl[ix] * f
            fbr = fbr + Vr[ix] * f
        end
    end
    set_node_vars!(ub, ubl, eq, 1, cell)
    set_node_vars!(ub, ubr, eq, 2, cell)
    if Bflux # evaluate: flux of the extrapolated boundary state
        xl = xc[cell] - half * dx[cell]
        xr = xc[cell] + half * dx[cell]
        set_node_vars!(Fb, flux(xl, ubl, eq), eq, 1, cell)
        set_node_vars!(Fb, flux(xr, ubr, eq), eq, 2, cell)
    else
        set_node_vars!(Fb, fbl, eq, 1, cell)
        set_node_vars!(Fb, fbr, eq, 2, cell)
    end
end

# Two-pass face residual, mirroring the pattern FR2D.jl already uses
# correctly: pass 1 computes the numerical flux once per face and stores it
# into Fb (no race: each face is touched by exactly one thread); pass 2 is
# an element-parallel gather that reads Fb and updates res.
@kernel function face_flux_kernel!(Fb, @Const(ua), @Const(ub), eq, xf, num_flux)
    i = @index(Global) # face i, between cells i-1 and i; i in 1:nx+1
    ual, uar = get_node_vars(ua, eq, i - 1), get_node_vars(ua, eq, i)
    Fl, Fr = get_node_vars(Fb, eq, 2, i - 1), get_node_vars(Fb, eq, 1, i)
    Ul, Ur = get_node_vars(ub, eq, 2, i - 1), get_node_vars(ub, eq, 1, i)
    Fn = num_flux(xf[i], ual, uar, Fl, Fr, Ul, Ur, eq, 1)
    set_node_vars!(Fb, Fn, eq, 2, i - 1)
    set_node_vars!(Fb, Fn, eq, 1, i)
end

@kernel function face_gather_kernel!(res, @Const(Fb), eq, bl, br, dx, dt)
    ix, cell = @index(Global, NTuple)
    fac = dt / dx[cell]
    Fl, Fr = get_node_vars(Fb, eq, 1, cell), get_node_vars(Fb, eq, 2, cell)
    multiply_add_to_node_vars!(res, fac * br[ix], Fr, fac * bl[ix], Fl, eq, ix, cell)
end

# Ub/Fb ghost values: `periodic`, `neumann`, `reflect` (`dirichlet` needs the
# user's `boundary_value(x, t)` called on the host, not yet supported here --
# see docs/GPU.md). All device-to-device, so a kernel rather than the shared
# `update_ghost_values_periodic!`'s `copyto!(..., CartesianIndices(...))` --
# that breaks under an OffsetArray-wrapped GPU array (falls back to scalar
# host indexing; verified directly against Metal, not just assumed). 2
# threads, one per side.
@kernel function face_ghost_kernel!(Ub, Fb, eq, nx, periodic_x, left_reflect,
                                    right_reflect)
    side = @index(Global)
    if side == 1
        u = periodic_x ? get_node_vars(Ub, eq, 2, nx) : get_node_vars(Ub, eq, 1, 1)
        f = periodic_x ? get_node_vars(Fb, eq, 2, nx) : get_node_vars(Fb, eq, 1, 1)
        if left_reflect
            u = SVector(u[1], -u[2], u[3])
            f = SVector(-f[1], f[2], -f[3])
        end
        set_node_vars!(Ub, u, eq, 2, 0)
        set_node_vars!(Fb, f, eq, 2, 0)
    else
        u = periodic_x ? get_node_vars(Ub, eq, 1, 1) : get_node_vars(Ub, eq, 2, nx)
        f = periodic_x ? get_node_vars(Fb, eq, 1, 1) : get_node_vars(Fb, eq, 2, nx)
        if right_reflect
            u = SVector(u[1], -u[2], u[3])
            f = SVector(-f[1], f[2], -f[3])
        end
        set_node_vars!(Ub, u, eq, 1, nx + 1)
        set_node_vars!(Fb, f, eq, 1, nx + 1)
    end
end

# Shared by the standalone `compute_cell_average!` dispatch below and
# `compute_residual_rkfr_gpu!`.
function compute_cell_average_gpu!(ua, u1, eq, grid, problem, op, backend)
    nx = grid.size
    left, right = problem.boundary_condition
    cell_average_kernel!(backend)(ua, u1, eq, op.wg; ndrange = nx)
    cell_average_ghost_kernel!(backend)(ua, eq, nx, problem.periodic_x, left == reflect,
                                        right == reflect; ndrange = 1)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# `compute_cell_average!`/`set_initial_condition!` are equation-agnostic (no
# `eq`-specific dispatch needed), so unlike `compute_time_step` (added per
# equation, e.g. EqEuler1D.jl) these new, more specific methods -- selected
# on `u1`'s array type -- can live here without waiting for any equation
# module to be defined.
function compute_cell_average!(ua, u1::OffsetArray{<:Any, <:Any, <:AbstractGPUArray}, t,
                               eq::AbstractEquations{1}, grid, problem, scheme, aux, op)
    compute_cell_average_gpu!(ua, u1, eq, grid, problem, op,
                              KernelAbstractions.get_backend(parent(u1)))
end

function set_initial_condition!(u::OffsetArray{<:Any, <:Any, <:AbstractGPUArray},
                                eq::AbstractEquations{1}, grid, op, problem)
    println("Setting initial condition")
    # Built on the host (arbitrary user `initial_value` function; this runs
    # once, not per timestep, so the transfer cost is negligible) and copied
    # to the device in one shot.
    @unpack initial_value = problem
    u_host = OffsetArray(zeros(eltype(u), size(u)...), u.offsets)
    nx, nd = grid.size, length(op.xg)
    for i in 1:nx, ii in 1:nd
        dx, xc = grid.dx[i], grid.xc[i]
        x = xc - dx / 2 + op.xg[ii] * dx
        u_host[:, ii, i] .= eltype(u).(initial_value(x))
    end
    copyto!(parent(u), parent(u_host))
    return nothing
end

"""
    compute_residual_rkfr_gpu!(eq, grid, op, problem, scheme, t, dt, cache)

GPU-kernel path for one RK-stage residual evaluation: cell average, cell
residual, boundary extrapolation, ghost values, face residual. Requires
`scheme.limiter.name == "none"`, `scheme.bound_limit == "no"`, and
`problem.boundary_condition` to be `periodic`, `neumann`, or `reflect` (not
`dirichlet` yet) -- see docs/GPU.md for what's ported so far.
"""
function compute_residual_rkfr_gpu!(eq, grid, op, problem, scheme, t, dt, cache)
    @unpack limiter, bound_limit, numerical_flux, bflux = scheme
    @assert limiter.name=="none" "GPU path only supports limiter = \"none\" so far"
    @assert bound_limit=="no" "GPU path only supports bound_limit = \"no\" so far"

    backend = cache.backend
    @unpack u1, ua, res, Fb, ub, xc_d, dx_d, xf_d = cache
    @unpack xg, wg, D1, Vl, Vr, bl, br = op
    nx = grid.size
    nd = length(xg)
    RealT = eltype(xg)

    fill!(res, zero(RealT))
    fill!(Fb, zero(RealT))
    fill!(ub, zero(RealT))

    left, right = problem.boundary_condition
    compute_cell_average_gpu!(ua, u1, eq, grid, problem, op, backend)

    cell_residual_kernel!(backend)(res, u1, eq, D1, xg, xc_d, dx_d, dt; ndrange = (nd, nx))
    bflux_evaluate = Val(bflux.bflux_ind == evaluate)
    boundary_extrapolate_kernel!(backend)(ub, Fb, u1, eq, Vl, Vr, xg, xc_d, dx_d,
                                          bflux_evaluate; ndrange = nx)
    KernelAbstractions.synchronize(backend)

    if !problem.periodic_x
        @assert left in (neumann, reflect) && right in (neumann, reflect) "GPU path doesn't support dirichlet yet"
    end
    face_ghost_kernel!(backend)(ub, Fb, eq, nx, problem.periodic_x, left == reflect,
                                right == reflect; ndrange = 2)
    KernelAbstractions.synchronize(backend)

    face_flux_kernel!(backend)(Fb, ua, ub, eq, xf_d, numerical_flux; ndrange = nx + 1)
    KernelAbstractions.synchronize(backend)
    face_gather_kernel!(backend)(res, Fb, eq, bl, br, dx_d, dt; ndrange = (nd, nx))
    KernelAbstractions.synchronize(backend)
    return nothing
end
