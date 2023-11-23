module EqEuler2D

( # Methods to be extended in this module
 # Extended methods are also marked with Tenkai. Example - Tenkai.flux
 import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
                eigmatrix,
                apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
                blending_flux_factors, zhang_shu_flux_fix,
                write_soln!, compute_time_step, post_process_soln,
                update_ghost_values_rkfr!, update_ghost_values_lwfr!)

(using Tenkai: get_filename, minmod, @threaded,
               periodic, dirichlet, neumann, reflect,
               update_ghost_values_fn_blend!,
               get_node_vars,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!, multiply_add_set_node_vars!,
               comp_wise_mutiply_node_vars!, AbstractEquations)

using Tenkai.CartesianGrids: CartesianGrid2D, save_mesh_file
using Tenkai.FR: limit_variable_slope

using Polyester
using StaticArrays
using LoopVectorization
using TimerOutputs
using UnPack
using WriteVTK
using LinearAlgebra
using MuladdMacro
using Printf
using EllipsisNotation
using HDF5: h5open, attributes
using Tenkai

using Tenkai.FR2D: correct_variable!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct Euler2D{HLLSpeeds <: Function} <: AbstractEquations{2, 4}
    γ::Float64
    γ_minus_1::Float64
    hll_speeds::HLLSpeeds
    nvar::Int64
    name::String
    initial_values::Dict{String, Function}
    numfluxes::Dict{String, Function}
end

#------------------------------------------------------------------------------

# Extending the flux function
@inline @inbounds function Tenkai.flux(x, y, U, eq::Euler2D, orientation::Integer)
    @unpack γ_minus_1 = eq
    ρ, ρ_u1, ρ_u2, ρ_e = U
    u1 = ρ_u1 / ρ
    u2 = ρ_u2 / ρ
    p = γ_minus_1 * (ρ_e - 0.5 * (ρ_u1 * u1 + ρ_u2 * u2))
    if orientation == 1
        F1 = ρ_u1
        F2 = ρ_u1 * u1 + p
        F3 = ρ_u1 * u2
        F4 = (ρ_e + p) * u1
        return SVector(F1, F2, F3, F4)
    else
        G1 = ρ_u2
        G2 = ρ_u2 * u1
        G3 = ρ_u2 * u2 + p
        G4 = (ρ_e + p) * u2
        return SVector(G1, G2, G3, G4)
    end
end

# Extending the flux function
@inline @inbounds function Tenkai.flux(x, y, U, eq::Euler2D)
    @unpack γ_minus_1 = eq
    ρ, ρ_u1, ρ_u2, ρ_e = U
    u1 = ρ_u1 / ρ
    u2 = ρ_u2 / ρ
    p = γ_minus_1 * (ρ_e - 0.5 * (ρ_u1 * u1 + ρ_u2 * u2))

    F1 = ρ_u1
    F2 = ρ_u1 * u1 + p
    F3 = ρ_u1 * u2
    F4 = (ρ_e + p) * u1
    F = SVector(F1, F2, F3, F4)

    G1 = ρ_u2
    G2 = ρ_u2 * u1
    G3 = ρ_u2 * u2 + p
    G4 = (ρ_e + p) * u2
    G = SVector(G1, G2, G3, G4)

    return F, G
end

# function converting primitive variables to PDE variables
function Tenkai.prim2con(eq::Euler2D, prim) # primitive, gas constant
    @unpack γ_minus_1 = eq
    ρ, v1, v2, p = prim
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    ρ_e = p / γ_minus_1 + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2)
    return SVector(ρ, ρ_v1, ρ_v2, ρ_e)
end

function Tenkai.prim2con!(eq::Euler2D, ua)
    @unpack γ, γ_minus_1 = eq
    ρ, v1, v2, p = ua
    ua[2] = ρ_v1 = ρ * v1
    ua[3] = ρ_v2 = ρ * v2
    ua[4] = p / γ_minus_1 + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2)
    return nothing
end

# function converting pde variables to primitive variables
function Tenkai.con2prim(eq::Euler2D, U)
    @unpack γ_minus_1 = eq
    ρ, ρ_u1, ρ_u2, ρ_e = U
    u1 = ρ_u1 / ρ
    u2 = ρ_u2 / ρ
    p = γ_minus_1 * (ρ_e - 0.5 * (ρ_u1 * u1 + ρ_u2 * u2))
    primitives = SVector(ρ, u1, u2, p)
    return primitives
end

function Tenkai.con2prim!(eq::Euler2D, ua, ua_)
    @unpack γ_minus_1 = eq
    ρ, ρ_u1, ρ_u2, ρ_e = ua
    u1, u2 = ρ_u1 / ρ, ρ_u2 / ρ
    p = γ_minus_1 * (ρ_e - 0.5 * (ρ_u1 * u1 + ρ_u2 * u2))
    ua_[1], ua_[2], ua_[3], ua_[4] = (ρ, u1, u2, p)
    return nothing
end

function Tenkai.con2prim!(eq::Euler2D, ua)
    @unpack γ_minus_1 = eq
    ρ, ρ_u1, ρ_u2, ρ_e = ua
    u1, u2 = ρ_u1 / ρ, ρ_u2 / ρ
    p = γ_minus_1 * (ρ_e - 0.5 * (ρ_u1 * u1 + ρ_u2 * u2))
    ua[1], ua[2], ua[3], ua[4] = (ρ, u1, u2, p)
    return nothing
end

@inline function get_density(eq::Euler2D, u::AbstractArray)
    ρ = u[1]
    return ρ
end

@inline function get_pressure(eq::Euler2D, u::AbstractArray)
    @unpack γ_minus_1 = eq
    ρ, ρ_v1, ρ_v2, ρ_e = u
    p = γ_minus_1 * (ρ_e - 0.5 * (ρ_v1 * ρ_v1 + ρ_v2 * ρ_v2) / ρ)
    return p
end

function Tenkai.is_admissible(eq::Euler2D, u::AbstractVector)
    ρ, vel_x, vel_y, p = con2prim(eq, u)
    if ρ > 1e-12 && p > 1e-12
        return true
    else
        @debug ρ, p
        return false
    end
end

#-------------------------------------------------------------------------------
# Scheme information
#-------------------------------------------------------------------------------
function Tenkai.compute_time_step(eq::Euler2D, grid, aux, op, cfl, u1, ua)
    @timeit aux.timer "Time Step computation" begin
    #! format: noindent
    @unpack dx, dy = grid
    nx, ny = grid.size
    @unpack γ = eq
    @unpack wg = op
    den = 0.0
    corners = ((0, 0), (nx + 1, 0), (0, ny + 1), (nx + 1, ny + 1))
    for element in CartesianIndices((0:(nx + 1), 0:(ny + 1)))
        el_x, el_y = element[1], element[2]
        if (el_x, el_y) ∈ corners # KLUDGE - Temporary hack
            continue
        end
        u_node = get_node_vars(ua, eq, el_x, el_y)
        rho, v1, v2, p = con2prim(eq, u_node)
        c = sqrt(γ * p / rho)
        sx, sy = abs(v1) + c, abs(v2) + c
        den = max(den, abs(sx) / dx[el_x] + abs(sy) / dy[el_y] + 1e-12)
        # Code for using polynomial values in place of cell average values
        # TOTHINK - Decide whether to keep them or not
        # for j in 1:nd, i in 1:nd
        #    u_node = get_node_vars(u1, eq, i, j, el_x, el_y)
        #    rho, v1, v2, p = con2prim(eq, u_node)
        #    c = sqrt(γ*p/rho)
        #    sx, sy = abs(v1) + c, abs(v2) + c
        #    den = max(den, abs(sx)/dx[el_x] + abs(sy)/dy[el_y] + 1e-12)
        #    # speed_x, speed_y = abs(sx)/dx[el_x], abs(sy)/dy[el_y]
        #    # λy = speed_y/(speed_x + speed_y + 1e-13)
        #    # λx = 1.0 - λy
        #    # w = min(wg[1], wg[nd])
        #    # den_blend = max(den_blend, abs(sx)/(dx[el_x]*λx*w) + 1e-12)
        #    # den_blend = max(den_blend, abs(sy)/(λy*dy[el_y]*w) + 1e-12)
        # end
    end

    dt = cfl / den
    return dt
    end # timer
end

function dwave(x, y)
    γ = 1.4 # RETHINK!
    ρ = 1.0 + 0.98 * sinpi(2.0 * (x + y))
    v1 = 0.1
    v2 = 0.2
    p = 1.0
    ρ_v1, ρ_v2 = ρ * v1, ρ * v2
    ρ_e = p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2)
    return SVector(ρ, ρ * v1, ρ * v2, ρ_e)
end

function constant_state(x, y)
    γ = 1.4 # RETHINK!
    ρ = 1.0
    v1 = 0.1
    v2 = 0.2
    p = 1.0
    ρ_v1, ρ_v2 = ρ * v1, ρ * v2
    ρ_e = p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2)
    return SVector(ρ, ρ * v1, ρ * v2, ρ_e)
end

function constant_state_exact(x, y, t)
    return constant_state(x, y)
end

function isentropic_iv(x, y)
    γ = 1.4
    β = 5.0
    M = 0.5
    α = 45.0 * (pi / 180.0)
    x0, y0 = 0.0, 0.0
    u0, v0 = M * cos(α), M * sin(α)
    r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0)

    a1 = 0.5 * β / π
    a2 = 0.5 * (γ - 1.0) * a1 * a1 / γ

    ρ = (1.0 - a2 * exp(1.0 - r2))^(1.0 / (γ - 1.0))
    v1 = u0 - a1 * (y - y0) * exp(0.5 * (1.0 - r2))
    v2 = v0 + a1 * (x - x0) * exp(0.5 * (1.0 - r2))
    p = ρ^γ
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

# initial_value_ref, final_time, ic_name = Eq.dwave_data
function isentropic_exact(x, y, t)
    xmin, xmax = -10.0, 10.0
    ymin, ymax = -10.0, 10.0
    Lx = xmax - xmin
    Ly = ymax - ymin
    theta = 45.0 * (pi / 180.0)
    M = 0.5
    u0 = M * cos(theta)
    v0 = M * sin(theta)
    q1, q2 = x - u0 * t, y - v0 * t
    if q1 > xmax
        q1 = q1 - Lx * floor((q1 + xmin) / Lx)
    elseif q1 < xmin
        q1 = q1 + Lx * floor((xmax - q1) / Lx)
    end
    if q2 > ymax
        q2 = q2 - Ly * floor((q2 + ymin) / Ly)
    elseif q2 < ymin
        q2 = q2 + Ly * floor((ymax - q2) / Ly)
    end

    return isentropic_iv(q1, q2)
end

zero_bv(x, t) = 0.0

@inline @inbounds function double_mach_reflection_bv(x::Real, y::Real, t::Real)
    γ = 1.4

    if x < 1.0 / 6.0 + (y + 20.0 * t) / sqrt(3.0)
        phi = pi / 6.0
        sin_phi, cos_phi = sincos(phi)
        ρ = 8.0
        v1 = 8.25 * cos_phi
        v2 = -8.25 * sin_phi
        p = 116.5
    else
        ρ = 1.4
        v1 = 0.0
        v2 = 0.0
        p = 1.0
    end
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector{4, Float64}(ρ, ρ_v1, ρ_v2,
                               p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

function isentropic_dumbser_iv(x, y)
    γ = 1.4
    rho_inf, u_inf, v_inf, p_inf, T_inf = 1.0, 1.0, 1.0, 1.0, 1.0
    x0, y0 = 5.0, 5.0
    ε = 5.0
    r_sqr = (x - x0) * (x - x0) + (y - y0) * (y - y0)
    δu, δv = (ε / (2.0 * π) * exp((1.0 - r_sqr) / 2.0),
              ε / (2.0 * π) * exp((1.0 - r_sqr) / 2.0))
    δu *= -(y - y0)
    δv *= x - x0
    δT = -((γ - 1.0) * ε * ε) / (8.0 * γ * π * π) * exp(1.0 - r_sqr)
    T = T_inf + δT
    ρ = T^(1.0 / (γ - 1.0))
    u = u_inf + δu
    v = v_inf + δv
    p = ρ^γ
    ρ_v1 = ρ * u
    ρ_v2 = ρ * v
    return SVector(ρ, ρ * u, ρ * v, p / (γ - 1.0) + 0.5 * (ρ_v1 * u + ρ_v2 * v))
end

# initial_value_ref, final_time, ic_name = Eq.dwave_data
isentropic_dumbser_exact(x, y, t) = isentropic_dumbser_iv(x, y)

isentropic_dumbser_data = (isentropic_dumbser_iv, isentropic_dumbser_exact, 10.0)

double_mach_reflection_iv(x, y) = double_mach_reflection_bv(x, y, 0.0)

function kh_iv(x, y)
    γ = 1.4

    slope = 15
    amplitude = 0.02
    B = tanh(slope * y + 7.5) - tanh(slope * y - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x)
    p = 1.0
    ρ_v1 = rho * v1
    ρ_v2 = rho * v2
    return SVector(rho, rho * v1, rho * v2,
                   p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

kevin_helmholtz_data = (kh_iv, (x, y, t) -> kh_iv(x, y))

function kh_iv_schaal(x, y)
    γ = 7.0 / 5.0
    w0 = 0.1
    sigma = 0.05 / sqrt(2.0)
    p = 2.5
    if 0.25 < y < 0.75
        rho = 2.0
        v1 = 0.5
    else
        rho = 1.0
        v1 = -0.5
    end
    v2 = w0 * sin(4.0 * pi * x)
    exp_term = exp(-(y - 0.25)^2 / (2.0 * sigma^2))
    exp_term += exp(-(y - 0.75)^2 / (2.0 * sigma^2))
    v2 *= exp_term
    ρ_v1 = rho * v1
    ρ_v2 = rho * v2
    return SVector(rho, rho * v1, rho * v2,
                   p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

kevin_helmholtz_schaal_data = (kh_iv_schaal, (x, y, t) -> kh_iv_schaal(x, y))

function sedov_iv(x, y)
    r = sqrt(x^2 + y^2)
    γ = 1.4

    v1 = v2 = 0.0
    σ_ρ = 0.25
    ρ0 = 1.0
    ρ = ρ0 + 0.25 / (π * σ_ρ^2) * exp(-0.5 * r^2 / σ_ρ^2)

    σ_p = 0.15
    p0 = 1.0e-5
    p = p0 + 0.25 * (γ - 1.0) / (π * σ_p^2) * exp(-0.5 * r^2 / (σ_p^2))

    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

sedov_data = (sedov_iv, (x, y, t) -> sedov_iv(x, y))

function shock_ref_iv(x, y)
    γ = 1.4

    ρ = 1.0
    v1 = 2.9
    v2 = 0.0
    p = 1.0 / γ
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

function shock_ref_bv(x, y, t)
    γ = 1.4

    if x < 1e-13 && y < 1.0
        ρ = 1.0
        v1 = 2.9
        v2 = 0.0
        p = 1.0 / γ
    else
        ρ = 1.69997
        v1 = 2.61934
        v2 = -0.50632
        p = 1.52819
    end
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

shock_ref_data = (shock_ref_iv, shock_ref_bv)

function shock_vortex_iv(x, y)
    γ = 1.4

    x0 = 0.5
    xc, yc = 0.25, 0.5
    beta = 0.204
    ep = 0.3
    rc = 0.05

    rhol = 1.0
    ul = sqrt(γ)
    vl = 0.0
    pl = 1.0

    pr = 1.3
    rhor = rhol * ((γ - 1.0) + (γ + 1.0) * pr) / ((γ + 1.0) + (γ - 1.0) * pr)
    ur = sqrt(γ) + sqrt(2.0) * ((1.0 - pr) / sqrt(γ - 1.0 + pr * (γ + 1.0)))
    vr = 0.0

    if x <= x0
        r = ((x - xc)^2 + (y - yc)^2) / (rc * rc)
        du = ep * ((y - yc) / rc) * exp(beta * (1.0 - r))
        dv = -ep * ((x - xc) / rc) * exp(beta * (1.0 - r))
        dtheta = -((γ - 1.0) / (4.0 * beta * γ)) * ep^2 * exp(2.0 * beta * (1.0 - r))
        ρ = (pl / rhol + dtheta)^(1.0 / (γ - 1.0))
        v1 = ul + du
        v2 = vl + dv
        p = (pl / rhol + dtheta)^(γ / (γ - 1.0))
    else
        ρ = rhor
        v1 = ur
        v2 = vr
        p = pr
    end
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

shock_vortex_data = (shock_vortex_iv, (x, y, t) -> shock_vortex_iv(x, y))

function taylor_green_exact_solution(x, y, t)
    γ = 1.4
    xmin, xmax = 0.0, 2.0 * π
    ymin, ymax = 0.0, 2.0 * π
    Lx = xmax - xmin
    Ly = ymax - ymin

    if x > xmax
        x = x - Lx * floor((x + xmin) / Lx)
    elseif x < xmin
        x = x + Lx * floor((xmax - x) / Lx)
    else
        x = x
    end
    if y > ymax
        y = y - Ly * floor((y + ymin) / Ly)
    elseif y < ymin
        y = y + Ly * floor((ymax - y) / Ly)
    else
        y = y
    end

    ρ = 1.0
    v1 = cos(x) * sin(y)
    v2 = -sin(x) * cos(y)
    p = 500.0 - 0.25 * (cos(2.0 * x) + cos(2.0 * y))

    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

taylor_green_data = ((x, y) -> taylor_green_exact_solution(x, y, 0.0),
                     taylor_green_exact_solution)

function vortex_convection_initial_value(x, y)
    γ = 1.4

    ρ_inf = 1.0
    M_inf = 0.1
    u_inf, v_inf = 1.0, 0.0
    pre_inf = ρ_inf * (u_inf / M_inf)^2 / γ
    Rc = 1.0
    C0 = 0.02 * u_inf * Rc
    x0, y0 = 0.0, 0.0

    r2 = ((x - x0)^2 + (y - y0)^2) / Rc^2
    v1 = u_inf - C0 * (y - y0) * exp(-0.5 * r2) / Rc^2
    v2 = v_inf + C0 * (x - x0) * exp(-0.5 * r2) / Rc^2
    ρ = ρ_inf
    p = pre_inf - 0.5 * ρ_inf * (C0 / Rc)^2 * exp(-r2)
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

function vortex_convection_exact_solution(x, y, t)
    xmin, xmax = -8.0, 8.0
    ymin, ymax = -8.0, 8.0
    Lx = xmax - xmin
    Ly = ymax - ymin
    u_inf, v_inf = 1.0, 0.0
    u0 = u_inf
    v0 = v_inf
    q1, q2 = x - u0 * t, y - v0 * t
    if q1 > xmax
        q1 = q1 - Lx * floor((q1 + xmin) / Lx)
    elseif q1 < xmin
        q1 = q1 + Lx * floor((xmax - q1) / Lx)
    end
    if q2 > ymax
        q2 = q2 - Ly * floor((q2 + ymin) / Ly)
    elseif q2 < ymin
        q2 = q2 + Ly * floor((ymax - q2) / Ly)
    end

    return vortex_convection_initial_value(q1, q2)
end

vortex_convection_data = (vortex_convection_initial_value,
                          vortex_convection_exact_solution)

dwave_data = (dwave, (x, y, t) -> dwave(x - 0.1 * t, y - 0.2 * t), 1.0, "dwave")

initial_values = Dict{String, Function}("dwave" => dwave)

function initial_value_sedov_zhang_shu(x, nx, ny)
    ρ = 1.0
    v1 = 0.0
    v2 = 0.0
    dx, dy = 1.0 / nx, 1.0 / ny
    if x[1] > dx || x[2] > dy
        E = 10^(-12)
    else
        E = 0.244816 / (dx * dy)
    end
    return SVector(ρ, ρ * v1, ρ * v2, E)
end

rp_datas = ([(1.0, 0.0, 0.0, 1.0), (0.5197, -0)], # 1
            [], # 2
            [], # 3
            [(1.1, 0, 0, 1.1), (0.5065, 0.8939, 0, 0.35),
                (1.1, 0.8939, 0.8939, 1.1), (0.5065, 0, 0.8939, 0.35)], # 4
            [(1, -0.75, -0.5, 1), (2, -0.75, 0.5, 1.0),
                (1, 0.75, 0.5, 1.0), (3, 0.75, -0.5, 1)], # 5
            [(1, -0.75, -0.5, 1), (2, -0.75, 0.5, 1.0),
                (1, 0.75, 0.5, 1.0), (3, 0.75, -0.5, 1)], # 6
            [], # 7
            [], # 8
            [], # 9
            [], # 10
            [], # 11
            [(0.5313, 0, 0, 0.4), (1.0, 0.7276, 0, 1.0),
                (0.8, 0, 0, 1.0), (1.0, 0, 0.7276, 1.0)], # 12
            [(1, 0, -0.3, 1), (2, 0, 0.3, 1),
                (1.0625, 0, 0.8145, 0.4), (0.5313, 0, 0.4276, 0.4)], # 13
            [])

function riemann_problem(x, y, eq::Euler2D, prim_ur, prim_ul, prim_dl, prim_dr)
    @unpack γ = eq
    if x >= 0.5 && y >= 0.5
        ρ, v1, v2, p = prim_ur
    elseif x <= 0.5 && y >= 0.5
        ρ, v1, v2, p = prim_ul
    elseif x <= 0.5 && y <= 0.5
        ρ, v1, v2, p = prim_dl
    elseif x >= 0.5 && y <= 0.5
        ρ, v1, v2, p = prim_dr
    end
    ρ_v1 = ρ * v1
    ρ_v2 = ρ * v2
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1.0) + 0.5 * (ρ_v1 * v1 + ρ_v2 * v2))
end

zs_nx = zs_ny = 160

initial_value_sedov_zhang_shu(x, y) = initial_value_sedov_zhang_shu((x, y), zs_nx,
                                                                    zs_ny)
exact_solution_sedov_zhang_shu(x, y, t) = initial_value_sedov_zhang_shu(x, y)

sedov2d_zhang_shu_data = (zs_nx, zs_ny,
                          initial_value_sedov_zhang_shu,
                          exact_solution_sedov_zhang_shu)

#-------------------------------------------------------------------------------
# Numerical Fluxes
#-------------------------------------------------------------------------------

function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler2D, dir)
    nvar = eq.nvar
    γ = eq.γ
    ρ_ll, v1_ll, v2_ll, p_ll = con2prim(eq, ual)
    ρ_rr, v1_rr, v2_rr, p_rr = con2prim(eq, uar)
    cl, cr = sqrt(γ * p_ll / ρ_ll), sqrt(γ * p_rr / ρ_rr)                   # sound speed
    if dir == 1
        λ = max(abs(v1_ll), abs(v1_rr)) + max(cl, cr)
        # λ = max(abs(v1_ll)+cl, abs(v1_rr+cr))
    else
        λ = max(abs(v2_ll), abs(v2_rr)) + max(cl, cr)
        # λ = max(abs(v2_ll)+cl, abs(v2_rr+cr))
    end
    F1 = 0.5 * (Fl[1] + Fr[1]) - 0.5 * λ * (Ur[1] - Ul[1])
    F2 = 0.5 * (Fl[2] + Fr[2]) - 0.5 * λ * (Ur[2] - Ul[2])
    F3 = 0.5 * (Fl[3] + Fr[3]) - 0.5 * λ * (Ur[3] - Ul[3])
    F4 = 0.5 * (Fl[4] + Fr[4]) - 0.5 * λ * (Ur[4] - Ul[4])
    return SVector(F1, F2, F3, F4)
end

# Taken from Trixi

@inline @inbounds function hll_speeds(ual, uar, dir, eq)
    # Calculate primitive variables and speed of sound
    @unpack γ_minus_1 = eq
    rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = ual
    rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = uar

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    p_ll = γ_minus_1 * (rho_e_ll - 1 / 2 * rho_ll * (v1_ll * v1_ll + v2_ll * v2_ll))
    c_ll = sqrt(eq.γ * p_ll / rho_ll)

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    p_rr = γ_minus_1 * (rho_e_rr - 1 / 2 * rho_rr * (v1_rr * v1_rr + v2_rr * v2_rr))
    c_rr = sqrt(eq.γ * p_rr / rho_rr)

    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
    if dir == 1 # x-direction
        vel_L = v1_ll
        vel_R = v1_rr
        ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2
    elseif dir == 2 # y-direction
        vel_L = v2_ll
        vel_R = v2_rr
        ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2
    end
    vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
    ekin_roe = 0.5 * (vel_roe * vel_roe + ekin_roe / (sum_sqrt_rho * sum_sqrt_rho))
    H_ll = (rho_e_ll + p_ll) / rho_ll
    H_rr = (rho_e_rr + p_rr) / rho_rr
    H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
    c_roe = sqrt(γ_minus_1 * (H_roe - ekin_roe))
    Ssl = min(vel_L - c_ll, vel_roe - c_roe)
    Ssr = max(vel_R + c_rr, vel_roe + c_roe)

    return Ssl, Ssr
end

@inline @inbounds function hllc(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler2D, dir::Integer)
    @unpack γ_minus_1 = eq
    # Calculate primitive variables and speed of sound
    Ssl, Ssr = hll_speeds(ual, uar, dir, eq)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = Ul
    rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = Ur
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    e_ll = rho_e_ll / rho_ll
    p_ll = γ_minus_1 * (rho_e_ll - 1 / 2 * rho_ll * (v1_ll * v1_ll + v2_ll * v2_ll))

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    e_rr = rho_e_rr / rho_rr
    p_rr = γ_minus_1 * (rho_e_rr - 1 / 2 * rho_rr * (v1_rr * v1_rr + v2_rr * v2_rr))

    # Obtain left and right fluxes
    if dir == 1 # x-direction
        vel_L = v1_ll
        vel_R = v1_rr
    elseif dir == 2 # y-direction
        vel_L = v2_ll
        vel_R = v2_rr
    end
    sMu_L = Ssl - vel_L
    sMu_R = Ssr - vel_R

    if Ssl >= 0.0
        f1 = Fl[1]
        f2 = Fl[2]
        f3 = Fl[3]
        f4 = Fl[4]
        return SVector(f1, f2, f3, f4)
    elseif Ssr <= 0.0
        f1 = Fr[1]
        f2 = Fr[2]
        f3 = Fr[3]
        f4 = Fr[4]
        return SVector(f1, f2, f3, f4)
    else
        SStar = (p_rr - p_ll + rho_ll * vel_L * sMu_L - rho_rr * vel_R * sMu_R) /
                (rho_ll * sMu_L - rho_rr * sMu_R)
        if Ssl <= 0.0 <= SStar
            densStar = rho_ll * sMu_L / (Ssl - SStar)
            enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
            UStar1 = densStar
            UStar4 = densStar * enerStar
            if dir == 1 # x-direction
                UStar2 = densStar * SStar
                UStar3 = densStar * v2_ll
            elseif dir == 2 # y-direction
                UStar2 = densStar * v1_ll
                UStar3 = densStar * SStar
            end
            f1 = Fl[1] + Ssl * (UStar1 - rho_ll)
            f2 = Fl[2] + Ssl * (UStar2 - rho_v1_ll)
            f3 = Fl[3] + Ssl * (UStar3 - rho_v2_ll)
            f4 = Fl[4] + Ssl * (UStar4 - rho_e_ll)
            return SVector(f1, f2, f3, f4)
        else
            densStar = rho_rr * sMu_R / (Ssr - SStar)
            enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
            UStar1 = densStar
            UStar4 = densStar * enerStar
            if dir == 1 # x-direction
                UStar2 = densStar * SStar
                UStar3 = densStar * v2_rr
            elseif dir == 2 # y-direction
                UStar2 = densStar * v1_rr
                UStar3 = densStar * SStar
            end
            f1 = Fr[1] + Ssr * (UStar1 - rho_rr)
            f2 = Fr[2] + Ssr * (UStar2 - rho_v1_rr)
            f3 = Fr[3] + Ssr * (UStar3 - rho_v2_rr)
            f4 = Fr[4] + Ssr * (UStar4 - rho_e_rr)
            return SVector(f1, f2, f3, f4)
        end
    end
end

@inline @inbounds function roe_avg(ual, uar, eq, dir)
    @unpack γ, γ_minus_1 = eq
    # Calculate primitive variables and speed of sound
    rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = ual
    rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = uar

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    p_ll = γ_minus_1 * (rho_e_ll - 1 / 2 * rho_ll * (v1_ll * v1_ll + v2_ll * v2_ll))
    c_ll = sqrt(γ * p_ll / rho_ll)

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    p_rr = γ_minus_1 * (rho_e_rr - 1 / 2 * rho_rr * (v1_rr * v1_rr + v2_rr * v2_rr))
    c_rr = sqrt(γ * p_rr / rho_rr)

    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    density_roe = sqrt_rho_ll * sqrt_rho_rr
    sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
    if dir == 1 # x-direction
        vel_L = v1_ll
        vel_R = v1_rr
        ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2
    elseif dir == 2 # y-direction
        vel_L = v2_ll
        vel_R = v2_rr
        ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2
    end
    vel_x_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr) / sum_sqrt_rho
    vel_y_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr) / sum_sqrt_rho
    vn_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
    ekin_roe = 0.5 * (vn_roe * vn_roe + ekin_roe / (sum_sqrt_rho * sum_sqrt_rho))
    H_ll = (rho_e_ll + p_ll) / rho_ll
    H_rr = (rho_e_rr + p_rr) / rho_rr
    H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
    c_roe = sqrt(γ_minus_1 * (H_roe - ekin_roe))

    return density_roe, vn_roe, vel_x_roe, vel_y_roe, H_roe, c_roe, ekin_roe
end

@inline @inbounds function eroe(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler2D, dir::Integer)
    # @inline @inbounds function eroe(x, Ul, Ur, Fl, Fr, ual, uar, eq::Euler2D, dir::Integer)

    @unpack γ = eq

    # Compute dissipation quantities

    (ρ_roe, vel_n, vel_roe_x, vel_roe_y, h, c, ekin_roe) = roe_avg(ual, uar, eq, dir)

    # Central quantities
    rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = Ul
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    e_ll = rho_e_ll / rho_ll
    p_ll = γ_minus_1 * (rho_e_ll - 0.5 * rho_ll * (v1_ll * v1_ll + v2_ll * v2_ll))
    ekin_l = v1_ll^2 + v2_ll^2

    rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = Ur
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    e_rr = rho_e_rr / rho_rr
    p_rr = γ_minus_1 * (rho_e_rr - 0.5 * rho_rr * (v1_rr * v1_rr + v2_rr * v2_rr))
    ekin_r = v1_rr^2 + v2_rr^2

    dv_x = v1_rr - v1_ll
    dv_y = v2_rr - v2_ll
    v_dot_dv = vel_roe_x * dv_x + vel_roe_y * dv_y

    if dir == 1
        vel_l_n = v1_ll
        vel_r_n = v1_rr
    else
        vel_l_n = v2_ll
        vel_r_n = v2_rr
    end

    H_ll = (rho_e_ll + p_ll) / rho_ll
    H_rr = (rho_e_rr + p_rr) / rho_rr

    drho = Ur[1] - Ul[1]
    dp = p_rr - p_ll
    dvn = vel_r_n - vel_l_n

    a1 = (dp - ρ_roe * c * dvn) / (2.0 * c * c)
    a2 = drho - dp / (c * c)
    a3 = (dp + ρ_roe * c * dvn) / (2.0 * c * c)

    l1 = abs(vel_n - c)
    l2 = abs(vel_n)
    l3 = abs(vel_n + c)

    delta = 0.1 * c
    if l1 < delta
        l1 = 0.5 * (l1 * l1 / delta + delta)
    end
    if l3 < delta
        l3 = 0.5 * (l3 * l3 / delta + delta)
    end

    # Dflux = Dissipative flux

    Dflux_rho = l1 * a1 + l2 * a2 + l3 * a3
    F1 = 0.5 * (Fl[1] + Fr[1] - Dflux_rho)

    Dflux_E = (l1 * a1 * (h - c * vel_n)
               + l2 * a2 * 0.5 * ekin_roe
               + l2 * ρ_roe * (v_dot_dv - vel_n * dvn)
               + l3 * a3 * (h + c * vel_n))
    F4 = 0.5 * (Fl[4] + Fr[4] - Dflux_E)
    if dir == 1
        normal_x, normal_y = 1.0, 0.0
    else
        normal_x, normal_y = 0.0, 1.0
    end

    p_avg = 0.5 * (p_ll + p_rr)
    Dflux_vel_x = ((vel_roe_x - normal_x * c) * l1 * a1
                   + vel_roe_x * l2 * a2
                   + (dv_x - normal_x * dvn) * l2 * ρ_roe
                   + (vel_roe_x + normal_x * c) * l3 * a3)
    F2 = 0.5 * (Fl[2] + Fr[2] - Dflux_vel_x)

    Dflux_vel_y = ((vel_roe_y - normal_y * c) * l1 * a1
                   + vel_roe_y * l2 * a2
                   + (dv_y - normal_y * dvn) * l2 * ρ_roe
                   + (vel_roe_y + normal_y * c) * l3 * a3)
    F3 = 0.5 * (Fl[3] + Fr[3] - Dflux_vel_y)

    Fn = SVector(F1, F2, F3, F4)
end

#------------------------------------------------------------------------------
# Limiters
#------------------------------------------------------------------------------
# Zhang-Shu limiting procedure for one variable
function Tenkai.apply_bound_limiter!(eq::Euler2D, grid, scheme, param, op, ua,
                                     u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end
    @unpack eps = param
    @timeit aux.timer "Bound limiter" begin
    #! format: noindent
    # variables = (get_density, get_pressure)
    # for variable in variables
    #    correct_variable!(eq, variable, op, aux, grid, u1, ua)
    # end # KLUDGE Fix the type instability and do it with a loop
    # https://github.com/trixi-framework/Trixi.jl/blob/0fd86e4bd856d894de6a7514edcb9758bf6f8e1e/src/callbacks_stage/positivity_zhang_shu.jl#L39   correct_variable!(eq, get_density,  op, aux, grid, u1, ua)
    correct_variable!(eq, get_density, op, aux, grid, u1, ua, eps)
    correct_variable!(eq, get_pressure, op, aux, grid, u1, ua, eps)
    return nothing
    end # timer
end

# Eigen matrices taken from
#  https://github.com/cpraveen/dflo/blob/918a6c54acaeac57f56bfb66f0e139ba70fd7a7f/src/equation.h#L218

function eigmatrix(eq::Euler2D, U)
    nvar = nvariables(eq)
    @unpack γ, γ_minus_1 = eq

    g1 = γ_minus_1
    rho = U[1]
    E = U[nvar]
    u = U[2] / rho
    v = U[3] / rho
    q2 = u * u + v * v
    p = g1 * (E - 0.5 * rho * q2)
    c2 = γ * p / rho
    c = sqrt(c2)
    beta = 0.5 / c2
    phi2 = 0.5 * g1 * q2
    h = c2 / g1 + 0.5 * q2

    M11 = 1
    M12 = 0
    M13 = 1
    M14 = 1
    M21 = u
    M22 = 0
    M23 = u + c
    M24 = u - c
    M31 = v
    M32 = -1
    M33 = v
    M34 = v
    M41 = 0.5 * q2
    M42 = -v
    M43 = h + c * u
    M44 = h - c * u

    Rx = SMatrix{nvariables(eq), nvariables(eq)}(M11, M21, M31, M41,
                                                 M12, M22, M32, M42,
                                                 M13, M23, M33, M43,
                                                 M14, M24, M34, M44)

    M11 = 1
    M12 = 0
    M13 = 1
    M14 = 1
    M21 = u
    M22 = 1
    M23 = u
    M24 = u
    M31 = v
    M32 = 0
    M33 = v + c
    M34 = v - c
    M41 = 0.5 * q2
    M42 = u
    M43 = h + c * v
    M44 = h - c * v

    Ry = SMatrix{nvariables(eq), nvariables(eq)}(M11, M21, M31, M41,
                                                 M12, M22, M32, M42,
                                                 M13, M23, M33, M43,
                                                 M14, M24, M34, M44)

    M11 = 1 - phi2 / c2
    M12 = g1 * u / c2
    M13 = g1 * v / c2
    M14 = -g1 / c2
    M21 = v
    M22 = 0
    M23 = -1
    M24 = 0
    M31 = beta * (phi2 - c * u)
    M32 = beta * (c - g1 * u)
    M33 = -beta * g1 * v
    M34 = beta * g1
    M41 = beta * (phi2 + c * u)
    M42 = -beta * (c + g1 * u)
    M43 = -beta * g1 * v
    M44 = beta * g1

    Lx = SMatrix{nvariables(eq), nvariables(eq)}(M11, M21, M31, M41,
                                                 M12, M22, M32, M42,
                                                 M13, M23, M33, M43,
                                                 M14, M24, M34, M44)

    M11 = 1 - phi2 / c2
    M12 = g1 * u / c2
    M13 = g1 * v / c2
    M14 = -g1 / c2
    M21 = -u
    M22 = 1
    M23 = 0
    M24 = 0
    M31 = beta * (phi2 - c * v)
    M32 = -beta * g1 * u
    M33 = beta * (c - g1 * v)
    M34 = beta * g1
    M41 = beta * (phi2 + c * v)
    M42 = -beta * g1 * u
    M43 = -beta * (c + g1 * v)
    M44 = beta * g1

    Ly = SMatrix{nvariables(eq), nvariables(eq)}(M11, M21, M31, M41,
                                                 M12, M22, M32, M42,
                                                 M13, M23, M33, M43,
                                                 M14, M24, M34, M44)
    return Lx, Ly, Rx, Ry
end

# dflo version

function minmod_β(a, b)
    s1, s2 = sign(a), sign(b)
    if (s1 != s2)
        return 0.0
    else
        slope = s1 * min(abs(a), abs(b))
        return slope
    end
end

function minmod_β(a, b, c, Mdx2)
    if abs(a) < Mdx2
        return a
    end
    s1, s2, s3 = sign(a), sign(b), sign(c)
    if (s1 != s2) || (s2 != s3)
        return 0.0
    else
        # slope = s1 * slope
        # return slope
        slope = s1 * min(abs(a), abs(b), abs(c))
        return slope
    end
end

function Tenkai.apply_tvb_limiterβ!(eq::Euler2D, problem, scheme, grid, param,
                                    op, ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack dx, dy = grid
    @unpack tvbM, cache, beta = scheme.limiter
    @unpack nvar = eq
    nd = length(wg)

    refresh!(u) = fill!(u, zero(eltype(u)))
    # Pre-allocate for each thread

    # Loop over cells
    @threaded for ij in CartesianIndices((1:nx, 1:ny))
        id = Threads.threadid()
        el_x, el_y = ij[1], ij[2]
        # face averages
        (ul, ur, ud, uu,
        dux, dur, duy, duu,
        dual, duar, duad, duau,
        char_dux, char_dur, char_duy, char_duu,
        char_dual, char_duar, char_duad, char_duau,
        duxm, durm, duym, duum) = cache[id]
        u1_ = @view u1[:, :, :, el_x, el_y]
        ua_, ual, uar, uad, uau = (get_node_vars(ua, eq, el_x, el_y),
                                   get_node_vars(ua, eq, el_x - 1, el_y),
                                   get_node_vars(ua, eq, el_x + 1, el_y),
                                   get_node_vars(ua, eq, el_x, el_y - 1),
                                   get_node_vars(ua, eq, el_x, el_y + 1))
        Lx, Ly, Rx, Ry = eigmatrix(eq, ua_)
        refresh!.((ul, ur, ud, uu))
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_ = get_node_vars(u1_, eq, i, j)
            multiply_add_to_node_vars!(ul, Vl[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ur, Vr[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ud, Vl[j] * wg[i], u_, eq, 1)
            multiply_add_to_node_vars!(uu, Vr[j] * wg[i], u_, eq, 1)
        end
        # KLUDGE - Give better names to these quantities
        # slopes b/w centres and faces
        ul_, ur_ = get_node_vars(ul, eq, 1), get_node_vars(ur, eq, 1)
        ud_, uu_ = get_node_vars(ud, eq, 1), get_node_vars(uu, eq, 1)
        ual_, uar_ = get_node_vars(ual, eq, 1), get_node_vars(uar, eq, 1)
        uad_, uau_ = get_node_vars(uad, eq, 1), get_node_vars(uau, eq, 1)

        multiply_add_set_node_vars!(dux, 1.0, ur_, -1.0, ul_, eq, 1)
        multiply_add_set_node_vars!(duy, 1.0, uu_, -1.0, ud_, eq, 1)

        multiply_add_set_node_vars!(dual, 1.0, ua_, -1.0, ual_, eq, 1)
        multiply_add_set_node_vars!(duar, 1.0, uar_, -1.0, ua_, eq, 1)
        multiply_add_set_node_vars!(duad, 1.0, ua_, -1.0, uad_, eq, 1)
        multiply_add_set_node_vars!(duau, 1.0, uau_, -1.0, ua_, eq, 1)

        dux_ = get_node_vars(dux, eq, 1)
        dual_, duar_ = get_node_vars(dual, eq, 1), get_node_vars(duar, eq, 1)
        duy_ = get_node_vars(duy, eq, 1)
        duad_, duau_ = get_node_vars(duad, eq, 1), get_node_vars(duau, eq, 1)

        # Convert to characteristic variables
        mul!(char_dux, Lx, dux_)
        mul!(char_dual, Lx, dual_)
        mul!(char_duar, Lx, duar_)
        mul!(char_duy, Ly, duy_)
        mul!(char_duad, Ly, duad_)
        mul!(char_duau, Ly, duau_)
        Mdx2, Mdy2 = tvbM * dx[el_x]^2, tvbM * dy[el_y]^2
        for n in Base.OneTo(nvar)
            duxm[n] = minmod_β(char_dux[n], beta * char_dual[n],
                               beta * char_duar[n], Mdx2)
            duym[n] = minmod_β(char_duy[n], beta * char_duad[n],
                               beta * char_duau[n], Mdy2)
        end

        jump_x = jump_y = 0.0
        duxm_ = get_node_vars(duxm, eq, 1)
        duym_ = get_node_vars(duym, eq, 1)
        for n in 1:nvar
            jump_x += abs(char_dux[n] - duxm_[n])
            jump_y += abs(char_duy[n] - duym_[n])
        end
        jump_x /= nvar
        jump_y /= nvar
        if jump_x + jump_y > 1e-10
            mul!(dux, Rx, duxm_)
            mul!(duy, Ry, duym_)
            dux_, duy_ = get_node_vars(dux, eq, 1), get_node_vars(duy, eq, 1)
            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                multiply_add_set_node_vars!(u1_,
                                            1.0, ua_,
                                            xg[i] - 0.5,
                                            dux_,
                                            xg[j] - 0.5,
                                            duy_,
                                            eq, i, j)
            end
        end
    end
    return nothing
    end # timer
end

function Tenkai.apply_tvb_limiter!(eq::Euler2D, problem, scheme, grid, param, op,
                                   ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack dx, dy = grid
    @unpack tvbM, cache, beta = scheme.limiter
    @unpack nvar = eq
    nd = length(wg)

    refresh!(u) = fill!(u, zero(eltype(u)))
    # Pre-allocate for each thread

    # Loop over cells
    @threaded for ij in CartesianIndices((1:nx, 1:ny))
        id = Threads.threadid()
        el_x, el_y = ij[1], ij[2]
        # face averages
        (ul, ur, ud, uu,
        dul, dur, dud, duu,
        dual, duar, duad, duau,
        char_dul, char_dur, char_dud, char_duu,
        char_dual, char_duar, char_duad, char_duau,
        dulm, durm, dudm, duum,
        duxm, duym, dux, duy) = cache[id]
        u1_ = @view u1[:, :, :, el_x, el_y]
        ua_, ual, uar, uad, uau = (get_node_vars(ua, eq, el_x, el_y),
                                   get_node_vars(ua, eq, el_x - 1, el_y),
                                   get_node_vars(ua, eq, el_x + 1, el_y),
                                   get_node_vars(ua, eq, el_x, el_y - 1),
                                   get_node_vars(ua, eq, el_x, el_y + 1))
        Lx, Ly, Rx, Ry = eigmatrix(eq, ua_)
        refresh!.((ul, ur, ud, uu))
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_ = get_node_vars(u1_, eq, i, j)
            multiply_add_to_node_vars!(ul, Vl[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ur, Vr[i] * wg[j], u_, eq, 1)
            multiply_add_to_node_vars!(ud, Vl[j] * wg[i], u_, eq, 1)
            multiply_add_to_node_vars!(uu, Vr[j] * wg[i], u_, eq, 1)
        end
        # KLUDGE - Give better names to these quantities
        # slopes b/w centres and faces
        ul_, ur_ = get_node_vars(ul, eq, 1), get_node_vars(ur, eq, 1)
        ud_, uu_ = get_node_vars(ud, eq, 1), get_node_vars(uu, eq, 1)
        ual_, uar_ = get_node_vars(ual, eq, 1), get_node_vars(uar, eq, 1)
        uad_, uau_ = get_node_vars(uad, eq, 1), get_node_vars(uau, eq, 1)

        multiply_add_set_node_vars!(dul, 1.0, ua_, -1.0, ul_, eq, 1)
        multiply_add_set_node_vars!(dur, 1.0, ur_, -1.0, ua_, eq, 1)
        multiply_add_set_node_vars!(dud, 1.0, ua_, -1.0, ud_, eq, 1)
        multiply_add_set_node_vars!(duu, 1.0, uu_, -1.0, ua_, eq, 1)

        multiply_add_set_node_vars!(dual, 1.0, ua_, -1.0, ual_, eq, 1)
        multiply_add_set_node_vars!(duar, 1.0, uar_, -1.0, ua_, eq, 1)
        multiply_add_set_node_vars!(duad, 1.0, ua_, -1.0, uad_, eq, 1)
        multiply_add_set_node_vars!(duau, 1.0, uau_, -1.0, ua_, eq, 1)

        dul_, dur_ = get_node_vars(dul, eq, 1), get_node_vars(dur, eq, 1)
        dual_, duar_ = get_node_vars(dual, eq, 1), get_node_vars(duar, eq, 1)
        dud_, duu_ = get_node_vars(dud, eq, 1), get_node_vars(duu, eq, 1)
        duad_, duau_ = get_node_vars(duad, eq, 1), get_node_vars(duau, eq, 1)

        # Convert to characteristic variables
        mul!(char_dul, Lx, dul_)
        mul!(char_dur, Lx, dur_)
        mul!(char_dual, Lx, dual_)
        mul!(char_duar, Lx, duar_)
        mul!(char_dud, Ly, dud_)
        mul!(char_duu, Ly, duu_)
        mul!(char_duad, Ly, duad_)
        mul!(char_duau, Ly, duau_)
        Mdx2, Mdy2 = tvbM * dx[el_x]^2, tvbM * dy[el_y]^2
        for n in Base.OneTo(nvar)
            dulm[n] = minmod(char_dul[n], char_dual[n], char_duar[n], Mdx2)
            durm[n] = minmod(char_dur[n], char_dual[n], char_duar[n], Mdx2)
            dudm[n] = minmod(char_dud[n], char_duad[n], char_duau[n], Mdy2)
            duum[n] = minmod(char_duu[n], char_duad[n], char_duau[n], Mdy2)
        end

        jump_x = jump_y = 0.0
        dulm_, durm_ = get_node_vars(dulm, eq, 1), get_node_vars(durm, eq, 1)
        dudm_, duum_ = get_node_vars(dudm, eq, 1), get_node_vars(duum, eq, 1)
        for n in 1:nvar
            jump_x += 0.5 *
                      (abs(char_dul[n] - dulm_[n]) + abs(char_dur[n] - durm_[n]))
            jump_y += 0.5 *
                      (abs(char_dud[n] - dudm_[n]) + abs(char_duu[n] - duum_[n]))
        end
        jump_x /= nvar
        jump_y /= nvar
        if jump_x + jump_y > 1e-10
            dulm_, durm_, dudm_, duum_ = (get_node_vars(dulm, eq, 1),
                                          get_node_vars(durm, eq, 1),
                                          get_node_vars(dudm, eq, 1),
                                          get_node_vars(duum, eq, 1))
            multiply_add_set_node_vars!(duxm,
                                        0.5, dulm_, 0.5, durm_,
                                        eq, 1)
            multiply_add_set_node_vars!(duym,
                                        0.5, dudm_, 0.5, duum_,
                                        eq, 1)
            duxm_, duym_ = get_node_vars(duxm, eq, 1), get_node_vars(duym, eq, 1)
            mul!(dux, Rx, duxm_)
            mul!(duy, Ry, duym_)
            dux_, duy_ = get_node_vars(dux, eq, 1), get_node_vars(duy, eq, 1)
            for j in Base.OneTo(nd), i in Base.OneTo(nd)
                multiply_add_set_node_vars!(u1_,
                                            1.0, ua_,
                                            2.0 * (xg[i] - 0.5),
                                            dux_,
                                            2.0 * (xg[j] - 0.5),
                                            duy_,
                                            eq, i, j)
            end
        end
    end
    return nothing
    end # timer
end

#------------------------------------------------------------------------------
# Blending limiter
#------------------------------------------------------------------------------

@inbounds @inline function primitive_indicator!(un, eq::Euler2D)
    @unpack γ_minus_1 = eq # Is this inefficient?
    nd_p2 = size(un, 2)
    for iy in 1:nd_p2, ix in 1:nd_p2 # loop over dofs and faces
        ρ, ρ_u1, ρ_u2, ρ_e = @view un[:, ix, iy]
        u1, u2 = ρ_u1 / ρ, ρ_u2 / ρ
        p = γ_minus_1 * (ρ_e - 0.5 * (ρ_u1 * u1 + ρ_u2 * u2))
        un[:, ix, iy] .= ρ, u1, u2, p
    end
    n_ind_var = nvar
    return n_ind_var
end

@inbounds @inline function rho_indicator!(ue, ::Euler2D)
    # Use only density as indicating variables
    n_ind_var = 1
    return n_ind_var
end

@inbounds @inline function rho_p_indicator!(un, eq::Euler2D)
    nd_p2 = size(un, 2) # nd + 2
    @unpack γ_minus_1 = eq # Is this inefficient?
    for iy in 1:nd_p2, ix in 1:nd_p2 # loop over dofs and faces
        ρ, ρ_v1, ρ_v2, ρ_e = @view un[:, ix, iy] # USE GET NODE VARS!!
        p = γ_minus_1 * (ρ_e - 0.5 * (ρ_v1^2 + ρ_v2^2) / ρ)
        un[1, ix, iy] = ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

@inbounds @inline function p_indicator!(un, ::Euler2D)
    @unpack γ_minus_1 = eq
    nd_p2 = size(un, 2)
    for iy in 1:nd_p2, ix in 1:nd_p2 # loop over dofs and faces
        ρ, ρ_v1, ρ_v2, ρ_e = @view un[:, ix, iy]
        p = γ_minus_1 * (ρ_e - 0.5 * (ρ_v1^2 + ρ_v2^2) / ρ)
        un[1, ix, iy] = p
    end
    n_ind_var = 1
    return n_ind_var
end

function Tenkai.blending_flux_factors(eq::Euler2D, ua, dx, dy)
    # This method is done differently for different equations

    # TODO - temporary hack. FIX!
    return 0.5, 0.5

    @unpack γ = eq
    rho, vel_x, vel_y, p = con2prim(eq, ua)
    c = sqrt(γ * p / rho)

    speed_x = (abs(vel_x) + c) / dx
    speed_y = (abs(vel_y) + c) / dy

    λx = speed_x / (speed_x + speed_y + 1e-13)
    λy = 1.0 - λx # not used

    return λx, λy
end

function Tenkai.limit_slope(eq::Euler2D, slope, ufl, u_star_ll, ufr, u_star_rr,
                            ue, xl, xr, el_x = nothing, el_y = nothing)

    # The MUSCL-Hancock scheme is guaranteed to be admissibility preserving if
    # slope is chosen so that
    # u_star_l = ue + 2.0*slope*xl, u_star_r = ue+2.0*slope*xr are admissible
    # ue is already admissible and we know we can find sequences of thetas
    # to make theta*u_star_l+(1-theta)*ue is admissible.
    # This is equivalent to replacing u_star_l by
    # u_star_l = ue + 2.0*theta*s*xl.
    # Thus, we simply have to update the slope by multiplying by theta.

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, get_density, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    slope, u_star_ll, u_star_rr = limit_variable_slope(eq, get_pressure, slope,
                                                       u_star_ll, u_star_rr, ue, xl, xr)

    ufl = ue + slope * xl
    ufr = ue + slope * xr

    return ufl, ufr, slope
end

function Tenkai.zhang_shu_flux_fix(eq::Euler2D,
                                   uprev,    # Solution at previous time level
                                   ulow,     # low order update
                                   Fn,       # Blended flux candidate
                                   fn_inner, # Inner part of flux
                                   fn,       # low order flux
                                   c)
    uhigh = uprev - c * (Fn - fn_inner) # First candidate for high order update
    ρ_low, ρ_high = get_density(eq, ulow), get_density(eq, uhigh)
    eps = 0.1 * ρ_low
    ratio = abs(eps - ρ_low) / (abs(ρ_high - ρ_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Second candidate for flux
    end
    uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
    p_low, p_high = get_pressure(eq, ulow), get_pressure(eq, uhigh)
    eps = 0.1 * p_low
    ratio = abs(eps - p_low) / (abs(p_high - p_low) + 1e-13)
    theta = min(ratio, 1.0)
    if theta < 1.0
        Fn = theta * Fn + (1.0 - theta) * fn # Final flux
    end
    return Fn
end

#------------------------------------------------------------------------------
# Ghost values functions
#------------------------------------------------------------------------------
function Tenkai.update_ghost_values_rkfr!(problem, scheme, eq::Euler2D, grid,
                                          aux, op, cache, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, ub = cache
    update_ghost_values_periodic!(eq, problem, Fb, ub)

    @unpack periodic_x, periodic_y = problem
    if periodic_x && periodic_y
        return nothing
    end

    nx, ny = grid.size
    @unpack degree, xg = op
    nd = degree + 1
    nvar = nvariables(eq)
    @unpack dx, dy, xf, yf = grid
    @unpack boundary_value, boundary_condition = problem
    left, right, bottom, top = boundary_condition

    if left == dirichlet
        x1 = xf[1]
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                y1 = yf[j] + xg[k] * dy[j]
                ub_value = boundary_value(x1, y1, t)
                set_node_vars!(ub, ub_value, eq, k, 2, 0, j)
                fb_value = flux(x1, y1, ub_value, eq, 1)
                set_node_vars!(Fb, fb_value, eq, k, 2, 0, j)

                # Purely upwind at boundary
                set_node_vars!(ub, ub_value, eq, k, 1, 1, j)
                set_node_vars!(Fb, fb_value, eq, k, 1, 1, j)
            end
        end
    elseif left in (neumann, reflect)
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                ub_node = get_node_vars(ub, eq, k, 1, 1, j)
                fb_node = get_node_vars(Fb, eq, k, 1, 1, j)
                set_node_vars!(ub, ub_node, eq, k, 2, 0, j)
                set_node_vars!(Fb, fb_node, eq, k, 2, 0, j)
                if left == reflect
                    ub[2, k, 2, 0, j] *= -1.0 # vel_x
                    Fb[1, k, 2, 0, j] *= -1.0 # ρ * vel_x
                    Fb[3, k, 2, 0, j] *= -1.0 # ρ * vel_x * vel_y
                    Fb[4, k, 2, 0, j] *= -1.0 # (ρ_e + p) * vel_x
                end
            end
        end
    else
        println("Incorrect bc specified at left.")
        @assert false
    end

    if right == dirichlet
        x2 = xf[nx + 1]
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                y2 = yf[j] + xg[k] * dy[j]
                ub_value = boundary_value(x2, y2, t)
                set_node_vars!(ub, ub_value, eq, k, 1, nx + 1, j)
                fb_value = flux(x2, y2, ub_value, eq, 1)
                set_node_vars!(Fb, fb_value, eq, k, 1, nx + 1, j)

                # Purely upwind
                # set_node_vars!(ub, ub_value, eq, k, 2, nx, j)
                # set_node_vars!(Fb, fb, eq, k, 2, nx, j)
            end
        end
    elseif right in (neumann, reflect)
        @threaded for j in 1:ny
            for k in 1:nd
                ub_node = get_node_vars(ub, eq, k, 2, nx, j)
                fb_node = get_node_vars(Fb, eq, k, 2, nx, j)
                set_node_vars!(ub, ub_node, eq, k, 1, nx + 1, j)
                set_node_vars!(Fb, fb_node, eq, k, 1, nx + 1, j)

                if right == reflect
                    ub[2, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[1, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[3, k, 1, nx + 1, j] *= -1.0 # ρ*u1*u2
                    Fb[4, k, 1, nx + 1, j] *= -1.0 # (ρ_e + p) * u1
                end
            end
        end
    else
        println("Incorrect bc specified at right.")
        @assert false
    end

    if bottom == dirichlet
        y3 = yf[1]
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                x3 = xf[i] + xg[k] * dx[i]
                ub_value = boundary_value(x3, y3, t)
                fb_value = flux(x3, y3, ub_value, eq, 2)
                set_node_vars!(ub, ub_value, eq, k, 4, i, 0)
                set_node_vars!(Fb, fb_value, eq, k, 4, i, 0)

                # Purely upwind

                # set_node_vars!(Ub, ub, eq, k, 3, i, 1)
                # set_node_vars!(Fb, fb, eq, k, 3, i, 1)
            end
        end
    elseif bottom in (neumann, reflect)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                ub_node = get_node_vars(ub, eq, k, 3, i, 1)
                fb_node = get_node_vars(Fb, eq, k, 3, i, 1)
                set_node_vars!(ub, ub_node, eq, k, 4, i, 0)
                set_node_vars!(Fb, fb_node, eq, k, 4, i, 0)
                if bottom == reflect
                    ub[3, k, 4, i, 0] *= -1.0 # ρ*vel_y
                    Fb[1, k, 4, i, 0] *= -1.0 # ρ*vel_y
                    Fb[2, k, 4, i, 0] *= -1.0 # ρ*vel_x*vel_y
                    Fb[4, k, 4, i, 0] *= -1.0 # (ρ_e + p) * vel_y
                end
            end
        end
    else
        @assert typeof(bottom)<:Tuple{Any, Any, Any} "$(typeof(bottom))"
        bc! = bottom[1]
        bc!(grid, eq, op, Fb, ub, aux)
    end

    if top == dirichlet
        y4 = yf[ny + 1]
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                x4 = xf[i] + xg[k] * dx[i]
                ub_value = boundary_value(x4, y4, t)
                fb_value = flux(x4, y4, ub_value, eq, 2)
                set_node_vars!(ub, ub_value, eq, k, 3, i, ny + 1)
                set_node_vars!(Fb, fb_value, eq, k, 3, i, ny + 1)

                # Purely upwind
                # set_node_vars!(Ub, ub, eq, k, 4, i, ny)
                # set_node_vars!(Fb, fb, eq, k, 4, i, ny)
            end
        end
    elseif top in (neumann, reflect)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                ub_node = get_node_vars(ub, eq, k, 4, i, ny)
                fb_node = get_node_vars(Fb, eq, k, 4, i, ny)
                set_node_vars!(ub, ub_node, eq, k, 3, i, ny + 1)
                set_node_vars!(Fb, fb_node, eq, k, 3, i, ny + 1)
                if top == reflect
                    ub[3, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
                    Fb[1, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
                    Fb[2, k, 3, i, ny + 1] *= -1.0 # ρ * vel_x * vel_y
                    Fb[4, k, 3, i, ny + 1] *= -1.0 # (ρ_e + p) * vel_y
                end
            end
        end
    else
        println("Incorrect bc specified at top")
        @assert false
    end

    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    return nothing
    end # timer
end

function Tenkai.update_ghost_values_lwfr!(problem, scheme, eq::Euler2D,
                                          grid, aux, op, cache, t, dt,
                                          scaling_factor = 1.0)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, Ub, ua = cache
    update_ghost_values_periodic!(eq, problem, Fb, Ub)

    @unpack periodic_x, periodic_y = problem
    if periodic_x && periodic_y
        return nothing
    end

    nx, ny = grid.size
    nvar = nvariables(eq)
    @unpack degree, xg, wg = op
    nd = degree + 1
    @unpack dx, dy, xf, yf = grid
    @unpack boundary_condition, boundary_value = problem
    left, right, bottom, top = boundary_condition

    refresh!(u) = fill!(u, zero(eltype(u)))

    pre_allocated = cache.ghost_cache

    # Julia bug occuring here. Below, we have unnecessarily named
    # x1,y1, x2, y2,.... We should have been able to just call them x,y
    # Otherwise we were getting a type instability and variables were
    # called Core.Box. This issue is probably related to
    # https://discourse.julialang.org/t/type-instability-of-nested-function/57007
    # https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
    # https://github.com/JuliaLang/julia/issues/15276
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured-1

    dt_scaled = scaling_factor * dt
    wg_scaled = scaling_factor * wg

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
    if left == dirichlet
        x1 = xf[1]
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                y1 = yf[j] + xg[k] * dy[j]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ub_value = problem.boundary_value(x1, y1, tq)
                    fb_value = flux(x1, y1, ub_value, eq, 1)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ub_value, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fb_value, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 2, 0, j)
                set_node_vars!(Fb, fb_node, eq, k, 2, 0, j)

                # # Put hllc flux values
                # Ul = get_node_vars(Ub, eq, k, 2, 0, j)
                # Fl = get_node_vars(Fb, eq, k, 2, 0, j)
                # Ur = get_node_vars(Ub, eq, k, 1, 1, j)
                # Fr = get_node_vars(Fb, eq, k, 1, 1, j)
                # ual, uar = get_node_vars(ua, eq, 0, j), get_node_vars(ua, eq, 1, j)
                # X = SVector{2}(x1, y1)
                # Fn = hllc(X, ual, uar, Fl, Fr, Ul, Ur, eq, 1)
                # set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
                # set_node_vars!(Fb, Fn     , eq, k, 1, 1, j)

                # Purely upwind at boundary
                # if abs(y1) < 0.055
                set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
                set_node_vars!(Fb, fb_node, eq, k, 1, 1, j)
                # end
            end
        end
    elseif left in (neumann, reflect)
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 1, 1, j)
                Fb_node = get_node_vars(Fb, eq, k, 1, 1, j)
                set_node_vars!(Ub, Ub_node, eq, k, 2, 0, j)
                set_node_vars!(Fb, Fb_node, eq, k, 2, 0, j)
                if left == reflect
                    Ub[2, k, 2, 0, j] *= -1.0 # ρ*u1
                    Fb[1, k, 2, 0, j] *= -1.0 # ρ*u1
                    Fb[3, k, 2, 0, j] *= -1.0 # ρ*u1*u2
                    Fb[4, k, 2, 0, j] *= -1.0 # (ρ_e + p) * u1
                end
            end
        end
    else
        println("Incorrect bc specified at left.")
        @assert false
    end

    if right == dirichlet
        x2 = xf[nx + 1]
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                y2 = yf[j] + xg[k] * dy[j]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x2, y2, tq)
                    fbvalue = flux(x2, y2, ubvalue, eq, 1)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 1, nx + 1, j)
                set_node_vars!(Fb, fb_node, eq, k, 1, nx + 1, j)

                # Purely upwind
                # set_node_vars!(Ub, ub_node, eq, k, 2, nx, j)
                # set_node_vars!(Fb, fb_node, eq, k, 2, nx, j)
            end
        end
    elseif right in (reflect, neumann)
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 2, nx, j)
                Fb_node = get_node_vars(Fb, eq, k, 2, nx, j)
                set_node_vars!(Ub, Ub_node, eq, k, 1, nx + 1, j)
                set_node_vars!(Fb, Fb_node, eq, k, 1, nx + 1, j)

                if right == reflect
                    Ub[2, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[1, k, 1, nx + 1, j] *= -1.0 # ρ*u1
                    Fb[3, k, 1, nx + 1, j] *= -1.0 # ρ*u1*u2
                    Fb[4, k, 1, nx + 1, j] *= -1.0 # (ρ_e + p) * u1
                end
            end
        end
    else
        println("Incorrect bc specified at right.")
        @assert false
    end

    if bottom == dirichlet
        y3 = yf[1]
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                x3 = xf[i] + xg[k] * dx[i]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x3, y3, tq)
                    fbvalue = flux(x3, y3, ubvalue, eq, 2)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 4, i, 0)
                set_node_vars!(Fb, fb_node, eq, k, 4, i, 0)

                # Purely upwind

                # set_node_vars!(Ub, ub, eq, k, 3, i, 1)
                # set_node_vars!(Fb, fb, eq, k, 3, i, 1)
            end
        end
    elseif bottom in (reflect, neumann)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 3, i, 1)
                Fb_node = get_node_vars(Fb, eq, k, 3, i, 1)
                set_node_vars!(Ub, Ub_node, eq, k, 4, i, 0)
                set_node_vars!(Fb, Fb_node, eq, k, 4, i, 0)
                if bottom == reflect
                    Ub[3, k, 4, i, 0] *= -1.0 # ρ * vel_y
                    Fb[1, k, 4, i, 0] *= -1.0 # ρ * vel_y
                    Fb[2, k, 4, i, 0] *= -1.0 # ρ * vel_x * vel_y
                    Fb[4, k, 4, i, 0] *= -1.0 # (ρ_e + p) * vel_y
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert typeof(bottom) <: Tuple{Any, Any, Any}
        bc! = bottom[1]
        bc!(grid, eq, op, Fb, Ub, aux)
    end

    if top == dirichlet
        y4 = yf[ny + 1]
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                x4 = xf[i] + xg[k] * dx[i]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x4, y4, tq)
                    fbvalue = flux(x4, y4, ubvalue, eq, 2)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
                end
                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 3, i, ny + 1)
                set_node_vars!(Fb, fb_node, eq, k, 3, i, ny + 1)

                # Purely upwind
                # set_node_vars!(Ub, ub_node, eq, k, 4, i, ny)
                # set_node_vars!(Fb, fb_node, eq, k, 4, i, ny)
            end
        end
    elseif top in (reflect, neumann)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(Ub, eq, k, 4, i, ny)
                Fb_node = get_node_vars(Fb, eq, k, 4, i, ny)
                set_node_vars!(Ub, Ub_node, eq, k, 3, i, ny + 1)
                set_node_vars!(Fb, Fb_node, eq, k, 3, i, ny + 1)
                if top == reflect
                    Ub[3, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
                    Fb[1, k, 3, i, ny + 1] *= -1.0 # ρ * vel_y
                    Fb[2, k, 3, i, ny + 1] *= -1.0 # ρ * vel_x * vel_y
                    Fb[4, k, 3, i, ny + 1] *= -1.0 # (ρ_e + p) * vel_y
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert false "Incorrect bc specific at top"
    end
    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    if scheme.limiter.name == "blend"
        update_ghost_values_fn_blend!(eq, problem, grid, aux)
    end

    return nothing
    end # timer
end

#------------------------------------------------------------------------------
# Boundary conditions
#------------------------------------------------------------------------------
## Double Mach reflection

# For Ub, Fb
function wall_double_mach_reflection_fb_ub!(grid, eq, op, Fb, Ub, aux)
    @unpack nvar = eq
    @unpack degree, xg = op
    @unpack dx, dy, xf = grid

    nx = grid.size[1]
    nd = degree + 1
    nvar = nvariables(eq)

    @threaded for i in 1:nx
        for k in Base.OneTo(nd)
            # Outflow upto [0,1/6]
            # KLUDGE - Don't do this, it's stupid.
            # This is only needed when handling non-linear
            # kernels. LoopVectorization is the right way for this
            Ub_node = get_node_vars(Ub, eq, k, 3, i, 1)
            Fb_node = get_node_vars(Fb, eq, k, 3, i, 1)
            set_node_vars!(Ub, Ub_node, eq, k, 4, i, 0)
            set_node_vars!(Fb, Fb_node, eq, k, 4, i, 0)

            x = xf[i] + xg[k] * dx[i]

            # Solid wall after 1/6
            if x >= 1.0 / 6.0
                Ub[3, k, 4, i, 0] *= -1.0 # ρ * vel_y
                Fb[1, k, 4, i, 0] *= -1.0 # ρ * vel_y
                Fb[2, k, 4, i, 0] *= -1.0 # ρ * vel_x * vel_y
                Fb[4, k, 4, i, 0] *= -1.0 # (ρ_e + p) * vel_y
            end
        end
    end
end

function wall_double_mach_reflection_ua!(grid, eq, ua)
    @unpack xf = grid

    nx = grid.size[1]

    for i in 1:nx
        ua_node = get_node_vars(ua, eq, i, 1)
        set_node_vars!(ua, ua_node, eq, i, 0)
        x = xf[i]
        if x >= 1.0 / 6.0
            ua[3, i, 0] = -ua[3, i, 1]
        end
    end
end

function wall_double_mach_reflection_u1!(op, grid, u1)
    @unpack xf, dx = grid
    nx, ny = grid.size
    @unpack degree, xg = op
    nd = degree + 1
    u1[:, :, :, :, 0] .= @view u1[:, :, :, :, 1]
    for i in 1:nx
        for ix in 1:nd
            x = xf[i] + ix * dx[i] * xg[ix]
            if x >= 1.0 / 6.0
                for jy in 1:nd
                    u1[3, ix, jy, i, 0] *= -1.0
                end
            end
        end
    end
end

# Put them in a tuple
double_mach_bottom = (wall_double_mach_reflection_fb_ub!,
                      wall_double_mach_reflection_ua!,
                      wall_double_mach_reflection_u1!)

# Perform HLLC upwinding in x flux of MUSCL-Hancock reconstruction
function hllc_upwinding_super_x(u1, eq, op, xf, y, jy, el_x, el_y, Fn)
    if el_x > 20
        return Fn
    else
        @unpack xg, Vl = op
        nd = length(xg)
        ul = get_node_vars(u1, eq, nd, jy, el_x - 1, el_y)
        # ur = get_node_vars(u1, eq, 1 , jy, el_x  , el_y)
        u = @view u1[:, :, jy, el_x, el_y]
        # ur = get_node_vars(Ub, eq, jy, 1, el_x, el_y)
        @views ur = SVector{nvariables(eq)}(dot(u[n, :], Vl) for n in eachvariable(eq))
        if !(Tenkai.is_admissible(eq, ur))
            ur = get_node_vars(u1, eq, 1, jy, el_x, el_y)
        end
        fl, fr = flux(xf, y, ul, eq, 1), flux(xf, y, ur, eq, 1)
        X = SVector(xf, y)
        # fn = fl
        fn = hllc(X, ul, ur, fl, fr, ul, ur, eq, 1)
        # Repetetition block
        # Fn2 = get_blended_flux_x(el_x, el_y, jy, eq, dt, grid,
        #                          blend, scheme, xf, y, u1, ua, fn2, Fn, op)
        return fn
    end
end

function hllc_upwinding_normal_x(u1, eq, op, xf, y, jy, el_x, el_y, Fn)
    if el_x > 2
        return Fn
    else
        @unpack xg, Vl = op
        nd = length(xg)
        ul = get_node_vars(u1, eq, nd, jy, el_x - 1, el_y)
        ur = get_node_vars(u1, eq, 1, jy, el_x, el_y)
        # u = @view u1[:,:,jy,el_x,el_y]
        # ur = get_node_vars(Ub, eq, jy, 1, el_x, el_y)
        # @views ur = SVector{nvariables(eq)}(dot(u[n,:], Vl) for n in eachvariable(eq))
        # if !(Tenkai.is_admissible(eq, ur))
        #    ur = get_node_vars(u1, eq, 1, jy, el_x, el_y)
        # end
        fl, fr = flux(xf, y, ul, eq, 1), flux(xf, y, ur, eq, 1)
        X = SVector(xf, y)
        # fn = fl
        fn = hllc(X, ul, ur, fl, fr, ul, ur, eq, 1)
        # Repetetition block
        # Fn2 = get_blended_flux_x(el_x, el_y, jy, eq, dt, grid,
        #                          blend, scheme, xf, y, u1, ua, fn2, Fn, op)
        return fn
    end
end

function hllc_upwinding_weak_x(u1, eq, op, xf, y, jy, el_x, el_y, Fn)
    if el_x > 1
        return Fn
    else
        @unpack xg, Vl = op
        nd = length(xg)
        ul = get_node_vars(u1, eq, nd, jy, el_x - 1, el_y)
        # ur = get_node_vars(u1, eq, 1 , jy, el_x  , el_y)
        u = @view u1[:, :, jy, el_x, el_y]
        # ur = get_node_vars(Ub, eq, jy, 1, el_x, el_y)
        @views ur = SVector{nvariables(eq)}(dot(u[n, :], Vl) for n in eachvariable(eq))
        if !(Tenkai.is_admissible(eq, ur))
            ur = get_node_vars(u1, eq, 1, jy, el_x, el_y)
        end
        fl, fr = flux(xf, y, ul, eq, 1), flux(xf, y, ur, eq, 1)
        X = SVector(xf, y)
        # fn = fl
        fn = hllc(X, ul, ur, fl, fr, ul, ur, eq, 1)
        # Repetetition block
        # Fn2 = get_blended_flux_x(el_x, el_y, jy, eq, dt, grid,
        #                          blend, scheme, xf, y, u1, ua, fn2, Fn, op)
        return fn
    end
end
#-------------------------------------------------------------------------------
# Write solution to a vtk file
#-------------------------------------------------------------------------------
function Tenkai.initialize_plot(eq::Euler2D, op, grid, problem, scheme, timer, u1, ua)
    return nothing
end

function write_poly(eq::Euler2D, grid, op, u1, fcount)
    filename = get_filename("output/sol", 3, fcount)
    @show filename
    @unpack xf, yf, dx, dy = grid
    nx, ny = grid.size
    @unpack degree, xg = op
    nd = degree + 1
    # Clear and re-create output directory

    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    Mx, My = nx * nu, ny * nu
    grid_x = zeros(Mx)
    grid_y = zeros(My)
    for i in 1:nx
        i_min = (i - 1) * nu + 1
        i_max = i_min + nu - 1
        # grid_x[i_min:i_max] .= LinRange(xf[i], xf[i+1], nu)
        grid_x[i_min:i_max] .= xf[i] .+ dx[i] * xg
    end

    for j in 1:ny
        j_min = (j - 1) * nu + 1
        j_max = j_min + nu - 1
        # grid_y[j_min:j_max] .= LinRange(yf[j], yf[j+1], nu)
        grid_y[j_min:j_max] .= yf[j] .+ dy[j] * xg
    end

    vtk_sol = vtk_grid(filename, grid_x, grid_y)

    u_equi = zeros(Mx, My)
    u = zeros(nu)
    for j in 1:ny
        for i in 1:nx
            # to get values in the equispaced thing
            for jy in 1:nd
                i_min = (i - 1) * nu + 1
                i_max = i_min + nu - 1
                u_ = @view u1[1, :, jy, i, j]
                mul!(u, Vu, u_)
                j_index = (j - 1) * nu + jy
                u_equi[i_min:i_max, j_index] .= @view u1[1, :, jy, i, j]
            end
        end
    end
    vtk_sol["sol"] = u_equi

    println("Wrote pointwise solution to $filename")

    out = vtk_save(vtk_sol)
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::Euler2D,
                            grid, problem, param, op,
                            z, u1, aux, ndigits = 3)
    @timeit aux.timer "Write solution" begin
    #! format: noindent
    @unpack final_time = problem
    # Clear and re-create output directory
    if fcount == 0
        run(`rm -rf output`)
        run(`mkdir output`)
        save_mesh_file(grid, "output")
    end

    nx, ny = grid.size
    @unpack exact_solution = problem
    exact(x) = exact_solution(x[1], x[2], time)
    @unpack xc, yc = grid
    filename = get_filename("output/avg", ndigits, fcount)
    # filename = string("output/", filename)
    vtk = vtk_grid(filename, xc, yc)
    xy = [[xc[i], yc[j]] for i in 1:nx, j in 1:ny]
    # KLUDGE - Do it efficiently
    prim = @views copy(z[:, 1:nx, 1:ny])
    exact_data = exact.(xy)
    for j in 1:ny, i in 1:nx
        @views con2prim!(eq, z[:, i, j], prim[:, i, j])
    end
    density_arr = prim[1, 1:nx, 1:ny]
    velx_arr = prim[2, 1:nx, 1:ny]
    vely_arr = prim[3, 1:nx, 1:ny]
    pres_arr = prim[4, 1:nx, 1:ny]
    vtk["sol"] = density_arr
    vtk["Density"] = density_arr
    vtk["Velocity_x"] = velx_arr
    vtk["Velocity_y"] = vely_arr
    vtk["Pressure"] = pres_arr
    for j in 1:ny, i in 1:nx
        @views con2prim!(eq, exact_data[i, j], prim[:, i, j])
    end
    @views vtk["Exact Density"] = prim[1, 1:nx, 1:ny]
    @views vtk["Exact Velocity_x"] = prim[2, 1:nx, 1:ny]
    @views vtk["Exact Velocity_y"] = prim[3, 1:nx, 1:ny]
    @views vtk["Exact Pressure"] = prim[4, 1:nx, 1:ny]
    # @views vtk["Exact Density"] = exact_data[1:nx,1:ny][1]
    # @views vtk["Exact Velocity_x"] = exact_data[1:nx,1:ny][2]
    # @views vtk["Exact Velocity_y"] = exact_data[1:nx,1:ny][3]
    # @views vtk["Exact Pressure"] = exact_data[1:nx,1:ny][4]
    vtk["CYCLE"] = iter
    vtk["TIME"] = time
    out = vtk_save(vtk)
    println("Wrote file ", out[1])
    write_poly(eq, grid, op, u1, fcount)
    if final_time - time < 1e-10
        cp("$filename.vtr", "./output/avg.vtr")
        println("Wrote final average solution to avg.vtr.")
    end

    fcount += 1

    # HDF5 file
    element_variables = Dict()
    element_variables[:density] = vec(density_arr)
    element_variables[:velocity_x] = vec(velx_arr)
    element_variables[:velocity_y] = vec(vely_arr)
    element_variables[:pressure] = vec(pres_arr)
    # element_variables[:indicator_shock_capturing] = vec(aux.blend.cache.alpha[1:nx,1:ny])
    filename = save_solution_file(u1, time, dt, iter, grid, eq, op,
                                  element_variables) # Save h5 file
    println("Wrote ", filename)
    return fcount
    end # timer
end

function save_solution_file(u_, time, dt, iter,
                            mesh,
                            equations, op,
                            element_variables = Dict{Symbol, Any}();
                            system = "")
    # Filename without extension based on current time step
    output_directory = "output"
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%06d.h5", iter))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%06d.h5", system, iter))
    end

    solution_variables(u) = con2prim(equations, u) # For broadcasting

    nx, ny = mesh.size
    u = @view u_[:, :, :, 1:nx, 1:ny] # Don't plot ghost cells

    # Convert to different set of variables if requested
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    # OffsetArray(reinterpret(eltype(ua), con2prim_.(reinterpret(SVector{nvariables(equation), eltype(ua)}, ua))))
    u_static_reinter = reinterpret(SVector{nvariables(equations), eltype(u)}, u)
    data = Array(reinterpret(eltype(u), solution_variables.(u_static_reinter)))

    # Find out variable count by looking at output from `solution_variables` function
    n_vars = size(data, 1)

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = 2
        attributes(file)["equations"] = "2D Euler Equations"
        attributes(file)["polydeg"] = op.degree
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nx * ny
        attributes(file)["mesh_type"] = "StructuredMesh" # For Trixi2Vtk
        attributes(file)["mesh_file"] = "mesh.h5"
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = iter

        # Store each variable of the solution data
        var_names = ("Density", "Velocity x", "Velocity y", "Pressure")
        for v in 1:n_vars
            # Convert to 1D array
            file["variables_$v"] = vec(data[v, .., :])

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = var_names[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            file["element_variables_$v"] = element_variable

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

function get_equation(γ; hll_wave_speeds = "toro")
    name = "2d Euler Equations"
    numfluxes = Dict{String, Function}()
    nvar = 4
    if hll_wave_speeds == "toro"
        hll_speeds(x) = nothing
    else
        println("Wave speed not implemented!")
        @assert false
    end
    fprime(x) = nothing
    return Euler2D(γ, γ - 1, hll_speeds, nvar, name, initial_values, numfluxes)
end

(export flux, update_ghost_values_rkfr!,
        update_ghost_values_lwfr!)
end # @muladd

end
