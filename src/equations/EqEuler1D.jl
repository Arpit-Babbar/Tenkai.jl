module EqEuler1D

import GZip
using DelimitedFiles
using Plots
using LinearAlgebra
using SimpleUnPack
using Printf
using TimerOutputs
using StaticArrays
using Polyester
using LoopVectorization
using JSON3

using Trixi: Trixi

using Tenkai
using Tenkai.Basis

import Tenkai: admissibility_tolerance

(import Tenkai: flux, prim2con, prim2con!, con2prim, con2prim!,
                eigmatrix,
                limit_slope, zhang_shu_flux_fix,
                apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln)

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars, sum_node_vars_1d,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!)

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct Euler1D{RealT <: Real, HLLSpeeds <: Function} <: AbstractEquations{1, 3}
    γ::RealT
    hll_speeds::HLLSpeeds
    nvar::Int64
    name::String
    initial_values::Dict{String, Function}
    numfluxes::Dict{String, Function}
end

function tenkai2trixiequation(equation::EqEuler1D.Euler1D)
    Trixi.CompressibleEulerEquations1D(equation.γ)
end

#-------------------------------------------------------------------------------
# PDE Information
#-------------------------------------------------------------------------------

@inbounds @inline function flux(x, U, eq::Euler1D)
    rho, rho_v1, rho_e = U
    v1 = rho_v1 / rho
    p = (eq.γ - 1) * (rho_e - 0.5 * rho_v1 * v1)
    # Ignore orientation since it is always "1" in 1D
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = (rho_e + p) * v1
    return SVector(f1, f2, f3)
end

@inbounds @inline flux(U, eq::Euler1D) = flux(1.0, U, eq)

# The matrix fprime(U)
function fprime(eq::Euler1D, x, U)
    @unpack γ = eq
    ρ = U[1]        # density
    u = U[2] / U[1] # velocity
    E = U[3]        # energy

    p = (γ - 1.0) * (E - 0.5 * ρ * u * u) # pressure

    H = (E + p) / ρ

    A = [0.0 1.0 0.0;
         0.5*(γ-3.0)*u*u (3.0 - γ)*u γ-1.0;
         u*(0.5 * (γ - 1.0) * u * u - H) H-(γ - 1.0) * u * u γ*u]
    return A
end

# function converting primitive variables to PDE variables
function prim2con(eq::Euler1D, prim) # primitive, gas constant
    @unpack γ = eq
    U = SVector(prim[1], prim[1] * prim[2],
                prim[3] / (γ - 1.0) + 0.5 * prim[1] * prim[2]^2)
    #           ρ    ,     ρ*u     ,        p/(γ-1.0) +     ρ*u^2/2.0
    return U
end

function prim2con!(eq::Euler1D, ua)
    @unpack γ = eq
    ua[3] = ua[3] / (γ - 1.0) + 0.5 * ua[1] * ua[2]^2
    ua[2] *= ua[1]
    return nothing
end

# function converting pde variables to primitive variables
@inbounds @inline function con2prim(eq::Euler1D, U)
    @unpack γ = eq
    primitives = SVector(U[1], U[2] / U[1], (γ - 1.0) * (U[3] - 0.5 * U[2]^2 / U[1]))
    #                   [ρ ,   u        , p]
    return primitives
end

function con2prim!(eq::Euler1D, ua, ua_)
    @unpack γ = eq
    ua_[1], ua_[2], ua_[3] = (ua[1], ua[2] / ua[1],
                              (γ - 1.0) * (ua[3]
                                           -
                                           0.5 * ua[2]^2 / ua[1]))
    return nothing
end

function con2prim!(eq::Euler1D, ua)
    @unpack γ = eq
    ua[3] = (γ - 1.0) * (ua[3] - 0.5 * ua[2]^2 / ua[1])
    ua[2] /= ua[1]
    return nothing
end

@inline function get_density(::Euler1D, u::AbstractArray)
    ρ = u[1]
    return ρ
end

@inline function get_pressure(eq::Euler1D, u::AbstractArray)
    @unpack γ = eq
    p = (γ - 1.0) * (u[3] - 0.5 * u[2]^2 / u[1])
    return p
end

function Tenkai.eigmatrix(eq::Euler1D, u)
    @unpack γ = eq
    g1 = γ - 1.0
    g2 = 0.5 * g1

    d = u[1]
    v = u[2] / d
    p = (γ - 1.0) * (u[3] - 0.5 * u[2]^2 / u[1])
    c = sqrt(γ * p / d)
    h = c^2 / g1 + 0.5 * v^2
    f = 0.5 * d / c

    # Inverse eigenvector-matrix
    L11 = 1.0 - g2 * v^2 / c^2
    L21 = (g2 * v^2 - v * c) / (d * c)
    L31 = -(g2 * v^2 + v * c) / (d * c)

    L12 = g1 * v / c^2
    L22 = (c - g1 * v) / (d * c)
    L32 = (c + g1 * v) / (d * c)

    L13 = -g1 / c^2
    L23 = g1 / (d * c)
    L33 = -g1 / (d * c)

    L = SMatrix{nvariables(eq), nvariables(eq)}(L11, L21, L31,
                                                L12, L22, L32,
                                                L13, L23, L33)

    # Eigenvector matrix
    R11 = 1.0
    R21 = v
    R31 = 0.5 * v^2

    R12 = f
    R22 = (v + c) * f
    R32 = (h + v * c) * f

    R13 = -f
    R23 = -(v - c) * f
    R33 = -(h - v * c) * f

    R = SMatrix{nvariables(eq), nvariables(eq)}(R11, R21, R31,
                                                R12, R22, R32,
                                                R13, R23, R33)

    return R, L
end

#-------------------------------------------------------------------------------
# Scheme information
#-------------------------------------------------------------------------------
function compute_time_step(eq::Euler1D, problem, grid, aux, op, cfl, u1, ua)
    nx = grid.size
    dx = grid.dx
    den = 0.0
    for i in 1:nx
        u = get_node_vars(ua, eq, i)
        rho, v, p = con2prim(eq, u)
        c = sqrt(eq.γ * p / rho)
        smax = abs(v) + c
        den = max(den, smax / dx[i])
    end
    dt = cfl / den
    return dt
end

#-------------------------------------------------------------------------------
# Initial Values
#-------------------------------------------------------------------------------
function blast(x)
    γ = 1.4
    if x <= 0.1
        rho = 1.0
        v = 0.0
        p = 1000.0
    elseif x > 0.1 && x <= 0.9
        rho = 1.0
        v = 0.0
        p = 0.01
    else
        rho = 1.0
        v = 0.0
        p = 100.0
    end
    U = [rho, rho * v, p / (γ - 1.0) + 0.5 * rho * v^2]
    return U
end

exact_blast(x, t) = blast(x)

function sedov_iv(x, dx)
    γ = 1.4
    rho = 1.0
    v = 0.0
    if abs(x) <= 0.5 * dx
        p = (γ - 1.0) * (3.2 * 10^6) / dx
    else
        p = (γ - 1.0) * 1e-12
    end
    U = [rho, rho * v, p / (γ - 1.0) + 0.5 * rho * v^2]
    return U
end

xmin_sedov, xmax_sedov = -2.0, 2.0
nx_sedov = 201
dx_sedov = (xmax_sedov - xmin_sedov) / 201

sedov1d(x) = sedov_iv(x, dx_sedov)

sedov_data = (xmin_sedov, xmax_sedov, nx_sedov, dx_sedov,
              sedov1d, (x, t) -> sedov_iv(x, dx_sedov))

function shuosher(x)
    γ = 1.4
    if x < -4.0
        rho = 3.857143
        v = 2.629369
        p = 10.333333
    else
        rho = 1.0 + 0.2 * sin(5.0 * x)
        v = 0.0
        p = 1.0
    end
    U = [rho, rho * v, p / (γ - 1.0) + 0.5 * rho * v^2]
    return U
end

exact_solution_shuosher(x, t) = shuosher(x)

function riemann_problem(ul, ur, xs, x)
    γ = 1.4
    if x < xs
        prim = ul
    else
        prim = ur
    end
    U = SVector(prim[1], prim[1] * prim[2],
                prim[3] / (γ - 1.0) + 0.5 * prim[1] * prim[2]^2)
    return U
end

# All Riemann problems
lax_iv(x) = riemann_problem([0.445, 0.698, 3.528], [0.5, 0.0, 0.571], 0.0, x)
lax_data = (lax_iv, (x, t) -> lax_iv(x), 1.3, "lax")

sod_iv(x) = riemann_problem([1.0, 0.0, 1.0], [0.125, 0.0, 0.1], 0.5, x)
sod_data = (sod_iv, (x, t) -> sod_iv(x), 0.2, "sod")

toro5_iv(x) = riemann_problem([1.0, -19.59745, 1000.0],
                              [1.0, -19.59745, 0.01], 0.8, x)
toro5_data = (toro5_iv, (x, t) -> toro5_iv(x), 0.012, "toro5")

double_rarefaction_iv(x) = riemann_problem((7.0, -1.0, 0.2),
                                           (7.0, 1.0, 0.2),
                                           0.0, x)
double_rarefaction_data = (double_rarefaction_iv,
                           (x, t) -> double_rarefaction_iv(x),
                           0.6, "double_rarefaction")

leblanc_iv(x) = riemann_problem((2.0, 0.0, 10.0^9),
                                (0.001, 0.0, 1.0),
                                0.0, x)

leblanc_data = (leblanc_iv, (x, t) -> leblanc_iv(x), 0.0001, "leblanc")

function dwave(x)
    γ = 1.4
    rho = 1.0 + 0.5 * sinpi(2.0 * x)
    vel = 1.0
    p = 1.0
    return [rho, rho * vel, p / (γ - 1.0) + 0.5 * rho * vel^2]
end

dwave_data = (dwave, (x, t) -> dwave(x - t), 1.0, "dwave")

dummy_zero_boundary_value(x, t) = 0.0

function initial_value_titarev_toro(x)
    γ = 1.4
    if -5.0 <= x <= -4.5
        rho, v, p = 1.515695, 0.523346, 1.805
    else
        rho, v, p = 1.0 + 0.1 * sinpi(20 * x), 0.0, 1.0
    end
    return SVector(rho, rho * v, p / (γ - 1.0) + 0.5 * rho * v^2)
end

function initial_value_larger_density(x)
    γ = 1.4
    if x < 0.3
        rho, v, p = 1000.0, 0.0, 1000.0
    else
        rho, v, p = 1.0, 0.0, 1.0
    end
    return SVector(rho, rho * v, p / (γ - 1.0) + 0.5 * rho * v^2)
end

initial_values = Dict{String, Function}()
for data in [lax_data, sod_data, toro5_data, dwave_data]
    initial_value, exact_solution, final_time, name = data
    initial_values[name] = initial_value
end
initial_values["sedov1d"] = sedov1d
initial_values["blast"], initial_values["shuosher"] = blast, shuosher
initial_values["double_rarefaction"] = double_rarefaction_iv
initial_values["leblanc"] = leblanc_iv
initial_values["titarev_toro"] = initial_value_titarev_toro
initial_values["larger_density"] = initial_value_larger_density
#-------------------------------------------------------------------------------
# Numerical Fluxes
#-------------------------------------------------------------------------------
@inline function ln_mean(x, y)
    epsilon_f2 = 1.0e-4
    f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y) # f2 = f^2
    if f2 < epsilon_f2
        return (x + y) / @evalpoly(f2, 2, 2/3, 2/5, 2/7)
    else
        return (y - x) / log(y / x)
    end
end

# Chandrashekar flux

function chandrashekar!(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler1D, dir, F)
    rho_ll, v1_ll, p_ll = eq.con2prim(Ul)
    rho_rr, v1_rr, p_rr = eq.con2prim(Ur)
    beta_ll = 0.5 * rho_ll / p_ll
    beta_rr = 0.5 * rho_rr / p_rr
    specific_kin_ll = 0.5 * (v1_ll^2)
    specific_kin_rr = 0.5 * (v1_rr^2)

    # Compute the necessary mean values
    rho_avg = 0.5 * (rho_ll + rho_rr)
    rho_mean = ln_mean(rho_ll, rho_rr)
    beta_mean = ln_mean(beta_ll, beta_rr)
    beta_avg = 0.5 * (beta_ll + beta_rr)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    p_mean = 0.5 * rho_avg / beta_avg
    velocity_square_avg = specific_kin_ll + specific_kin_rr

    # Calculate fluxes
    # Ignore orientation since it is always "1" in 1D
    F[1] = rho_mean * v1_avg
    F[2] = F[1] * v1_avg + p_mean
    F[3] = F[1] * 0.5 * (1 / (eq.γ - 1.0) / beta_mean - velocity_square_avg) +
           F[2] * v1_avg
    return nothing
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler1D, dir)
    @unpack γ = eq
    rho_ll, rho_v1_ll, rho_e_ll = ual
    rho_rr, rho_v1_rr, rho_e_rr = uar
    v1_ll = rho_v1_ll / rho_ll
    v_mag_ll = abs(v1_ll)
    p_ll = (γ - 1) * (rho_e_ll - 1 / 2 * rho_ll * v_mag_ll^2)
    c_ll = sqrt(γ * p_ll / rho_ll)
    v1_rr = rho_v1_rr / rho_rr
    v_mag_rr = abs(v1_rr)
    p_rr = (γ - 1) * (rho_e_rr - 1 / 2 * rho_rr * v_mag_rr^2)
    c_rr = sqrt(γ * p_rr / rho_rr)
    # ρl, ul, pl = con2prim(eq, ual)
    # ρr, ur, pr = con2prim(eq, uar)
    # cl, cr = sqrt(γ*pl/ρl), sqrt(γ*pr/ρr)                   # sound speed
    # λ = maximum(abs.([ul, ul-cl, ul+cl, ur, ur-cr, ur+cr])) # local wave speed

    λ = max(abs(v1_ll) + c_ll, abs(v1_rr) + c_rr) # local wave speed
    f1 = 0.5 * (Fl[1] + Fr[1]) - 0.5 * λ * (Ur[1] - Ul[1])
    f2 = 0.5 * (Fl[2] + Fr[2]) - 0.5 * λ * (Ur[2] - Ul[2])
    f3 = 0.5 * (Fl[3] + Fr[3]) - 0.5 * λ * (Ur[3] - Ul[3])
    return SVector(f1, f2, f3)
end

@inbounds @inline function rusanov_rk(x, Ul, Ur, eq::Euler1D)
    @unpack γ = eq
    rho_ll, rho_v1_ll, rho_e_ll = Ul
    rho_rr, rho_v1_rr, rho_e_rr = Ur

    Fl, Fr = flux(x, Ul, eq), flux(x, Ur, eq)

    v1_ll = rho_v1_ll / rho_ll
    v_mag_ll = abs(v1_ll)
    p_ll = (γ - 1) * (rho_e_ll - 1 / 2 * rho_ll * v_mag_ll^2)
    c_ll = sqrt(γ * p_ll / rho_ll)
    v1_rr = rho_v1_rr / rho_rr
    v_mag_rr = abs(v1_rr)
    p_rr = (γ - 1) * (rho_e_rr - 1 / 2 * rho_rr * v_mag_rr^2)
    c_rr = sqrt(γ * p_rr / rho_rr)
    # ρl, ul, pl = con2prim(eq, ual)
    # ρr, ur, pr = con2prim(eq, uar)
    # cl, cr = sqrt(γ*pl/ρl), sqrt(γ*pr/ρr)                   # sound speed
    # λ = maximum(abs.([ul, ul-cl, ul+cl, ur, ur-cr, ur+cr])) # local wave speed

    λ = max(abs(v1_ll) + c_ll, abs(v1_rr) + c_rr) # local wave speed
    f1 = 0.5 * (Fl[1] + Fr[1]) - 0.5 * λ * (Ur[1] - Ul[1])
    f2 = 0.5 * (Fl[2] + Fr[2]) - 0.5 * λ * (Ur[2] - Ul[2])
    f3 = 0.5 * (Fl[3] + Fr[3]) - 0.5 * λ * (Ur[3] - Ul[3])
    return SVector(f1, f2, f3)
end

# Roe's flux
function roe(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler1D, dir)
    γ = eq.γ
    ρl, ul, pl = con2prim(eq, ual)
    ρr, ur, pr = con2prim(eq, uar)
    sρl, sρr = sqrt(ρl), sqrt(ρr)            # pre-compute for efficiency
    Hl, Hr = γ * pl / ((γ - 1.0) * ρl) + 0.5 * ul^2,
             γ * pr / ((γ - 1.0) * ρr) + 0.5 * ur^2 # enthl
    u = (sρl * ul + sρr * ur) / (sρl + sρr)      # roe avg velocity
    H = (sρl * Hl + sρr * Hr) / (sρl + sρr)      # roe avg enthalpy
    c = sqrt((γ - 1.0) * (H - 0.5 * u * u))        # sound speed
    # Computing R |L| inv(R) ΔU efficiently
    dU1, dU2, dU3 = Ur[1] - Ul[1], Ur[2] - Ul[2], Ur[3] - Ul[3]
    α2 = (γ - 1.0) / c^2 * ((H - u * u) * dU1 + u * dU2 - dU3)
    α1 = 1.0 / (2.0 * c) * ((u + c) * dU1 - dU2 - c * α2)
    α3 = dU1 - α1 - α2
    l1, l2, l3 = abs(u - c), abs(u), abs(u + c)
    # Eigenvectors are as follows, but we don't store them to avoid allocations
    # r1,r2,r3 = [1.0, u-c, H-u*c ], [1.0, u, 0.5*u^2 ], [1.0, u+c, H+u*c ]
    # Flux is F = 0.5*(Fl+Fr) - 0.5(∑αi*li*ri)
    F1 = 0.5 * (Fl[1] + Fr[1]) - 0.5 * (α1 * l1 + α2 * l2 + α3 * l3)
    F2 = 0.5 * (Fl[2] + Fr[2]) -
         0.5 * (α1 * l1 * (u - c) + α2 * l2 * u
                + α3 * l3 * (u + c))
    F3 = 0.5 * (Fl[3] + Fr[3]) -
         0.5 * (α1 * l1 * (H - u * c) + 0.5 * α2 * l2 * u * u
                + α3 * l3 * (H + u * c))

    Fn = SVector(F1, F2, F3)
    return Fn
end

# Roe's flux with entropy correction
function eroe(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler1D, dir)
    γ = eq.γ
    ϵ = 0.1 # For entropy correction
    ρl, ul, pl = eq.con2prim(ual)
    ρr, ur, pr = eq.con2prim(uar)
    sρl, sρr = sqrt(ρl), sqrt(ρr)            # pre-compute for efficiency
    Hl, Hr = γ * pl / ((γ - 1.0) * ρl) + 0.5 * ul^2,
             γ * pr / ((γ - 1.0) * ρr) + 0.5 * ur^2 # enthl
    u = (sρl * ul + sρr * ur) / (sρl + sρr)      # roe avg velocity
    H = (sρl * Hl + sρr * Hr) / (sρl + sρr)      # roe avg enthalpy
    c = sqrt((γ - 1.0) * (H - 0.5 * u * u))        # sound speed
    # Computing R |L| inv(R) ΔU efficiently
    dU1, dU2, dU3 = Ur[1] - Ul[1], Ur[2] - Ul[2], Ur[3] - Ul[3]
    α2 = (γ - 1.0) / c^2 * ((H - u * u) * dU1 + u * dU2 - dU3)
    α1 = 1.0 / (2.0 * c) * ((u + c) * dU1 - dU2 - c * α2)
    α3 = dU1 - α1 - α2
    l1, l2, l3 = abs(u - c), abs(u), abs(u + c)
    δ = c * ϵ
    if abs(l1) < 2.0 * ϵ
        l1 = 0.5 * (l1^2 / δ + δ)
    end # entropy correction
    if abs(l3) < 2.0 * ϵ
        l3 = 0.5 * (l3^2 / δ + δ)
    end # entropy correction
    # Eigenvectors are follows, but we don't store them to avoid allocations
    # r1,r2,r3 = [1.0, u-c, H-u*c ], [1.0, u, 0.5*u^2 ], [1.0, u+c, H+u*c ]
    # We perform F = 0.5*(Fl+Fr) - 0.5(∑αi*li*ri) for eigenvectors ri's
    F1 = 0.5 * (Fl[1] + Fr[1]) - 0.5 * (α1 * l1 + α2 * l2 + α3 * l3)
    F2 = 0.5 * (Fl[2] + Fr[2]) -
         0.5 * (α1 * l1 * (u - c) + α2 * l2 * u
                + α3 * l3 * (u + c))
    F3 = 0.5 * (Fl[3] + Fr[3]) -
         0.5 * (α1 * l1 * (H - u * c) + 0.5 * α2 * l2 * u * u
                + α3 * l3 * (H + u * c))
    return SVector(F1, F2, F3)
end

# HLL/HLLC Wave speed estimates from Toro2009, DOI : 10.1007/b79761
function hll_speeds_toro(ual, uar, eq)
    γ = eq.γ
    ρl, ul, pl = con2prim(eq, ual)
    ρr, ur, pr = con2prim(eq, uar)
    cl, cr = sqrt(γ * pl / ρl), sqrt(γ * pr / ρr) # Sound speed
    ρa = 0.5 * (ρl + ρr) # Average density
    ca = 0.5 * (cl + cr)
    pstar = 0.5 * (pl + pr) - 0.5 * (ur - ul) * ρa * ca
    # vstar = 0.5*(ur + ul) - 0.5*(pr - pl)/(ρa*ca)
    if pstar < pr
        qr = 1.0
    else
        qr = sqrt(1.0 + ((γ + 1.0) / (2.0 * γ)) * (pstar / pr - 1.0))
    end

    if pstar < pl
        ql = 1.0
    else
        ql = sqrt(1.0 + ((γ + 1.0) / (2.0 * γ)) * (pstar / pl - 1.0))
    end

    sl = ul - cl * ql
    sr = ur + cr * qr
    return sl, sr
end

function hll!(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler1D, dir, F)
    nvar = eq.nvar
    sl, sr = eq.hll_speeds(ual, uar, eq)
    if sl > 0
        for n in 1:nvar
            F[n] = Fl[n]
        end
    elseif sr < 0
        for n in 1:nvar
            F[n] = Fr[n]
        end
    else
        for n in 1:nvar
            F[n] = (sr * Fl[n] - sl * Fr[n] + sl * sr * (Ur[n] - Ul[n])) / (sr - sl)
        end
    end
    return nothing
end

function hllc!(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler1D, dir, F)
    # Compute speeds using ual, uar
    nvar = eq.nvar
    sl, sr = eq.hll_speeds(ual, uar, eq)
    # Supersonic cases
    if sl > 0.0
        for n in 1:nvar
            F[n] = Fl[n]
        end
        return nothing
    elseif sr < 0.0
        for n in 1:nvar
            F[n] = Fr[n]
        end
        return nothing
    end
    # It is subsonic now, the intermediary states are given by
    # u⋆ = (Sr*Ur[2]-Fr[2])-(Sl*Ul[2]-Fl[2])/((Sr*Ur[1]-Fr[1])-(Sl*U[1]-Fl[1]))
    # p⋆ = ( (Sr*Ur[2]-Fr[2])(Sl*U[1]-Fl[1])-(Sl*Ul[2]-Fl[2])(Sr*Ur[1]-Fr[1]) )
    #     / ( (Sr*Ur[1]-Fr[1])-(Sl*U[1]-Fl[1]) )
    # E⋆l = (p⋆u⋆+Sl*El-Fl[3])/(Sl-u⋆)

    # Pre-compute coefficients for efficiency
    al1, al2, al3 = sl * Ul[1] - Fl[1], sl * Ul[2] - Fl[2], sl * Ul[3] - Fl[3]
    ar1, ar2, ar3 = sr * Ur[1] - Fr[1], sr * Ur[2] - Fr[2], sr * Ur[3] - Fr[3]
    # Intermediate velocity and pressure
    us = (ar2 - al2) / (ar1 - al1)
    ps = (ar2 * al1 - al2 * ar1) / (ar1 - al1)
    # Compute flux
    if us > 0.0
        dsl = sl - us
        rsl = al1 / dsl
        Esl = (ps * us + al3) / dsl
        F[1] = Fl[1] + sl * (rsl - Ul[1])
        F[2] = Fl[2] + sl * (rsl * us - Ul[2])
        F[3] = Fl[3] + sl * (Esl - Ul[3])
    else
        dsr = sr - us
        rsr = ar1 / dsr
        Esr = (ps * us + ar3) / dsr
        F[1] = Fr[1] + sr * (rsr - Ur[1])
        F[2] = Fr[2] + sr * (rsr * us - Ur[2])
        F[3] = Fr[3] + sr * (Esr - Ur[3])
    end
    return nothing
end

function hllc(x, ual, uar, Fl, Fr, Ul, Ur, eq::Euler1D, dir)
    # Compute speeds using ual, uar
    sl, sr = hll_speeds_toro(ual, uar, eq)
    # Supersonic cases
    if sl > 0.0
        f1, f2, f3 = Fl[1], Fl[2], Fl[3]
        return SVector(f1, f2, f3)
    elseif sr < 0.0
        f1, f2, f3 = Fr[1], Fr[2], Fr[3]
        return SVector(f1, f2, f3)
    end
    # It is subsonic now, the intermediary states are given by
    # u⋆ = (Sr*Ur[2]-Fr[2])-(Sl*Ul[2]-Fl[2])/((Sr*Ur[1]-Fr[1])-(Sl*U[1]-Fl[1]))
    # p⋆ = ( (Sr*Ur[2]-Fr[2])(Sl*U[1]-Fl[1])-(Sl*Ul[2]-Fl[2])(Sr*Ur[1]-Fr[1]) )
    #     / ( (Sr*Ur[1]-Fr[1])-(Sl*U[1]-Fl[1]) )
    # E⋆l = (p⋆u⋆+Sl*El-Fl[3])/(Sl-u⋆)

    # Pre-compute coefficients for efficiency
    al1, al2, al3 = sl * Ul[1] - Fl[1], sl * Ul[2] - Fl[2], sl * Ul[3] - Fl[3]
    ar1, ar2, ar3 = sr * Ur[1] - Fr[1], sr * Ur[2] - Fr[2], sr * Ur[3] - Fr[3]
    # Intermediate velocity and pressure
    us = (ar2 - al2) / (ar1 - al1)
    ps = (ar2 * al1 - al2 * ar1) / (ar1 - al1)
    # Compute flux
    if us > 0.0
        dsl = sl - us
        rsl = al1 / dsl
        Esl = (ps * us + al3) / dsl
        f1 = Fl[1] + sl * (rsl - Ul[1])
        f2 = Fl[2] + sl * (rsl * us - Ul[2])
        f3 = Fl[3] + sl * (Esl - Ul[3])
        return SVector(f1, f2, f3)
    else
        dsr = sr - us
        rsr = ar1 / dsr
        Esr = (ps * us + ar3) / dsr
        f1 = Fr[1] + sr * (rsr - Ur[1])
        f2 = Fr[2] + sr * (rsr * us - Ur[2])
        f3 = Fr[3] + sr * (Esr - Ur[3])
        return SVector(f1, f2, f3)
    end
end

#-------------------------------------------------------------------------------
# Limiters
#-------------------------------------------------------------------------------
function Tenkai.apply_bound_limiter!(eq::Euler1D, grid, scheme, param, op, ua,
                                     u1, aux)
    if scheme.bound_limit == "no"
        return nothing
    end

    @timeit aux.timer "Positivity limiter" begin
    #! format: noindent
    @unpack Vl, Vr = op
    nx = grid.size
    @unpack γ = eq
    nd = op.degree + 1

    variables = (get_density, get_pressure)

    # Looping over tuple of functions like this is only type stable if
    # there are only two. For tuples with than two functions, see
    # https://github.com/trixi-framework/Trixi.jl/blob/0fd86e4bd856d894de6a7514edcb9758bf6f8e1e/src/callbacks_stage/positivity_zhang_shu.jl#L39

    # Find a minimum for all variables
    eps = 1e-10
    for variable in variables
        for i in 1:nx
            ua_ = get_node_vars(ua, eq, i)
            var = variable(eq, ua_)
            eps = min(eps, var)
        end
        if eps < 0.0
            println("Fatal: Negative states in cell averages")
            @show variable
            println("       minimum cell average = $eps")
            throw(DomainError(eps, "Positivity limiter failed"))
        end
    end

    for variable in variables
        for element in 1:nx
            var_ll = var_rr = 0.0
            var_min = 1e20
            for i in Base.OneTo(nd)
                u_node = get_node_vars(u1, eq, i, element)
                var = variable(eq, u_node)
                var_ll += var * Vl[i]
                var_rr += var * Vr[i]
                var_min = min(var_min, var)
            end
            var_min = min(var_min, var_ll, var_rr)

            # In order to correct the solution at the faces, we need to extrapolate it to faces
            # and then correct it.
            ul = sum_node_vars_1d(Vl, u1, eq, 1:nd, element) # ul = ∑ Vl*u
            ur = sum_node_vars_1d(Vr, u1, eq, 1:nd, element) # ur = ∑ Vr*u
            var_u_ll, var_u_rr = variable(eq, ul), variable(eq, ur)
            var_min = min(var_min, var_u_ll, var_u_rr)
            ua_ = get_node_vars(ua, eq, element)
            var_avg = variable(eq, ua_)
            ratio = abs(eps - var_avg) / (abs(var_min - var_avg) + 1e-13)
            theta = min(ratio, 1.0) # theta for preserving positivity of density
            if theta < 1.0
                for i in 1:nd
                    u_node = get_node_vars(u1, eq, i, element)
                    multiply_add_set_node_vars!(u1,
                                                theta, u_node,
                                                1 - theta, ua_,
                                                eq, i, element)
                end
            end
        end
    end
    end # timer
end

function Tenkai.apply_tvb_limiter!(eq::Euler1D, problem, scheme, grid, param, op, ua,
                                   u1, aux)
    @timeit aux.timer "TVB limiter" begin
    #! format: noindent
    nx = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack limiter = scheme
    @unpack tvbM, cache = limiter
    left_bc, right_bc = problem.boundary_condition
    nd = length(wg)
    nvar = nvariables(eq)
    # face values
    (uimh, uiph, Δul, Δur, Δual, Δuar, char_Δul, char_Δur, char_Δual, char_Δuar,
    dulm, durm, du) = cache

    # Loop over cells
    for cell in 1:nx
        ual, ua_, uar = (get_node_vars(ua, eq, cell - 1),
                         get_node_vars(ua, eq, cell),
                         get_node_vars(ua, eq, cell + 1))
        R, L = eigmatrix(eq, ua_)
        fill!(uimh, zero(eltype(uimh)))
        fill!(uiph, zero(eltype(uiph)))
        Mdx2 = tvbM * grid.dx[cell]^2
        if left_bc == neumann && right_bc == neumann && (cell == 1 || cell == nx)
            Mdx2 = 0.0 # Force TVD on boundary for Shu-Osher
        end
        # end # timer
        for ii in 1:nd
            u_ = get_node_vars(u1, eq, ii, cell)
            multiply_add_to_node_vars!(uimh, Vl[ii], u_, eq, 1)
            multiply_add_to_node_vars!(uiph, Vr[ii], u_, eq, 1)
        end
        # Get views of needed cell averages
        # slopes b/w centres and faces

        uimh_ = get_node_vars(uimh, eq, 1)
        uiph_ = get_node_vars(uiph, eq, 1)

        # We will set
        # Δul[n] = ua_[n] - uimh[n]
        # Δur[n] = uiph[n] - ua_[n]
        # Δual[n] = ua_[n] - ual[n]
        # Δuar[n] = uar[n] - ua_[n]

        set_node_vars!(Δul, ua_, eq, 1)
        set_node_vars!(Δur, uiph_, eq, 1)
        set_node_vars!(Δual, ua_, eq, 1)
        set_node_vars!(Δuar, uar, eq, 1)

        subtract_from_node_vars!(Δul, uimh_, eq)
        subtract_from_node_vars!(Δur, ua_, eq)
        subtract_from_node_vars!(Δual, ual, eq)
        subtract_from_node_vars!(Δuar, ua_, eq)

        Δul_ = get_node_vars(Δul, eq, 1)
        Δur_ = get_node_vars(Δur, eq, 1)
        Δual_ = get_node_vars(Δual, eq, 1)
        Δuar_ = get_node_vars(Δuar, eq, 1)
        mul!(char_Δul, L, Δul_)   # char_Δul = L*Δul
        mul!(char_Δur, L, Δur_)   # char_Δur = L*Δur
        mul!(char_Δual, L, Δual_) # char_Δual = L*Δual
        mul!(char_Δuar, L, Δuar_) # char_Δuar = L*Δuar

        char_Δul_ = get_node_vars(char_Δul, eq, 1)
        char_Δur_ = get_node_vars(char_Δur, eq, 1)
        char_Δual_ = get_node_vars(char_Δual, eq, 1)
        char_Δuar_ = get_node_vars(char_Δuar, eq, 1)
        for n in eachvariable(eq)
            dulm[n] = minmod(char_Δul_[n], char_Δual_[n], char_Δuar_[n], Mdx2)
            durm[n] = minmod(char_Δur_[n], char_Δual_[n], char_Δuar_[n], Mdx2)
        end

        # limit if jumps are detected
        dulm_ = get_node_vars(dulm, eq, 1)
        durm_ = get_node_vars(durm, eq, 1)
        jump_l = jump_r = 0.0
        for n in 1:nvar
            jump_l += abs(char_Δul_[n] - dulm_[n])
            jump_r += abs(char_Δur_[n] - durm_[n])
        end
        jump_l /= nvar
        jump_r /= nvar

        if jump_l > 1e-06 || jump_r > 1e-06
            add_to_node_vars!(durm, dulm_, eq, 1) # durm = durm + dulm
            # We want durm = 0.5 * (dul + dur), we adjust 0.5 later
            mul!(du, R, durm)            # du = R * (dulm+durm)
            for ii in Base.OneTo(nd)
                du_ = get_node_vars(du, eq, 1)
                set_node_vars!(u1, ua_ + (xg[ii] - 0.5) * du_, # 2.0 adjusted with 0.5 above
                               eq, ii,
                               cell)
            end
        end
    end
    return nothing
    end # timer
end

#-------------------------------------------------------------------------------
# Blending Limiter
#-------------------------------------------------------------------------------
@inbounds @inline function primitive_indicator!(un, eq::Euler1D)
    for ix in 1:size(un, 2) # loop over dofs and faces
        un[3, ix] = eq.γ * (un[3, ix] - 0.5 * un[2, ix]^2 / un[1, ix])
        un[2, ix] /= un[1, ix]
    end
    n_ind_var = nvariables(eq)
    return n_ind_var
end

@inbounds @inline function rho_indicator!(ue, eq::Euler1D)
    # Use only density as indicating variables
    n_ind_var = 1
    return n_ind_var
end

@inbounds @inline function rho_p_indicator!(un, eq::Euler1D)
    for ix in 1:size(un, 2) # loop over dofs and faces
        p = (eq.γ - 1.0) * (un[3, ix] - 0.5 * un[2, ix]^2 / un[1, ix])
        un[1, ix] *= p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

@inbounds @inline function p_indicator!(un, eq::Euler1D)
    for ix in 1:size(un, 2) # loop over dofs and faces
        p = (eq.γ - 1.0) * (un[3, ix] - 0.5 * un[2, ix]^2 / un[1, ix])
        un[1, ix] = p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

function Tenkai.is_admissible(eq::Euler1D, u::AbstractVector)
    prim = con2prim(eq, u)
    if prim[1] > 1e-10 && prim[3] > 1e-10
        return true
    else
        return false
    end
end

function Tenkai.conservative2characteristic_reconstruction!(mode, ua, eq::Euler1D)
    # R, L = eigmatrix(eq, ua)
    # temp = L * mode
    # mode .= temp
    @assert false "not implemented"
end

function Tenkai.characteristic2conservative_reconstruction!(mode, ua, eq::Euler1D)
    # R, L = eigmatrix(eq, ua)
    # temp = R * mode
    # mode .= temp
    @assert false "not implemented"
end

admissibility_tolerance(eq::Euler1D) = 1e-10

function Tenkai.limit_slope(eq::Euler1D, s, ufl, u_s_l, ufr, u_s_r, ue, xl, xr)
    @unpack γ = eq
    eps = admissibility_tolerance(eq)

    variables = (get_density, get_pressure)

    for variable in variables
        var_star_tuple = (variable(eq, u_s_l), variable(eq, u_s_r))
        var_low = variable(eq, ue)

        theta = 1.0
        for var_star in var_star_tuple
            if var_star < eps
                # TOTHINK - Replace eps here by 0.1*var_low
                ratio = abs(0.1 * var_low - var_low) / (abs(var_star - var_low) + 1e-13)
                theta = min(ratio, theta)
            end
        end
        s *= theta
        u_s_l = ue + 2.0 * xl * s
        u_s_r = ue + 2.0 * xr * s
    end

    ufl = ue + xl * s
    ufr = ue + xr * s

    return ufl, ufr
end

function Tenkai.zhang_shu_flux_fix(eq::Euler1D,
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

#-------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------

function Tenkai.initialize_plot(eq::Euler1D, op, grid, problem, scheme, timer, u1,
                                ua)
    @timeit timer "Write solution" begin
    #! format: noindent
    @timeit timer "Initialize write solution" begin
    #! format: noindent
    # Clear and re-create output directory
    rm("output", force = true, recursive = true)
    mkdir("output")

    xc = grid.xc
    nx = grid.size
    @unpack xg = op
    nd = op.degree + 1
    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    xf = grid.xf
    nvar = eq.nvar
    # Create plot objects to be later collected as subplots

    # Creating a subplot for title
    p_title = plot(title = "Cell averages plot, $nx cells, t = 0.0",
                   grid = false, showaxis = false, bottom_margin = 0Plots.px)
    # Initialize subplots for density, velocity and pressure
    p_ua, p_u1 = [plot() for _ in 1:nvar], [plot() for _ in 1:nvar]
    labels = ["Density", "Velocity", "Pressure"]
    plot_type = Float64
    y = zeros(plot_type, nx) # put dummy to fix plotly bug with OffsetArrays
    for n in 1:nvar
        @views plot!(p_ua[n], xc, y, label = "Approximate",
                     linestyle = :dot, seriestype = :scatter,
                     color = :blue, markerstrokestyle = :dot,
                     markershape = :circle, markersize = 2,
                     markerstrokealpha = 0)
        xlabel!(p_ua[n], "x")
        ylabel!(p_ua[n], labels[n])
    end
    l_super = @layout[a{0.01h}; b c d] # Selecting layout for p_title being title
    p_ua = plot(p_title, p_ua[1], p_ua[2], p_ua[3], layout = l_super,
                size = (1500, 500)) # Make subplots

    # Set up p_u1 to contain polynomial approximation as a different curve
    # for each cell
    x = LinRange(xf[1], xf[2], nu)
    up1 = zeros(plot_type, nvar, nd)
    u = zeros(plot_type, nu)
    for ii in 1:nd
        @views con2prim!(eq, u1[:, ii, 1], up1[:, ii]) # store prim form in up1
    end

    for n in 1:nvar
        u = @views Vu * up1[n, :]
        plot!(p_u1[n], x, u, color = :red, legend = false)
        xlabel!(p_u1[n], "x")
        ylabel!(p_u1[n], labels[n])
    end
    for i in 2:nx
        for ii in 1:nd
            @views con2prim!(eq, u1[:, ii, i], up1[:, ii]) # store prim form in up1
        end
        x = LinRange(xf[i], xf[i + 1], nu)
        for n in 1:nvar
            u = @views Vu * up1[n, :]
            plot!(p_u1[n], x, u, color = :red, label = nothing, legend = false)
        end
    end

    l = @layout[a{0.01h}; b c d] # Selecting layout for p_title being title
    p_u1 = plot(p_title, p_u1[1], p_u1[2], p_u1[3], layout = l,
                size = (1700, 500)) # Make subplots

    anim_ua, anim_u1 = Animation(), Animation() # Initialize animation objects
    plot_data = PlotData(p_ua, anim_ua, p_u1, anim_u1)
    return plot_data
    end # timer
    end # timer
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::Euler1D, grid,
                            problem, param, op, ua, u1, aux, ndigits = 3)
    @timeit aux.timer "Write solution" begin
    #! format: noindent
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
    nvar = eq.nvar
    @unpack save_time_interval, save_iter_interval, animate = param
    avg_file = open("$avg_filename.txt", "w")
    plot_type = Float64
    up_ = zeros(plot_type, nvar)
    ylims = [[Inf, -Inf] for _ in 1:nvar] # set ylims for plots of all variables
    for i in 1:nx
        @views con2prim!(eq, ua[:, i], up_) # store primitve form in up_
        @printf(avg_file, "%e %e %e %e\n", xc[i], up_[1], up_[2], up_[3])
        # TOTHINK - Check efficiency of printf
        for n in 1:(eq.nvar)
            p_ua[n + 1][1][:y][i] = @views up_[n]    # Update y-series
            ylims[n][1] = min(ylims[n][1], up_[n]) # Compute ymin
            ylims[n][2] = max(ylims[n][2], up_[n]) # Compute ymax
        end
    end
    close(avg_file)
    for n in 1:nvar # set ymin, ymax for ua, u1 plots
        ylims!(p_ua[n + 1], (ylims[n][1] - 0.1, ylims[n][2] + 0.1))
        ylims!(p_u1[n + 1], (ylims[n][1] - 0.1, ylims[n][2] + 0.1))
    end
    t = round(time; digits = 3)
    title!(p_ua[1], "Cell averages plot, $nx cells, t = $t")
    sol_filename = get_filename("output/sol", ndigits, fcount)
    sol_file = open(sol_filename * ".txt", "w")
    up1 = zeros(plot_type, nvar, nd)

    u = zeros(plot_type, nvar, nu)
    x = zeros(plot_type, nu)
    for i in 1:nx
        for ii in 1:nd
            @views con2prim!(eq, u1[:, ii, i], up1[:, ii]) # store prim form in up1
        end
        @. x = grid.xf[i] + grid.dx[i] * xu
        @views mul!(u, up1, Vu')
        for n in 1:nvar
            p_u1[n + 1][i][:y] = u[n, :]
        end
        for ii in 1:nu
            @printf(sol_file, "%e %e %e %e\n", x[ii], u[1, ii], u[2, ii], u[3, ii])
        end
    end
    close(sol_file)
    title!(p_u1[1], "Numerical Solution, $nx cells, t = $t")
    println("Wrote $sol_filename.txt, $avg_filename.txt")
    if problem.final_time - time < 1e-10
        cp("$avg_filename.txt", "./output/avg.txt", force = true)
        cp("$sol_filename.txt", "./output/sol.txt", force = true)
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
            k1, k2 = ceil(time / animate_time_interval),
                     floor(time / animate_time_interval)
            if (abs(time - k1 * animate_time_interval) < 1e-10 ||
                abs(time - k2 * animate_time_interval) < 1e-10)
                frame(anim_ua, p_ua)
                frame(anim_u1, p_u1)
            end
        end
    end
    fcount += 1
    return fcount
    end # timer
end

function exact_solution_data(test_case)
    if test_case == "sod"
        file = GZip.open("$data_dir/sod.dat.gz")
        exact_data = readdlm(file)
    elseif test_case == "lax"
        file = GZip.open("$data_dir/lax.dat.gz")
        exact_data = readdlm(file)
        exact_data[:, 1] .-= 5.0
    elseif test_case == "toro5"
        file = GZip.open("$data_dir/toro_5_exact.dat.gz")
        exact_data = readdlm(file, skipstart = 9)
        # Swap third and fourth row because of structure of data
        temp = copy(exact_data[:, 3])
        exact_data[:, 3] = copy(exact_data[:, 4])
        exact_data[:, 4] = temp
    elseif test_case == "sedov1d"
        file = GZip.open("$data_dir/sedov1d.dat.gz")
        exact_data = readdlm(file)
    elseif test_case == "blast"
        file = GZip.open("$data_dir/blast.dat.gz")
        exact_data = readdlm(file)
    elseif test_case == "shuosher"
        file = GZip.open("$data_dir/shuosher.dat.gz")
        exact_data = readdlm(file)
    elseif test_case == "double_rarefaction"
        file = GZip.open("$data_dir/double_rarefaction_exact.dat.gz")
        exact_data = readdlm(file, skipstart = 9)
        # Swap third and fourth row because of structure of data
        temp = copy(exact_data[:, 3])
        exact_data[:, 3] = copy(exact_data[:, 4])
        exact_data[:, 4] = temp
    elseif test_case == "leblanc"
        file = GZip.open("$data_dir/leblanc_exact.dat.gz")
        exact_data = readdlm(file, skipstart = 9)
        # Swap third and fourth row because of structure of data
        temp = copy(exact_data[:, 3])
        exact_data[:, 3] = copy(exact_data[:, 4])
        exact_data[:, 4] = temp
    elseif test_case == "dwave"
        nx = 1000
        exact_data = zeros(RealT, nx, 4)
        exact_data[:, 1] .= LinRange(0.0, 1.0, nx)
        for i in 1:nx
            exact_data[i, 2] = 1.0 + 0.5 * sinpi(2.0 * exact_data[i, 1])
            exact_data[i, 3:4] .= 1.0
        end
    elseif test_case == "titarev_toro"
        exact_data = readdlm("$data_dir/$test_case.txt", skipstart = 0)
    elseif test_case == "larger_density"
        exact_data = readdlm("$data_dir/$test_case.txt", skipstart = 0)
    else
        @warn "Exact solution does not set!"
        return nothing
    end
    return exact_data
end

function Tenkai.post_process_soln(eq::Euler1D, aux, problem, param, scheme)
    @unpack timer, error_file = aux
    @timeit timer "Write solution" begin
    #! format: noindent
    println("Post processing solution")
    nvar = eq.nvar
    @unpack plot_data = aux
    @unpack p_ua, p_u1, anim_ua, anim_u1 = plot_data
    @unpack animate, saveto = param
    initial_values = eq.initial_values
    if problem.initial_value in values(initial_values) # Using ready made tests
        initial_value_string, = [a
                                 for (a, b) in initial_values
                                 if
                                 b == problem.initial_value]
        exact_data = exact_solution_data(initial_value_string)

        for n in 1:nvar
            @views plot!(p_ua[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact",
                         color = :black)
            @views plot!(p_u1[n + 1], exact_data[:, 1], exact_data[:, n + 1],
                         label = "Exact",
                         color = :black, legend = true)
            ymin = min(minimum(p_ua[n + 1][1][:y]), minimum(exact_data[:, n + 1]))
            ymax = max(maximum(p_ua[n + 1][1][:y]), maximum(exact_data[:, n + 1]))
            ylims!(p_ua[n + 1], (ymin - 0.1, ymax + 0.1))
            ylims!(p_u1[n + 1], (ymin - 0.1, ymax + 0.1))
        end
    end
    savefig(p_ua, "output/avg.png")
    savefig(p_u1, "output/sol.png")
    savefig(p_ua, "output/avg.html")
    savefig(p_u1, "output/sol.html")
    if animate == true
        gif(anim_ua, "output/avg.mp4", fps = 5)
        gif(anim_u1, "output/sol.mp4", fps = 5)
    end
    println("Wrote avg, sol in gif,html,png format to output directory.")
    plot(p_ua)
    plot(p_u1)

    close(error_file)
    if saveto != "none"
        if saveto[end] == "/"
            saveto = saveto[1:(end - 1)]
        end
        mkpath(saveto)
        for file in readdir("./output")
            cp("./output/$file", "$saveto/$file", force = true)
        end
        cp("./error.txt", "$saveto/error.txt", force = true)
        println("Saved output files to $saveto")
    end
    end # timer

    # Print timer data on screen
    print_timer(aux.timer, sortby = :firstexec)
    print("\n")
    show(aux.timer)
    print("\n")
    println("Time outside write_soln = "
            *
            "$(( TimerOutputs.tottime(timer)
                - TimerOutputs.time(timer["Write solution"]) ) * 1e-9)s")
    println("─────────────────────────────────────────────────────────────────────────────────────────")
    timer_file = open("./output/timer.json", "w")
    JSON3.write(timer_file, TimerOutputs.todict(timer))
    close(timer_file)
    return nothing
end

function get_equation(γ; hll_wave_speeds = "toro")
    name = "1d Euler Equations"
    numfluxes = Dict("rusanov" => rusanov, "roe" => roe, "eroe" => eroe,
                     "hll" => hll!, "hllc" => hllc!, "chandrashekar" => chandrashekar!)
    nvar = 3
    if hll_wave_speeds == "toro"
        hll_speeds = hll_speeds_toro
    else
        println("Wave speed not implemented!")
        @assert false
    end
    return Euler1D(γ,
                   hll_speeds, nvar,
                   name, initial_values, numfluxes)
end
end # @muladd

end
