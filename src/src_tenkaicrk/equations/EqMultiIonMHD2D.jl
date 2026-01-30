module EqMultiIonMHD2D

import GZip
using Tenkai.DelimitedFiles
using Plots
using LinearAlgebra
using Printf
using TimerOutputs
using StaticArrays
using Tenkai.Polyester
using Tenkai.LoopVectorization
using Tenkai.JSON3
using Tenkai.SimpleUnPack
using Tenkai.WriteVTK
using Accessors

using Tenkai
using Tenkai.Basis

using Trixi: True, False, IdealGlmMhdMultiIonEquations2D

import Tenkai: admissibility_tolerance

import Tenkai.EqEuler1D: tenkai2trixiequation

import Tenkai: flux, prim2con, con2prim,
               eigmatrix,
               limit_slope, zhang_shu_flux_fix,
               apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
               write_soln!, compute_time_step, post_process_soln,
               update_ghost_values_cRK!, update_ghost_values_u1!,
               compute_cell_average!, get_trixi_equations,
               update_ghost_values_rkfr!

using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
              get_node_vars, sum_node_vars_1d,
              set_node_vars!,
              nvariables, eachvariable,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!,
              correct_variable_bound_limiter!

import ..TenkaicRK: calc_non_cons_gradient, calc_non_cons_Bu, non_conservative_equation,
                    update_ghost_values_ub_N!, AbstractNonConservativeEquations,
                    calc_non_cons_B, flux_central_nc

import Trixi

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct MultiIonMHDNonConservative2D{NVAR} <: AbstractNonConservativeEquations{2, NVAR}
end

struct MultiIonMHD2D{TrixiEquations, NVAR, RealT <: Real} <:
       AbstractNonConservativeEquations{2, NVAR}
    trixi_equations::TrixiEquations
    glm_scale::RealT
    name::String
    non_conservative_part::MultiIonMHDNonConservative2D{NVAR}
end

tenkai2trixiequation(eq::MultiIonMHD2D) = eq.trixi_equations

get_trixi_equations(semi, eq::MultiIonMHD2D) = tenkai2trixiequation(eq)

non_conservative_equation(eq::MultiIonMHD2D) = eq.non_conservative_part

function calc_non_cons_gradient(u_node, x, y, t, eq::MultiIonMHD2D)
    return u_node
end

@inbounds @inline function flux(x, y, u, eq::MultiIonMHD2D, orientation)
    Trixi.flux(u, orientation, eq.trixi_equations)
end

@inbounds @inline flux(u, eq::MultiIonMHD2D, orientation) = flux(1.0, 1.0, u, eq,
                                                                 orientation)

@inbounds @inline function flux(x, y, u, eq::MultiIonMHD2D)
    return flux(x, y, u, eq, 1), flux(x, y, u, eq, 2)
end

function prim2con(eq::MultiIonMHD2D, prim)
    return Trixi.prim2cons(prim, eq.trixi_equations)
end

function con2prim(eq::MultiIonMHD2D, con)
    return Trixi.cons2prim(con, eq.trixi_equations)
end

function density(eq::MultiIonMHD2D, u)
    return Trixi.density(u, eq.trixi_equations)
end

function pressure(eq::MultiIonMHD2D, u)
    return Trixi.pressure(u, eq.trixi_equations)
end

function total_pressure(eq::MultiIonMHD2D, u)
    return sum(pressure(eq, u))
end

function density_pressure(eq::MultiIonMHD2D, u)
    dens = density(eq, u)
    press = total_pressure(eq, u)
    return dens * press
end

@inbounds @inline function rho_p_indicator!(un, eq::MultiIonMHD2D)
    nd_p2 = size(un, 2) # nd + 2
    for iy in 1:nd_p2, ix in 1:nd_p2 # loop over dofs and faces
        u_node = get_node_vars(un, eq, ix, iy)
        rho_p = density_pressure(eq, u_node)
        un[1, ix, iy] = rho_p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

function velocity(eq::MultiIonMHD2D, u)
    return Trixi.velocity(u, eq.trixi_equations)
end

function Tenkai.is_admissible(eq::MultiIonMHD2D, u::AbstractVector)
    dens = Trixi.density(u, eq.trixi_equations)
    press = Trixi.pressure(u, eq.trixi_equations)
    return all(p -> p > 0, dens) && all(p -> p > 0, press)
end

function flux_central_nc(ul, ur, orientation, eq::IdealGlmMhdMultiIonEquations2D)
    Trixi.flux_nonconservative_central(ul, ur, orientation, eq)
end

@inline function flux_nonconservative_central_simpler(u_ll, u_rr, orientation::Integer,
                                                      equations::IdealGlmMhdMultiIonEquations2D)
    # Some commented parts are kept to indicate changes made from the Trixi.jl code

    @unpack charge_to_mass = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = Trixi.magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = Trixi.magnetic_field(u_rr, equations)
    psi_ll = Trixi.divergence_cleaning_field(u_ll, equations)
    psi_rr = Trixi.divergence_cleaning_field(u_rr, equations)

    # Compute important averages
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2

    # Electron pressure
    pe_ll = equations.electron_pressure(u_ll, equations)
    pe_rr = equations.electron_pressure(u_rr, equations)

    # Compute charge ratio of u_ll
    charge_ratio_ll = zero(MVector{Trixi.ncomponents(equations), eltype(u_ll)})
    total_electron_charge = zero(real(equations))
    for k in Trixi.eachcomponent(equations)
        rho_k = u_ll[3 + (k - 1) * 5 + 1]
        charge_ratio_ll[k] = rho_k * charge_to_mass[k]
        total_electron_charge += charge_ratio_ll[k]
    end
    charge_ratio_ll ./= total_electron_charge

    # Compute auxiliary variables
    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = Trixi.charge_averaged_velocities(u_ll,
                                                                                                                 equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = Trixi.charge_averaged_velocities(u_rr,
                                                                                                                 equations)

    f = zero(MVector{Trixi.nvariables(equations), eltype(u_ll)})

    B1_ = B1_rr
    B2_ = B2_rr

    if orientation == 1
        # Entries of Godunov-Powell term for induction equation
        f[1] = v1_plus_ll * B1_
        f[2] = v2_plus_ll * B1_
        f[3] = v3_plus_ll * B1_
        for k in Trixi.eachcomponent(equations)
            # Compute Lorentz term
            f2 = charge_ratio_ll[k] * (
                  #   (0.5f0 * mag_norm_ll - B1_ll * B1_ll + pe_ll) +
                  (0.5f0 * mag_norm_rr - B1_rr * B1_rr + pe_rr))
            # @assert false f2, charge_ratio_ll[1], mag_norm_rr, B1_rr, pe_rr
            f3 = charge_ratio_ll[k] * (
                  # (-B1_ll * B2_ll) +
                  (-B1_rr * B2_rr))
            f4 = charge_ratio_ll[k] * (
                  # (-B1_ll * B3_ll) +
                  (-B1_rr * B3_rr))
            f5 = vk1_plus_ll[k] * (
                  # pe_ll +
                  pe_rr)
            # Compute multi-ion term, which vanishes for NCOMP==1
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            f5 += (B2_ll * (
                    # (vk1_minus_ll * B2_ll - vk2_minus_ll * B1_ll) +
                    (vk1_minus_rr * B2_rr - vk2_minus_rr * B1_rr)) +
                   B3_ll * (
                    # (vk1_minus_ll * B3_ll - vk3_minus_ll * B1_ll) +
                    (vk1_minus_rr * B3_rr - vk3_minus_rr * B1_rr)))

            # Compute Godunov-Powell term
            f2 += charge_ratio_ll[k] * B1_ll * B1_
            f3 += charge_ratio_ll[k] * B2_ll * B1_
            f4 += charge_ratio_ll[k] * B3_ll * B1_
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) *
                  B1_

            # Compute GLM term for the energy
            f5 += v1_plus_ll * psi_ll * (
                                         # psi_ll +
                                         psi_rr)
            # Append to the flux vector
            Trixi.set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v1_plus_ll * (
                  # psi_ll +
                  psi_rr)

    else #if orientation == 2
        # Entries of Godunov-Powell term for induction equation
        f[1] = v1_plus_ll * B2_
        f[2] = v2_plus_ll * B2_
        f[3] = v3_plus_ll * B2_

        for k in Trixi.eachcomponent(equations)
            # Compute Lorentz term
            f2 = charge_ratio_ll[k] * (
                  # (-B2_ll * B1_ll) +
                  (-B2_rr * B1_rr))
            f3 = charge_ratio_ll[k] * (
                  #   (-B2_ll * B2_ll + 0.5f0 * mag_norm_ll + pe_ll) +
                  (-B2_rr * B2_rr + 0.5f0 * mag_norm_rr + pe_rr))
            f4 = charge_ratio_ll[k] * (
                  # (-B2_ll * B3_ll) +
                  (-B2_rr * B3_rr))
            f5 = vk2_plus_ll[k] * (
                  # pe_ll +
                  pe_rr)

            # Compute multi-ion term (vanishes for NCOMP==1)
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            f5 += (B1_ll * (
                    # (vk2_minus_ll * B1_ll - vk1_minus_ll * B2_ll) +
                    (vk2_minus_rr * B1_rr - vk1_minus_rr * B2_rr)) +
                   B3_ll * (
                    # (vk2_minus_ll * B3_ll - vk3_minus_ll * B2_ll) +
                    (vk2_minus_rr * B3_rr - vk3_minus_rr * B2_rr)))

            # Compute Godunov-Powell term
            f2 += charge_ratio_ll[k] * B1_ll * B2_
            f3 += charge_ratio_ll[k] * B2_ll * B2_
            f4 += charge_ratio_ll[k] * B3_ll * B2_
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) *
                  B2_

            # Compute GLM term for the energy
            f5 += v2_plus_ll * psi_ll * (
                                         # psi_ll +
                                         psi_rr)

            # Append to the flux vector
            Trixi.set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v2_plus_ll * (
                  # psi_ll +
                  psi_rr)
    end

    return SVector(f)
end

function calc_non_cons_Bu(u, u_nc, x, y, t, orientation::Int,
                          eq::MultiIonMHD2D{<:Any, <:Any})
    return flux_nonconservative_central_simpler(u, u_nc, orientation,
                                                eq.trixi_equations)
end

function calc_non_cons_Bu_hardcoded(u, u_nc, x, y, t, orientation::Int,
                                    eq::MultiIonMHD2D{<:Any, <:Any})
    @assert false "not implemented"
end

@inbounds @inline function max_abs_eigen_value(eq::MultiIonMHD2D, u, dir)
    @unpack trixi_equations = eq
    # @assert false
    cf = Trixi.calc_fast_wavespeed(u, dir, trixi_equations)

    v = zero(real(trixi_equations))

    if dir == 1
        for k in Trixi.eachcomponent(trixi_equations)
            rho, rho_v1, _, _ = Trixi.get_component(k, u, trixi_equations)
            v = max(v, abs(rho_v1 / rho))
        end
    else
        for k in Trixi.eachcomponent(trixi_equations)
            rho, _, rho_v2, _ = Trixi.get_component(k, u, trixi_equations)
            v = max(v, abs(rho_v2 / rho))
        end
    end

    return abs(v) + cf
end

function compute_time_step(eq::MultiIonMHD2D, problem, grid, aux, op, cfl, u1, ua)
    @timeit aux.timer "Time Step computation" begin
    #! format: noindent
    @unpack dx, dy = grid
    @unpack glm_scale = eq
    nx, ny = grid.size
    @unpack wg = op
    den = 0.0
    den_const_speed = 0.0
    corners = ((0, 0), (nx + 1, 0), (0, ny + 1), (nx + 1, ny + 1))
    for element in CartesianIndices((0:(nx + 1), 0:(ny + 1)))
        el_x, el_y = element[1], element[2]
        if (el_x, el_y) ∈ corners # KLUDGE - Temporary hack
            continue
        end
        u_node = get_node_vars(ua, eq, el_x, el_y)
        sx, sy = max_abs_eigen_value(eq, u_node, 1),
                 max_abs_eigen_value(eq, u_node, 2)
        den = max(den, abs(sx) / dx[el_x] + abs(sy) / dy[el_y] + 1e-12)
        den_const_speed = max(den_const_speed,
                              1.0 / dx[el_x] + 1.0 / dy[el_y] + 1e-12)
    end

    dt = cfl / den
    dt_const_speed = cfl / den_const_speed

    c_h = glm_scale * dt_const_speed / dt
    @unpack trixi_equations = eq
    @reset trixi_equations.c_h = c_h
    @reset eq.trixi_equations = trixi_equations
    # eq.trixi_equations.c_h = glm_scale * dt_const_speed / dt

    return dt, eq
    end # timer
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::MultiIonMHD2D, dir)
    # λ = max(max_abs_eigen_value(eq, ual, dir), max_abs_eigen_value(eq, uar, dir)) # local wave speed, another option but gives slightly larger errors
    λ = Trixi.max_abs_speed_naive(ual, uar, dir, eq.trixi_equations)
    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function Tenkai.initialize_plot(eq::MultiIonMHD2D, op, grid, problem, scheme,
                                timer, u1,
                                ua)
    nothing
end

function write_poly(eq::MultiIonMHD2D, grid, op, u1, fcount)
    filename = get_filename("output/sol", 3, fcount)
    @unpack xf, yf, dx, dy = grid
    nx, ny = grid.size
    @unpack degree, xg = op
    nvar = nvariables(eq)
    nd = degree + 1
    # Clear and re-create output directory

    Mx, My = nx * nd, ny * nd
    grid_x = zeros(Mx)
    grid_y = zeros(My)
    for i in 1:nx
        i_min = (i - 1) * nd + 1
        i_max = i_min + nd - 1
        grid_x[i_min:i_max] .= xf[i] .+ dx[i] * xg
    end

    for j in 1:ny
        j_min = (j - 1) * nd + 1
        j_max = j_min + nd - 1
        grid_y[j_min:j_max] .= yf[j] .+ dy[j] * xg
    end

    vtk_sol = vtk_grid(filename, grid_x, grid_y)

    u_arr = zeros(nvar, Mx, My)
    for j in 1:ny
        for i in 1:nx
            # to get values in the equispaced thing
            for jy in 1:nd, ix in 1:nd
                i_index = (i - 1) * nd + ix
                j_index = (j - 1) * nd + jy
                u_node = get_node_vars(u1, eq, ix, jy, i, j)
                u_arr[:, i_index, j_index] .= con2prim(eq, u_node)
            end
        end
    end

    for (i, var) in enumerate(Trixi.varnames(Trixi.cons2prim, eq.trixi_equations))
        @views vtk_sol[var] = u_arr[i, :, :]
    end

    out = vtk_save(vtk_sol)
    println("Wrote file ", out[1])
end

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::MultiIonMHD2D, grid,
                            problem, param, op, ua, u1, aux, ndigits = 3)
    @timeit aux.timer "Write solution" begin
    #! format: noindent
    @unpack final_time = problem
    # Clear and re-create output directory
    if fcount == 0
        run(`rm -rf output`)
        run(`mkdir output`)
        # save_mesh_file(grid, "output")
    end

    nvar = nvariables(eq)

    nx, ny = grid.size
    @unpack exact_solution = problem
    exact(x) = exact_solution(x[1], x[2], time)
    @unpack xc, yc = grid
    filename = get_filename("output/avg", ndigits, fcount)
    # filename = string("output/", filename)
    vtk = vtk_grid(filename, xc, yc)
    xy = [[xc[i], yc[j]] for i in 1:nx, j in 1:ny]
    # KLUDGE - Do it efficiently
    prim = @views copy(ua[:, 1:nx, 1:ny])
    exact_data = exact.(xy)
    for j in 1:ny, i in 1:nx
        @views prim[:, i, j] .= con2prim(eq, ua[:, i, j])
    end

    for (i, var) in enumerate(Trixi.varnames(Trixi.cons2prim, eq.trixi_equations))
        vtk[var] = prim[i, 1:nx, 1:ny]
    end
    vtk["CYCLE"] = iter
    vtk["TIME"] = time
    out = vtk_save(vtk)
    println("Wrote file ", out[1])
    # write_poly(eq, grid, op, u1, fcount)
    if final_time - time < 1e-10
        cp("$filename.vtr", "./output/avg.vtr")
        println("Wrote final average solution to avg.vtr.")
    end

    write_poly(eq, grid, op, u1, fcount)

    fcount += 1

    return fcount
    end # timer
end

# TODO - This is only for the KHI test case, should be generalized. Maybe allow
# the user to specify the x_reflect
@inline @inbounds x_reflect(u, eq::MultiIonMHD2D) = SVector(-u[1], u[2], u[3], u[4],
                                                            -u[5], u[6], u[7], u[8],
                                                            u[9], -u[10],
                                                            u[11], u[12], u[13], u[14])

@inline @inbounds y_reflect(u, eq::MultiIonMHD2D) = SVector(u[1], -u[2], u[3], u[4],
                                                            u[5], -u[6], u[7], u[8],
                                                            u[9], u[10],
                                                            -u[11], u[12], u[13], u[14])

function update_ghost_values_cRK!(problem, scheme, eq::MultiIonMHD2D,
                                  grid, aux,
                                  op, cache,
                                  t, dt, scaling_factor = 1)
    update_ghost_values_u1!(eq, problem, grid, op, cache.u1, aux, t)

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

                # set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
                # set_node_vars!(Fb, fb_node, eq, k, 1, 1, j)
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
                    Ub_node = get_node_vars(Ub, eq, k, 2, 0, j)
                    set_node_vars!(Ub, x_reflect(Ub_node, eq), eq, k, 2, 0, j)
                    x1 = xf[1]
                    y1 = yf[j] + xg[k] * dy[j]
                    # TODO - Not correct, we need to reflect the flux appropriately.
                    # This is first order
                    set_node_vars!(Fb, flux(x1, y1, Ub_node, eq, 1), eq, k, 2, 0, j)
                end
            end
        end
    else
        @assert left in (periodic,) "Incorrect bc specified at left."
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
                    Ub_node = get_node_vars(Ub, eq, k, 1, nx + 1, j)
                    Ub_reflect = x_reflect(Ub_node, eq)
                    set_node_vars!(Ub, Ub_reflect, eq, k, 1, nx + 1, j)
                    x2 = xf[nx + 1]
                    y2 = yf[j] + xg[k] * dy[j]
                    # TODO - Not correct, we need to reflect the flux appropriately.
                    # This is first order
                    set_node_vars!(Fb, flux(x2, y2, Ub_reflect, eq, 1), eq, k, 1,
                                   nx + 1,
                                   j)
                end
            end
        end
    else
        @assert right in (periodic,) "Incorrect bc specified at right."
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
                    Ub_node = get_node_vars(Ub, eq, k, 4, i, 0)
                    Ub_reflect = y_reflect(Ub_node, eq)
                    set_node_vars!(Ub, Ub_reflect, eq, k, 4, i, 0)
                    y3 = yf[1]
                    x3 = xf[i] + xg[k] * dx[i]
                    # TODO - Not correct, we need to reflect the flux appropriately
                    # This is first order
                    set_node_vars!(Fb, flux(x3, y3, Ub_reflect, eq, 2), eq, k, 4, i, 0)
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
                    Ub_node = get_node_vars(Ub, eq, k, 3, i, ny + 1)
                    Ub_reflect = y_reflect(Ub_node, eq)
                    set_node_vars!(Ub, Ub_reflect, eq, k, 3, i, ny + 1)
                    y4 = yf[ny + 1]
                    x4 = xf[i] + xg[k] * dx[i]
                    # TODO - Not correct, we need to reflect the flux appropriately
                    # This is first order
                    set_node_vars!(Fb, flux(x4, y4, Ub_reflect, eq, 2), eq, k, 3, i,
                                   ny + 1)
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert false "Incorrect bc specific at top"
    end

    return nothing
end

function update_ghost_values_u1!(eq::MultiIonMHD2D, problem, grid, op, u1, aux, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    nx, ny = grid.size
    nd = op.degree + 1
    @unpack xg = op
    nvar = size(u1, 1)
    @views if problem.periodic_x
        @turbo u1[:, :, :, 0, 1:ny] .= u1[:, :, :, nx, 1:ny]
        @turbo u1[:, :, :, nx + 1, 1:ny] .= u1[:, :, :, 1, 1:ny]
    end

    @views if problem.periodic_y
        @turbo u1[:, :, :, 1:nx, 0] .= u1[:, :, :, 1:nx, ny]
        @turbo u1[:, :, :, 1:nx, ny + 1] .= u1[:, :, :, 1:nx, 1]
    end

    @views if problem.periodic_x && problem.periodic_y
        # Corners
        @turbo u1[:, :, :, 0, 0] .= u1[:, :, :, nx, 0]
        @turbo u1[:, :, :, nx + 1, 0] .= u1[:, :, :, 1, 0]
        @turbo u1[:, :, :, 0, ny + 1] .= u1[:, :, :, nx, ny + 1]
        @turbo u1[:, :, :, nx + 1, ny + 1] .= u1[:, :, :, 1, ny + 1]
        return nothing
    end

    left, right, bottom, top = problem.boundary_condition
    boundary_value = problem.boundary_value
    @unpack dx, dy, xf, yf = grid
    if left == dirichlet
        x = xf[1]
        for j in 1:ny
            for k in 1:nd
                y = yf[j] + xg[k] * dy[j]
                ub = boundary_value(x, y, t)
                for ix in 1:nd, n in 1:nvar
                    u1[n, ix, k, 0, j] = ub[n]
                end
            end
        end
    elseif left == neumann
        for j in 1:ny
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, 0, j] = u1[n, ix, jy, 1, j]
            end
        end
    elseif left == reflect
        for j in 1:ny
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, 0, j] = u1[n, ix, jy, 1, j]
            end
            for jy in 1:nd, ix in 1:nd
                u_node = get_node_vars(u1, eq, nd - ix + 1, jy, 0, j)
                set_node_vars!(u1, x_reflect(u_node, eq), eq, ix, jy, 0, j)
            end
        end
    else
        @assert left in (periodic,) "Incorrect bc specified at left."
    end

    if right == dirichlet
        for j in 1:ny
            for k in 1:nd
                x = xf[nx + 1]
                y = yf[j] + xg[k] * dy[j]
                ub = boundary_value(x, y, t)
                for ix in 1:nd, n in 1:nvar
                    u1[n, ix, k, nx + 1, j] = ub[n]
                end
            end
        end
    elseif right == neumann
        for j in 1:ny
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, nx + 1, j] = u1[n, ix, jy, nx, j]
            end
        end
    elseif right == reflect
        for j in 1:ny
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, nx + 1, j] = u1[n, ix, jy, nx, j]
            end
            for jy in 1:nd, ix in 1:nd
                u_node = get_node_vars(u1, eq, nd - ix + 1, jy, nx + 1, j)
                set_node_vars!(u1, x_reflect(u_node, eq), eq, ix, jy, nx + 1, j)
            end
        end
    else
        @assert right in (periodic,) "Incorrect bc specified at right."
    end

    if bottom in (dirichlet,)
        y = yf[1]
        for i in 1:nx
            for k in 1:nd
                x = xf[i] + xg[k] * dx[i]
                ub = boundary_value(x, y, t)
                for jy in 1:nd, n in 1:nvar
                    u1[n, k, jy, i, 0] = ub[n]
                end
            end
        end
    elseif bottom == neumann
        for i in 1:nx
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, i, 0] = u1[n, ix, jy, i, 1]
            end
        end
    elseif bottom == reflect
        for i in 1:nx
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, i, 0] = u1[n, ix, jy, i, 1]
            end
            for jy in 1:nd, ix in 1:nd
                u_node = get_node_vars(u1, eq, ix, nd - jy + 1, i, 0)
                set_node_vars!(u1, y_reflect(u_node, eq), eq, ix, jy, i, 0)
            end
        end
    else
        @assert bottom in (periodic,) "Incorrect bc specified at bottom."
    end

    if top == dirichlet
        y = yf[ny + 1]
        for i in 1:nx
            for k in 1:nd
                x = xf[i] + xg[k] * dx[i]
                ub = boundary_value(x, y, t)
                for jy in 1:nd, n in 1:nvar
                    u1[n, k, jy, i, ny + 1] = ub[n]
                end
            end
        end
    elseif top == neumann
        for i in 1:nx
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, i, ny + 1] = u1[n, ix, jy, i, ny]
            end
        end
    elseif top == reflect
        for i in 1:nx
            for jy in 1:nd, ix in 1:nd, n in 1:nvar
                u1[n, ix, jy, i, ny + 1] = u1[n, ix, jy, i, ny]
            end
            for jy in 1:nd, ix in 1:nd
                u_node = get_node_vars(u1, eq, ix, nd - jy + 1, i, ny + 1)
                set_node_vars!(u1, y_reflect(u_node, eq), eq, ix, jy, i, ny + 1)
            end
        end
    else
        @assert top in (periodic,) "Incorrect bc specified at top."
    end

    # Corners (Probably not needed)
    @views if problem.periodic_x
        @turbo u1[:, :, :, 0, 0] .= u1[:, :, :, nx, 0]
        @turbo u1[:, :, :, nx + 1, 0] .= u1[:, :, :, 1, 0]
        @turbo u1[:, :, :, 0, ny + 1] .= u1[:, :, :, nx, ny + 1]
        @turbo u1[:, :, :, nx + 1, ny + 1] .= u1[:, :, :, 1, ny + 1]
    else
        # TOTHINK - Reflect bc and stuff for corners as well?
        @turbo u1[:, :, :, 0, 0] .= u1[:, :, :, 1, 0]
        @turbo u1[:, :, :, nx + 1, 0] .= u1[:, :, :, nx, 0]
        @turbo u1[:, :, :, 0, ny + 1] .= u1[:, :, :, 1, ny + 1]
        @turbo u1[:, :, :, nx + 1, ny + 1] .= u1[:, :, :, nx, ny + 1]
    end

    return nothing
    end # timer
end

function update_ghost_values_rkfr!(problem, scheme,
                                   eq::MultiIonMHD2D,
                                   grid, aux, op, cache, t)
    @timeit aux.timer "Update ghost values" begin
    #! format: noindent
    @unpack Fb, ub, ua = cache

    update_ghost_values_periodic!(eq, problem, Fb, ub)
    update_ghost_values_u1!(eq, problem, grid, op, cache.u1, aux, t)

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

    # Julia bug occuring here. Below, we have unnecessarily named
    # x1,y1, x2, y2,.... We should have been able to just call them x,y
    # Otherwise we were getting a type instability and variables were
    # called Core.Box. This issue is probably related to
    # https://discourse.julialang.org/t/type-instability-of-nested-function/57007
    # https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
    # https://github.com/JuliaLang/julia/issues/15276
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured-1

    # For Dirichlet bc, use upwind flux at faces by assigning both physical
    # and ghost cells through the bc.
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
                Ub_node = get_node_vars(ub, eq, k, 1, 1, j)
                Fb_node = get_node_vars(Fb, eq, k, 1, 1, j)
                set_node_vars!(ub, Ub_node, eq, k, 2, 0, j)
                set_node_vars!(Fb, Fb_node, eq, k, 2, 0, j)
                if left == reflect
                    Ub_node = get_node_vars(ub, eq, k, 2, 0, j)
                    set_node_vars!(ub, x_reflect(Ub_node, eq), eq, k, 2, 0, j)
                    x1 = xf[1]
                    y1 = yf[j] + xg[k] * dy[j]
                    set_node_vars!(Fb, flux(x1, y1, Ub_node, eq, 1), eq, k, 2, 0, j)
                end
            end
        end
    else
        @assert left in (periodic,) "Incorrect bc specified at left."
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
    elseif right in (reflect, neumann)
        @threaded for j in 1:ny
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(ub, eq, k, 2, nx, j)
                Fb_node = get_node_vars(Fb, eq, k, 2, nx, j)
                set_node_vars!(ub, Ub_node, eq, k, 1, nx + 1, j)
                set_node_vars!(Fb, Fb_node, eq, k, 1, nx + 1, j)

                if right == reflect
                    Ub_node = get_node_vars(ub, eq, k, 1, nx + 1, j)
                    Ub_reflect = x_reflect(Ub_node, eq)
                    set_node_vars!(ub, Ub_reflect, eq, k, 1, nx + 1, j)
                    x2 = xf[nx + 1]
                    y2 = yf[j] + xg[k] * dy[j]
                    # TODO - Not correct, we need to reflect the flux appropriately.
                    # This is first order
                    set_node_vars!(Fb, flux(x2, y2, Ub_reflect, eq, 1), eq, k, 1,
                                   nx + 1,
                                   j)
                end
            end
        end
    else
        @assert right in (periodic,) "Incorrect bc specified at right."
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
    elseif bottom in (reflect, neumann)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(ub, eq, k, 3, i, 1)
                Fb_node = get_node_vars(Fb, eq, k, 3, i, 1)
                set_node_vars!(ub, Ub_node, eq, k, 4, i, 0)
                set_node_vars!(Fb, Fb_node, eq, k, 4, i, 0)
                if bottom == reflect
                    Ub_node = get_node_vars(ub, eq, k, 4, i, 0)
                    Ub_reflect = y_reflect(Ub_node, eq)
                    set_node_vars!(ub, Ub_reflect, eq, k, 4, i, 0)
                    y3 = yf[1]
                    x3 = xf[i] + xg[k] * dx[i]
                    # TODO - Not correct, we need to reflect the flux appropriately
                    # This is first order
                    set_node_vars!(Fb, flux(x3, y3, Ub_reflect, eq, 2), eq, k, 4, i, 0)
                end
            end
        end
    elseif periodic_y
        nothing
    else
        @assert typeof(bottom) <: Tuple{Any, Any, Any}
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
    elseif top in (reflect, neumann)
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                Ub_node = get_node_vars(ub, eq, k, 4, i, ny)
                Fb_node = get_node_vars(Fb, eq, k, 4, i, ny)
                set_node_vars!(ub, Ub_node, eq, k, 3, i, ny + 1)
                set_node_vars!(Fb, Fb_node, eq, k, 3, i, ny + 1)
                if top == reflect
                    Ub_node = get_node_vars(ub, eq, k, 3, i, ny + 1)
                    Ub_reflect = y_reflect(Ub_node, eq)
                    set_node_vars!(ub, Ub_reflect, eq, k, 3, i, ny + 1)
                    y4 = yf[ny + 1]
                    x4 = xf[i] + xg[k] * dx[i]
                    # TODO - Not correct, we need to reflect the flux appropriately
                    # This is first order
                    set_node_vars!(Fb, flux(x4, y4, Ub_reflect, eq, 2), eq, k, 3, i,
                                   ny + 1)
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

    return nothing

    end # timer
end

function Tenkai.apply_bound_limiter!(eq::MultiIonMHD2D, grid, scheme, param, op,
                                     ua, u1, aux)
    return nothing
end

function compute_cell_average!(ua, u1, t, eq::MultiIonMHD2D, grid,
                               problem, scheme, aux, op)
    @timeit aux.timer "Cell averaging" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack limiter = scheme
    @unpack xc = grid
    @unpack xg, wg, Vl, Vr = op
    @unpack periodic_x, periodic_y = problem
    @unpack boundary_condition, boundary_value = problem
    left, right, bottom, top = boundary_condition
    nd = length(wg)
    fill!(ua, zero(eltype(ua)))
    # Compute cell averages
    @threaded for element in CartesianIndices((1:nx, 1:ny))
        el_x, el_y = element[1], element[2]
        u1_ = @view u1[:, :, :, el_x, el_y]
        for j in Base.OneTo(nd), i in Base.OneTo(nd)
            u_node = get_node_vars(u1_, eq, i, j)
            multiply_add_to_node_vars!(ua, wg[i] * wg[j], u_node, eq, el_x, el_y)
            # Maybe put w2d in op and use it here
        end
    end

    # Update ghost values of ua by periodicity or with neighbours
    if periodic_x
        for j in 1:ny
            ua_node = get_node_vars(ua, eq, nx, j)
            set_node_vars!(ua, ua_node, eq, 0, j)
            ua_node = get_node_vars(ua, eq, 1, j)
            set_node_vars!(ua, ua_node, eq, nx + 1, j)
        end
    else
        if left == dirichlet
            @threaded for el_y in 1:ny
                x = grid.xf[1]
                for j in Base.OneTo(nd)
                    y = grid.yf[el_y] + grid.dy[el_y] * xg[j]
                    uval = boundary_value(x, y, t)
                    for i in Base.OneTo(nd)
                        multiply_add_to_node_vars!(ua, wg[i] * wg[j], uval, eq, 0,
                                                   el_y)
                    end
                end
            end
        else
            @threaded for el_y in 1:ny
                ua_ = get_node_vars(ua, eq, 1, el_y)
                set_node_vars!(ua, ua_, eq, 0, el_y)
            end
        end

        if right == dirichlet
            @threaded for el_y in 1:ny
                x = grid.xf[nx + 1]
                for j in Base.OneTo(nd)
                    y = grid.yf[el_y] + grid.dy[el_y] * xg[j]
                    uval = boundary_value(x, y, t)
                    for i in Base.OneTo(nd)
                        multiply_add_to_node_vars!(ua, wg[i] * wg[j], uval, eq,
                                                   nx + 1,
                                                   el_y)
                    end
                end
            end
        else
            @threaded for el_y in 1:ny
                ua_ = get_node_vars(ua, eq, nx, el_y)
                set_node_vars!(ua, ua_, eq, nx + 1, el_y)
            end
        end

        if left == reflect
            for el_y in 1:ny
                ua_node = get_node_vars(ua, eq, 1, el_y)
                set_node_vars!(ua, x_reflect(ua_node, eq), eq, 0, el_y)
            end
        end

        if right == reflect
            for el_y in 1:ny
                ua_node = get_node_vars(ua, eq, nx, el_y)
                set_node_vars!(ua, x_reflect(ua_node, eq), eq, nx + 1, el_y)
            end
        end
    end

    if periodic_y
        # Bottom ghost cells
        for el_x in Base.OneTo(nx)
            ua_node = get_node_vars(ua, eq, el_x, ny)
            set_node_vars!(ua, ua_node, eq, el_x, 0)
            ua_node = get_node_vars(ua, eq, el_x, 1)
            set_node_vars!(ua, ua_node, eq, el_x, ny + 1)
        end
    else
        if bottom in (reflect, neumann, dirichlet)
            if bottom in (dirichlet,)
                @threaded for el_x in 1:nx
                    y = grid.yf[1]
                    for i in Base.OneTo(nd)
                        x = grid.xf[el_x] + grid.dx[el_x] * xg[i]
                        uval = boundary_value(x, y, t)
                        for j in Base.OneTo(nd)
                            multiply_add_to_node_vars!(ua, wg[j] * wg[i], uval, eq,
                                                       el_x, 0)
                        end
                    end
                end
            else
                for el_x in Base.OneTo(nx)
                    ua_ = get_node_vars(ua, eq, el_x, 1)
                    set_node_vars!(ua, ua_, eq, el_x, 0)
                end
                if bottom == reflect
                    for i in 1:nx
                        ua_node = get_node_vars(ua, eq, i, 1)
                        set_node_vars!(ua, y_reflect(ua_node, eq), eq, i, 0)
                    end
                end
            end
        else
            @assert typeof(bottom) <: Tuple{Any, Any, Any}
            ua_bc! = bottom[2]
            ua_bc!(grid, eq, ua)
        end

        if top in (reflect, neumann, dirichlet)
            if top == dirichlet
                @threaded for el_x in 1:nx
                    y = grid.yf[ny + 1]
                    for i in Base.OneTo(nd)
                        x = grid.xf[el_x] + grid.dx[el_x] * xg[i]
                        uval = boundary_value(x, y, t)
                        for j in Base.OneTo(nd)
                            multiply_add_to_node_vars!(ua, wg[j] * wg[i], uval, eq,
                                                       el_x, ny + 1)
                        end
                    end
                end
            else
                for el_x in Base.OneTo(nx)
                    ua_ = get_node_vars(ua, eq, el_x, ny)
                    set_node_vars!(ua, ua_, eq, el_x, ny + 1)
                end
                if top == reflect
                    for i in 1:nx
                        ua_node = get_node_vars(ua, eq, i, ny)
                        set_node_vars!(ua, y_reflect(ua_node, eq), eq, i, ny + 1)
                    end
                end
            end
        else
            @assert typeof(top) <: Tuple{Any, Any, Any}
            bc_ua! = top[2]
            bc_ua!(grid, eq, ua)
        end
    end
    end # timer
    return nothing
end

struct ExactSolutionAlfvenWave{TrixiEquations}
    equations::MultiIonMHD2D{TrixiEquations}
end

function (exact_solution_alfven_wave::ExactSolutionAlfvenWave)(x, y, t)
    @unpack equations = exact_solution_alfven_wave
    @unpack trixi_equations = equations

    # Set up the Alfven wave initial condition
    return Trixi.initial_condition_convergence_test(SVector(x, y), t, trixi_equations)
end

function get_equation(trixi_equations; glm_scale = 0.5)
    name = "Ideal Multi-Ion MHD 2D"
    nc_part = MultiIonMHDNonConservative2D{Trixi.nvariables(trixi_equations)}()
    return MultiIonMHD2D(trixi_equations, glm_scale, name, nc_part)
end
end # @muladd

end # module EqMHD2D
