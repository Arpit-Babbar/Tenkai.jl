module EqMHD2D

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
using Accessors: @reset

using Tenkai
using Tenkai.Basis

using Trixi: True, False

import Tenkai: admissibility_tolerance

(import Tenkai: flux, prim2con, con2prim,
                eigmatrix,
                limit_slope, zhang_shu_flux_fix,
                apply_tvb_limiter!, apply_bound_limiter!, initialize_plot,
                write_soln!, compute_time_step, post_process_soln,
                update_ghost_values_rkfr!)

(using Tenkai: PlotData, data_dir, get_filename, neumann, minmod,
               get_node_vars, sum_node_vars_1d,
               set_node_vars!,
               nvariables, eachvariable,
               add_to_node_vars!, subtract_from_node_vars!,
               multiply_add_to_node_vars!,
               correct_variable_bound_limiter!)

import ..TenkaicRK: calc_non_cons_gradient, calc_non_cons_Bu, non_conservative_equation,
                    update_ghost_values_ub_N!, AbstractNonConservativeEquations,
                    calc_non_cons_B, flux_central_nc

import Tenkai.EqEuler1D: tenkai2trixiequation

import Trixi

using MuladdMacro

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct MHDNonConservative2D <: AbstractNonConservativeEquations{2, 9}
end

struct MHD2D{TrixiEquations, RealT <: Real, StaticBool} <:
       AbstractNonConservativeEquations{2, 9}
    trixi_equations::TrixiEquations
    glm_scale::RealT
    name::String
    non_conservative_part::MHDNonConservative2D
    non_conservative_activated::StaticBool
end

tenkai2trixiequation(eq::MHD2D) = eq.trixi_equations

non_conservative_equation(eq::MHD2D) = eq.non_conservative_part

function calc_non_cons_gradient(u_node, x, y, t, eq::MHD2D)
    return u_node
end

@inbounds @inline function flux(x, y, u, eq::MHD2D, orientation)
    Trixi.flux(u, orientation, eq.trixi_equations)
end

@inbounds @inline flux(u, eq::MHD2D, orientation) = flux(1.0, 1.0, u, eq, orientation)

@inbounds @inline function flux(x, y, u, eq::MHD2D)
    return flux(x, y, u, eq, 1), flux(x, y, u, eq, 2)
end

function prim2con(eq::MHD2D, prim)
    return Trixi.prim2cons(prim, eq.trixi_equations)
end

function con2prim(eq::MHD2D, con)
    return Trixi.cons2prim(con, eq.trixi_equations)
end

function density(eq::MHD2D, u)
    return Trixi.density(u, eq.trixi_equations)
end

function pressure(eq::MHD2D, u)
    return Trixi.pressure(u, eq.trixi_equations)
end

function velocity(eq::MHD2D, u)
    return Trixi.velocity(u, eq.trixi_equations)
end

function Tenkai.is_admissible(eq::MHD2D, u::AbstractVector)
    return density(eq, u) > 0.0 && pressure(eq, u) > 0.0
end

function flux_central_nc(ul, ur, orientation, eq::Trixi.IdealGlmMhdEquations2D)
    return Trixi.flux_nonconservative_powell_local_symmetric(ul, ur, orientation, eq)
end

@inbounds @inline function rho_p_indicator!(un, eq::MHD2D)
    nd_p2 = size(un, 2) # nd + 2
    for iy in 1:nd_p2, ix in 1:nd_p2 # loop over dofs and faces
        u_node = get_node_vars(un, eq, ix, iy)
        rho_p = Trixi.density_pressure(u_node, eq.trixi_equations)
        un[1, ix, iy] = rho_p # ρ * p
    end
    n_ind_var = 1
    return n_ind_var
end

function calc_non_cons_Bu(u, u_nc, x, y, t, orientation::Int,
                          eq::MHD2D{<:Any, <:Any, <:True})
    ul = u
    ur = u_nc
    return Trixi.flux_nonconservative_powell(ul, ur, orientation, eq.trixi_equations)

    # ur = u_nc - u
    # return Trixi.flux_nonconservative_powell_local_symmetric(ul, ur, orientation, eq.trixi_equations)
end

function calc_non_cons_Bu_hardcoded(u, u_nc, x, y, t, orientation::Int,
                                    eq::MHD2D{<:Any, <:Any, <:True})
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    _, _, _, _, _, B1_nc, B2_nc, _, psi_nc = u_nc
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    if orientation == 1
        B_nc = B1_nc
        v = v1
    else
        B_nc = B2_nc
        v = v2
    end

    B_v = v1 * B1 + v2 * B2 + v3 * B3

    U_mhd = SVector(0.0, B1, B2, B3, B_v, v1, v2, v3, 0.0) * B_nc
    U_glm = SVector(0.0, 0.0, 0.0, 0.0, v * psi, 0.0, 0.0, 0.0, v) * psi_nc

    return U_mhd + U_glm
end

function calc_non_cons_Bu(u, u_nc, x, y, t, orientation::Int,
                          eq::MHD2D{<:Any, <:Any, <:False})
    return zero(u)
end

function calc_non_cons_B(u, x, y, t, orientation::Int, eq::MHD2D)
    e1 = SVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e2 = SVector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e3 = SVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e4 = SVector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e5 = SVector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    e6 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    e7 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    e8 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    e9 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    SMatrix{9, 9}(calc_non_cons_Bu(u, e1, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e2, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e3, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e4, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e5, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e6, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e7, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e8, x, y, t, orientation, eq)...,
                  calc_non_cons_Bu(u, e9, x, y, t, orientation, eq)...)
end

function calc_non_cons_B(u, x, y, t, eq::MHD2D)
    return calc_non_cons_B(u, x, y, t, 1, eq), calc_non_cons_B(u, x, y, t, 2, eq)
end

@inbounds @inline function max_abs_eigen_value(eq::MHD2D, u, dir)
    cf = Trixi.calc_fast_wavespeed(u, dir, eq.trixi_equations)
    v = u[dir + 1] / u[1]
    return abs(v) + cf
end

function eigmatrix(eq::MHD2D, u)
    e1 = SVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e2 = SVector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e3 = SVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e4 = SVector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e5 = SVector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    e6 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    e7 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    e8 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    e9 = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    Id = SMatrix{9, 9}(e1...,
                       e2...,
                       e3...,
                       e4...,
                       e5...,
                       e6...,
                       e7...,
                       e8...,
                       e9...)
    return Id, Id, Id, Id
end

function Tenkai.apply_tvb_limiter!(eq::MHD2D, problem, scheme, grid, param, op,
                                   ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
    #! format: noindent
    nx, ny = grid.size
    @unpack xg, wg, Vl, Vr = op
    @unpack dx, dy = grid
    @unpack tvbM, cache, beta = scheme.limiter
    nvar = nvariables(eq)
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
            dux_ = duy_ = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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

function compute_time_step(eq::MHD2D, problem, grid, aux, op, cfl, u1, ua)
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
        # den = max(den, abs(sx) / dx[el_x] + 1e-12)
    end

    dt = cfl / den
    dt_const_speed = cfl / den_const_speed

    # @show dt, dt_const_speed

    c_h = glm_scale * (dt_const_speed / dt)
    @unpack trixi_equations = eq
    @reset trixi_equations.c_h = c_h
    # eq.trixi_equations.c_h = c_h
    @reset eq.trixi_equations = trixi_equations
    @show c_h

    return dt, eq
    end # timer
end

@inbounds @inline function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::MHD2D,
                                   dir)
    λ = max(max_abs_eigen_value(eq, ual, dir), max_abs_eigen_value(eq, uar, dir)) # local wave speed
    # λ = Trixi.max_abs_speed_naive(uar, ual, dir, eq.trixi_equations)
    return 0.5 * (Fl + Fr - λ * (Ur - Ul))
end

function Tenkai.initialize_plot(eq::MHD2D, op, grid, problem, scheme,
                                timer, u1,
                                ua)
    nothing
end

function write_poly(eq::MHD2D, grid, op, u1, fcount)
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

function Tenkai.write_soln!(base_name, fcount, iter, time, dt, eq::MHD2D, grid,
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

function update_ghost_values_rkfr!(problem, scheme,
                                   eq::MHD2D,
                                   grid, aux, op, cache, t)
    @unpack Fb, ub = cache
    update_ghost_values_periodic!(eq, problem, Fb, ub)
end

@inbounds @inline function primitive_indicator!(un, eq::MHD2D)
    # @unpack γ_minus_1 = eq # Is this inefficient?
    nd_p2 = size(un, 2)
    for iy in 1:nd_p2, ix in 1:nd_p2 # loop over dofs and faces
        # ρ, ρ_u1, ρ_u2, ρ_e = @view un[:, ix, iy]
        u_node = get_node_vars(un, eq, ix, iy)
        # u1, u2 = ρ_u1 / ρ, ρ_u2 / ρ
        # p = γ_minus_1 * (ρ_e - 0.5 * (ρ_u1 * u1 + ρ_u2 * u2))
        # un[:, ix, iy] .= ρ, u1, u2, p
        set_node_vars!(un, con2prim(eq, u_node), eq, ix, iy)
    end
    n_ind_var = nvariables(eq)
    return n_ind_var
end

function Tenkai.apply_bound_limiter!(eq::MHD2D, grid, scheme, param, op,
                                     ua, u1, aux)
    return nothing
end

struct ExactSolutionAlfvenWave{TrixiEquations}
    equations::MHD2D{TrixiEquations}
end

function (exact_solution_alfven_wave::ExactSolutionAlfvenWave)(x, y, t)
    @unpack equations = exact_solution_alfven_wave
    @unpack trixi_equations = equations

    # Set up the Alfven wave initial condition
    return Trixi.initial_condition_convergence_test(SVector(x, y), t, trixi_equations)
end

function get_equation(gamma; glm_scale = 0.5, activate_nc = True())
    name = "Ideal GLM MHD 2D"
    trixi_equations = Trixi.IdealGlmMhdEquations2D(gamma)
    nc_part = MHDNonConservative2D()
    return MHD2D(trixi_equations, glm_scale, name, nc_part, activate_nc)
end
end # @muladd

end # module EqMHD2D
