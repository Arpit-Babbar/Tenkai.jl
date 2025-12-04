using Tenkai

import Tenkai: compute_time_step, correct_variable!, apply_tvb_limiter!,
               apply_tvb_limiterβ!, write_poly, write_soln!,
               compute_time_step

import Tenkai.EqEuler2D: save_solution_file

using Tenkai.EqEuler2D: Euler2D, con2prim!, get_pressure, con2prim

using EllipsisNotation

function compute_time_step(eq::Euler2D, problem, grid::StepGrid, aux, op, cfl, u1, ua)
    @timeit aux.timer "Time Step computation" begin
        @unpack dx, dy = grid
        nx_tuple, ny_tuple = grid.size
        @unpack γ = eq
        @unpack wg = op
        den = 0.0
        corners = ((0, 0), (nx_tuple[2] + 1, 0), (0, ny_tuple[2] + 1),
                   (nx_tuple[2] + 1, ny_tuple[2] + 1))
        # @threaded for element in element_iterator_with_ghosts(grid)
        for element_index in element_iterator(grid) # Loop over cells
            element = element_indices(element_index, grid)
            # TODO - This doesn't include ghost cell info!
            el_x, el_y = element[1], element[2]
            if (el_x, el_y) ∈ corners # KLUDGE - Temporary hack
                continue
            end
            u_node = get_node_vars(ua, eq, el_x, el_y)
            rho, v1, v2, p = con2prim(eq, u_node)
            c = sqrt(γ * p / rho)
            sx, sy = abs(v1) + c, abs(v2) + c
            den = max(den, abs(sx) / dx[el_x] + abs(sy) / dy[el_y] + 1e-12)
        end

        dt = cfl / den
        return dt
    end # timer
end

#------------------------------------------------------------------------------
# Limiters
#------------------------------------------------------------------------------
function apply_tvb_limiterβ!(eq::Euler2D, problem, scheme, grid::StepGrid, param,
                             op, ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
        nx, ny = grid.size
        @unpack xg, wg, Vl, Vr = op
        @unpack dx, dy = grid
        @unpack tvbM, cache, beta = scheme.limiter
        @unpack nvar = eq
        nd = length(wg)

        refresh!(u) = fill!(u, zero(eltype(u)))
        # Pre-allocate for each thread

        # Loop over cells
        @threaded for element_index in element_iterator(grid)
            id = Threads.threadid()
            element = element_indices(element_index, grid)
            el_x, el_y = element[1], element[2]
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
                duxm[n] = minmod_β(char_dux[n], beta * char_dual[n], beta * char_duar[n],
                                   Mdx2)
                duym[n] = minmod_β(char_duy[n], beta * char_duad[n], beta * char_duau[n],
                                   Mdy2)
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

function apply_tvb_limiter!(eq::Euler2D, problem, scheme, grid::StepGrid, param, op,
                            ua, u1, aux)
    @timeit aux.timer "TVB Limiter" begin
        nx, ny = grid.size
        @unpack xg, wg, Vl, Vr = op
        @unpack dx, dy = grid
        @unpack tvbM, cache, beta = scheme.limiter
        @unpack nvar = eq
        nd = length(wg)

        refresh!(u) = fill!(u, zero(eltype(u)))
        # Pre-allocate for each thread

        # Loop over cells
        @threaded for element_index in element_iterator(grid)
            id = Threads.threadid()
            element = element_indices(element_index, grid)
            el_x, el_y = element[1], element[2]
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
                jump_x += 0.5 * (abs(char_dul[n] - dulm_[n]) + abs(char_dur[n] - durm_[n]))
                jump_y += 0.5 * (abs(char_dud[n] - dudm_[n]) + abs(char_duu[n] - duum_[n]))
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

function write_poly(eq::Euler2D, grid::StepGrid, op, u1, fcount)
    filename = get_filename("output/sol", 3, fcount)
    @show filename
    @unpack xf, yf, dx, dy = grid
    nx_tuple, ny_tuple = grid.size
    @unpack degree, xg = op
    nd = degree + 1
    # Clear and re-create output directory

    nu = max(nd, 2)
    xu = LinRange(0.0, 1.0, nu)
    Vu = Vandermonde_lag(xg, xu)
    Mx, My = nx_tuple[2] * nu, ny_tuple[2] * nu
    grid_x = zeros(Mx)
    grid_y = zeros(My)
    for i in 1:nx_tuple[2]
        i_min = (i - 1) * nu + 1
        i_max = i_min + nu - 1
        # grid_x[i_min:i_max] .= LinRange(xf[i], xf[i+1], nu)
        grid_x[i_min:i_max] .= xf[i] .+ dx[i] * xg
    end

    for j in 1:ny_tuple[2]
        j_min = (j - 1) * nu + 1
        j_max = j_min + nu - 1
        # grid_y[j_min:j_max] .= LinRange(yf[j], yf[j+1], nu)
        grid_y[j_min:j_max] .= yf[j] .+ dy[j] * xg
    end

    vtk_sol = vtk_grid(filename, grid_x, grid_y)

    u_equi = zeros(Mx, My)
    u_equi_pres = copy(u_equi)
    u_equi_vx = copy(u_equi)
    u_equi_vy = copy(u_equi)
    u = zeros(nu)
    for j in 1:ny_tuple[2]
        for i in 1:nx_tuple[2]
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
    for j in 1:ny_tuple[2]
        for i in 1:nx_tuple[2]
            # to get values in the equispaced thing
            for jy in 1:nd
                i_min = (i - 1) * nu + 1
                i_max = i_min + nu - 1
                u_ = @view u1[1, :, jy, i, j]
                mul!(u, Vu, u_)
                j_index = (j - 1) * nu + jy
                for ix in 1:nd
                    u_node = get_node_vars(u1, eq, ix, jy, i, j)
                    u_equi_pres[i_min + ix - 1, j_index] = get_pressure(eq, u_node)
                    u_equi_vx[i_min + ix - 1, j_index] = u_node[2] / u_node[1]
                    u_equi_vy[i_min + ix - 1, j_index] = u_node[3] / u_node[1]
                end
            end
        end
    end
    # Now we get the pressure variable
    vtk_sol["sol"] = u_equi
    vtk_sol["pressure"] = u_equi_pres
    vtk_sol["vx"] = u_equi_vx
    vtk_sol["vy"] = u_equi_vy

    println("Wrote pointwise solution to $filename")

    out = vtk_save(vtk_sol)
end

function write_soln!(base_name, fcount, iter, time, dt, eq::Euler2D,
                     grid::StepGrid, problem, param, op,
                     z, u1, aux, ndigits = 3)
    @timeit aux.timer "Write solution" begin
        @unpack final_time = problem
        # Clear and re-create output directory
        if fcount == 0
            run(`rm -rf output`)
            run(`mkdir output`)
            save_mesh_file(grid, "output")
        end

        nx_tuple, ny_tuple = grid.size
        @unpack exact_solution = problem
        exact(x) = exact_solution(x[1], x[2], time)
        @unpack xc, yc = grid
        filename = get_filename("output/avg", ndigits, fcount)
        # filename = string("output/", filename)
        vtk = vtk_grid(filename, xc, yc)
        xy = [[xc[i], yc[j]] for i in 1:nx_tuple[2], j in 1:ny_tuple[2]]
        # KLUDGE - Do it efficiently
        prim = @views copy(z[:, 1:nx_tuple[2], 1:ny_tuple[2]])
        exact_data = exact.(xy)
        for j in 1:ny_tuple[2], i in 1:nx_tuple[2]
            @views con2prim!(eq, z[:, i, j], prim[:, i, j])
        end
        density_arr = prim[1, 1:nx_tuple[2], 1:ny_tuple[2]]
        velx_arr = prim[2, 1:nx_tuple[2], 1:ny_tuple[2]]
        vely_arr = prim[3, 1:nx_tuple[2], 1:ny_tuple[2]]
        pres_arr = prim[4, 1:nx_tuple[2], 1:ny_tuple[2]]
        vtk["sol"] = density_arr
        vtk["Density"] = density_arr
        vtk["Velocity_x"] = velx_arr
        vtk["Velocity_y"] = vely_arr
        vtk["Pressure"] = pres_arr
        for j in 1:ny_tuple[2], i in 1:nx_tuple[2]
            @views con2prim!(eq, exact_data[i, j], prim[:, i, j])
        end
        @views vtk["Exact Density"] = prim[1, 1:nx_tuple[2], 1:ny_tuple[2]]
        @views vtk["Exact Velocity_x"] = prim[2, 1:nx_tuple[2], 1:ny_tuple[2]]
        @views vtk["Exact Velocity_y"] = prim[3, 1:nx_tuple[2], 1:ny_tuple[2]]
        @views vtk["Exact Pressure"] = prim[4, 1:nx_tuple[2], 1:ny_tuple[2]]
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
        # element_variables[:indicator_shock_capturing] = vec(aux.blend.cache.alpha[1:nx_tuple[2],1:ny_tuple[2]])
        filename = save_solution_file(u1, time, dt, iter, grid, eq, op, element_variables) # Save h5 file
        println("Wrote ", filename)
        return fcount
    end # timer
end
