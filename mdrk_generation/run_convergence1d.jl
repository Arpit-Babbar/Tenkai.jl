using Tenkai
using Tenkai: base_dir
using TrixiBase: trixi_include
using DelimitedFiles

function bf2str(bflux)
    if bflux == evaluate
        return "EA"
    else
        @assert bflux == extrapolate
        return "AE"
    end
end

mdrk_data_dir = joinpath(base_dir, "mdrk_results")
function trixi_convergence(test, solver, nx_array;
                           bflux = evaluate, corr = "radau", diss = "2", points = "gl",
                           degree = 3,
                           outdir = nothing)
    if outdir === nothing
        my_base_dir = joinpath(mdrk_data_dir, test)
        mkpath(my_base_dir)
        filename = "$(solver)$(degree)_$(bf2str(bflux))_D$(diss)_$(corr)_$(points).txt"
        outdir = joinpath(my_base_dir, filename)
    end
    M = length(nx_array)
    data = zeros(M, 2)
    for i in eachindex(nx_array)
        sol = trixi_include("$(@__DIR__)/run_files/run_$test.jl",
                             solver = solver, degree = 3, bflux = bflux,
                             solution_points = points, diss = diss,
                             correction_function = corr, nx = nx_array[i])
        data[i,1] = nx_array[i]
        data[i,2] = sol["errors"]["l2_error"]
    end
    writedlm(outdir, data)
    return
end

nx_array = [10, 20, 40, 80, 160]

### burg1d

# gl, radau

# Diss = 1
trixi_convergence("burg1d", "mdrk", nx_array, bflux = evaluate, diss = "1")
trixi_convergence("burg1d", "mdrk", nx_array, bflux = extrapolate, diss = "1")
trixi_convergence("burg1d", "lwfr", nx_array, bflux = evaluate, diss = "1")
trixi_convergence("burg1d", "lwfr", nx_array, bflux = extrapolate, diss = "1")

# Diss = 2
trixi_convergence("burg1d", "mdrk", nx_array, bflux = evaluate)
trixi_convergence("burg1d", "mdrk", nx_array, bflux = extrapolate)
trixi_convergence("burg1d", "lwfr", nx_array, bflux = evaluate)
trixi_convergence("burg1d", "lwfr", nx_array, bflux = extrapolate)

# rkfr

trixi_convergence("burg1d", "rkfr", nx_array, bflux = evaluate)

## gll, g2
# Diss = 1
trixi_convergence("burg1d", "mdrk", nx_array, bflux = evaluate,
                  diss = "1", points = "gll", corr = "g2")
trixi_convergence("burg1d", "mdrk", nx_array, bflux = extrapolate,
                  diss = "1", points = "gll", corr = "g2")
trixi_convergence("burg1d", "lwfr", nx_array, bflux = evaluate,
                  diss = "1", points = "gll", corr = "g2")
trixi_convergence("burg1d", "lwfr", nx_array, bflux = extrapolate,
                  diss = "1", points = "gll", corr = "g2")

# Diss = 2
trixi_convergence("burg1d", "mdrk", nx_array, bflux = evaluate, points = "gll", corr = "g2")
trixi_convergence("burg1d", "mdrk", nx_array, bflux = extrapolate, points = "gll", corr = "g2")
trixi_convergence("burg1d", "lwfr", nx_array, bflux = evaluate, points = "gll", corr = "g2")
trixi_convergence("burg1d", "lwfr", nx_array, bflux = extrapolate, points = "gll", corr = "g2")

# RKFR
trixi_convergence("burg1d", "rkfr", nx_array, bflux = evaluate, points = "gll", corr = "g2")

## gl, radau
# Diss = 1
trixi_convergence("linadv1d", "mdrk", nx_array, bflux = extrapolate, diss = "1")
trixi_convergence("linadv1d", "lwfr", nx_array, bflux = extrapolate, diss = "1")

# Diss = 2
trixi_convergence("linadv1d", "mdrk", nx_array, bflux = extrapolate)
trixi_convergence("linadv1d", "lwfr", nx_array, bflux = extrapolate)

# RKFR
trixi_convergence("linadv1d", "rkfr", nx_array, bflux = evaluate)

## gll, g2
# Diss = 1
trixi_convergence("linadv1d", "mdrk", nx_array, bflux = extrapolate,
                  diss = "1", points = "gll", corr = "g2")
trixi_convergence("linadv1d", "lwfr", nx_array, bflux = extrapolate,
                  diss = "1", points = "gll", corr = "g2")


# diss = 2
trixi_convergence("linadv1d", "mdrk", nx_array, bflux = extrapolate, points = "gll", corr = "g2")
trixi_convergence("linadv1d", "lwfr", nx_array, bflux = extrapolate, points = "gll", corr = "g2")

# rkfr

trixi_convergence("linadv1d", "rkfr", nx_array, points = "gll", corr = "g2")

### Variable Advection
# Diss = 1
trixi_convergence("or2", "mdrk", nx_array, bflux = evaluate, diss = "1")
trixi_convergence("or2", "mdrk", nx_array, bflux = extrapolate, diss = "1")
trixi_convergence("or2", "lwfr", nx_array, bflux = evaluate, diss = "1")
trixi_convergence("or2", "lwfr", nx_array, bflux = extrapolate, diss = "1")

# Diss = 2
trixi_convergence("or2", "mdrk", nx_array, bflux = evaluate)
trixi_convergence("or2", "mdrk", nx_array, bflux = extrapolate)
trixi_convergence("or2", "lwfr", nx_array, bflux = evaluate)
trixi_convergence("or2", "lwfr", nx_array, bflux = extrapolate)

# RKFR
trixi_convergence("or2", "rkfr", nx_array, bflux = evaluate)

