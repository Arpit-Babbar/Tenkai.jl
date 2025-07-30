using TrixiBase
using Tenkai
using DelimitedFiles
using Test

overwrite_errors = false

test_data_dir = joinpath(@__DIR__, "data")

function get_errors(sol)
    return [sol["errors"]["l1_error"], sol["errors"]["l2_error"], sol["errors"]["energy"]]'
end

function compare_errors(sol, l1_error, l2_error, energy; tol = 1e-14)
    @test isapprox(sol["errors"]["l1_error"], l1_error, atol = tol, rtol = tol)
    @test isapprox(sol["errors"]["l2_error"], l2_error, atol = tol, rtol = tol)
    @test isapprox(sol["errors"]["energy"], energy, atol = tol, rtol = tol)
end

function compare_errors_txt(sol, testname; tol = 1e-14,
                            overwrite_errors = false)
    datafile = joinpath(test_data_dir, testname)
    if overwrite_errors == true
        println("Overwriting $datafile, this should not be triggered in actual testing.")
        writedlm(datafile, get_errors(sol))
    end
    data = readdlm(datafile)
    l1_error, l2_error, energy = data[1], data[2], data[3]
    @test isapprox(sol["errors"]["l1_error"], l1_error, atol = tol, rtol = tol)
    @test isapprox(sol["errors"]["l2_error"], l2_error, atol = tol, rtol = tol)
    @test isapprox(sol["errors"]["energy"], energy, atol = tol, rtol = tol)
end

@testset "Burgers' equation 1D" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_burg1d_smooth_sin.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "burg1d_smooth_sine.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Blast 1-D" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_blast.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "blast.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Bucklev" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_bucklev.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "bucklev.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Burg 1D hat" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_burg1d_hat.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "burg1d_hat.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MHD 1D alfven" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_mhd_alfven_wave.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "alfven_mhd.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Burgers' equation 2D smooth sin" begin
    trixi_include(joinpath(examples_dir(), "2d", "run_burg2d_smooth_sin.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 5, ny = 5)
    data_name = "burg2d_smooth_sine.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Isentropic 2D" begin
    local solver_degrees = Dict("mdrk" => [3],
                                "lwfr" => [1, 2, 3, 4],
                                "rkfr" => [1, 2, 3, 4])
    for solver in ["mdrk", "lwfr", "rkfr"], degree in solver_degrees[solver]
        trixi_include(joinpath(examples_dir(), "2d", "run_isentropic.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver, degree = degree,
                      limiter = setup_limiter_none(),
                      animate = false, final_time = 1.0, nx = 5, ny = 5)
        data_name = "isentropic_2d_$(solver)_$(degree).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end

    for time_scheme in ["RK4", "Tsit5"]
        trixi_include(joinpath(examples_dir(), "2d", "run_isentropic.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = "rkfr", degree = 4,
                      time_scheme = time_scheme,
                      limiter = setup_limiter_none(),
                      animate = false, final_time = 1.0, nx = 5, ny = 5)
        data_name = "isentropic_2d_rkfr_4_$(time_scheme).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end

@testset "Ten Moment 2D" begin
    for solver in ["mdrk", "lwfr", "rkfr", cRK33()]
        trixi_include(joinpath(examples_dir(), "2d", "run_tenmom_dwave.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver, degree = 3,
                      limiter = setup_limiter_none(),
                      animate = false, final_time = 1.0, nx = 5, ny = 5)
        data_name = "tenmom_dwave_2d_$(solver)_$(degree).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

        trixi_include(joinpath(examples_dir(), "2d", "run_tenmom_near_vacuum.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver, degree = 3,
                      # eq comes from the previous test
                      limiter = setup_limiter_tvbβ(eq; tvbM = 0.0, beta = 0.9),
                      animate = false, final_time = 0.02, nx = 5, ny = 5)
        data_name = "tenmom_near_vacuum_2d_$(solver)_$(degree).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end
