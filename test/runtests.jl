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
