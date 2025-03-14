using TrixiBase
using Tenkai
using Test

function get_errors(sol)
    return sol["errors"]["l1_error"], sol["errors"]["l2_error"], sol["errors"]["energy"]
end

function compare_errors(sol, l1_error, l2_error, energy; tol = 1e-14)
    @test isapprox(sol["errors"]["l1_error"], l1_error, atol = tol, rtol = tol)
    @test isapprox(sol["errors"]["l2_error"], l2_error, atol = tol, rtol = tol)
    @test isapprox(sol["errors"]["energy"], energy, atol = tol, rtol = tol)
end

@testset "Burgers' equation 1D" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_burg1d_smooth_sin.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    compare_errors(sol, 0.00442859103683705, 0.005717202714983488, 0.4999471330000709)
    # @show get_errors(sol)
end
