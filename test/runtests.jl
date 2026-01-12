using TrixiBase
using Tenkai
using DelimitedFiles
using Test
import Tenkai.Trixi
using Tenkai.TenkaicRK

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

solver2string(s::String) = s
solver2string(s::cRK22) = "cRK22"
solver2string(s::cRK33) = "cRK33"
solver2string(s::cRK44) = "cRK44"
solver2string(s::cIMEX111) = "cIMEX111"
solver2string(s::cHT112) = "cHT112"
solver2string(s::cSSP2IMEX433) = "cSSP2IMEX433"
solver2string(s::cSSP2IMEX222) = "cSSP2IMEX222"
solver2string(s::cSSP2IMEX332) = "cSSP2IMEX332"
solver2string(s::LWTDEltWise) = "LWTDEltWise"

# Returns `all` if the key `TENKAI_TEST` is not found
const TENKAI_TEST = get(ENV, "TENKAI_TEST", "all")

println("Running tests for TENKAI_TEST = $TENKAI_TEST")

if TENKAI_TEST == "part_1" || TENKAI_TEST == "all"
    include("test_part1.jl")
end

if TENKAI_TEST == "crk_imex_paper" || TENKAI_TEST == "all"
    include("test_crk_imex_paper.jl")
end
