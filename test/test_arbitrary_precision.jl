# Claude written tests for Tenkai in arbitrary floating point arithmetic. The number
# type is taken from the domain of the `Problem` and threaded through the grid,
# the quadrature nodes and FR operators, the solution arrays and the error
# computation. The headline test is an order of accuracy study that keeps showing
# the optimal rate down to errors of about 1e-22, far below what `Float64` can
# resolve. `Float64x2` is from MultiFloats.jl and `Double64` from DoubleFloats.jl;
# both carry about 32 decimal digits.

using Test
using Printf
using StaticArrays
using DelimitedFiles: readdlm

using MultiFloats
using DoubleFloats

using Tenkai
using Tenkai: EqLinAdv1D, EqLinAdv2D, EqBurg1D, EqBurg2D, EqEuler1D, EqEuler2D

# MultiFloats has no native transcendental functions; this falls back to
# `BigFloat`. Only the initial condition and the exact solution use them.
MultiFloats.use_bigfloat_transcendentals()
setprecision(BigFloat, 256)

const F64x2 = Float64x2

#-------------------------------------------------------------------------------
# Test problems, written so that every constant is formed in the arithmetic in
# use rather than rounded to `Float64`. The amplitudes and velocities are chosen
# so that the CFL condition, and not the final time, sets the time step even on
# the coarsest grid; otherwise every grid takes one step of the same size and the
# temporal error stops converging.
#-------------------------------------------------------------------------------
heat_capacity_ratio(::Type{RealT}) where {RealT} = RealT(7) / 5

la_iv(x) = sinpi(2 * x)
la_exact(x, t) = la_iv(x - t)
la2_iv(x, y) = SVector(sinpi(2 * x) * sinpi(2 * y))
la2_exact(x, y, t) = la2_iv(x - t, y - t)

# Burgers. The characteristics of `sin(2 pi x) / 2` first cross at t = 1/pi in 1D
# and t = 1/(2 pi) in 2D, so the solution is smooth for the times used here and
# follows from tracing them back.
burg_iv(x) = sinpi(2 * x) / 2
burg2_iv(x, y) = SVector(sinpi(2 * (x + y)) / 2)

function newton_characteristic(u, residual, derivative)
    for _ in 1:100
        du = residual(u) / derivative(u)
        u -= du
        iszero(du) && break
    end
    return u
end

function burg_exact(x, t)
    newton_characteristic(burg_iv(x), u -> u - burg_iv(x - t * u),
                          u -> 1 + t * oftype(u, pi) * cospi(2 * (x - t * u)))
end

function burg2_exact(x, y, t)
    s = x + y
    u = newton_characteristic(burg2_iv(x, y)[1],
                              u -> u - sinpi(2 * (s - 2 * t * u)) / 2,
                              u -> 1 + 2 * t * oftype(u, pi) * cospi(2 * (s - 2 * t * u)))
    return SVector(u)
end

# Euler density wave. The advection velocity is of the same order as the sound
# speed, so the wave travels a useful distance in the time the acoustic CFL
# condition allows.
function euler_iv(x)
    RealT = typeof(x)
    γ = heat_capacity_ratio(RealT)
    ρ, v, p = 1 + sinpi(2 * x) / 2, one(RealT), one(RealT)
    return SVector(ρ, ρ * v, p / (γ - 1) + ρ * v^2 / 2)
end
euler_exact(x, t) = euler_iv(x - t)

function euler2_iv(x, y)
    RealT = typeof(x)
    γ = heat_capacity_ratio(RealT)
    ρ, v, p = 1 + sinpi(2 * (x + y)) / 2, one(RealT), one(RealT)
    return SVector(ρ, ρ * v, ρ * v, p / (γ - 1) + ρ * v^2)
end
euler2_exact(x, y, t) = euler2_iv(x - t, y - t)

# (initial value, exact solution, equation, numerical flux) for each test problem
function test_problem(dimension, eqname, ::Type{RealT}) where {RealT}
    γ = heat_capacity_ratio(RealT)
    if dimension == 1
        eqname == "linadv" &&
            return la_iv, la_exact, EqLinAdv1D.get_equation(one), EqLinAdv1D.rusanov
        eqname == "burg" &&
            return burg_iv, burg_exact, EqBurg1D.get_equation(), EqBurg1D.rusanov
        eqname == "euler" &&
            return euler_iv, euler_exact, EqEuler1D.get_equation(γ), EqEuler1D.rusanov
    else
        eqname == "linadv" &&
            return la2_iv, la2_exact,
                   EqLinAdv2D.get_equation((x, y) -> SVector(one(x), one(y))),
                   EqLinAdv2D.rusanov
        eqname == "burg" &&
            return burg2_iv, burg2_exact, EqBurg2D.get_equation(), EqBurg2D.rusanov
        eqname == "euler" &&
            return euler2_iv, euler2_exact, EqEuler2D.get_equation(γ), EqEuler2D.rusanov
    end
    error("unknown $(dimension)d equation $eqname")
end

#-------------------------------------------------------------------------------
# Drivers
#-------------------------------------------------------------------------------
function solver_object(name)
    name == "cRK22" ? cRK22() :
    name == "cRK33" ? cRK33() :
    name == "cRK44" ? cRK44() : name
end

"""
    l2_error(RealT, dimension, eqname, solver, degree, nx, final_time)

Run a periodic test problem on an `nx^dimension` grid in the arithmetic `RealT`
and return the L2 error of the first conservative variable, in that arithmetic.
"""
function l2_error(::Type{RealT}, dimension, eqname, solver, degree, nx, final_time;
                  time_scheme = "by degree", cfl_safety_factor = 0.98) where {RealT}
    initial_value, exact_solution, equation, numerical_flux = test_problem(dimension,
                                                                           eqname, RealT)
    ends = (zero(RealT), one(RealT))
    domain = dimension == 1 ? [ends...] : [ends..., ends...]
    problem = Problem(domain, initial_value, exact_solution,
                      ntuple(_ -> periodic, 2 * dimension), RealT(final_time),
                      exact_solution)
    scheme = Scheme(solver_object(solver), degree, "gl", "radau", numerical_flux, "no",
                    setup_limiter_none(), evaluate)
    param = Parameters(dimension == 1 ? nx : [nx, nx], zero(RealT),
                       ([-RealT(Inf)], [RealT(Inf)]), 0, zero(RealT), 0;
                       time_scheme, cfl_safety_factor = RealT(cfl_safety_factor))
    return Tenkai.solve(equation, problem, scheme, param)["errors"]["l2_error"]
end

# Tenkai's own Runge-Kutta schemes are used for RKFR rather than the
# OrdinaryDiffEq ones (`Tsit5`, `SSPRK54`), which need `Base` methods that
# `MultiFloat` does not provide (`rem`). `get_cfl` warns that its CFL numbers are
# calibrated for `SSPRK54` at degrees 3 and 4; a smaller safety factor is the
# remedy it suggests.
function rk_time_scheme(solver, degree)
    solver != "rkfr" ? "by degree" :
    degree == 1 ? "SSPRK22" :
    degree == 2 ? "SSPRK33" : "RK4"
end
rk_cfl_safety_factor(solver, degree) = (solver == "rkfr" && degree >= 3) ? 0.4 : 0.98

# Order of the fully discrete scheme: the degree gives order `degree + 1` in
# space, and the time integrator can cap it.
function expected_order(solver, degree)
    solver == "lwfr" && return degree + 1  # order degree + 1 in space and time
    solver == "rkfr" && return min(degree + 1, degree == 1 ? 2 : degree == 2 ? 3 : 4)
    solver == "cRK22" && return min(degree + 1, 2)
    solver == "cRK33" && return min(degree + 1, 3)
    return min(degree + 1, 4)  # mdrk, cRK44
end

# Enough cases to run every solver in both dimensions, and every degree and
# every equation somewhere, without the full product of the four. LWFR appears at
# all four degrees because its Lax-Wendroff residual and boundary flux are a
# different function for each degree; the other solvers have one residual each,
# so one case per dimension covers them.
const CASES = [(1, "linadv", "lwfr", 1), (1, "burg", "lwfr", 2),
    (1, "euler", "lwfr", 3), (1, "linadv", "lwfr", 4),
    (2, "linadv", "lwfr", 1), (2, "burg", "lwfr", 2),
    (2, "euler", "lwfr", 3), (2, "linadv", "lwfr", 4),
    (1, "burg", "rkfr", 3), (1, "euler", "mdrk", 3),
    (1, "linadv", "cRK22", 2), (1, "burg", "cRK33", 3),
    (1, "euler", "cRK44", 4),
    (2, "burg", "rkfr", 3), (2, "euler", "mdrk", 3),
    (2, "linadv", "cRK22", 2), (2, "burg", "cRK33", 3),
    (2, "euler", "cRK44", 4)]

# Experimental orders of accuracy between successive grids, each refining the
# previous one by a factor of two
convergence_rates(errors) = [log2(errors[i - 1] / errors[i]) for i in 2:length(errors)]

"""
    check_order(errors, expected_order; label, atol_rate)

Print the convergence table and assert that the mean experimental order of
accuracy is within `atol_rate` of `expected_order`. Printed rather than logged so
the tables appear in order in the test output, which is the point of these tests.
"""
function check_order(errors, expected_order; label = "", atol_rate = 0.35)
    rates = convergence_rates(errors)
    mean_rate = sum(rates) / length(rates)
    @printf("CONVERGENCE  %-46s order %.3f (expected %d)  errors %s  rates %s\n",
            label, mean_rate, expected_order,
            join([@sprintf("%.3e", e) for e in errors], " "),
            join([@sprintf("%.2f", r) for r in rates], " "))
    flush(stdout)
    @test mean_rate > expected_order - atol_rate
    return mean_rate
end

#-------------------------------------------------------------------------------
# Stored L2 errors for the solver/degree/equation/dimension sweep, regenerated by
# setting `overwrite_errors = true` in test/runtests.jl as elsewhere in the
# suite. One file rather than one per case. The
# values are written and read back at full precision, so the file does not lose
# any of the digits the double-double arithmetic produced.
#-------------------------------------------------------------------------------
const OVERWRITE_ERRORS = @isdefined(overwrite_errors) ? overwrite_errors : false
const REFERENCE_FILE = joinpath(@__DIR__, "data", "arbitrary_precision_errors.txt")

function reference_errors(::Type{RealT}) where {RealT}
    isfile(REFERENCE_FILE) || return Dict{String, RealT}()
    data = readdlm(REFERENCE_FILE, String)
    return Dict(data[i, 1] => RealT(parse(BigFloat, data[i, 2])) for i in axes(data, 1))
end

function write_reference_errors(errors)
    println("Overwriting $REFERENCE_FILE, this should not be triggered in actual testing.")
    open(REFERENCE_FILE, "w") do io
        for key in sort(collect(keys(errors)))
            @printf(io, "%-28s %.32e\n", key, BigFloat(errors[key]))
        end
    end
end

#-------------------------------------------------------------------------------

@testset "Arbitrary precision arithmetic" begin
    # The building blocks must be accurate to the precision of the number type,
    # not to `Float64` precision.
    @testset "Quadrature nodes and weights" begin
        for RealT in (Float64, F64x2, Double64), points in ("gl", "gll"), n in 2:6
            x, w = Tenkai.Basis.weights_and_points(n, points, RealT)
            # `n`-point Gauss-Legendre on [0,1] is exact up to degree 2n-1,
            # Lobatto up to 2n-3
            for p in 0:(points == "gl" ? 2n - 1 : 2n - 3)
                @test isapprox(sum(w[i] * x[i]^p for i in 1:n), one(RealT) / (p + 1),
                               atol = 8 * eps(RealT), rtol = 8 * eps(RealT))
            end
        end
    end

    @testset "FR operators" begin
        # `Float64x4` has twice the precision of `Float64x2` and is the
        # reference. If any part of the setup fell back to `Float64` the
        # difference would be about 1e-16 instead of about 1e-31.
        for degree in 1:4
            reference = Tenkai.fr_operators(degree, "gl", "radau", Float64x4)
            for RealT in (F64x2, Double64)
                op = Tenkai.fr_operators(degree, "gl", "radau", RealT)
                for key in (:xg, :wg, :Vl, :Vr, :bl, :br, :Dm, :D1)
                    @test maximum(abs,
                                  BigFloat.(getproperty(op, key)) .-
                                  BigFloat.(getproperty(reference, key))) <
                          1000 * eps(RealT)
                end
            end
        end
    end

    @testset "Uniform grid coordinates" begin
        # Regression test: building these with `LinRange` caps them at `Float64`.
        for RealT in (F64x2, Double64), nx in (16, 160)
            problem = Problem([zero(RealT), one(RealT)], la_iv, la_exact,
                              (periodic, periodic), RealT(1) / 10, la_exact)
            grid = Tenkai.make_cartesian_grid(problem, nx)
            for i in 1:(nx + 1)
                @test abs(grid.xf[i] - BigFloat(i - 1) / nx) < 8 * eps(RealT)
            end
            for i in 1:nx
                @test abs(grid.xc[i] - (BigFloat(i) - 1 // 2) / nx) < 8 * eps(RealT)
            end
        end
    end

    # Each case run once and checked against a stored error. This fixes the
    # values, and with them that each solver runs at all in an arithmetic other
    # than `Float64`. The order of accuracy is demonstrated by the studies below.
    @testset "Solvers, degrees, equations and dimensions" begin
        reference = reference_errors(F64x2)
        computed = Dict{String, Any}()
        for (dimension, eqname, solver, degree) in CASES
            key = "$(dimension)d_$(eqname)_$(solver)_$(degree)"
            computed[key] = l2_error(F64x2, dimension, eqname, solver, degree,
                                     dimension == 1 ? 40 : 20, 1 // 20;
                                     time_scheme = rk_time_scheme(solver, degree),
                                     cfl_safety_factor = rk_cfl_safety_factor(solver,
                                                                              degree))
            OVERWRITE_ERRORS || @test isapprox(computed[key], reference[key],
                                               rtol = 1e-9)
        end
        # Lower precision still has to work
        computed["float32_1d_linadv_lwfr_2"] = l2_error(Float32, 1, "linadv", "lwfr", 2,
                                                        40, 1 // 20)
        OVERWRITE_ERRORS ||
            @test isapprox(Float64(computed["float32_1d_linadv_lwfr_2"]),
                           Float64(reference["float32_1d_linadv_lwfr_2"]), rtol = 1e-5)
        OVERWRITE_ERRORS && write_reference_errors(computed)
    end

    # The point of the exercise: with a double-double arithmetic the errors keep
    # falling at the optimal rate far below `eps(Float64)`, where a `Float64` run
    # would flatten out at about 1e-16.
    @testset "Convergence below Float64 precision" begin
        for (RealT, grids, bound) in ((F64x2, (160, 320, 640, 1280, 2560, 5120, 10240),
                                       2e-21),
                                      (Double64, (160, 320, 640, 1280, 2560), 2e-18))
            errors = [Float64(l2_error(RealT, 1, "linadv", "lwfr", 4, nx, 1 // 100))
                      for nx in grids]
            check_order(errors, 5; label = "1D linear advection, LWFR degree 4, $RealT")
            # The table has to actually reach these errors, which is what makes
            # this a test of the arithmetic and not just of the scheme, and every
            # refinement must still reduce the error, i.e. there is no floor.
            @test errors[end] < bound
            @test all(errors[i] < errors[i - 1] / 20 for i in 2:length(errors))
        end
    end

    @testset "Deep convergence, 2D" begin
        # In 2D the error is dominated by the interpolation error of the initial
        # condition, so reaching below eps(Float64) would need of the order of
        # 1000 x 1000 cells. This checks the optimal order survives to about
        # 1e-14.
        errors = [Float64(l2_error(F64x2, 2, "linadv", "lwfr", 4, nx, 1 // 2000))
                  for nx in (40, 80, 160, 320)]
        check_order(errors, 5; label = "2D linear advection, LWFR degree 4, Float64x2")
        @test errors[end] < 1e-13
        @test all(errors[i] < errors[i - 1] / 20 for i in 2:length(errors))
    end
end
