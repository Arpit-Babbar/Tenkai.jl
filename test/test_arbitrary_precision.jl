#-------------------------------------------------------------------------------
# Tests for running Tenkai in an arbitrary floating point arithmetic.
#
# Everything in Tenkai is generic in the number type: the type is taken from the
# domain of the `Problem` and is threaded through the grid, the quadrature nodes
# and FR operators, the solution arrays and the error computation. These tests
# use the double-double types `Float64x2` (MultiFloats.jl) and `Double64`
# (DoubleFloats.jl), which carry about 32 decimal digits.
#
# The headline test is an order of accuracy study that keeps showing the optimal
# convergence rate down to errors of about 1e-22, i.e. six orders of magnitude
# below what `Float64` can resolve at all.
#-------------------------------------------------------------------------------

using Test
using Printf
using StaticArrays
using LinearAlgebra: norm

using MultiFloats
using DoubleFloats

using Tenkai
using Tenkai: EqLinAdv1D, EqLinAdv2D, EqBurg1D, EqBurg2D, EqEuler1D, EqEuler2D

# `MultiFloats` does not implement transcendental functions natively; this makes
# them fall back to `BigFloat`, which is accurate but slow. Only the initial
# condition and the exact solution use them, so this does not affect the cost of
# the time stepping.
MultiFloats.use_bigfloat_transcendentals()
setprecision(BigFloat, 256)

const F64x2 = Float64x2

#-------------------------------------------------------------------------------
# Test problems. They are written generically so that every constant is formed
# in the arithmetic being used, rather than being rounded to `Float64`.
#-------------------------------------------------------------------------------

# 1D linear advection with unit velocity
la_velocity(x) = one(x)
la_iv(x) = sinpi(2 * x)
la_exact(x, t) = la_iv(x - t)

# 1D Burgers. The characteristics of `u0(x) = sin(2 pi x) / 2` first cross at
# t = 1/pi, so the solution is smooth for the times used here and follows from
# tracing the characteristics back. The amplitude also has to be large enough
# that the CFL condition, and not the final time, sets the time step on the
# coarsest grid; otherwise every grid takes a single step of the same size and
# the temporal error stops converging.
burg_iv(x) = sinpi(2 * x) / 2
function burg_exact(x, t)
    u = burg_iv(x)
    for _ in 1:100
        f = u - burg_iv(x - t * u)
        df = 1 + t * oftype(u, pi) * cospi(2 * (x - t * u))
        du = f / df
        u -= du
        iszero(du) && break
    end
    return u
end

# 1D Euler density wave
heat_capacity_ratio(::Type{RealT}) where {RealT} = RealT(7) / 5
function euler_iv(x)
    RealT = typeof(x)
    γ = heat_capacity_ratio(RealT)
    ρ = 1 + sinpi(2 * x) / 2
    v, p = one(RealT), one(RealT)
    return SVector(ρ, ρ * v, p / (γ - 1) + ρ * v^2 / 2)
end
euler_exact(x, t) = euler_iv(x - t)

# 2D linear advection with unit velocity in both directions
la2_velocity(x, y) = SVector(one(x), one(y))
la2_iv(x, y) = SVector(sinpi(2 * x) * sinpi(2 * y))
la2_exact(x, y, t) = la2_iv(x - t, y - t)

# 2D Burgers. The flux is (u^2/2, u^2/2), so the characteristics move with speed
# u in both directions and the first crossing is at t = 1/(2 pi).
burg2_iv(x, y) = SVector(sinpi(2 * (x + y)) / 2)
function burg2_exact(x, y, t)
    u = burg2_iv(x, y)[1]
    for _ in 1:100
        f = u - sinpi(2 * (x + y - 2 * t * u)) / 2
        df = 1 + 2 * t * oftype(u, pi) * cospi(2 * (x + y - 2 * t * u))
        du = f / df
        u -= du
        iszero(du) && break
    end
    return SVector(u)
end

# 2D Euler density wave. The advection velocity is of the same order as the
# sound speed, so that the wave actually travels a useful distance in the time
# the acoustic CFL condition allows; with a much slower wave the error stays
# dominated by the initial interpolation and the measured order is degraded.
function euler2_iv(x, y)
    RealT = typeof(x)
    γ = heat_capacity_ratio(RealT)
    ρ = 1 + sinpi(2 * (x + y)) / 2
    v1, v2 = one(RealT), one(RealT)
    p = one(RealT)
    return SVector(ρ, ρ * v1, ρ * v2, p / (γ - 1) + ρ * (v1^2 + v2^2) / 2)
end
euler2_exact(x, y, t) = euler2_iv(x - t, y - t)

#-------------------------------------------------------------------------------
# Drivers
#-------------------------------------------------------------------------------
function solver_object(name)
    name == "cRK22" ? cRK22() :
    name == "cRK33" ? cRK33() :
    name == "cRK44" ? cRK44() : name
end

function equation_data_1d(eqname, ::Type{RealT}) where {RealT}
    if eqname == "linadv"
        return la_iv, la_exact, EqLinAdv1D.get_equation(la_velocity),
               EqLinAdv1D.rusanov
    elseif eqname == "burg"
        return burg_iv, burg_exact, EqBurg1D.get_equation(), EqBurg1D.rusanov
    elseif eqname == "euler"
        return euler_iv, euler_exact,
               EqEuler1D.get_equation(heat_capacity_ratio(RealT)), EqEuler1D.rusanov
    end
    error("unknown 1d equation $eqname")
end

function equation_data_2d(eqname, ::Type{RealT}) where {RealT}
    if eqname == "linadv"
        return la2_iv, la2_exact, EqLinAdv2D.get_equation(la2_velocity),
               EqLinAdv2D.rusanov
    elseif eqname == "burg"
        return burg2_iv, burg2_exact, EqBurg2D.get_equation(), EqBurg2D.rusanov
    elseif eqname == "euler"
        return euler2_iv, euler2_exact,
               EqEuler2D.get_equation(heat_capacity_ratio(RealT)), EqEuler2D.rusanov
    end
    error("unknown 2d equation $eqname")
end

"""
    l2_error_1d(RealT, eqname, solver, degree, nx, final_time)

Run a 1D periodic test problem in the arithmetic `RealT` and return the L2 error
of the first conservative variable, in that same arithmetic.
"""
function l2_error_1d(::Type{RealT}, eqname, solver, degree, nx, final_time;
                     time_scheme = "by degree",
                     cfl_safety_factor = 0.98) where {RealT}
    initial_value, exact_solution, equation, numerical_flux = equation_data_1d(eqname,
                                                                               RealT)
    domain = [zero(RealT), one(RealT)]
    problem = Problem(domain, initial_value, exact_solution, (periodic, periodic),
                      RealT(final_time), exact_solution)
    scheme = Scheme(solver_object(solver), degree, "gl", "radau", numerical_flux, "no",
                    setup_limiter_none(), evaluate)
    param = Parameters(nx, zero(RealT), ([-RealT(Inf)], [RealT(Inf)]), 0,
                       zero(RealT), 0; time_scheme,
                       cfl_safety_factor = RealT(cfl_safety_factor))
    sol = Tenkai.solve(equation, problem, scheme, param)
    return sol["errors"]["l2_error"]
end

function l2_error_2d(::Type{RealT}, eqname, solver, degree, nx, final_time;
                     time_scheme = "by degree",
                     cfl_safety_factor = 0.98) where {RealT}
    initial_value, exact_solution, equation, numerical_flux = equation_data_2d(eqname,
                                                                               RealT)
    domain = [zero(RealT), one(RealT), zero(RealT), one(RealT)]
    boundary_condition = (periodic, periodic, periodic, periodic)
    problem = Problem(domain, initial_value, exact_solution, boundary_condition,
                      RealT(final_time), exact_solution)
    scheme = Scheme(solver_object(solver), degree, "gl", "radau", numerical_flux, "no",
                    setup_limiter_none(), evaluate)
    param = Parameters([nx, nx], zero(RealT), ([-RealT(Inf)], [RealT(Inf)]), 0,
                       zero(RealT), 0; time_scheme,
                       cfl_safety_factor = RealT(cfl_safety_factor))
    sol = Tenkai.solve(equation, problem, scheme, param)
    return sol["errors"]["l2_error"]
end

# Runge-Kutta time integration for the RKFR solver. Tenkai's own schemes are
# used rather than the OrdinaryDiffEq ones (`Tsit5`, `SSPRK54`), because those
# require `Base` methods that `MultiFloat` does not provide (`rem`).
function rk_time_scheme(solver, degree)
    solver == "rkfr" || return "by degree"
    return degree == 1 ? "SSPRK22" : degree == 2 ? "SSPRK33" : "RK4"
end

# `get_cfl` returns the stability limit of the time integrator that "by degree"
# would have picked, which for degrees 3 and 4 is the five stage `SSPRK54`. The
# classical `RK4` used here takes fewer stages per step and so has a smaller
# stability region, and needs a correspondingly smaller time step.
rk_cfl_safety_factor(solver, degree) = (solver == "rkfr" && degree >= 3) ? 0.4 : 0.98

# Order of the fully discrete scheme: the degree of the solution space gives
# order `degree + 1` in space, and the time integrator can cap that.
function expected_order(solver, degree)
    if solver == "rkfr"
        time_order = degree == 1 ? 2 : degree == 2 ? 3 : 4
        return min(degree + 1, time_order)
    elseif solver == "mdrk"
        return min(degree + 1, 4)
    elseif solver == "cRK22"
        return min(degree + 1, 2)
    elseif solver == "cRK33"
        return min(degree + 1, 3)
    elseif solver == "cRK44"
        return min(degree + 1, 4)
    else # LWFR is order degree + 1 in space and time simultaneously
        return degree + 1
    end
end

"""
    solver_degree_pairs(dimension)

Every (solver, degree) combination that Tenkai supports on Cartesian grids.
`cRK44` with degree below 3 is skipped in 1D: its cell data cache is sized for
the higher degrees and indexing it out of bounds is a pre-existing failure,
independent of the arithmetic (it fails the same way in `Float64`).
"""
function solver_degree_pairs(dimension)
    pairs = Tuple{String, Int}[]
    for solver in ("lwfr", "rkfr", "mdrk", "cRK22", "cRK33", "cRK44"),
        degree in 1:4

        dimension == 1 && solver == "cRK44" && degree < 3 && continue
        push!(pairs, (solver, degree))
    end
    return pairs
end

"""
    convergence_rates(errors, refinements)

Experimental orders of accuracy between successive grids, each of which refines
the previous one by a factor of two.
"""
convergence_rates(errors) = [log2(errors[i - 1] / errors[i]) for i in 2:length(errors)]

"""
    check_order(errors, expected_order; label, atol_rate)

Report the convergence table and assert that the mean experimental order of
accuracy is within `atol_rate` of `expected_order`.
"""
function check_order(errors, expected_order; label = "", atol_rate = 0.35)
    rates = convergence_rates(errors)
    mean_rate = sum(rates) / length(rates)
    # Printed rather than logged so that the convergence tables appear in the
    # test output in order, which is the point of these tests.
    @printf("CONVERGENCE  %-46s order %.3f (expected %d)  errors %s  rates %s\n",
            label, mean_rate, expected_order,
            join([@sprintf("%.3e", e) for e in errors], " "),
            join([@sprintf("%.2f", r) for r in rates], " "))
    flush(stdout)
    @test mean_rate > expected_order - atol_rate
    return mean_rate
end

#-------------------------------------------------------------------------------

@testset "Arbitrary precision arithmetic" begin
    #-------------------------------------------------------------------------------
    # The building blocks: quadrature, FR operators and grid coordinates must all be
    # accurate to the precision of the number type, not to `Float64` precision.
    #-------------------------------------------------------------------------------
    @testset "Quadrature nodes and weights" begin
        for RealT in (Float64, F64x2, Double64), points in ("gl", "gll"), n in 2:6
            x, w = Tenkai.Basis.weights_and_points(n, points, RealT)
            # An n-point Gauss-Legendre rule on [0,1] is exact for polynomials up to
            # degree 2n-1; Gauss-Lobatto up to degree 2n-3.
            max_degree = points == "gl" ? 2n - 1 : 2n - 3
            for p in 0:max_degree
                quadrature = sum(w[i] * x[i]^p for i in 1:n)
                @test isapprox(quadrature, one(RealT) / (p + 1),
                               atol = 8 * eps(RealT), rtol = 8 * eps(RealT))
            end
        end
    end

    @testset "FR operators" begin
        # `Float64x4` carries twice the precision of `Float64x2` and is used as the
        # reference. If any part of the operator setup fell back to `Float64` the
        # difference would be about 1e-16 instead of about 1e-31.
        for degree in 1:4
            reference = Tenkai.fr_operators(degree, "gl", "radau", Float64x4)
            for RealT in (F64x2, Double64)
                op = Tenkai.fr_operators(degree, "gl", "radau", RealT)
                for key in (:xg, :wg, :Vl, :Vr, :bl, :br, :Dm, :D1)
                    computed = BigFloat.(getproperty(op, key))
                    expected = BigFloat.(getproperty(reference, key))
                    @test maximum(abs, computed .- expected) < 1000 * eps(RealT)
                end
            end
        end
    end

    @testset "Uniform grid coordinates" begin
        # Regression test: building the coordinates with `LinRange` interpolates
        # with a `Float64` parameter and silently caps the grid at `Float64`
        # accuracy for any higher precision number type.
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

    #-------------------------------------------------------------------------------
    # Order of accuracy in 1D, over the solvers, degrees and equations that Tenkai
    # supports on Cartesian grids.
    #-------------------------------------------------------------------------------
    @testset "Order of accuracy, 1D" begin
        final_time = 1 // 20
        grids = (20, 40, 80)
        for eqname in ("linadv", "burg", "euler"),
            (solver, degree) in solver_degree_pairs(1)

            errors = [Float64(l2_error_1d(F64x2, eqname, solver, degree, nx, final_time;
                                          time_scheme = rk_time_scheme(solver, degree),
                                          cfl_safety_factor = rk_cfl_safety_factor(solver,
                                                                                   degree)))
                      for nx in grids]
            check_order(errors, expected_order(solver, degree);
                        label = "1D $eqname, $solver, degree $degree")
        end
    end

    #-------------------------------------------------------------------------------
    # Order of accuracy in 2D
    #-------------------------------------------------------------------------------
    @testset "Order of accuracy, 2D" begin
        final_time = 1 // 20
        grids = (10, 20, 40)
        for eqname in ("linadv", "burg", "euler"),
            (solver, degree) in solver_degree_pairs(2)

            errors = [Float64(l2_error_2d(F64x2, eqname, solver, degree, nx, final_time;
                                          time_scheme = rk_time_scheme(solver, degree),
                                          cfl_safety_factor = rk_cfl_safety_factor(solver,
                                                                                   degree)))
                      for nx in grids]
            check_order(errors, expected_order(solver, degree);
                        label = "2D $eqname, $solver, degree $degree")
        end
    end

    #-------------------------------------------------------------------------------
    # The point of the whole exercise: with a double-double arithmetic the errors
    # keep falling at the optimal rate far below `eps(Float64)`. In `Float64` this
    # table would flatten out at about 1e-16.
    #-------------------------------------------------------------------------------
    @testset "Convergence below Float64 precision" begin
        final_time = 1 // 100
        grids = (160, 320, 640, 1280, 2560, 5120, 10240)
        errors = [Float64(l2_error_1d(F64x2, "linadv", "lwfr", 4, nx, final_time))
                  for nx in grids]
        check_order(errors, 5; label = "1D linear advection, LWFR degree 4, Float64x2")

        # The table has to actually get down to ~1e-22, which is what makes this a
        # test of the arithmetic and not just of the scheme.
        @test errors[end] < 2e-21
        # ... and every refinement must still be reducing the error, i.e. there is
        # no round-off floor anywhere in the table.
        @test all(errors[i] < errors[i - 1] / 20 for i in 2:length(errors))
    end

    @testset "Convergence below Float64 precision, Double64" begin
        # The same study with the other double-double implementation, to show that
        # nothing in Tenkai is tied to one particular number type.
        final_time = 1 // 100
        grids = (160, 320, 640, 1280, 2560)
        errors = [Float64(l2_error_1d(Double64, "linadv", "lwfr", 4, nx, final_time))
                  for nx in grids]
        check_order(errors, 5; label = "1D linear advection, LWFR degree 4, Double64")
        @test errors[end] < 2e-18
    end

    @testset "Deep convergence, 2D" begin
        # In 2D the error is dominated by the interpolation error of the initial
        # condition, so pushing it below eps(Float64) would need of the order of
        # 1000 x 1000 cells. This checks that the optimal order survives four
        # refinements down to about 1e-14, well past the point where the errors are
        # small enough that a `Float64` run would be dominated by round-off in the
        # error computation itself.
        final_time = 1 // 2000
        grids = (40, 80, 160, 320)
        errors = [Float64(l2_error_2d(F64x2, "linadv", "lwfr", 4, nx, final_time))
                  for nx in grids]
        check_order(errors, 5; label = "2D linear advection, LWFR degree 4, Float64x2")
        @test errors[end] < 1e-13
        @test all(errors[i] < errors[i - 1] / 20 for i in 2:length(errors))
    end

    #-------------------------------------------------------------------------------
    # Lower precision still has to work.
    #-------------------------------------------------------------------------------
    @testset "Float32" begin
        errors = [Float64(l2_error_1d(Float32, "linadv", "lwfr", 2, nx, 1 // 20))
                  for nx in (20, 40, 80)]
        check_order(errors, 3; label = "1D linear advection, LWFR degree 2, Float32")
    end
end
