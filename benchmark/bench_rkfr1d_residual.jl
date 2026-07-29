# CPU-vs-GPU benchmark for the full RK-1D per-stage residual pipeline
# (cell average, cell residual, boundary extrapolation, ghost values, face
# residual -- everything compute_residual_rkfr_gpu!/compute_residual_rkfr!
# do for one stage), not just the isolated flux functions in
# bench_euler1d_flux.jl.

using Tenkai
using Metal
using KernelAbstractions

isdefined(@__MODULE__, :BenchHarness) || include("harness.jl")
using .BenchHarness

Eq = Tenkai.EqEuler1D

function rkfr1d_setup(nx; backend = KernelAbstractions.CPU())
    RealT = backend isa KernelAbstractions.CPU ? Float64 : Float32
    domain = RealT[0, 1]
    problem = Problem(domain, Eq.dwave, Eq.dummy_zero_boundary_value,
                      (periodic, periodic), RealT(0.1), (x, t) -> Eq.dwave(x - t))
    equation = Eq.get_equation(RealT(1.4))
    scheme = Scheme("rkfr", 3, "gl", "radau", Eq.rusanov, "no", setup_limiter_none(),
                    evaluate)
    param = Parameters(nx, RealT(0), (RealT[-Inf], RealT[Inf]), 0, RealT(0), 0)

    grid = Tenkai.make_cartesian_grid(problem, param.grid_size)
    op = Tenkai.fr_operators(scheme.degree, scheme.solution_points,
                             scheme.correction_function, RealT)
    cache = (; Tenkai.setup_arrays(grid, scheme, equation; backend)...)
    aux = Tenkai.create_auxiliaries(equation, op, grid, problem, scheme, param, cache)
    Tenkai.set_initial_condition!(cache.u1, equation, grid, op, problem)

    dt = RealT(1e-3)
    return (; equation, problem, scheme, param, grid, op, cache, aux, dt)
end

function run_rkfr1d_residual_benchmarks(; backend = Metal.MetalBackend(),
                                        sizes = (20, 100, 500, 2000))
    results = BenchResult[]
    for nx in sizes
        s_cpu = rkfr1d_setup(nx)
        s_gpu = rkfr1d_setup(nx; backend)

        run_cpu! = _ -> Tenkai.compute_residual_rkfr!(s_cpu.equation, s_cpu.problem,
                                                       s_cpu.grid, s_cpu.op, s_cpu.scheme,
                                                       s_cpu.param, s_cpu.aux, 0.0,
                                                       s_cpu.dt, 0, 0, s_cpu.cache,
                                                       s_cpu.cache.u1, s_cpu.cache.Fb,
                                                       s_cpu.cache.ub, s_cpu.cache.ua,
                                                       s_cpu.cache.res)
        run_gpu! = (_, backend) -> Tenkai.compute_residual_rkfr_gpu!(s_gpu.equation,
                                                                     s_gpu.grid, s_gpu.op,
                                                                     s_gpu.problem,
                                                                     s_gpu.scheme, 0.0f0,
                                                                     s_gpu.dt, s_gpu.cache)

        r = bench_pair("rkfr1d_residual", nx; setup_cpu = _ -> nothing,
                       setup_gpu = _ -> nothing, run_cpu!, run_gpu!, backend)
        push!(results, r)
        println(r.name, " nx=", r.size, " speedup=", round(r.speedup, digits = 2), "x")
    end
    return results
end
