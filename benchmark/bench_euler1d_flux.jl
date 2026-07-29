# CPU-vs-GPU benchmarks for every GPU-kernel-safe function in EqEuler1D.jl
# (flux, rusanov, roe, hllc). See docs/GPU.md for the ceiling methodology.

using Tenkai
using Tenkai: get_node_vars, set_node_vars!
using StaticArrays: SVector
using Metal
using KernelAbstractions

isdefined(@__MODULE__, :BenchHarness) || include("harness.jl")
using .BenchHarness

const EQ = Tenkai.EqEuler1D.get_equation(1.4f0)

function admissible_state(n)
    u = zeros(Float32, 3, n)
    for i in 1:n
        prim = SVector(1.0f0 + rand(Float32), (rand(Float32) - 0.5f0) * 0.5f0,
                       1.0f0 + rand(Float32))
        set_node_vars!(u, Tenkai.EqEuler1D.prim2con(EQ, prim), EQ, i)
    end
    return u
end

@kernel function flux_kernel!(out, @Const(u), eq)
    i = @index(Global)
    set_node_vars!(out, Tenkai.EqEuler1D.flux(0.0f0, get_node_vars(u, eq, i), eq), eq, i)
end

@kernel function numflux_kernel!(out, @Const(u), eq, f)
    i = @index(Global)
    n = size(u, 2)
    ul, ur = get_node_vars(u, eq, i), get_node_vars(u, eq, mod1(i + 1, n))
    fl, fr = Tenkai.EqEuler1D.flux(0.0f0, ul, eq), Tenkai.EqEuler1D.flux(0.0f0, ur, eq)
    set_node_vars!(out, f(0.0f0, ul, ur, fl, fr, ul, ur, eq, 1), eq, i)
end

function cpu_flux!(out, u)
    for i in 1:size(u, 2)
        set_node_vars!(out, Tenkai.EqEuler1D.flux(0.0f0, get_node_vars(u, EQ, i), EQ), EQ,
                       i)
    end
end

function cpu_numflux!(out, u, f)
    n = size(u, 2)
    for i in 1:n
        ul, ur = get_node_vars(u, EQ, i), get_node_vars(u, EQ, mod1(i + 1, n))
        fl, fr = Tenkai.EqEuler1D.flux(0.0f0, ul, EQ), Tenkai.EqEuler1D.flux(0.0f0, ur, EQ)
        set_node_vars!(out, f(0.0f0, ul, ur, fl, fr, ul, ur, EQ, 1), EQ, i)
    end
end

function run_euler1d_flux_benchmarks(; backend = Metal.MetalBackend(),
                                     sizes = (10^3, 10^4, 10^5, 10^6))
    results = BenchResult[]
    report!(name, n, run_cpu!, run_gpu!) = begin
        r = bench_pair("euler1d_$name", n; setup_cpu = admissible_state,
                       setup_gpu = n -> MtlArray(admissible_state(n)), run_cpu!, run_gpu!,
                       backend)
        push!(results, r)
        println(r.name, " n=", r.size, " speedup=", round(r.speedup, digits = 2), "x")
    end

    for n in sizes
        report!("flux", n, u -> cpu_flux!(similar(u), u),
                (u, backend) -> flux_kernel!(backend)(similar(u), u, EQ;
                                                      ndrange = size(u, 2)))
    end

    for (name, f) in (("rusanov", Tenkai.EqEuler1D.rusanov), ("roe", Tenkai.EqEuler1D.roe),
                      ("hllc", Tenkai.EqEuler1D.hllc)), n in sizes
        report!(name, n, u -> cpu_numflux!(similar(u), u, f),
                (u, backend) -> numflux_kernel!(backend)(similar(u), u, EQ, f;
                                                         ndrange = size(u, 2)))
    end

    return results
end
