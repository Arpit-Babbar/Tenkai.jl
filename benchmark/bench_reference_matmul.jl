# Reference "ceiling" benchmarks that GPU kernels elsewhere are optimized
# toward (docs/GPU.md). Two, deliberately: `matmul_ceiling` times the exact
# batched-GEMM shape of Tenkai's per-cell D1-contraction (CPU BLAS vs Metal);
# `bandwidth_ceiling` covers the memory-bound case matmul can't represent,
# since Tenkai's polynomial degree (nd=3-6) is usually too small to be
# compute-bound.

using LinearAlgebra
using Metal
using KernelAbstractions
using Printf
using BenchmarkTools

# Guarded: included both standalone and from runbenchmarks.jl (which already
# includes harness.jl), and including it twice redefines `module
# BenchHarness`.
isdefined(@__MODULE__, :BenchHarness) || include("harness.jl")
using .BenchHarness

# CPU-BLAS-mul! vs Metal-mul! for D1 (nd x nd) * B (nd x nvar*ncells), the
# batched-GEMM shape of Tenkai's per-cell D1-contraction.
function matmul_ceiling(nd, ncells, nvar; backend, dtype = Float32, samples = 50)
    D1_cpu = rand(dtype, nd, nd)
    B_cpu = rand(dtype, nd, nvar * ncells)
    C_cpu = similar(B_cpu)

    D1_gpu = MtlArray(D1_cpu)
    B_gpu = MtlArray(B_cpu)
    C_gpu = MtlArray(C_cpu)

    cpu_trial = @benchmark mul!($C_cpu, $D1_cpu, $B_cpu) samples=samples evals=1
    gpu_trial = @benchmark begin
        mul!($C_gpu, $D1_gpu, $B_gpu)
        KernelAbstractions.synchronize($backend)
    end samples=samples evals=1

    cpu_ns = minimum(cpu_trial.times)
    gpu_ns = minimum(gpu_trial.times)
    return (; nd, ncells, nvar, cpu_ns, gpu_ns, speedup = cpu_ns / gpu_ns)
end

# Achieved Metal bandwidth (GB/s) for y = 2x + y (2 reads + 1 write/elem).
function bandwidth_ceiling(n; backend, dtype = Float32, samples = 50)
    x_gpu = Metal.ones(dtype, n)
    y_gpu = Metal.ones(dtype, n)

    gpu_trial = @benchmark begin
        $y_gpu .= 2 .* $x_gpu .+ $y_gpu
        KernelAbstractions.synchronize($backend)
    end samples=samples evals=1

    gpu_ns = minimum(gpu_trial.times)
    bytes_moved = 3 * n * sizeof(dtype) # 2 reads + 1 write
    gbps = bytes_moved / (gpu_ns * 1e-9) / 1e9
    return (; n, gpu_ns, gbps)
end

function run_reference_benchmarks(; backend = Metal.MetalBackend())
    results = BenchResult[]

    # nd sweeps Tenkai's typical polynomial degrees (N=2..5, i.e. nd=3..6);
    # ncells sweeps small-to-large problem sizes.
    for nd in (3, 4, 5, 6), ncells in (10^3, 10^4, 10^5)
        nvar = 3 # e.g. compressible Euler 1D
        m = matmul_ceiling(nd, ncells, nvar; backend)
        push!(results,
              BenchResult("matmul_ceiling(nd=$nd,nvar=$nvar)", ncells, m.cpu_ns,
                          m.gpu_ns, m.speedup, nothing, nothing))
        @printf("matmul  nd=%d ncells=%-7d speedup=%.2fx\n", nd, ncells, m.speedup)
    end

    # Not folded into `results`: BenchResult's `speedup` column is unitless
    # "x", not GB/s -- printed only, to avoid mislabeling the markdown table.
    for n in (10^5, 10^6, 10^7, 10^8)
        b = bandwidth_ceiling(n; backend)
        @printf("bandwidth n=%-9d %.1f GB/s\n", n, b.gbps)
    end

    return results
end
