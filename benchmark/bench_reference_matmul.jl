# Reference "ceiling" benchmarks: how fast can this hardware go on the
# shapes that actually occur in Tenkai's hot loops, on CPU (BLAS) vs GPU
# (Metal)? GPU-ported kernels elsewhere in `benchmark/` are optimized toward
# these numbers rather than an arbitrary target -- see docs/GPU.md.
#
# Two ceilings are reported, deliberately:
#
#   * `matmul_ceiling` -- the per-cell D1-contraction in
#     `compute_cell_residual_rkfr!` (`res[:, iix, cell] += D1[iix, ix] *
#     flux[:, ix, cell]` summed over `ix`) is, across all cells at once, a
#     batched GEMM: `D1 (nd x nd) * B (nd x nvar*ncells)`. Timing that exact
#     shape on CPU BLAS vs Metal gives a fair "how fast could this
#     arithmetic go" number.
#   * `bandwidth_ceiling` -- Tenkai's polynomial degree `nd` is small
#     (typically 3-6), so most kernels here move far more bytes than they do
#     FLOPs: they're memory-bound, not compute-bound. matmul is a compute-
#     bound benchmark, so for small `nd` it sets an unreachably optimistic
#     target. Achieved-bandwidth-vs-peak is the more honest ceiling for
#     those kernels, and is reported alongside so each phase can cite the
#     correct one instead of being judged against the wrong number.

using LinearAlgebra
using Metal
using KernelAbstractions
using Printf

include("harness.jl")
using .BenchHarness

"""
    matmul_ceiling(nd, ncells, nvar; backend, dtype = Float32, samples = 50)

CPU-BLAS-`mul!` vs Metal-`mul!` speedup for `D1 (nd x nd) * B (nd x
nvar*ncells)`, i.e. the true batched-GEMM shape of Tenkai's per-cell
D1-contraction.
"""
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

"""
    bandwidth_ceiling(n; backend, dtype = Float32, samples = 50)

Achieved Metal memory bandwidth (GB/s) for a large elementwise `y = 2x + y`
kernel (2 reads + 1 write per element), reported against a rough peak
estimate so memory-bound kernels have a meaningful target distinct from
`matmul_ceiling`.
"""
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

    for n in (10^5, 10^6, 10^7, 10^8)
        b = bandwidth_ceiling(n; backend)
        @printf("bandwidth n=%-9d %.1f GB/s\n", n, b.gbps)
    end

    return results
end
