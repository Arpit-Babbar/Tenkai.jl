# Shared CPU-vs-GPU benchmarking harness for every bench_*.jl file here. See
# docs/GPU.md for the ceiling methodology.
module BenchHarness

using BenchmarkTools
using KernelAbstractions
using JSON3
using Printf
using Dates

export BenchResult, bench_pair, to_markdown, save_results

# One row: min wall time on each side, the resulting speedup, and -- when
# available -- the reference ceiling speedup and fraction of it achieved.
struct BenchResult
    name::String
    size::Int
    cpu_ns::Float64
    gpu_ns::Float64
    speedup::Float64
    ceiling::Union{Float64, Nothing}
    frac_of_ceiling::Union{Float64, Nothing}
end

# Benchmarks run_cpu!(setup_cpu(size)) vs run_gpu!(setup_gpu(size), backend).
# setup_* allocates once, outside the timed region (a real solve keeps
# arrays resident, not reallocated per step). `synchronize` is called
# *inside* the GPU-timed region -- without it, BenchmarkTools only measures
# async kernel-launch overhead, not real execution time.
function bench_pair(name, size; setup_cpu, setup_gpu, run_cpu!, run_gpu!, backend,
                    ceiling = nothing, samples = 30)
    cpu_state = setup_cpu(size)
    gpu_state = setup_gpu(size)

    cpu_trial = @benchmark $run_cpu!($cpu_state) samples=samples evals=1
    gpu_trial = @benchmark begin
        $run_gpu!($gpu_state, $backend)
        KernelAbstractions.synchronize($backend)
    end samples=samples evals=1

    cpu_ns = minimum(cpu_trial.times)
    gpu_ns = minimum(gpu_trial.times)
    speedup = cpu_ns / gpu_ns
    frac = ceiling === nothing ? nothing : speedup / ceiling

    return BenchResult(name, size, cpu_ns, gpu_ns, speedup, ceiling, frac)
end

function to_markdown(results::Vector{BenchResult})
    io = IOBuffer()
    println(io,
            "| kernel | size | CPU (ms) | GPU (ms) | speedup | ceiling | % of ceiling |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|")
    for r in results
        ceil_str = r.ceiling === nothing ? "-" : @sprintf("%.1fx", r.ceiling)
        frac_str = r.frac_of_ceiling === nothing ? "-" :
                   @sprintf("%.0f%%", 100*r.frac_of_ceiling)
        @printf(io, "| %s | %d | %.3f | %.3f | %.2fx | %s | %s |\n",
                r.name, r.size, r.cpu_ns/1e6, r.gpu_ns/1e6, r.speedup,
                ceil_str, frac_str)
    end
    return String(take!(io))
end

# Writes `results` as JSON, tagged with git SHA and timestamp. Never
# hand-edit these files -- regenerate with runbenchmarks.jl.
function save_results(results::Vector{BenchResult}, path; sha = nothing)
    payload = [(; name = r.name, size = r.size, cpu_ns = r.cpu_ns, gpu_ns = r.gpu_ns,
                speedup = r.speedup, ceiling = r.ceiling,
                frac_of_ceiling = r.frac_of_ceiling) for r in results]
    open(path, "w") do io
        JSON3.write(io, (; sha, timestamp = string(now()), results = payload))
    end
    return path
end

end # module
