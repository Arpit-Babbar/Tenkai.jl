# Regenerates benchmark/results/<git-sha>.json and latest.md. Usage:
#   julia --project=benchmark benchmark/runbenchmarks.jl
# Never hand-edit files under benchmark/results/.

using Metal
using Dates

include("harness.jl")
using .BenchHarness

include("bench_reference_matmul.jl")
include("bench_euler1d_flux.jl")
# More bench_*.jl files are included here as each solver tree is GPU-ported.

function git_sha()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function main()
    if !Metal.functional()
        error("Metal is not functional on this machine; benchmarks require a " *
              "real Metal-capable device (see docs/GPU.md). Run `Metal.functional()` " *
              "for details.")
    end
    backend = Metal.MetalBackend()

    results = BenchResult[]
    append!(results, run_reference_benchmarks(; backend))
    append!(results, run_euler1d_flux_benchmarks(; backend))
    # More `run_*_benchmarks(; backend)` calls are appended here as each
    # solver tree is GPU-ported.

    sha = git_sha()
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    save_results(results, joinpath(results_dir, "$sha.json"); sha)

    md = to_markdown(results)
    write(joinpath(results_dir, "latest.md"), md)
    println(md)

    return results
end

main()
