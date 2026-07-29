# Run every benchmark in this directory and regenerate:
#   - benchmark/results/<git-sha>.json   (raw data, one file per run)
#   - benchmark/results/latest.md        (markdown table, included into docs/GPU.md)
#
# Usage (from the repo root):
#   julia --project=benchmark benchmark/runbenchmarks.jl
#
# Never hand-edit files under benchmark/results/ -- regenerate them here.

using Metal
using Dates

include("harness.jl")
using .BenchHarness

include("bench_reference_matmul.jl")
# Per-kernel benchmark files (bench_cell_residual.jl, bench_face_residual.jl,
# bench_rk_stage.jl, ...) are `include`d here as each is added; none exist
# yet in Phase 0.

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
    # Per-kernel `run_*_benchmarks(; backend)` calls are appended here as
    # each phase adds its own bench_*.jl file.

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
