# GPU correctness tests: every ported kernel is checked against its existing
# CPU implementation, in Float32 (Metal has no Float64) at rtol~1e-5 rather
# than the Float64 regression suite's 1e-14. Skips (not fails) without a
# functional Metal device. Run with:
#   julia --project=test/gpu test/gpu/runtests.jl

using Test
using Tenkai

using Metal

const METAL_OK = Metal.functional()

@testset "GPU (Metal)" begin
    if !METAL_OK
        @info "Metal is not functional on this machine; skipping GPU tests." Metal.functional()
    else
        include("test_euler1d_flux.jl")
        # More files are added here phase-by-phase as solver trees are
        # GPU-ported; see docs/GPU.md for the current checklist.
    end
end
