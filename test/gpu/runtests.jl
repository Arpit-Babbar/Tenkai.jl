# GPU correctness tests: every ported kernel is checked against its existing
# CPU implementation. Skips cleanly (with an @info, not a failure) on any
# machine without a functional Metal device, so `TENKAI_TEST=gpu` stays safe
# to run in ordinary CI.
#
# Kernels are compared in Float32 (Metal has no Float64) against randomized
# *admissible* states -- see docs/GPU.md for why tolerances here (~1e-5) are
# far looser than the Float64 regression suite's 1e-14, and why admissible
# (rather than arbitrary) random states are required.

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
