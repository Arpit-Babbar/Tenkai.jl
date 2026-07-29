# GPU support

Tenkai.jl supports GPU-accelerated execution via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
currently targeting Metal (Apple Silicon). Opt-in and additive: `solve`
defaults to `backend = KernelAbstractions.CPU()`, so every existing code
path is unchanged unless a user passes a different backend.

**Status: no solve is GPU-accelerated yet.** So far this covers the
infrastructure (backend plumbing, benchmark harness) and makes `Euler1D`'s
`flux`/`rusanov`/`roe`/`hllc` `@kernel`-safe and benchmarked in isolation --
`solve(...; backend = Tenkai.gpu_backend(:metal))` on an RK/Euler1D problem
still runs entirely on the CPU today, since the cell-residual, face-residual,
and RK-stepping kernels that would actually use `backend` don't exist yet.
See the roadmap below.

## Why KernelAbstractions + a Metal extension

Kernels are written once, backend-agnostic, as `@kernel` functions in
`src/`. `Metal` is a weak dependency, resolved only when a user `using
Metal`s it, activating `ext/TenkaiMetalExt.jl` (no numerical code, just glue
to a working `Backend`) -- mirrors Trixi.jl's own `TrixiAMDGPUExt` pattern.

```julia
using Tenkai, Metal
solve(equation, problem, scheme, param; backend = Tenkai.gpu_backend(:metal))
```

## Making an equation struct GPU-kernel-safe

A `@kernel` function's arguments must be `isbits` -- no `Dict`/`String`/
`Vector` fields. `Euler1D` carried `name`/`initial_values`/`numfluxes` for
exactly this reason it broke. Fix: **remove the dead weight, don't add a
second "device" struct.** None of those fields earned their keep --
`numfluxes` was never read anywhere, `initial_values` was already a
module-level `Dict` the field only aliased, `name` was used in one log line
(now `equation_name(eq) = string(nameof(typeof(eq)))` in `FR.jl`). With
those gone, `Euler1D{RealT, HLLSpeeds}` is `isbits` on its own and passes
into `@kernel` functions unchanged -- no `Adapt.adapt_structure` needed.
Apply the same audit to each equation type before assuming a field needs
relocating.

## Gotchas

- **No `Float64` on Metal.** Sweep literals to be generic in `RealT`
  (`one(RealT)`, `RealT(0.5)`, ...) in every function on the kernel path,
  including inside comparisons -- Metal's compiler has no double-precision
  unit, so a stray `Float64` anywhere can fail to compile, not just promote.
- **Don't redefine a `@kernel` inside a loop.** Reusing a stale compiled
  kernel from a previous iteration is a real, silent wrong-answer bug this
  produced once. Define each kernel once at top level; pass the varying
  part (e.g. which numerical flux) as a runtime argument instead.

## Correctness tolerances

Metal has no `Float64`, so GPU-path tests compare in `Float32` at `rtol ~
1e-5`, not against the `Float64` regression suite's `1e-14` (`test/data/*.txt`,
unreachable in `Float32`). See `test/gpu/`.

## Benchmarks

Every GPU function is benchmarked against its CPU counterpart
(`benchmark/harness.jl`) and against a reference ceiling
(`benchmark/bench_reference_matmul.jl`): **matmul ceiling** times the exact
batched-GEMM shape of the per-cell D1-contraction (CPU BLAS vs Metal);
**bandwidth ceiling** covers the memory-bound case matmul can't represent,
since Tenkai's polynomial degree (nd=3-6) is usually too small to be
compute-bound. Kernels are optimized toward whichever applies. Regenerate
with `julia --project=benchmark benchmark/runbenchmarks.jl` -- see
`benchmark/results/latest.md` (never hand-edited).

## What stays CPU-only, and why

Nothing is permanently excluded -- these need real redesign, not a
mechanical port, so they're sequenced after a working baseline per tree:

- **Limiter/blending** (`Blend1D`/`Blend2D`): reuses ~20 mutable scratch
  arrays across cells, safe only because CPU iteration is serial. Needs
  per-thread private scratch and a bounded-iteration Newton solver.
- **AD-based residual variants** (`LWFR2D_ad.jl`, cRK equivalents): GPU AD
  is the least mature part of the Julia GPU ecosystem; approach TBD when
  each tree's AD phase is reached.

## Roadmap

| Solver tree | Dim | No-limiter baseline | Limiter | AD variant |
|---|---|---|---|---|
| RK   | 1D | in progress: `Euler1D` isbits + `flux`/`rusanov`/`roe`/`hllc` verified + benchmarked on Metal; cell/face residual + RK stepping not started | not started | n/a |
| RK   | 2D | not started | not started | n/a |
| LW   | 1D | not started | not started | not started |
| LW   | 2D | not started | not started | not started |
| MDRK | 1D/2D | not started | not started | not started |
| cRK  | 1D/2D | not started | not started | not started |
| src_tenkaicrk | 1D/2D | not started | not started | not started |
