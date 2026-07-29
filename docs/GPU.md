# GPU support

Tenkai.jl supports GPU-accelerated execution via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
currently targeting the Metal backend (Apple Silicon). GPU support is
additive and opt-in: every existing CPU code path is unchanged and remains
the default, since `solve` defaults to `backend = KernelAbstractions.CPU()`.

## Why KernelAbstractions + a Metal extension

Numerical kernels (flux/residual computation, time-stepping, limiters) are
written once as backend-agnostic `KernelAbstractions.@kernel` functions
inside `src/`. Vendor-specific GPU packages are kept out of Tenkai's hard
dependency graph -- `Metal` is a weak dependency, resolved only when a user
`using Metal` in their own session, which activates the
`ext/TenkaiMetalExt.jl` package extension. This mirrors the pattern
Trixi.jl itself uses for its own GPU backends (e.g. `TrixiAMDGPUExt`).

`ext/TenkaiMetalExt.jl` contains no numerical code -- only the glue needed
to turn `using Metal` into a working `KernelAbstractions.Backend`:

```julia
using Tenkai, Metal
backend = Tenkai.gpu_backend(:metal)
solve(equation, problem, scheme, param; backend = backend)
```

## Making an equation struct GPU-kernel-safe

A `@kernel` function's arguments must be `isbits` (no heap-allocated fields
like `Dict`/`String`/`Vector`) -- GPU kernels can't hold a pointer back into
host-managed memory. Equation structs like `Euler1D` originally carried
`name::String`, `initial_values::Dict{String,Function}`, and
`numfluxes::Dict{String,Function}`, which broke this.

**The fix is to remove the dead weight, not to introduce a second
"device" struct.** In `Euler1D`'s case, none of those fields were pulling
their weight: `numfluxes` was never read anywhere in the codebase (the one
place that could have consulted it, `ParseCommandLine`, is a no-op stub);
`initial_values` was already a module-level `Dict` in `EqEuler1D.jl` that
the field only ever aliased, so the one caller that used it
(`post_process_soln`, for plotting an exact-solution overlay) can reference
the module-level dict directly; `name` was used in exactly one place (a log
line), replaced by a generic `equation_name(eq) = string(nameof(typeof(eq)))`
fallback in `FR.jl`. Once those are gone, `Euler1D{RealT, HLLSpeeds}` is
`isbits` on its own -- `γ::RealT` and a plain (non-closure) `hll_speeds`
function are both `isbits` -- and passes into `@kernel` functions completely
unchanged, with no `Adapt.adapt_structure` or parallel type needed.

Apply the same audit to every other equation type as its solver tree is
reached: check what a `Dict`/`String`/similar field is actually used for
before assuming it needs relocating -- it may turn out to be dead weight, or
already available as a module-level binding.

## `Float64` and other portability gotchas

- **No `Float64` on Metal.** Every numeric function that runs inside a
  kernel needs its literals swept to be generic in the equation's `RealT`
  (`one(RealT)`, `zero(RealT)`, `RealT(0.5)`, ...), including inside
  comparisons like `if sl > 0.0` -- even though the *result* of a comparison
  is a `Bool`, Metal's shader compiler has no double-precision arithmetic
  unit at all, so a stray `Float64` literal anywhere in the expression can
  fail to compile, not just silently promote. This is a per-function,
  mechanical sweep to do before porting each function, not a one-time setup
  step.
- **Don't redefine a `@kernel` function inside a loop.** Defining
  `@kernel function foo!(...) ... end` repeatedly across loop iterations
  (e.g. once per numerical-flux option in a test) risks KernelAbstractions
  reusing a stale compiled kernel from a previous iteration -- this produced
  a real, silent wrong-answer bug during Phase 1 (a `roe`/`hllc` correctness
  test loop that appeared to run 3 different fluxes but sometimes compiled
  against the wrong one). Define each kernel once at top level (or module
  scope) and pass it around as a value instead.

## Correctness tolerances

Metal has no `Float64` support. Every GPU-path comparison in this package
therefore runs in `Float32` and compares against the CPU implementation at a
loosened tolerance (`rtol ~ 1e-5`), rather than against the existing
`Float64` regression suite in `test/data/*.txt` (which uses `1e-14` and is
unreachable in `Float32`). See `test/gpu/runtests.jl`.

## Benchmark methodology

Every GPU-ported function is benchmarked against its CPU counterpart with
the shared harness in `benchmark/harness.jl`, and against a reference
"ceiling" from `benchmark/bench_reference_matmul.jl`:

- **matmul ceiling** -- the per-cell differentiation-matrix contraction that
  recurs throughout this codebase (`D1 (nd x nd) * flux (nd x nvar)`, per
  cell) is, across all cells at once, a batched GEMM. Timing that exact
  shape as CPU-BLAS `mul!` vs Metal `mul!` gives a fair "how fast could this
  arithmetic go" ceiling.
- **bandwidth ceiling** -- Tenkai's polynomial degree `nd` is typically
  small (3-6), so most kernels here are memory-bound rather than
  compute-bound; matmul alone would set an unreachable target for them.
  Achieved memory bandwidth vs. hardware peak is reported alongside, and is
  the ceiling actually used for memory-bound kernels.

Kernels are optimized (memory layout, kernel fusion, workgroup sizing,
avoiding scalar/dynamic indexing) toward whichever ceiling applies, and the
choice is stated explicitly per kernel below rather than left implicit.
Results are regenerated with `julia --project=benchmark
benchmark/runbenchmarks.jl` -- see `benchmark/results/latest.md` for the
current numbers (never hand-edited).

## What stays CPU-only, and why

Nothing is permanently excluded from GPU support (see the roadmap below),
but two categories of code need real redesign, not a mechanical port, so
they are sequenced after a working baseline in each solver tree rather than
attempted first:

- **Limiter/blending code** (`Blend1D`/`Blend2D` and friends): the existing
  CPU implementation reuses ~20 mutable scratch arrays across cells, safe
  only because CPU iteration is serial. The GPU port gives each thread its
  own private scratch (KernelAbstractions `@private` storage, or an
  `MVector`/`SVector`-based rewrite) and makes the Newton solver
  (`newton_solver_tenkai`) GPU-kernel-safe (bounded iteration count, no
  dynamic allocation).
- **AD-based residual variants** (`LWFR2D_ad.jl` and the cRK-tree
  equivalents, using ForwardDiff/Enzyme/TaylorDiff): GPU-compatible
  automatic differentiation is the least mature part of the Julia GPU
  ecosystem. The concrete approach is worked out when each tree's AD phase
  is reached (Enzyme currently has the most GPU traction of the three).

## Roadmap / checklist

Updated at the end of every phase. See the plan history for the full
phase-by-phase execution order.

| Solver tree | Dim | No-limiter baseline | Limiter | AD variant |
|---|---|---|---|---|
| RK   | 1D | in progress: `Euler1D` isbits + `flux`/`rusanov`/`roe`/`hllc` verified on Metal (test/gpu/test_euler1d_flux.jl); cell/face residual + RK stepping kernels not started | not started | n/a |
| RK   | 2D | not started | not started | n/a |
| LW   | 1D | not started | not started | not started |
| LW   | 2D | not started | not started | not started |
| MDRK | 1D/2D | not started | not started | not started |
| cRK  | 1D/2D | not started | not started | not started |
| src_tenkaicrk | 1D/2D | not started | not started | not started |
