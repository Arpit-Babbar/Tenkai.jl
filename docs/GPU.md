# GPU support

Tenkai.jl supports GPU-accelerated execution via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
currently targeting Metal (Apple Silicon). Opt-in and additive: `solve`
defaults to `backend = KernelAbstractions.CPU()`, so every existing code
path is unchanged unless a user passes a different backend.

**Status: the RK/Euler1D, no-limiter, periodic-or-neumann-or-reflect-BC path
is real and verified.** `solve(equation, problem, scheme, param; backend =
Tenkai.gpu_backend(:metal))` runs the full cell-average / cell-residual /
boundary-extrapolation / ghost-value / face-residual / RK-stage-update loop
on Metal and matches the same problem solved on CPU in Float32 to within
floating-point roundoff (`test/gpu/test_rkfr1d_solve.jl`). Everything else
(other solver trees, 2D, limiters, non-periodic-non-reflect BCs) isn't
ported yet -- see the roadmap below.

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
- **`fr_operators` must produce `RealT`, not always `Float64`.** The
  differentiation/interpolation matrices (`D1`, `Vl`, `Vr`, ...) were always
  built in `Float64` regardless of the solve's own `RealT` -- harmless on
  CPU (transient promotion, silently rounds back to Float32 on write) but
  fatal for a Metal kernel, which can't contain a `Float64` anywhere, even a
  captured constant. `fr_operators` now takes a `RealT` type argument
  (`solve` passes `eltype(grid.xc)`) and rounds its Float64-computed
  quadrature/root-finding to it at the very end.
- **Scalar `t`/`dt`/`cfl` need the same sweep as arrays.** `t = 0.0` (a bare
  Float64 local) in the time-stepping loop, and `get_cfl`'s Float64 return
  value, both looked harmless until they reached a GPU kernel argument on
  exactly the last timestep (`dt = final_time - t`, computed once `t` had
  been silently Float64 the whole run). Any plain scalar that ends up as a
  kernel argument needs the same RealT-genericity as array elements.
- **`OffsetArray`-wrapped GPU arrays don't `.=`/`copyto!` as a whole.**
  Broadcasting or `copyto!`-ing an entire `OffsetArray{T,N,<:MtlArray}` (or
  the `CartesianIndices`-range form `update_ghost_values_periodic!` uses)
  falls back to scalar host indexing and errors -- verified directly against
  Metal, not assumed. `fill!` and per-element kernel access (`get_node_vars`/
  `set_node_vars!`) are unaffected. Operate on `parent(arr)` instead for
  whole-array copies/broadcasts (see `rk_copy!`/`rk_sub!`/`rk_combine!` in
  `RKFR.jl`); harmless for CPU too since `parent` of a CPU `OffsetArray` is
  just a plain `Array`.

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

Smaller, near-term gaps in the current RK/1D path specifically:

- **`dirichlet` boundary conditions**: `neumann`/`reflect`/`periodic` are
  device-to-device kernels; `dirichlet` needs the user's
  `boundary_value(x, t)` called on the host once per boundary per step,
  not yet wired up.
- **`u1`'s ghost cells (index 0 and nx+1) aren't populated** by the GPU
  path -- not needed by any kernel here (the D1-contraction and boundary
  extrapolation are both same-cell-only with `limiter = "none"`), unlike
  the CPU path where they end up filled as a side effect of unrelated
  bookkeeping. Only matters once something reads a neighbor's `u1` (e.g.
  a limiter), so it's deferred with that work rather than done now.
- **I/O and plotting** (`write_soln!`, `compute_error`, `initialize_plot`,
  `post_process_soln`): not GPU-ported; `solve` transfers `u1`/`ua` to the
  host (`to_host`) before calling them, which is fine since they only run
  at save/error intervals, not every step.

## Roadmap

| Solver tree | Dim | No-limiter baseline | Limiter | AD variant |
|---|---|---|---|---|
| RK   | 1D | done for periodic/neumann/reflect BCs: full cell-average/residual/face/RK-stage loop verified + benchmarked on Metal (`test/gpu/test_rkfr1d_solve.jl`); `dirichlet` BC not yet | not started | n/a |
| RK   | 2D | not started | not started | n/a |
| LW   | 1D | not started | not started | not started |
| LW   | 2D | not started | not started | not started |
| MDRK | 1D/2D | not started | not started | not started |
| cRK  | 1D/2D | not started | not started | not started |
| src_tenkaicrk | 1D/2D | not started | not started | not started |
