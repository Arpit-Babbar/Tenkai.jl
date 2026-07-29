# GPU backend infrastructure, shared across all solver trees.
#
# Tenkai does not depend on any specific GPU vendor package directly. Instead,
# KernelAbstractions.jl is used to write backend-agnostic `@kernel` functions
# throughout `src/`, and vendor packages (e.g. Metal.jl) are loaded as weak
# dependencies that activate a package extension (`ext/TenkaiMetalExt.jl`)
# supplying the actual `KernelAbstractions.Backend` instance. This keeps
# platform-specific packages like Metal.jl (macOS/Apple Silicon only) out of
# Tenkai's hard dependency graph.
#
# The default backend everywhere is `KernelAbstractions.CPU()`, so every
# existing code path behaves exactly as before unless a user explicitly opts
# into a GPU backend via the `backend` keyword of `solve`.

using KernelAbstractions: KernelAbstractions, Backend, CPU

export gpu_backend

"""
    gpu_backend(name::Symbol)

Return a `KernelAbstractions.Backend` for the requested GPU vendor `name`.
Currently only `name = :metal` is implemented.

Requires the corresponding vendor package to be loaded first so that its
package extension activates (e.g. `using Metal` for `:metal`); calling this
before that happens throws an informative error rather than silently falling
back to the CPU.

# Example
```julia
using Tenkai, Metal
backend = Tenkai.gpu_backend(:metal)
solve(equation, problem, scheme, param; backend = backend)
```
"""
gpu_backend(name::Symbol) = gpu_backend(Val(name))

function gpu_backend(::Val{T}) where {T}
    error("No GPU backend available for `:$T`. Load the corresponding vendor " *
          "package to activate it (e.g. `using Metal` for `:metal`).")
end
