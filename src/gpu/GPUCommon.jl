# GPU vendor packages (e.g. Metal.jl) are weak deps that activate a package
# extension (ext/TenkaiMetalExt.jl) supplying a KernelAbstractions.Backend,
# keeping platform-specific packages out of Tenkai's hard dependency graph.
# Default backend is KernelAbstractions.CPU(), so nothing changes unless a
# user opts in via `backend` on `solve`.

using KernelAbstractions: KernelAbstractions, Backend, CPU
using OffsetArrays: OffsetArrays, OffsetArray

export gpu_backend

"""
    gpu_backend(name::Symbol)

Return a `KernelAbstractions.Backend` for vendor `name` (only `:metal` so
far). Requires the vendor package loaded first (`using Metal`) so its
extension is active; errors otherwise instead of silently using the CPU.
"""
gpu_backend(name::Symbol) = gpu_backend(Val(name))

function gpu_backend(::Val{T}) where {T}
    error("No GPU backend available for `:$T`. Load the corresponding vendor " *
          "package to activate it (e.g. `using Metal` for `:metal`).")
end

export to_device, to_host

# Copies a plain (host) array onto `backend`. For `CPU()` this is `arr`
# itself, unchanged -- no copy, no new allocation.
to_device(::CPU, arr) = arr
function to_device(backend, arr)
    d = KernelAbstractions.zeros(backend, eltype(arr), size(arr)...)
    copyto!(d, arr)
    return d
end

# Inverse of `to_device`, preserving an OffsetArray wrapper if present. I/O
# and plotting (write_soln!, compute_error, post_process_soln) aren't
# GPU-ported -- call this on `u1`/`ua` before passing them in, rather than
# porting those too.
to_host(arr::Array) = arr
to_host(arr::OffsetArray) = OffsetArray(to_host(parent(arr)), arr.offsets)
to_host(arr) = Array(arr)

"""
    warn_if_gpu_backend_ignored(kwargs, context::String)

Only the conservative 1D/2D RK solver tree honors a non-CPU `backend` so far
(docs/GPU.md). Every other `setup_arrays`/`setup_arrays_rkfr` method just
absorbs `backend` via `kwargs...` and ignores it; call this from each so
that doing so isn't a silent no-op.
"""
function warn_if_gpu_backend_ignored(kwargs, context::String)
    backend = get(kwargs, :backend, CPU())
    if !(backend isa CPU)
        @warn "$context does not support the GPU backend $(typeof(backend)) yet; "*
        "falling back to plain CPU arrays. See docs/GPU.md for what's ported." backend
    end
    return nothing
end
