# Loads when both Tenkai and Metal are `using`d. No kernels here -- those
# live in `src/` as backend-agnostic @kernel functions -- just the glue to
# get a working MetalBackend.
module TenkaiMetalExt

using Metal: Metal, MetalBackend
using Tenkai: Tenkai

function Tenkai.gpu_backend(::Val{:metal})
    if !Metal.functional()
        error("Metal.jl is loaded but not functional on this system " *
              "(no Metal-capable GPU detected, or the Metal artifact failed " *
              "to initialize). Run `Metal.functional()` for details.")
    end
    return MetalBackend()
end

end # module
