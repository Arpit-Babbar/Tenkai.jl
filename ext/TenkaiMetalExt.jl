# Package extension activating Metal.jl-backed GPU support for Tenkai.jl.
#
# This extension loads automatically whenever both Tenkai and Metal are
# `using`d in the same session (Julia's package extension mechanism, driven
# by the `[weakdeps]`/`[extensions]` entries in Project.toml). It contains no
# numerical kernels of its own -- those live in `src/` as backend-agnostic
# `KernelAbstractions.@kernel` functions -- only the glue needed to obtain a
# working `MetalBackend` and to fail loudly if the hardware/driver isn't
# actually usable.
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
