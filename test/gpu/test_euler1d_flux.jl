# Every function here is called with the exact same Euler1D instance on
# host and device (no separate device type -- see the note on Euler1D in
# src/equations/EqEuler1D.jl).

using Test
using Tenkai
using Tenkai: get_node_vars, set_node_vars!
using StaticArrays: SVector
using Metal
using KernelAbstractions

@kernel function flux_kernel!(out, @Const(u), eq)
    i = @index(Global)
    set_node_vars!(out, Tenkai.EqEuler1D.flux(0.0f0, get_node_vars(u, eq, i), eq), eq, i)
end

@kernel function numflux_kernel!(out, @Const(u), eq, f)
    i = @index(Global)
    n = size(u, 2)
    ul, ur = get_node_vars(u, eq, i), get_node_vars(u, eq, mod1(i + 1, n))
    fl, fr = Tenkai.EqEuler1D.flux(0.0f0, ul, eq), Tenkai.EqEuler1D.flux(0.0f0, ur, eq)
    set_node_vars!(out, f(0.0f0, ul, ur, fl, fr, ul, ur, eq, 1), eq, i)
end

@testset "Euler1D flux kernels (Metal)" begin
    backend = Tenkai.gpu_backend(:metal)
    eq = Tenkai.EqEuler1D.get_equation(1.4f0)
    @test isbits(eq)

    nvar, n = 3, 2000
    u_cpu = zeros(Float32, nvar, n)
    for i in 1:n
        # admissible (rho, p > 0) -- arbitrary states give NaN in sqrt() on
        # both sides, masking real bugs
        prim = SVector(1.0f0 + rand(Float32), (rand(Float32) - 0.5f0) * 0.5f0,
                       1.0f0 + rand(Float32))
        set_node_vars!(u_cpu, Tenkai.EqEuler1D.prim2con(eq, prim), eq, i)
    end
    u_gpu = MtlArray(u_cpu)

    function gpu_run(kernel, args...)
        out = MtlArray(zeros(Float32, nvar, n))
        kernel(backend)(out, u_gpu, eq, args...; ndrange = n)
        KernelAbstractions.synchronize(backend)
        return Array(out)
    end

    @testset "flux" begin
        out_cpu = zeros(Float32, nvar, n)
        for i in 1:n
            unode = get_node_vars(u_cpu, eq, i)
            set_node_vars!(out_cpu, Tenkai.EqEuler1D.flux(0.0f0, unode, eq), eq, i)
        end
        @test isapprox(gpu_run(flux_kernel!), out_cpu; rtol = 1.0f-5)
    end

    for f in (Tenkai.EqEuler1D.rusanov, Tenkai.EqEuler1D.roe, Tenkai.EqEuler1D.hllc)
        @testset "$(nameof(f))" begin
            out_cpu = zeros(Float32, nvar, n)
            for i in 1:n
                ul, ur = get_node_vars(u_cpu, eq, i),
                         get_node_vars(u_cpu, eq, mod1(i + 1, n))
                fl, fr = Tenkai.EqEuler1D.flux(0.0f0, ul, eq),
                         Tenkai.EqEuler1D.flux(0.0f0, ur, eq)
                set_node_vars!(out_cpu, f(0.0f0, ul, ur, fl, fr, ul, ur, eq, 1), eq, i)
            end
            out_gpu = gpu_run(numflux_kernel!, f)
            @test isapprox(out_gpu, out_cpu; rtol = 1.0f-5)
            # Determinism check: cheap now (this kernel is a pure gather, so
            # no race is actually possible), becomes meaningful once a
            # scatter-shaped kernel reuses this pattern.
            @test out_gpu == gpu_run(numflux_kernel!, f)
        end
    end
end
