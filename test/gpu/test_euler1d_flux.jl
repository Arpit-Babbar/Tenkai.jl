# Correctness check for the numerical-flux functions used by the RK-1D GPU
# port: every function here is called with the *exact same* `Euler1D`
# instance on both host and device (no separate device type -- see the note
# on `Euler1D` in src/equations/EqEuler1D.jl), so a mismatch here would mean
# the kernel is silently computing something different from the CPU solver.

using Test
using Tenkai
using Tenkai: get_node_vars, set_node_vars!
using StaticArrays: SVector
using Metal
using KernelAbstractions

@testset "Euler1D flux kernels (Metal)" begin
    backend = Tenkai.gpu_backend(:metal)
    eq = Tenkai.EqEuler1D.get_equation(1.4f0)
    @test isbits(eq)

    nvar, n = 3, 2000
    # Randomized *admissible* states: rho > 0, small velocity, p > 0 -- an
    # arbitrary random conservative vector routinely has negative pressure,
    # which makes sqrt() in the Riemann solvers NaN on both CPU and GPU and
    # would mask real bugs rather than catch them.
    rho = 1.0f0 .+ rand(Float32, n)
    vel = (rand(Float32, n) .- 0.5f0) .* 0.5f0
    p = 1.0f0 .+ rand(Float32, n)
    u_cpu = zeros(Float32, nvar, n)
    for i in 1:n
        prim = SVector(rho[i], vel[i], p[i])
        set_node_vars!(u_cpu, Tenkai.EqEuler1D.prim2con(eq, prim), eq, i)
    end
    u_gpu = MtlArray(u_cpu)

    function cpu_reference(f)
        out = zeros(Float32, nvar, n)
        for i in 1:n
            ul = get_node_vars(u_cpu, eq, i)
            ur = get_node_vars(u_cpu, eq, mod1(i + 1, n)) # neighbor state
            fl = Tenkai.EqEuler1D.flux(0.0f0, ul, eq)
            fr = Tenkai.EqEuler1D.flux(0.0f0, ur, eq)
            fn = f(0.0f0, ul, ur, fl, fr, ul, ur, eq, 1)
            set_node_vars!(out, fn, eq, i)
        end
        return out
    end

    @kernel function flux_kernel!(out, @Const(u), eq)
        i = @index(Global)
        unode = get_node_vars(u, eq, i)
        set_node_vars!(out, Tenkai.EqEuler1D.flux(0.0f0, unode, eq), eq, i)
    end

    @testset "flux" begin
        out_cpu = zeros(Float32, nvar, n)
        for i in 1:n
            set_node_vars!(out_cpu, Tenkai.EqEuler1D.flux(0.0f0, get_node_vars(u_cpu, eq, i), eq),
                           eq, i)
        end
        out_gpu = MtlArray(zeros(Float32, nvar, n))
        flux_kernel!(backend)(out_gpu, u_gpu, eq; ndrange = n)
        KernelAbstractions.synchronize(backend)
        @test isapprox(Array(out_gpu), out_cpu; rtol = 1.0f-5)
    end

    # One @kernel per numerical flux, defined at top level (NOT redefined
    # inside a loop -- KernelAbstractions specializes/caches compiled
    # kernels per method, and redefining a same-named @kernel function on
    # every loop iteration is exactly the kind of thing that risks reusing a
    # stale compiled kernel from a previous iteration).
    @kernel function rusanov_kernel!(out, @Const(u), eq)
        i = @index(Global)
        n_ = size(u, 2)
        ul = get_node_vars(u, eq, i)
        ur = get_node_vars(u, eq, mod1(i + 1, n_))
        fl = Tenkai.EqEuler1D.flux(0.0f0, ul, eq)
        fr = Tenkai.EqEuler1D.flux(0.0f0, ur, eq)
        fn = Tenkai.EqEuler1D.rusanov(0.0f0, ul, ur, fl, fr, ul, ur, eq, 1)
        set_node_vars!(out, fn, eq, i)
    end

    @kernel function roe_kernel!(out, @Const(u), eq)
        i = @index(Global)
        n_ = size(u, 2)
        ul = get_node_vars(u, eq, i)
        ur = get_node_vars(u, eq, mod1(i + 1, n_))
        fl = Tenkai.EqEuler1D.flux(0.0f0, ul, eq)
        fr = Tenkai.EqEuler1D.flux(0.0f0, ur, eq)
        fn = Tenkai.EqEuler1D.roe(0.0f0, ul, ur, fl, fr, ul, ur, eq, 1)
        set_node_vars!(out, fn, eq, i)
    end

    @kernel function hllc_kernel!(out, @Const(u), eq)
        i = @index(Global)
        n_ = size(u, 2)
        ul = get_node_vars(u, eq, i)
        ur = get_node_vars(u, eq, mod1(i + 1, n_))
        fl = Tenkai.EqEuler1D.flux(0.0f0, ul, eq)
        fr = Tenkai.EqEuler1D.flux(0.0f0, ur, eq)
        fn = Tenkai.EqEuler1D.hllc(0.0f0, ul, ur, fl, fr, ul, ur, eq, 1)
        set_node_vars!(out, fn, eq, i)
    end

    for (name, f, kernel) in (("rusanov", Tenkai.EqEuler1D.rusanov, rusanov_kernel!),
                              ("roe", Tenkai.EqEuler1D.roe, roe_kernel!),
                              ("hllc", Tenkai.EqEuler1D.hllc, hllc_kernel!))
        @testset "$name" begin
            out_cpu = cpu_reference(f)

            out_gpu = MtlArray(zeros(Float32, nvar, n))
            kernel(backend)(out_gpu, u_gpu, eq; ndrange = n)
            KernelAbstractions.synchronize(backend)
            @test isapprox(Array(out_gpu), out_cpu; rtol = 1.0f-5)

            # Determinism: rerun and require bitwise-identical output. These
            # particular kernels are pure gathers (each thread only ever
            # writes its own output slot), so there's no scatter/write
            # conflict possible here regardless -- this check is a cheap
            # sanity net now and becomes a genuine race detector once
            # scatter-shaped kernels (e.g. the face-residual gather in a
            # later phase) reuse this same test pattern.
            out_gpu2 = MtlArray(zeros(Float32, nvar, n))
            kernel(backend)(out_gpu2, u_gpu, eq; ndrange = n)
            KernelAbstractions.synchronize(backend)
            @test Array(out_gpu) == Array(out_gpu2)
        end
    end
end
