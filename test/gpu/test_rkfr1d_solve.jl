# End-to-end test: a real Euler1D RK solve (cell average, cell residual,
# boundary extrapolation, ghost values, face residual, RK stage update, time
# step -- the full loop, not just isolated kernels) run entirely on Metal and
# compared against the same problem solved on the CPU in Float32.

using Test
using Tenkai
using Metal

Eq = Tenkai.EqEuler1D

function euler1d_dwave_setup(::Type{RealT}; time_scheme = "SSPRK22", degree = 1,
                             nx = 20) where {RealT}
    domain = RealT[0, 1]
    boundary_condition = (periodic, periodic)
    problem = Problem(domain, Eq.dwave, Eq.dummy_zero_boundary_value, boundary_condition,
                      RealT(0.1), (x, t) -> Eq.dwave(x - t))
    equation = Eq.get_equation(RealT(1.4))
    scheme = Scheme("rkfr", degree, "gl", "radau", Eq.rusanov, "no",
                    setup_limiter_none(), evaluate)
    param = Parameters(nx, RealT(0), (RealT[-Inf], RealT[Inf]), 0, RealT(0), 0;
                       time_scheme = time_scheme)
    return equation, problem, scheme, param
end

@testset "RKFR 1D end-to-end solve (Metal)" begin
    backend = Tenkai.gpu_backend(:metal)

    eq32, problem32, scheme32, param32 = euler1d_dwave_setup(Float32)
    sol_cpu = Tenkai.solve(eq32, problem32, scheme32, param32)
    sol_gpu = Tenkai.solve(eq32, problem32, scheme32, param32; backend = backend)

    # Physical cells only: index 1 (of the raw, 1-based parent array) is the
    # ghost cell at logical index 0, which the GPU path doesn't populate for
    # `u1` (not needed for these kernels -- see docs/GPU.md) while the CPU
    # path's happens to be, as a side effect of unrelated bookkeeping. Not a
    # GPU-vs-CPU numerical difference; excluded from the comparison on purpose.
    nx = param32.grid_size
    u_cpu = Array(parent(sol_cpu["u"]))[:, :, 2:(nx + 1)]
    u_gpu = Array(parent(sol_gpu["u"]))[:, :, 2:(nx + 1)]

    @test isapprox(u_cpu, u_gpu; rtol = 1.0f-5)
    @test isapprox(sol_cpu["errors"]["l2_error"], sol_gpu["errors"]["l2_error"];
                   rtol = 1.0f-4)
end
