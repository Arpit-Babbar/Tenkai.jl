using Test
using Tenkai
using Tenkai.TenkaicRK
using Tenkai
import Tenkai.Trixi
using Tenkai: VolumeIntegralFluxDifferencing
using TrixiBase
using Tenkai.DelimitedFiles

overwrite_errors = false

# Reactive Euler 1D

@testset "Reactive Euler 1D" begin
    # Pure FV blending test
    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_reactive_rp1.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 4,
                  solver = cSSP2IMEX433(),
                  bound_limit = "yes",
                  pure_fv = true,
                  bflux = extrapolate,
                  cfl_safety_factor = 0.9,
                  animate = false, final_time = 1.0, nx = 4)
    data_name = "reactive_rp1_pure_fv.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)

    # First order IMEX test without subcells
    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_reactive_rp1.jl"),
                  compute_error_interval = 0,
                  degree = 0,
                  solver = cIMEX111(),
                  bound_limit = "no",
                  limiter = setup_limiter_none(),
                  bflux = extrapolate,
                  cfl_safety_factor = 0.9,
                  save_iter_interval = 0, save_time_interval = 0.0,
                  animate = false, final_time = 1.0, nx = 4)
    data_name = "reactive_rp1_first_order.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)

    # cHT112 test, requiring lower CFL
    EqReactive = Tenkai.TenkaicRK.EqEulerReactive1D
    equation = Eq.get_equation(1.4, 25.0)
    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_reactive_rp1.jl"),
                  compute_error_interval = 0,
                  degree = 3,
                  solver = cHT112(),
                  bound_limit = "yes",
                  pure_fv = false,
                  bflux = extrapolate,
                  cfl_safety_factor = 0.1,
                  save_iter_interval = 0, save_time_interval = 0.0,
                  animate = false, final_time = 1.0, nx = 4)
    data_name = "reactive_rp1_ht112.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)

    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_reactive_rp1.jl"),
                  compute_error_interval = 0,
                  degree = 4,
                  solver = cSSP2IMEX433(),
                  bound_limit = "yes",
                  pure_fv = false,
                  bflux = extrapolate,
                  cfl_safety_factor = 0.9,
                  save_iter_interval = 0, save_time_interval = 0.0,
                  final_time = 1.0, nx = 4)
    data_name = "reactive_rp1_ssp433.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)

    # IMEXSSP433 test (used in the paper)
    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_reactive_rp1.jl"),
                  compute_error_interval = 0,
                  degree = 4,
                  solver = cSSP2IMEX433(),
                  bound_limit = "yes",
                  pure_fv = false,
                  bflux = extrapolate,
                  cfl_safety_factor = 0.9,
                  save_iter_interval = 0, save_time_interval = 0.0,
                  final_time = 1.0, nx = 4)
    data_name = "reactive_rp1_ssp433.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)

    # TVB limiter test
    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_reactive_rp1.jl"),
                  compute_error_interval = 0,
                  degree = 4,
                  solver = cSSP2IMEX433(),
                  bound_limit = "yes",
                  limiter = setup_limiter_tvb(equation; tvbM = 0.0),
                  bflux = extrapolate,
                  cfl_safety_factor = 0.9,
                  save_iter_interval = 0, save_time_interval = 0.0,
                  final_time = 1.0, nx = 4)
    data_name = "reactive_rp1_tvb.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)
end

@testset "Burg sin cHT112" begin
    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_burg1d_sin_source_smooth.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 1,
                  solver = cHT112(),
                  animate = false, final_time = 0.1, nx = 5)
    data_name = "burg1d_sin_source_smooth.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Jin-Xin" begin
    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_jin_xin_burg1d.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "jin_xin_burg1d.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 4e-13)

    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_jin_xin_burg1d_marco.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "jin_xin_burg1d_marco.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Burger 1D stiff double source non-linear" begin
    trixi_include(joinpath(cRK_examples_dir(), "1d",
                           "run_burg1d_stiff_source_non_linear.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.001, nx = 5)
    data_name = "burg1d_stiff_source_non_linear.txt"
    # TODO - Why is tolerance higher than in TenkaicRK.jl?
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 5e-8)
end

@testset "Shock diffraction" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_shock_diffraction.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.001, ny = 20)
    data_name = "shock_diffraction.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW roll wave" begin
    for solver in [cRK22(), "rkfr"]
        trixi_include(joinpath(cRK_examples_dir(), "1d", "run_ssw_roll_wave.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver,
                      animate = false, final_time = 0.1, nx = 5,
                      degree = 1, cfl_safety_factor = 0.98)
        data_name = "ssw_roll_wave_$(solver).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end

@testset "SSW convergence 1D" begin
    for solver in [cRK22(), "rkfr"]
        trixi_include(joinpath(cRK_examples_dir(), "1d", "run_ssw_accuracy.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver,
                      animate = false, final_time = 0.1, nx = 5,
                      degree = 1, cfl_safety_factor = 0.98)
        data_name = "ssw_accuracy_$(solver).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end

@testset "Ten Moment stiff rarefaction" begin
    # try
    #     trixi_include(joinpath(cRK_examples_dir(), "1d",
    #                            "run_tenmom_two_rare_source_stiff.jl"),
    #                   save_time_interval = 0.0, save_iter_interval = 0,
    #                   compute_error_interval = 0,
    #                   animate = false, final_time = 0.1, nx = 100,
    #                   solver = cRK44())
    # catch e
    #     @test isa(e.error, AssertionError)
    # end

    for solver in [cHT112(), cSSP2IMEX222(), cSSP2IMEX433(), cSSP2IMEX332()]
        trixi_include(joinpath(cRK_examples_dir(), "1d",
                               "run_tenmom_two_rare_source_stiff.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      animate = false, final_time = 0.1, nx = 10,
                      solver = solver)
        data_name = "tenmom_two_rare_source_stiff_$(solver).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)
    end

    trixi_include(joinpath(cRK_examples_dir(), "1d", "run_tenmom_two_rare_source_stiff.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1, nx = 10,
                  solver = cHT112())
    data_name = "tenmom_two_rare_source_stiff_cHT112_repeat.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-10)
end

@testset "Shock diffraction reactive" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d",
                           "run_reactive_euler_shock_diffraction.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.001, ny = 20)
    data_name = "reactive_euler_shock_diffraction.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Burg sin smooth 2D cHT112" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_burg2d_smooth_sin.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  solver = cHT112(),
                  animate = false, final_time = 0.1, nx = 5, ny = 5)
    data_name = "burg2d_smooth_sin.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D" begin
    for solver in [cRK22(), "rkfr"]
        trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      animate = false, final_time = 0.3, nx = 5, ny = 5)
        data_name = "ssw_accuracy_2d_$(solver).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end

@testset "SSW convergence 2D cRK33" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 2, solver = cRK33(),
                  animate = false, final_time = 0.3, nx = 5, ny = 5)
    data_name = "ssw_accuracy_2d_cRK33.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D cRK44" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 3, solver = cRK44(),
                  bflux = evaluate,
                  animate = false, final_time = 0.3, nx = 5, ny = 5)
    data_name = "ssw_accuracy_2d_cRK44.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D cHT112" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 1,
                  animate = false, final_time = 0.02, nx = 5, ny = 5,
                  solver = cHT112(), bflux = extrapolate)
    data_name = "ssw_accuracy_2d_cHT112.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D cSSP2IMEX222" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 1,
                  animate = false, final_time = 0.02, nx = 5, ny = 5,
                  solver = cSSP2IMEX222(), bflux = extrapolate)
    data_name = "ssw_accuracy_2d_cSSP2IMEX222.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D cBPR343" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 2,
                  animate = false, final_time = 0.02, nx = 5, ny = 5,
                  solver = cBPR343(), bflux = extrapolate)
    data_name = "ssw_accuracy_2d_cBPR343.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D cAGSA343" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 1,
                  animate = false, final_time = 0.02, nx = 5, ny = 5,
                  solver = cAGSA343(), bflux = extrapolate)
    data_name = "ssw_accuracy_2d_cAGSA343.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D pure FV" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.001, nx = 5, ny = 5,
                  limiter = :limiter_blend)
    data_name = "ssw_accuracy_2d_pure_fv.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D Dirichlet" begin
    for solver in [cRK22(), "rkfr"]
        trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_accuracy_dirichlet.jl"),
                      solver = solver, save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      animate = false, final_time = 0.001, nx = 5, ny = 5)
        data_name = "ssw_accuracy_dirichlet_2d_$(solver).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end

@testset "SSW convergence 2D-x" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_1dx_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.001, nx = 5, ny = 5)
    data_name = "ssw_1dx_accuracy_2d.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "SSW convergence 2D-y" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_ssw_1dy_accuracy.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.001, nx = 5, ny = 5)
    data_name = "ssw_1dy_accuracy_2d.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MHD2D alfven wave" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_mhd_alfven_wave.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.001, nx = 5, ny = 5)
    data_name = "mhd_alfven_wave.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_mhd_alfven_wave.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1,
                  solution_points = "gll", correction_function = "g2",
                  nx = 16, ny = 16,
                  degree = 1,
                  solver = TrixiRKSolver())
    data_name = "mhd_alfven_wave_trixirk.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    volume_flux = (Trixi.flux_central, Trixi.flux_nonconservative_powell_local_symmetric)
    volume_integral = Tenkai.VolumeIntegralFluxDifferencing(volume_flux)
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_mhd_alfven_wave.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1,
                  solution_points = "gll", correction_function = "g2",
                  nx = 16, ny = 16,
                  degree = 1,
                  solver = TrixiRKSolver(volume_integral))
    data_name = "mhd_alfven_wave_trixirk_flux_diff.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MHD2D alfven wave central" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_mhd_alfven_wave.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1, nx = 5, ny = 5,
                  solution_points = "gll",
                  degree = 1,
                  solver = cRK22(MyVolumeIntegralFluxDifferencing(1,
                                                                  flux_central_conservative,
                                                                  flux_central_non_conservative)))
    data_name = "mhd_alfven_wave_central.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MHD2D Orzag-Tang vortex" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_mhd_tang.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1, nx = 5, ny = 5)
    data_name = "mhd_tang.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    volume_flux = (Trixi.flux_central, Trixi.flux_nonconservative_powell_local_symmetric)
    volume_integral = Tenkai.VolumeIntegralFluxDifferencing(volume_flux)
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_mhd_tang.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1,
                  solution_points = "gll", correction_function = "g2",
                  nx = 16, ny = 16,
                  limiter = :limiter_blend,
                  degree = 3,
                  solver = TrixiRKSolver(volume_integral))
    data_name = "mhd_tang_trixirk.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MHD2D Orzag-Tang vortex flux differencing" begin
    function flux_conservative_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_hindenlang_gassner(u1, u2, i, eq.trixi_equations)
    end

    function flux_noncons_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_nonconservative_powell(u1, u2, i, eq.trixi_equations)
    end
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_mhd_tang.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1, nx = 5, ny = 5,
                  degree = 1,
                  solver = cRK22(MyVolumeIntegralFluxDifferencing(1,
                                                                  flux_conservative_hindenlang_gassner,
                                                                  flux_noncons_hindenlang_gassner)))
    data_name = "mhd_tang_flux_diff.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MultiIonMHD2D convergence" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_convergence.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  degree = 1,
                  solver = cRK22(),
                  animate = false, final_time = 0.1, nx = 5, ny = 5)
    data_name = "multiion_convergence.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MultiIonMHD2D KHI" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_khi.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1, nx = 5, ny = 5,
                  degree = 1, amax = 0.01)
    data_name = "multiion_khi.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MultiIonMHD2D KHI central RK22" begin
    function flux_conservative_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_ruedaramirez_etal(u1, u2, i, eq.trixi_equations)
    end

    function flux_noncons_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_nonconservative_ruedaramirez_etal(u1, u2, i, eq.trixi_equations)
    end
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_khi.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 5, ny = 5,
                  degree = 1, limiter = setup_limiter_none(),
                  solver = cRK22(MyVolumeIntegralFluxDifferencing(1,
                                                                  flux_conservative_hindenlang_gassner,
                                                                  flux_noncons_hindenlang_gassner)))
    data_name = "multiion_khi_central_rk22.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MultiIonMHD2D KHI central RK33" begin
    function flux_conservative_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_ruedaramirez_etal(u1, u2, i, eq.trixi_equations)
    end

    function flux_noncons_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_nonconservative_ruedaramirez_etal(u1, u2, i, eq.trixi_equations)
    end
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_khi.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 5, ny = 5,
                  degree = 1, limiter = setup_limiter_none(),
                  solver = cRK33(MyVolumeIntegralFluxDifferencing(1,
                                                                  flux_conservative_hindenlang_gassner,
                                                                  flux_noncons_hindenlang_gassner)))
    data_name = "multiion_khi_central_rk33.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MultiIonMHD2D KHI central RK44" begin
    function flux_conservative_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_ruedaramirez_etal(u1, u2, i, eq.trixi_equations)
    end

    function flux_noncons_hindenlang_gassner(u1, u2, i, eq)
        Trixi.flux_nonconservative_ruedaramirez_etal(u1, u2, i, eq.trixi_equations)
    end
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_khi.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 5, ny = 5,
                  degree = 1, limiter = setup_limiter_none(),
                  solver = cRK44(MyVolumeIntegralFluxDifferencing(1,
                                                                  flux_conservative_hindenlang_gassner,
                                                                  flux_noncons_hindenlang_gassner)))
    data_name = "multiion_khi_central_rk44.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MultiIonMHD2D Collisions" begin
    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_collisions.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  cfl_safety_factor = 0.37, # It crashes for any higher cfl_safety_factor
                  animate = false, final_time = 0.1, nx = 4, ny = 4)
    @show get_errors(sol)
    compare_errors_txt(sol, "multiion_collisions.txt"; overwrite_errors = overwrite_errors)

    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_collisions.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  solver = cHT112(implicit_solver = picard_solver),
                  cfl_safety_factor = 0.99,
                  bflux = extrapolate,
                  animate = false, final_time = 0.1, nx = 4, ny = 4)
    @show get_errors(sol)
    compare_errors_txt(sol, "multiion_collisions_ht112.txt";
                       overwrite_errors = overwrite_errors)

    # This does not work. Is there a bug?
    # trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_collisions.jl"),
    #               save_time_interval = 0.0, save_iter_interval = 0,
    #               compute_error_interval = 0,
    #               solver = cSSP2IMEX222(picard_solver),
    #               cfl_safety_factor = 0.99,
    #               bflux = extrapolate,
    #               animate = false, final_time = 0.1, nx = 4, ny = 4)
    # @show get_errors(sol)
    # compare_errors(sol, 0.0, 0.0, 0.0)

    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_collisions.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  solver = cARS222(implicit_solver = picard_solver),
                  cfl_safety_factor = 0.99,
                  bflux = extrapolate,
                  animate = false, final_time = 0.1, nx = 4, ny = 4)
    @show get_errors(sol)
    compare_errors_txt(sol, "multiion_collisions_ars222.txt";
                       overwrite_errors = overwrite_errors)

    trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_collisions.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  solver = cBPR343(implicit_solver = picard_solver),
                  cfl_safety_factor = 0.99,
                  bflux = extrapolate,
                  animate = false, final_time = 0.1, nx = 4, ny = 4)
    @show get_errors(sol)
    compare_errors_txt(sol, "multiion_collisions_bpr343.txt";
                       overwrite_errors = overwrite_errors)

    # Doesn't work. Is there a bug?
    # trixi_include(joinpath(cRK_examples_dir(), "2d", "run_multiion_collisions.jl"),
    #               save_time_interval = 0.0, save_iter_interval = 0,
    #               compute_error_interval = 0,
    #               solver = cAGSA343(implicit_solver = picard_solver),
    #               cfl_safety_factor = 0.1,
    #               bflux = extrapolate,
    #               animate = false, final_time = 0.1, nx = 4, ny = 4)
    # @show get_errors(sol)
    # compare_errors(sol, 0.0, 0.0, 0.0)
end
