@testset "Burgers' equation 1D" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_burg1d_smooth_sin.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "burg1d_smooth_sine.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Blast 1-D" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_blast.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "blast.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Bucklev" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_bucklev.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "bucklev.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Burg 1D hat" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_burg1d_hat.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5)
    data_name = "burg1d_hat.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "MHD 1D alfven" begin
    trixi_include(joinpath(examples_dir(), "1d", "run_mhd_alfven_wave.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01, nx = 5,
                  solver = cRK44())
    data_name = "alfven_mhd.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    trixi_include(joinpath(examples_dir(), "1d", "run_mhd_alfven_wave_trixirk.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 8,
                  solver = TrixiRKSolver())

    data_name = "alfven_mhd_trixirk.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    trixi_include(joinpath(examples_dir(), "1d", "run_mhd_alfven_wave_trixirk.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 8,
                  solution_points = "gl",
                  correction_function = "radau",
                  solver = TrixiRKSolver())

    data_name = "alfven_mhd_trixirk_gl.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    trixi_include(joinpath(examples_dir(), "1d", "run_mhd_alfven_wave_trixirk.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 8,
                  solver = TrixiRKSolver(VolumeIntegralFluxDifferencing(Trixi.flux_derigs_etal)))

    data_name = "alfven_mhd_trixirk_flux_diff.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Burgers' equation 2D smooth sin" begin
    trixi_include(joinpath(examples_dir(), "2d", "run_burg2d_smooth_sin.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.0, nx = 5, ny = 5)
    data_name = "burg2d_smooth_sine.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

@testset "Isentropic 2D" begin
    local solver_degrees = Dict("mdrk" => [3],
                                "lwfr" => [1, 2, 3, 4],
                                "rkfr" => [1, 2, 3, 4])
    for solver in ["mdrk", "lwfr", "rkfr"], degree in solver_degrees[solver]
        trixi_include(joinpath(examples_dir(), "2d", "run_isentropic.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver, degree = degree,
                      limiter = setup_limiter_none(),
                      animate = false, final_time = 1.0, nx = 5, ny = 5)
        data_name = "isentropic_2d_$(solver)_$(degree).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end

    for time_scheme in ["RK4", "Tsit5"]
        trixi_include(joinpath(examples_dir(), "2d", "run_isentropic.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = "rkfr", degree = 4,
                      time_scheme = time_scheme,
                      limiter = setup_limiter_none(),
                      animate = false, final_time = 1.0, nx = 5, ny = 5)
        data_name = "isentropic_2d_rkfr_4_$(time_scheme).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end

    solvers = (TrixiRKSolver(),
               TrixiRKSolver(Trixi.VolumeIntegralFluxDifferencing(Trixi.flux_ranocha)))
    solver_names = ("trixirk", "trixirk_flux_diff")
    for i in 1:2
        trixi_include(joinpath(examples_dir(), "2d", "run_isentropic_trixirk.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solvers[i], degree = 4,
                      limiter = setup_limiter_none(),
                      solution_points = "gll",
                      correction_function = "g2",
                      animate = false, final_time = 1.0, nx = 8, ny = 8,
                      cfl_safety_factor = 0.98)
        data_name = "isentropic_2d_$(solver_names[i])_4.txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end

@testset "Ten Moment 2D" begin
    for solver in ["mdrk", "lwfr", "rkfr", cRK33()]
        trixi_include(joinpath(examples_dir(), "2d", "run_tenmom_dwave.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver, degree = 3,
                      limiter = setup_limiter_none(),
                      animate = false, final_time = 1.0, nx = 5, ny = 5)
        data_name = "tenmom_dwave_2d_$(solver)_$(degree).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

        trixi_include(joinpath(examples_dir(), "2d", "run_tenmom_near_vacuum.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      solver = solver, degree = 3,
                      # eq comes from the previous test
                      limiter = setup_limiter_tvbβ(eq; tvbM = 0.0, beta = 0.9),
                      animate = false, final_time = 0.02, nx = 5, ny = 5)
        data_name = "tenmom_near_vacuum_2d_$(solver)_$(degree).txt"
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors, tol = 1e-12)
    end
end

# Shu-Osher test
# Shu-Osher test
@testset "Shu-Osher 1D" begin
    γ = 1.4
    Eq = Tenkai.EqEuler1D
    equation = Eq.get_equation(γ)

    trixi_include(joinpath(examples_dir(), "1d", "run_shuosher.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.8, nx = 16,
                  γ = γ, degree = 3,
                  blend_type = fo_blend(equation))

    data_name = "shuosher_1d_fo_blend.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    trixi_include(joinpath(examples_dir(), "1d", "run_shuosher.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.8, nx = 16,
                  γ = γ, degree = 3,
                  blend_type = mh_blend(equation))

    data_name = "shuosher_1d_mh_blend.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)

    trixi_include(joinpath(examples_dir(), "1d", "run_shuosher.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 1.8, nx = 16,
                  γ = γ,
                  solver = TrixiRKSolver(), degree = 3,
                  solution_points = "gll", correction_function = "g2",
                  limiter = setup_limiter_blend(blend_type = fo_blend(equation),
                                                indicating_variables = Eq.rho_p_indicator!,
                                                reconstruction_variables = conservative_reconstruction,
                                                indicator_model = "gassner",
                                                debug_blend = false,
                                                pure_fv = false))
    data_name = "shuosher_1d_trixi_rkfr_3.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

# Density wave 2D
@testset "Density wave 2D" begin
    Eq = Tenkai.EqEuler2D
    equation = Eq.get_equation(1.4)
    limiter_blend = setup_limiter_blend(blend_type = fo_blend(equation),
                                        indicating_variables = Eq.rho_p_indicator!,
                                        reconstruction_variables = conservative_reconstruction,
                                        indicator_model = "gassner",
                                        debug_blend = false,
                                        pure_fv = false,
                                        tvbM = Inf)
    filenames = ("dwave_2d_trixi_rkfr_3.txt", "dwave_2d_trixi_rkfr_3_blend.txt")
    limiters = (setup_limiter_none(), limiter_blend)
    cfl_safety_factors = (0.98, 0.95) # Blending limiter crashes with cfl_safety_factor = 0.98
    for i in 1:2
        trixi_include(joinpath(examples_dir(), "2d", "run_dwave2d.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      animate = false, final_time = 0.1, nx = 16, ny = 16,
                      solver = TrixiRKSolver(), degree = 3,
                      solution_points = "gll", correction_function = "g2",
                      limiter = limiters[i],
                      cfl_safety_factor = cfl_safety_factors[i])
        data_name = filenames[i]
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end

    # Test with GL points and Radau correction function
    trixi_include(joinpath(examples_dir(), "2d", "run_dwave2d.jl"),
                  save_time_interval = 0.0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.1, nx = 16, ny = 16,
                  solver = TrixiRKSolver(), degree = 3,
                  solution_points = "gl", correction_function = "radau",
                  limiter = setup_limiter_none(),
                  cfl_safety_factor = 0.98)
    data_name = "dwave_2d_trixi_rkfr_3_gl.txt"
    compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
end

# Double Mach reflection
@testset "Double Mach reflection" begin
    Eq = Tenkai.EqEuler2D
    equation = Eq.get_equation(1.4)
    filenames = ("dmr_fo_blend.txt", "dmr_mh_blend.txt", "dmr_mh_blend_crk.txt")
    blend_types = (fo_blend(equation), mh_blend(equation), mh_blend(equation))
    solvers = ("lwfr", "lwfr", cRK44())
    for i in 1:3
        trixi_include(joinpath(examples_dir(), "2d", "run_double_mach_reflection.jl"),
                      save_time_interval = 0.0, save_iter_interval = 0,
                      compute_error_interval = 0,
                      animate = false, final_time = 0.1, ny = 5,
                      degree = 3,
                      solver = solvers[i],
                      blend_type = blend_types[i])
        data_name = filenames[i]
        compare_errors_txt(sol, data_name; overwrite_errors = overwrite_errors)
    end
end

@testset "Float32 type preservation" begin
    # Test that Float32 is properly preserved throughout the computation
    trixi_include(joinpath(examples_dir(), "1d", "run_burg1d_float32.jl"),
                  save_time_interval = 0.0f0, save_iter_interval = 0,
                  compute_error_interval = 0,
                  animate = false, final_time = 0.01f0, nx = 10)
    
    # Verify Float32 types are preserved
    @test typeof(sol["problem"].domain) == Vector{Float32}
    @test eltype(sol["grid"].xc) == Float32
    @test eltype(sol["grid"].xf) == Float32
    @test eltype(sol["grid"].dx) == Float32
    
    # Check solution array type (sol has "u", not "u_f")
    if sol["u"] isa AbstractArray
        elem_type = eltype(eltype(sol["u"]))
        @test elem_type == Float32
        println("✓ Float32 type preserved in solution: ", elem_type)
    end
end
