module Tenkai

src_dir = @__DIR__ # Directory of file
data_dir = "$src_dir/../data/"
# base_dir = "$src_dir/../base"
eq_dir = "$src_dir/equations"
kernels_dir = "$src_dir/kernels"
grid_dir = "$src_dir/grids"
utils_dir = "$src_dir/../utils"
solvers_dir = "$src_dir/solvers"
fr_dir = "$solvers_dir/FR"
mdrk_dir = "$solvers_dir/MDRK"
rkfr_dir = "$solvers_dir/RK"
lwfr_dir = "$solvers_dir/LW"

export lwfr_dir, rkfr_dir

examples_dir() = "$src_dir/../Examples"
test_dir = "$src_dir/../test"

export examples_dir # TODO - Remove this export

# TODO - Move to a file "allmodules.jl"
include("$fr_dir/Basis.jl")
include("$grid_dir/CartesianGrids.jl")
include("$eq_dir/InitialValues.jl")
include("$eq_dir/Equations.jl")

using .Equations: nvariables, eachvariable, AbstractEquations

export nvariables, eachvariable, AbstractEquations

using .Basis: Vandermonde_lag

include("$fr_dir/FR.jl")

(using .FR: Problem, Scheme, Parameters, ParseCommandLine, solve, PlotData,
            get_filename, minmod, @threaded, periodic, dirichlet, neumann,
            conservative2conservative_reconstruction!,
            conservative2primitive_reconstruction!,
            primitive2conservative_reconstruction!,
            conservative2characteristic_reconstruction!,
            characteristic2conservative_reconstruction!,
            setup_limiter_none,
            reflect, extrapolate, evaluate,
            get_node_vars, set_node_vars!,
            add_to_node_vars!, subtract_from_node_vars!,
            multiply_add_to_node_vars!, multiply_add_set_node_vars!,
            get_first_node_vars, get_second_node_vars,
            comp_wise_mutiply_node_vars!,
            setup_limiter_blend,
            setup_limiter_hierarchical,
            ParseCommandLine)

(export
        get_filename, minmod, @threaded, Vandermonde_lag,
        Problem, Scheme, Parameters,
        ParseCommandLine,
        setup_limiter_none,
        setup_limiter_blend,
        setup_limiter_hierarchical,
        periodic, dirichlet, neumann, reflect,
        extrapolate, evaluate,
        update_ghost_values_periodic!,
        get_node_vars, set_node_vars!,
        get_first_node_vars, get_second_node_vars,
        add_to_node_vars!, subtract_from_node_vars!,
        multiply_add_to_node_vars!, multiply_add_set_node_vars!,
        comp_wise_mutiply_node_vars!)

(import .FR: flux, prim2con, prim2con!, con2prim, con2prim!, eigmatrix)

(import .FR: update_ghost_values_periodic!,
             update_ghost_values_u1!,
             update_ghost_values_fn_blend!,
             modal_smoothness_indicator,
             modal_smoothness_indicator_gassner,
             set_initial_condition!,
             compute_cell_average!,
             get_cfl,
             compute_time_step,
             compute_face_residual!,
             apply_bound_limiter!,
             apply_tvb_limiter!,
             apply_tvb_limiterβ!,
             setup_limiter_tvb,
             setup_limiter_tvbβ,
             Blend,
             set_blend_dt!,
             fo_blend,
             mh_blend,
             zhang_shu_flux_fix,
             limit_slope,
             no_upwinding_x,
             is_admissible,
             conservative_indicator!,
             apply_hierarchical_limiter!,
             Hierarchical,
             setup_arrays_lwfr,
             setup_arrays_rkfr,
             solve_lwfr,
             solve_rkfr,
             solve_mdrk,
             compute_error,
             initialize_plot,
             write_soln!,
             create_aux_cache,
             write_poly,
             write_soln!,
             post_process_soln)

# TODO - This situation disallows us from doing an allmodules.jl thing

include("$fr_dir/FR1D.jl")

(import .FR1D: update_ghost_values_periodic!,
               update_ghost_values_u1!,
               update_ghost_values_fn_blend!,
               modal_smoothness_indicator,
               modal_smoothness_indicator_gassner,
               set_initial_condition!,
               compute_cell_average!,
               get_cfl,
               compute_time_step,
               compute_face_residual!,
               apply_bound_limiter!,
               apply_tvb_limiter!,
               apply_tvb_limiterβ!,
               setup_limiter_tvb,
               setup_limiter_tvbβ,
 # blending limiter methods
               Blend,
               set_blend_dt!,
               fo_blend,
               mh_blend,
               zhang_shu_flux_fix,
               limit_slope,
               no_upwinding_x,
               is_admissible,
               apply_hierarchical_limiter!,
               Hierarchical,
               compute_error,
               initialize_plot,
               write_soln!,
               create_aux_cache,
               write_poly,
               write_soln!,
               post_process_soln)

include("$fr_dir/FR2D.jl")

(import .FR2D: update_ghost_values_periodic!,
               update_ghost_values_u1!,
               update_ghost_values_fn_blend!,
               modal_smoothness_indicator,
 # modal_smoothness_indicator_gassner, # yet to be implemented
               set_initial_condition!,
               compute_cell_average!,
               get_cfl,
               compute_time_step,
               compute_face_residual!,
               apply_bound_limiter!,
               apply_tvb_limiter!,
               apply_tvb_limiterβ!,
               setup_limiter_tvb,
               setup_limiter_tvbβ,
               Blend,
               set_blend_dt!,
               fo_blend,
               mh_blend,
               blending_flux_factors,
               zhang_shu_flux_fix,
               limit_slope,
               no_upwinding_x,
               is_admissible,
               apply_hierarchical_limiter!,
               Hierarchical, # not yet implemented
               compute_error,
               initialize_plot,
               write_soln!,
               create_aux_cache,
               write_poly,
               write_soln!,
               post_process_soln)

# 1D methods t

# Pack blending methods into containers for user API

# KLUDGE - Move reconstruction named tuples to FR.jl

conservative_reconstruction = (;
                               conservative2recon! = conservative2conservative_reconstruction!,
                               recon2conservative! = conservative2conservative_reconstruction!,
                               recon_string = "conservative")

primitive_reconstruction = (;
                            conservative2recon! = conservative2primitive_reconstruction!,
                            recon2conservative! = primitive2conservative_reconstruction!,
                            recon_string = "primitive")

characteristic_reconstruction = (;
                                 conservative2recon! = conservative2characteristic_reconstruction!,
                                 recon2conservative! = characteristic2conservative_reconstruction!,
                                 recon_string = "characteristic")

(export conservative_reconstruction, primitive_reconstruction,
        characteristic_reconstruction, conservative_indicator!,
        fo_blend, mh_blend)

(export
        setup_limiter_tvb,
        setup_limiter_tvbβ,
        ParseCommandLine)

include("$rkfr_dir/RKFR.jl")

(import .RKFR: setup_arrays_rkfr,
               solve_rkfr,
               compute_cell_residual_rkfr!,
               update_ghost_values_rkfr!)

include("$rkfr_dir/RKFR1D.jl")

(import .RKFR1D: setup_arrays_rkfr,
                 compute_cell_residual_rkfr!,
                 update_ghost_values_rkfr!,
                 flux)

include("$rkfr_dir/RKFR2D.jl")

(import .RKFR2D: setup_arrays_rkfr,
                 compute_cell_residual_rkfr!,
                 update_ghost_values_rkfr!,
                 flux)

# ( # RKFR API exported
# export setup_arrays_rkfr,
#        compute_cell_residual_rkfr!,
#        update_ghost_values_rkfr!
# )

include("$lwfr_dir/LWFR.jl")

(import .LWFR: setup_arrays_lwfr,
               compute_cell_residual_1!, compute_cell_residual_2!,
               compute_cell_residual_3!, compute_cell_residual_4!,
               update_ghost_values_lwfr!,
               eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
               extrap_bflux!)

include("$lwfr_dir/LWFR1D.jl")

(import .LWFR1D: setup_arrays_lwfr,
                 compute_cell_residual_1!, compute_cell_residual_2!,
                 compute_cell_residual_3!, compute_cell_residual_4!,
                 update_ghost_values_lwfr!,
                 eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
                 extrap_bflux!, flux)

include("$lwfr_dir/LWFR2D.jl")

(import .LWFR2D: setup_arrays_lwfr,
                 compute_cell_residual_1!, compute_cell_residual_2!,
                 compute_cell_residual_3!, compute_cell_residual_4!,
                 update_ghost_values_lwfr!,
                 eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
                 extrap_bflux!, flux)

# ( # LWFR API exported
# export setup_arrays_lwfr,
#        compute_cell_residual_1!, compute_cell_residual_2!,
#        compute_cell_residual_3!, compute_cell_residual_4!,
#        update_ghost_values_lwfr!,
#        eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
#        extrap_bflux!
# )

## Example equation files

include("$mdrk_dir/MDRK.jl")

# 1D
include("$eq_dir/EqLinAdv1D.jl")
include("$eq_dir/EqBurg1D.jl")
include("$eq_dir/EqBuckleyLeverett1D.jl")
include("$eq_dir/EqEuler1D.jl")
include("$eq_dir/EqTenMoment1D.jl")

# 2D
include("$eq_dir/EqBurg2D.jl")
include("$eq_dir/EqLinAdv2D.jl")
include("$eq_dir/EqEuler2D.jl")

# Utils

include("$utils_dir/Utils.jl")

end # module
