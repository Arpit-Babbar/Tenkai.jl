module Tenkai

src_dir = @__DIR__ # Directory of file
base_dir = "$src_dir/.."
data_dir = "$src_dir/../data/"
# base_dir = "$src_dir/../base"
eq_dir = "$src_dir/equations"
kernels_dir = "$src_dir/kernels"
grid_dir = "$src_dir/grids"
utils_dir = "$src_dir/../utils"
repro_dir = "$src_dir/../reproducibility"
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

include("$fr_dir/FR1D.jl")
include("$fr_dir/FR2D.jl")

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
        fo_blend, mh_blend, muscl_blend)

(export
        setup_limiter_tvb,
        setup_limiter_tvbβ)

include("$rkfr_dir/RKFR.jl")
include("$rkfr_dir/RKFR1D.jl")
include("$rkfr_dir/RKFR2D.jl")

# ( # RKFR API exported
# export setup_arrays_rkfr,
#        compute_cell_residual_rkfr!,
#        update_ghost_values_rkfr!
# )

include("$lwfr_dir/LWFR.jl")

include("$lwfr_dir/LWFR1D.jl")
include("$lwfr_dir/LWFR1D_ad.jl")
include("$lwfr_dir/LWFR2D.jl")
include("$lwfr_dir/LWFR2D_ad.jl")

# ( # LWFR API exported
# export setup_arrays_lwfr,
#        compute_cell_residual_1!, compute_cell_residual_2!,
#        compute_cell_residual_3!, compute_cell_residual_4!,
#        update_ghost_values_lwfr!,
#        eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
#        extrap_bflux!
# )

include("$mdrk_dir/MDRK.jl")

# 1D
include("$eq_dir/EqLinAdv1D.jl")
include("$eq_dir/EqBurg1D.jl")
include("$eq_dir/EqBuckleyLeverett1D.jl")
include("$eq_dir/EqEuler1D.jl")
include("$eq_dir/EqTenMoment1D.jl")
include("$eq_dir/EqMHD1D.jl")

# 2D
include("$eq_dir/EqBurg2D.jl")
include("$eq_dir/EqLinAdv2D.jl")
include("$eq_dir/EqEuler2D.jl")
include("$eq_dir/EqTenMoment2D.jl")

# cRK methods (developed similar to using Tenkai as a library)

include("$solvers_dir/cRK/cRK.jl")
include("$solvers_dir/cRK/cRK1D.jl")
include("$solvers_dir/cRK/cRK1D_DCSX.jl")
include("$solvers_dir/cRK/cRK2D.jl")

# RKTrixi methods
include("$solvers_dir/Trixi/RKTrixi.jl")

# Standard cRK solvers
export cRK11, cRK22, cRK33, cRK44, cRK65

# DCSX dissipation
export DCSX

# TrixiRKSolver
export TrixiRKSolver

export scheme_degree_plus_one, scheme_n_solution_points

export LWEnzymeTower, MDRKEnzymeTower

end # module
