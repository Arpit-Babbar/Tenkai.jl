module TenkaicRK

using Tenkai

include("equations/equations.jl")
include("solvers/cRK.jl")
include("solvers/cRK1D.jl")
include("solvers/cRK1D_non_conservative.jl")
include("solvers/cRK2D.jl")
include("solvers/cRK2D_non_conservative.jl")
include("grid/StepGrid.jl")
include("solvers/FR2D.jl")
include("equations/EqShallowWater1D.jl")
include("equations/EqShearShallowWater1D.jl")
include("equations/EqShearShallowWater2D.jl")
include("equations/EqJinXin1D.jl")
include("equations/EqEuler2D.jl")
include("equations/EqEulerReactive1D.jl")
include("equations/EqEulerReactive2D.jl")
include("equations/EqEulerReactive2DStepGrid.jl")
include("equations/EqMHD2D.jl")
include("equations/EqMultiIonMHD2D.jl")
include("equations/EqSupBurg1D.jl")

include("solvers/RKFR1D_non_conservative.jl")
include("solvers/RKFR2D_non_conservative.jl")
include("solvers/RKTrixi2D_non_conservative.jl")

include("equations/EqVarAdv.jl")

cRK_examples_dir() = joinpath(dirname(pathof(Tenkai)), "..", "Examples/examples_crk")
cRK_elixirs_dir() = joinpath(dirname(pathof(Tenkai)), "..", "elixirs")

# Compact Runge-Kutta Solvers
export cIMEX111, cHT112, cHT112Explicit, cSSP2IMEX222,
       cARS222, cBPR343, cAGSA343,
       cSSP2IMEX332, cSSP2IMEX433, DoublecRKSourceSolver

# Flux differencing stuff
export flux_central_conservative, flux_central_non_conservative,
       MyVolumeIntegralFluxDifferencing

# Implicit solvers
export newton_solver, picard_solver

export fo_blend_imex

# Directories
export cRK_examples_dir, cRK_elixirs_dir

import Trixi

end # module TenkaicRK
