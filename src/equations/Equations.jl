module Equations

   # TODO - Maybe this file should be called AbstractTypes

   abstract type AbstractEquations{NDIMS,NVAR} end # TODO - Unify Equations1D and Equations2D

   @inline nvariables(::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS

   @inline eachvariable(equations::AbstractEquations) = Base.OneTo(nvariables(equations))

   export nvariables, eachvariable, AbstractEquations

end