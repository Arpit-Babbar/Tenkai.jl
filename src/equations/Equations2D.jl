module Equations2D

   using MuladdMacro
   using ..FR2D: AbstractEquations, nvariables, eachvariable
   abstract type AbstractScalarEq2D <: AbstractEquations{2,1} end
   # abstract type AbstractSystemEq2D <: AbstractEq2D end
   abstract type AbstractEulerEq2D <: AbstractEquations{2,4} end

   # By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
   # Since these FMAs can increase the performance of many numerical algorithms,
   # we need to opt-in explicitly.
   # See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
   @muladd begin

   # Define dummy functions for modules to extend
   flux(x, u, eq, orientation) = @assert false "method not defined for equation"
   flux(x, u, eq) = @assert false "method not defined for equation"
   prim2con(eq, prim) = @assert false "method not defined for equation"
   prim2con!(eq, prim) = @assert false "method not defined for equation"
   prim2con!(eq, prim, U) = @assert false "method not defined for equation"
   con2prim(eq, U) = @assert false "method not defined for equation"
   con2prim!(eq, U) = @assert false "method not defined for equation"
   con2prim!(eq, U, prim) = @assert false "method not defined for equation"
   speed(x, u, eq) = @assert false "method not implemented for equation"

   include("EqLinAdv2D.jl")
   using .EqLinAdv2D: speed, flux

   include("EqBurg2D.jl")
   using .EqBurg2D: flux, speed

   include("EqEuler2D.jl")
   using .EqEuler2D: prim2con, prim2con!, con2prim, con2prim!

   (
   export AbstractEq2D, AbstractScalarEq2D, AbstractEquations, get_node_vars,
          set_node_vars!, nvariables, eachvariable
   )

   # Exporting submodules
   export EqLinAdv2D, EqBurg2D, EqEuler2D

   # Physical fluxes of all PDEs
   export speed, flux

   end # @muladd
end
