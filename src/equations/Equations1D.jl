module Equations1D

   using UnPack
   using StaticArrays


   import ..FR1D: AbstractEquations

   abstract type AbstractLinAdv1D <: AbstractEquations{1,1} end
   abstract type AbstractBurg1D <: AbstractEquations{1,1} end
   abstract type AbstractBuckleyLeverett1D <: AbstractEquations{1,1} end

   abstract type AbstractEulerEq1D <: AbstractEquations{1,3} end

   # Export types
   export AbstractEq1D, AbstractEquations, AbstractSystemEq1D

   # Export methods (TODO - Are they called methods or functions)
   (
   export is_admissible, con2recon!, recon2con!, limit_slope!
   )

end
