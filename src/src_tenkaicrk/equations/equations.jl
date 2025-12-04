using Tenkai: AbstractEquations

using Tenkai.EqBurg2D: Burg2D

abstract type AbstractNonConservativeEquations{NDIMS, NVAR} <:
              AbstractEquations{NDIMS, NVAR} end

function calc_non_cons_Bu(u, u_non_cons, x, y, t, orientation::Int,
                          eq::Burg2D)
    return zero(u)
end

# Empty method to be specified by the equations.
function flux_central_nc end
