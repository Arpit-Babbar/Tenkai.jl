module Grid

using Printf
using SimpleUnPack
using TimerOutputs
using OffsetArrays

struct CartesianGrid1D{RealT <: Real}
    domain::Vector{RealT}   # xmin,xmax
    size::Int64               # nx, ny
    xc::Array{RealT, 1}      # x coord of cell center
    xf::Array{RealT, 1}      # x coord of faces
    dx::OffsetVector{RealT, Vector{RealT}}      # cell size along x
end

struct CartesianGrid2D{RealT <: Real}
    domain::Vector{RealT}   # xmin,xmax,ymin,ymax
    size::Vector{Int64}       # nx, ny
    xc::Array{RealT, 1}      # x coord of cell center
    yc::Array{RealT, 1}      # y coord of cell center
    xf::Array{RealT, 1}      # x coord of faces
    yf::Array{RealT, 1}      # y coord of faces
    dx::OffsetVector{RealT, Vector{RealT}}      # cell size along x
    dy::OffsetVector{RealT, Vector{RealT}}      # cell size along y
end

# 1D/2D Uniform Cartesian grid
function make_cartesian_grid(problem, size::Int64)
    @unpack domain = problem
    println("Making 1D uniform Cartesian grid")
    xmin, xmax = domain
    nx = size
    dx1 = (xmax - xmin) / nx
    xc = collect(LinRange(xmin + 0.5 * dx1, xmax - 0.5 * dx1, nx))
    @printf("   Grid size = %d \n", nx)
    @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
    @printf("   dx        = %e\n", dx1)
    dx = OffsetArray(zeros(nx + 2), OffsetArrays.Origin(0))
    dx .= dx1
    xf = collect(LinRange(xmin, xmax, nx + 1))
    return CartesianGrid1D(domain, size, xc, xf, dx)
end

function make_cartesian_grid(problem, size::Vector{Int64})
    @unpack domain = problem
    println("Making 2D uniform Cartesian grid")
    xmin, xmax, ymin, ymax = domain
    nx, ny = size
    dx1 = (xmax - xmin) / nx
    dy1 = (ymax - ymin) / ny
    xc = collect(LinRange(xmin + 0.5 * dx1, xmax - 0.5 * dx1, nx))
    yc = collect(LinRange(ymin + 0.5 * dy1, ymax - 0.5 * dy1, ny))
    @printf("   Grid size = %d x %d\n", nx, ny)
    @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
    @printf("   ymin,ymax = %e, %e\n", ymin, ymax)
    @printf("   dx, dy    = %e, %e\n", dx1, dy1)
    dx = OffsetArray(zeros(nx + 2), OffsetArrays.Origin(0))
    dy = OffsetArray(zeros(ny + 2), OffsetArrays.Origin(0))
    dx .= dx1
    dy .= dy1
    xf = collect(LinRange(xmin, xmax, nx + 1))
    yf = collect(LinRange(ymin, ymax, ny + 1))
    return CartesianGrid2D(domain, size, xc, yc, xf, yf, dx, dy)
end

export make_cartesian_grid

end
