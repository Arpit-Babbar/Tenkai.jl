module Grid

using Printf
using SimpleUnPack
using TimerOutputs
using OffsetArrays

struct CartesianGrid1D
    domain::Vector{Float64}   # xmin,xmax
    size::Int64               # nx, ny
    xc::Array{Float64, 1}      # x coord of cell center
    xf::Array{Float64, 1}      # x coord of faces
    dx::OffsetVector{Float64, Vector{Float64}}      # cell size along x
end

struct CartesianGrid2D
    domain::Vector{Float64}   # xmin,xmax,ymin,ymax
    size::Vector{Int64}       # nx, ny
    xc::Array{Float64, 1}      # x coord of cell center
    yc::Array{Float64, 1}      # y coord of cell center
    xf::Array{Float64, 1}      # x coord of faces
    yf::Array{Float64, 1}      # y coord of faces
    dx::OffsetVector{Float64, Vector{Float64}}      # cell size along x
    dy::OffsetVector{Float64, Vector{Float64}}      # cell size along y
end

# 1D/2D Uniform Cartesian grid
function make_cartesian_grid(problem, size::Int64)
    @unpack domain = problem
    println("Making 1D uniform Cartesian grid")
    xmin, xmax = domain
    nx = size
    dx1 = (xmax - xmin) / nx
    xc = LinRange(xmin + 0.5 * dx1, xmax - 0.5 * dx1, nx)
    @printf("   Grid size = %d \n", nx)
    @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
    @printf("   dx        = %e\n", dx1)
    dx = OffsetArray(zeros(nx + 2), OffsetArrays.Origin(0))
    dx .= dx1
    xf = LinRange(xmin, xmax, nx + 1)
    return CartesianGrid1D(domain, size, xc, xf, dx)
end

function make_cartesian_grid(problem, size::Vector{Int64})
    @unpack domain = problem
    println("Making 2D uniform Cartesian grid")
    xmin, xmax, ymin, ymax = domain
    nx, ny = size
    dx1 = (xmax - xmin) / nx
    dy1 = (ymax - ymin) / ny
    xc = LinRange(xmin + 0.5 * dx1, xmax - 0.5 * dx1, nx)
    yc = LinRange(ymin + 0.5 * dy1, ymax - 0.5 * dy1, ny)
    @printf("   Grid size = %d x %d\n", nx, ny)
    @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
    @printf("   ymin,ymax = %e, %e\n", ymin, ymax)
    @printf("   dx, dy    = %e, %e\n", dx1, dy1)
    dx = OffsetArray(zeros(nx + 2), OffsetArrays.Origin(0))
    dy = OffsetArray(zeros(ny + 2), OffsetArrays.Origin(0))
    dx .= dx1
    dy .= dy1
    xf = LinRange(xmin, xmax, nx + 1)
    yf = LinRange(ymin, ymax, ny + 1)
    return CartesianGrid2D(domain, size, xc, yc, xf, yf, dx, dy)
end

export make_cartesian_grid

end
