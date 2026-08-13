module CartesianGrids

using Printf
using SimpleUnPack
using TimerOutputs
using OffsetArrays

using HDF5: h5open, attributes

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

# A version of `LinRange` that works in a general arithmetic. `Base.lerpi` builds
# the interpolation parameter `j / d` from integers, so it is always `Float64`,
# which caps the grid of a higher precision run at `Float64` accuracy. Here the
# parameter is formed in the wider of `RealT` and `Float64`, which reproduces
# `LinRange` bit for bit in `Float32` and `Float64`. (Comment written by Claude.)
function uniform_points(xmin::RealT, xmax::RealT, n) where {RealT <: Real}
    T = promote_type(RealT, Float64)
    a, b = T(xmin), T(xmax)
    d = T(max(n, 1))  # `n == 0` is a single point, as for `LinRange(a, b, 1)`
    return RealT[RealT((1 - T(i - 1) / d) * a + (T(i - 1) / d) * b)
                 for i in 1:(n + 1)]
end

# 1D/2D Uniform Cartesian grid
function make_cartesian_grid(problem, size::Int64)
    @unpack domain = problem
    println("Making 1D uniform Cartesian grid")
    xmin, xmax = domain
    nx = size
    dx1 = (xmax - xmin) / nx
    RealT = typeof(dx1)
    xf = uniform_points(xmin, xmax, nx)
    xc = uniform_points(xmin + dx1 / 2, xmax - dx1 / 2, nx - 1)
    @printf("   Grid size = %d \n", nx)
    @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
    @printf("   dx        = %e\n", dx1)
    dx = OffsetArray(zeros(RealT, nx + 2), OffsetArrays.Origin(0))
    dx .= dx1
    return CartesianGrid1D(domain, size, xc, xf, dx)
end

function make_cartesian_grid(problem, size::Vector{Int64})
    @unpack domain = problem
    println("Making 2D uniform Cartesian grid")
    xmin, xmax, ymin, ymax = domain
    nx, ny = size
    dx1 = (xmax - xmin) / nx
    dy1 = (ymax - ymin) / ny
    RealT = typeof(dx1)
    xf = uniform_points(xmin, xmax, nx)
    yf = uniform_points(ymin, ymax, ny)
    xc = uniform_points(xmin + dx1 / 2, xmax - dx1 / 2, nx - 1)
    yc = uniform_points(ymin + dy1 / 2, ymax - dy1 / 2, ny - 1)
    @printf("   Grid size = %d x %d\n", nx, ny)
    @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
    @printf("   ymin,ymax = %e, %e\n", ymin, ymax)
    @printf("   dx, dy    = %e, %e\n", dx1, dy1)
    dx = OffsetArray(zeros(RealT, nx + 2), OffsetArrays.Origin(0))
    dy = OffsetArray(zeros(RealT, ny + 2), OffsetArrays.Origin(0))
    dx .= dx1
    dy .= dy1
    return CartesianGrid2D(domain, size, xc, yc, xf, yf, dx, dy)
end

function save_mesh_file(mesh::CartesianGrid2D, output_directory)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    xmin, xmax, ymin, ymax = mesh.domain
    mapping(x, y) = (xmin + (xmax - xmin)ymin + (ymax - ymin) * y)

    # From src/meshes/structured_mesh.jl in Trixi.jl
    coordinates_min = (xmin, ymin)
    coordinates_max = (xmax, ymax)
    mapping_as_string = """
        coordinates_min = $coordinates_min
        coordinates_max = $coordinates_max
        mapping = coordinates2mapping(coordinates_min, coordinates_max)
        """

    filename = joinpath(output_directory, "mesh.h5")

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = "StructuredMesh" # For Trixi2Vtk
        attributes(file)["ndims"] = 2
        attributes(file)["size"] = mesh.size
        # HDF5 only understands the standard floating point types, so the
        # mesh extents are written out as `Float64`.
        attributes(file)["xmin"] = Float64(xmin)
        attributes(file)["xmax"] = Float64(xmax)
        attributes(file)["ymin"] = Float64(ymin)
        attributes(file)["ymax"] = Float64(ymax)
        attributes(file)["mapping"] = mapping_as_string
    end

    return filename
end

export make_cartesian_grid, save_mesh_file

end
