using Printf
using SimpleUnPack
using TimerOutputs
using OffsetArrays

using IterTools: product
using HDF5: h5open, attributes

import Tenkai: make_cartesian_grid, prolong_solution_to_face_and_ghosts!

using Tenkai: Scheme, AbstractEquations

struct StepGrid{NearCorners <: Tuple}
    domain::Vector{Float64}   # xmin,xmax,ymin,ymax
    size::Vector{NTuple{2, Int}}       # nx, ny
    xc::Array{Float64, 1}      # x coord of cell center
    yc::Array{Float64, 1}      # y coord of cell center
    xf::Array{Float64, 1}      # x coord of faces
    yf::Array{Float64, 1}      # y coord of faces
    dx::OffsetVector{Float64, Vector{Float64}}      # cell size along x
    dy::OffsetVector{Float64, Vector{Float64}}      # cell size along y
    element_indices::Vector{CartesianIndex{2}}
    n_elements::Int
    face_x_indices::Vector{CartesianIndex{2}}
    n_x_faces::Int
    face_y_indices::Vector{CartesianIndex{2}}
    n_y_faces::Int
    near_corners::NearCorners
end

function Tenkai.make_cartesian_grid(problem, grid_size::Vector{NTuple{2, Int}})
    @unpack domain = problem
    println("Making 2D grid with a step")
    xmin, xmax, ymin, ymax = domain
    nx_tuple, ny_tuple = grid_size
    nx, ny = nx_tuple[2], ny_tuple[2] #= Do get all the coordinates,
    we will restrict our updates by restricting our loops =#
    dx1 = (xmax - xmin) / nx
    dy1 = (ymax - ymin) / ny
    xc = [LinRange(xmin + 0.5 * dx1, xmax - 0.5 * dx1, nx)...]
    yc = [LinRange(ymin + 0.5 * dy1, ymax - 0.5 * dy1, ny)...]
    @printf("   Grid size = %d x %d\n", nx, ny)
    @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
    @printf("   ymin,ymax = %e, %e\n", ymin, ymax)
    @printf("   dx, dy    = %e, %e\n", dx1, dy1)
    dx = OffsetArray(zeros(nx + 2), OffsetArrays.Origin(0))
    dy = OffsetArray(zeros(ny + 2), OffsetArrays.Origin(0))
    dx .= dx1
    dy .= dy1
    xf = [LinRange(xmin, xmax, nx + 1)...]
    yf = [LinRange(ymin, ymax, ny + 1)...]

    # Make arrays for iterators
    loop_range_1_elements = CartesianIndices((1:nx_tuple[1],
                                              (ny_tuple[1] + 1):ny_tuple[2]))
    loop_range_2_elements = CartesianIndices(((nx_tuple[1] + 1):nx_tuple[2],
                                              1:ny_tuple[2]))
    loop_range_elements = [loop_range_1_elements..., loop_range_2_elements...]
    n_elements = length(loop_range_elements)

    loop_range_1_x_faces = CartesianIndices((1:nx_tuple[1], # nx_tuple[1] + 1 face will be handled in next
                                             (ny_tuple[1] + 1):ny_tuple[2]))
    loop_range_2_x_faces = CartesianIndices(((nx_tuple[1] + 1):(nx_tuple[2] + 1),
                                             1:ny_tuple[2]))

    loop_range_x_faces = [loop_range_1_x_faces..., loop_range_2_x_faces...]
    n_x_faces = length(loop_range_x_faces)

    loop_range_1_y_faces = CartesianIndices((1:nx_tuple[1],
                                             (ny_tuple[1] + 1):(ny_tuple[2] + 1)))
    loop_range_2_y_faces = CartesianIndices(((nx_tuple[1] + 1):nx_tuple[2],
                                             1:(ny_tuple[2] + 1)))

    loop_range_y_faces = [loop_range_1_y_faces..., loop_range_2_y_faces...]
    n_y_faces = length(loop_range_y_faces)

    near_step_corner_x_1 = (nx_tuple[1] - 1):(nx_tuple[1] + 1)
    near_step_corner_y_1 = (ny_tuple[1] - 1):(ny_tuple[1] + 1)

    near_step_corner_1 = Iterators.product(near_step_corner_x_1, near_step_corner_y_1)

    near_step_corner_x_2 = -1:+1
    near_step_corner_y_2 = (ny_tuple[1] - 1):(ny_tuple[1] + 1)

    near_step_corner_2 = Iterators.product(near_step_corner_x_2, near_step_corner_y_2)

    near_step_corner_x_3 = (nx_tuple[1] - 1):(nx_tuple[1] + 1)
    near_step_corner_y_3 = -1:+1

    near_step_corner_3 = Iterators.product(near_step_corner_x_3, near_step_corner_y_3)

    near_step_corner = (near_step_corner_1..., near_step_corner_2...,
                        near_step_corner_3...)

    near_bottom_right_corner_x = (nx_tuple[2] - 1):nx_tuple[2]
    near_bottom_right_corner_y = -1:+1

    near_bottom_right_corner = Iterators.product(near_bottom_right_corner_x,
                                                 near_bottom_right_corner_y)

    near_top_left_corner_x = -1:+1
    near_top_left_corner_y = (ny_tuple[2] - 1):ny_tuple[2]

    near_top_left_corner = Iterators.product(near_top_left_corner_x,
                                             near_top_left_corner_y)

    near_top_right_corner_x = (nx_tuple[2] - 1):nx_tuple[2]
    near_top_right_corner_y = (ny_tuple[2] - 1):ny_tuple[2]

    near_top_right_corner = Iterators.product(near_top_right_corner_x,
                                              near_top_right_corner_y)

    near_corners = (near_step_corner..., near_bottom_right_corner...,
                    near_top_right_corner..., near_top_left_corner...)

    return StepGrid(domain, grid_size, xc, yc, xf, yf, dx, dy,
                    loop_range_elements, n_elements,
                    loop_range_x_faces, n_x_faces,
                    loop_range_y_faces, n_y_faces,
                    near_corners)
end

function save_mesh_file(mesh::StepGrid, output_directory)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    xmin, xmax, ymin, ymax = mesh.domain

    filename = joinpath(output_directory, "mesh.h5")

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = "StructuredMesh" # For Trixi2Vtk
        attributes(file)["ndims"] = 2
        attributes(file)["size"] = [mesh.size[1][2], mesh.size[2][2]]
        attributes(file)["xmin"] = xmin
        attributes(file)["xmax"] = xmax
        attributes(file)["ymin"] = ymin
        attributes(file)["ymax"] = ymax
    end

    return filename
end

function element_iterator(grid::StepGrid)
    n_elements = grid.n_elements
    return Base.OneTo(n_elements)
end

element_indices(element, grid::StepGrid) = grid.element_indices[element]

function element_iterator_with_ghosts(grid::StepGrid)
    nx_tuple, ny_tuple = grid.size
    # The step region is (nx_tuple[1]+1:nx_tuple[2], 1:ny_tuple[1])
    loop_range_1 = CartesianIndices((1:nx_tuple[1], 1:ny_tuple[2]))
    loop_range_2 = CartesianIndices(((nx_tuple[1] + 1):nx_tuple[2],
                                     (ny_tuple[1] + 1):ny_tuple[2]))
    loop_range_ghosts = CartesianIndices((0:0, 1:1))
    loop_range = (loop_range_1..., loop_range_2..., loop_range_ghosts...)
    return loop_range
end

function face_x_iterator(grid::StepGrid)
    n_faces = grid.n_x_faces
    return Base.OneTo(n_faces)
end

face_x_indices(element, grid::StepGrid) = grid.face_x_indices[element]

function face_y_iterator(grid::StepGrid)
    n_faces = grid.n_y_faces
    return Base.OneTo(n_faces)
end

face_y_indices(element, grid::StepGrid) = grid.face_y_indices[element]

function bottom_horizontal_iterator(grid::StepGrid)
    nx_tuple, _ = grid.size
    return Base.OneTo(nx_tuple[2])
end

function bottom_vertical_iterator(grid::StepGrid)
    _, ny_tuple = grid.size
    return Base.OneTo(ny_tuple[1])
end

function bottom_physical_element(el_x, grid::StepGrid)
    nx_tuple, ny_tuple = grid.size
    if el_x <= nx_tuple[1]
        return ny_tuple[1] + 1
    else
        return 1
    end
end

function leftmost_physical_element(el_y, grid::StepGrid)
    nx_tuple, ny_tuple = grid.size
    if el_y <= ny_tuple[1]
        return nx_tuple[1] + 1
    else
        return 1
    end
end

function isnearcorners(el_x, el_y, grid::StepGrid)
    if (el_x, el_y) in grid.near_corners
        return true
    else
        return false
    end
end

function setup_arrays(grid::StepGrid, scheme::Scheme{<:cRKSolver}, eq::AbstractEquations{2})
    gArray(nvar, nx, ny) = OffsetArray(zeros(nvar, nx + 2, ny + 2),
                                       OffsetArrays.Origin(1, 0, 0))
    gArray(nvar, n1, n2, nx, ny) = OffsetArray(zeros(nvar, n1, n2, nx + 2, ny + 2),
                                               OffsetArrays.Origin(1, 1, 1, 0, 0))
    # Allocate memory
    @unpack degree, bflux = scheme
    @unpack bflux_ind = bflux
    @unpack nvar = eq
    nd = degree + 1
    nx_tuple, ny_tuple = grid.size
    nx, ny = nx_tuple[2], ny_tuple[2]
    u1 = gArray(nvar, nd, nd, nx, ny)
    u1x = gArray(nvar, nd, nd, nx, ny)
    u1y = gArray(nvar, nd, nd, nx, ny)
    ua = gArray(nvar, nx, ny)
    res = gArray(nvar, nd, nd, nx, ny)
    Fb = gArray(nvar, nd, 4, nx, ny)
    Ub = gArray(nvar, nd, 4, nx, ny)
    u1_b = copy(Ub)

    # Cell residual cache

    nt = Threads.nthreads()
    cell_array_sizes = Dict(1 => 10, 2 => 11, 3 => 14, 4 => 15)
    big_eval_data_sizes = Dict(1 => 12, 2 => 32, 3 => 40, 4 => 56)
    small_eval_data_sizes = Dict(1 => 4, 2 => 4, 3 => 4, 4 => 4)
    if bflux_ind == extrapolate
        cell_array_size = cell_array_sizes[degree]
        big_eval_data_size = 2
        small_eval_data_size = 2
    elseif bflux_ind == evaluate
        cell_array_size = cell_array_sizes[degree]
        big_eval_data_size = big_eval_data_sizes[degree]
        small_eval_data_size = small_eval_data_sizes[degree]
    else
        @assert false "Incorrect bflux"
    end

    # Construct `cache_size` number of objects with `constructor`
    # and store them in an SVector
    function alloc(constructor, cache_size)
        SVector{cache_size}(constructor(undef) for _ in Base.OneTo(cache_size))
    end

    # Create the result of `alloc` for each thread. Basically,
    # for each thread, construct `cache_size` number of objects with
    # `constructor` and store them in an SVector
    function alloc_for_threads(constructor, cache_size)
        nt = Threads.nthreads()
        SVector{nt}([alloc(constructor, cache_size) for _ in Base.OneTo(nt)])
    end

    MArr = MArray{Tuple{nvariables(eq), nd, nd}, Float64}
    cell_arrays = alloc_for_threads(MArr, cell_array_size)

    MEval = MArray{Tuple{nvariables(eq), nd}, Float64}
    eval_data_big = alloc_for_threads(MEval, big_eval_data_size)

    MEval_small = MArray{Tuple{nvariables(eq), 1}, Float64}
    eval_data_small = alloc_for_threads(MEval_small, small_eval_data_size)

    eval_data = (; eval_data_big, eval_data_small)

    # Ghost values cache

    Marr = MArray{Tuple{nvariables(eq), 1}, Float64}

    ghost_cache = alloc_for_threads(Marr, 2)

    # KLUDGE - Rename this to LWFR cache
    cache = (; u1, u1x, u1y, ua, res, Fb, Ub, eval_data, cell_arrays, ghost_cache, u1_b)
    return cache
end

function prolong_solution_to_face_and_ghosts!(u1, cache, eq::AbstractEquations{2},
                                              grid::StepGrid,
                                              op, problem, scheme, aux, t, dt)
    return nothing # Use cell average for face residual in StepGrid functions so this is not needed
end

export make_step_grid, save_mesh_file, StepGrid
