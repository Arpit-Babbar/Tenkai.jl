# Tenkai.jl

# 1d/2d FR solver

## Setup (5-10 minutes)

Set parameters in any of the `run` files in `Examples` directly, like `run_const_linadv1d.jl` in `Examples/1d`. Some of the options we have are

Scheme: RKFR, LWFR

Correction: radau, g2

Degree: 1,2,3,4

Then, enter the Julia REPL as

```shell
julia --project=.
```
or by starting plain `julia` REPL and then entering `import Pkg; Pkg.activate(".")`. Install all dependencies (only needed the first time) with
```julia
julia> import Pkg; Pkg.instantiate()
```

For the first time, to precompile parts of code to local drive, it is also recommended that you run

```julia
julia> using Tenkai
```

At this point, in principle you can exit the REPL and always run your code directly through the shell as
```shell
julia --project=. Examples/1d/run_const_linadv1d.jl
```

However, since Julia uses a Just-In-Time (JIT) compiler, this is not recommended. If you wish to run this file again or run a different file, Julia will have to compile most of the code again. The recommended way to use Julia is within the Julia REPL where, within each session, the compiled code is stored in memory.

## Usage

Assuming you have already selected the `run_const_linadv1d.jl` file and modified parameters if needed, within the REPL, run it as
```julia
julia> include("Examples/1d/run_const_linadv1d.jl")
```

## Visualization

For 1-D, you can see `png`, `.txt` and interactive HTML files of the final solution in `output` directory.

For 2-D, plot the solution using visit as

```shell
visit -o output/sol*.vtr
```

If you wish to suppress on-screen printing during the simulation, you can redirect the printing to a `string`

```julia
julia> using Suppressor
julia> out = @capture_out include("Examples/1d/run_const_linadv1d.jl") # Capture output in variable out
julia> println(out)                         # Print at the end if needed
```

If you have a 4 core CPU, you can use 4 threads by starting REPL as

```shell
julia --project=. --threads=4
```
