using Trixi

lw_fourier_cfl(N) = (0.259, 0.170, 0.103, 0.069)[N]

# Use this as safety factor to get LW CFL with same safety factor
function trixi2lw(cfl_number, dg)
    N = Trixi.polydeg(dg)
    nnodes = N + 1
    fourier_cfl = lw_fourier_cfl(N)
    cfl_number = cfl_number * nnodes * fourier_cfl
    return cfl_number
end
