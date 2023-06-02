using SSFR
FR = SSFR.FR2D
Eq = SSFR.EqEuler2D
using StaticArrays
#------------------------------------------------------------------------------
xmin, xmax = -10.0, 10.0
ymin, ymax = -10.0, 10.0

boundary_value     = Eq.zero_bv # dummy
boundary_condition = (periodic, periodic, periodic, periodic)
γ = 1.4

   function initial_value_isentropic(x,y)
      γ      = 1.4
      β      = 5.0
      M      = 0.5
      α      = 45.0 * (pi/180.0)
      x0, y0 = 0.0, 0.0
      u0, v0 = M*cos(α), M*sin(α)
      r2     = (x-x0)^2 + (y-y0)^2

      a1 = 0.5 * β / π
      a2 = 0.5*(γ-1.0) * a1^2 / γ

      ρ  = ( 1.0 - a2 * exp(1.0-r2) ) ^ (1.0/(γ-1.0))
      v1 = u0 - a1*(y-y0) * exp(0.5*(1.0-r2))
      v2 = v0 + a1*(x-x0) * exp(0.5*(1.0-r2))
      p = ρ^γ
      ρ_v1 = ρ*v1
      ρ_v2 = ρ*v2
      return SVector(ρ, ρ*v1, ρ*v2, p/(γ-1.0) + 0.5*(ρ_v1*v1+ρ_v2*v2))
   end

   # initial_value_ref, final_time, ic_name = Eq.dwave_data
   function exact_solution_(x,y,t)
      xmin, xmax = -10.0, 10.0
      ymin, ymax = -10.0, 10.0
      Lx = xmax - xmin
      Ly = ymax - ymin
      theta = 45.0 * (pi/180.0)
      M = 0.5
      u0 = M * cos(theta)
      v0 = M * sin(theta)
      q1, q2 = x - u0*t, y - v0*t
      if q1 > xmax
         q1 = q1 - Lx*floor((q1+xmin)/Lx)
      elseif q1 < xmin
         q1 = q1 + Lx*floor((xmax-q1)/Lx)
      end
      if q2 > ymax
         q2 = q2 - Ly*floor((q2+ymin)/Ly)
      elseif q2 < ymin
         q2 = q2 + Ly*floor((ymax-q2)/Ly)
      end

      return initial_value_isentropic(q1, q2)
   end

   degree = 4
   solver = "lwfr"
   solution_points = "gl"
   correction_function = "radau"
   numerical_flux = Eq.rusanov

   bound_limit = "yes"
   bflux = evaluate
   final_time = 20 * sqrt(2.0) / 0.5

   nx, ny = 160, 160
   cfl = 0.0
   bounds = ([-Inf],[Inf]) # Not used in Euler
   tvbM = 0.0
   save_iter_interval = 0
   save_time_interval = final_time / 30.0
   animate = true # Factor on save_iter_interval or save_time_interval
   compute_error_interval = 0

   #------------------------------------------------------------------------------
   grid_size = [nx, ny]
   domain = [xmin, xmax, ymin, ymax]
   equation = Eq.get_equation(γ)
   problem = Problem(domain, initial_value_isentropic, boundary_value, boundary_condition,
                        final_time, exact_solution_)
   limiter = setup_limiter_none()
   scheme = Scheme(solver, degree, solution_points, correction_function,
                     numerical_flux, bound_limit, limiter, bflux)
   param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                        save_time_interval, compute_error_interval,
                        animate = animate)
sol = SSFR.solve(equation, problem, scheme, param);

println(sol["errors"])

return sol;


