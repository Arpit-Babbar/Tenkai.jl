using Plots
using Setfield
using Tenkai: utils_dir

include("$utils_dir/Utils.jl")

ARGS = ["--grid_size", "60"]
include("../Examples/1d/run_blast.jl")
empty!(ARGS)

param = setproperties(param, (grid_size = 400, cfl_safety_factor = 0.95))

degree = 3

limiter_fo = setup_limiter_blend(blend_type = fo_blend(equation),
                                 # indicating_variables = Eq.rho_p_indicator!,
                                 indicating_variables = Eq.rho_p_indicator!,
                                 reconstruction_variables = conservative_reconstruction,
                                 indicator_model = indicator_model,
                                 debug_blend = debug_blend,
                                 pure_fv = pure_fv)
limiter_mh = setup_limiter_blend(blend_type = mh_blend(equation),
                                 # indicating_variables = Eq.rho_p_indicator!,
                                 indicating_variables = Eq.rho_p_indicator!,
                                 reconstruction_variables = conservative_reconstruction,
                                 indicator_model = indicator_model,
                                 debug_blend = debug_blend,
                                 pure_fv = pure_fv)
tvbM = 300;
limiter_tvb = setup_limiter_tvb(equation; tvbM = tvbM)

scheme = setproperties(scheme,
                       (degree = degree, limiter = limiter_mh,
                        numerical_flux = Eq.roe))

sol1 = Tenkai.solve(equation, problem, scheme, param);
p1 = sol1["plot_data"].p_ua

scheme = setproperties(scheme,
                       (degree = degree, limiter = limiter_tvb,
                        numerical_flux = Eq.roe))
sol2 = Tenkai.solve(equation, problem, scheme, param);
p2 = sol2["plot_data"].p_ua

leg1 = "MH"
leg2 = "TVB"
title = ""
# p_ua_compared = join_plots_scalar(deepcopy(p1), deepcopy(p2), leg1, leg2, title)
p_ua_compared = join_plots(deepcopy(p1), deepcopy(p2), leg1, leg2, title, 3)

p = plot(p_ua_compared, legend = true)
empty!(ARGS)
savefig(p, "compared.html")
