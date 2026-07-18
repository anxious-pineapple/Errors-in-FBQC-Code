using Plots
using Distributions 

d = Exponential(1)
d = Normal(0, 1)
d = Uniform(0, 1)
td = truncated(d, 0.0, Inf)
n_series = rand(td, 10)'
n_series = n_series ./ sum(n_series)
V_exact = sum(n_series.^2)

F_exact = (1 + V_exact + V_exact^2 + sum(n_series.^4)) / (6 - 2V_exact)

F_up(V) = (1 + V + 2V^2)/(6-2V)
F_down(V) =  (1 + 4V + 3V^2)/(12-4V)


V = 0.12:0.01:0.21
plot!(V, F_up.(V), label="Upper bound", xlabel="V", ylabel="Fidelity", title="Bell State Fidelity", lw=0.5)
plot!(V, F_down.(V), label="Lower bound", lw=0.5)
plot!([V_exact], [F_exact], seriestype="scatter", label="Exact value", markersize=1, color="black")