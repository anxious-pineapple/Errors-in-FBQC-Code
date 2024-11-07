using Plots
import MPOFuncs

gamma = 1
dt = 0.01
t_final = 10
dep_array = [0:0.001:0.03;]
max_cav = 21
data_array = Array{Float64}(undef, size(dep_array)[1], max_cav-1)


for dep in dep_array
    gf, eival = MPOFuncs.g2(gamma, dep, t_final, dt, 2; reverse=false)
    eival = eival/sum(eival)
    # plot!(eival)
    ideal = 0.5*(1 - sum(eival.^2))
    for no_cavs in 2:max_cav
        err  = ideal - (0.5*(1 - (sum(eival[1:no_cavs].^2)/(sum(eival[1:no_cavs])^2))))
        relerr = 100 * err/ideal
        data_array[ findall(x->x==dep,dep_array)[1] , no_cavs-1] = relerr
    end
end



p = plot(heatmap(abs.(data_array)))
Plots.savefig(p, "Plots/DephasingErrorHeatmap.pdf")
p2 = plot(heatmap(abs.(data_array).<5))
Plots.savefig(p2, "Plots/DephasingErrorHeatmapLessthan5.pdf")