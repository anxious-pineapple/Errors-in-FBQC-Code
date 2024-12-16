#
    # Code generates the graph of relative error in distinguishability vs dephasing rate for 2 to max_cav cavities
    # New graph is generated and saved in /Plots 
    # Inputs : max_cavs rest is default
    # Outputs : RelativeAnalyticalErrorPerCavs.pdf
    # can also generate heatmaps of relative error, relevant to find no of cavities needed for
    # say less than 5% relative error
    # can also generate heatmap/contour graphs of actual error

#

using Plots
import MPOFuncs

gamma = 1
dt = 0.01
t_final = 10
dep_array = [0:0.01:0.50;]
max_cav = 21
data_array = Array{Float64}(undef, size(dep_array)[1], max_cav-1)
err_array = Array{Float64}(undef, size(dep_array)[1], max_cav-1)

for dep in dep_array
    gf, eival = MPOFuncs.g2(gamma, dep, t_final, dt, 2; reverse=false)
    eival = eival/sum(eival)
    # plot!(eival)
    ideal = 0.5*(1 - sum(eival.^2))
    for no_cavs in 2:max_cav
        err  = ideal - (0.5*(1 - (sum(eival[1:no_cavs].^2)/(sum(eival[1:no_cavs])^2))))
        relerr = 100 * err/ideal
        data_array[ findall(x->x==dep,dep_array)[1] , no_cavs-1] = relerr
        err_array[ findall(x->x==dep,dep_array)[1] , no_cavs-1] = err
    end
end

# deph_array[1,:] ignored below, as for very low dephasing, the entire wave function is absorbed 
# in the first cavity, and the error is dominated by any photon number lost in the simulation 
# Giving rise to very high relative error, 175%+, but this is again just cause the ideal g_34 is very small itself

###
    p = plot(dep_array[2:end], abs.((data_array))[2:end,1:end])
    plot!(xlabel="Dephasing Rate", ylabel="Relative Error (in g34)", legend = :outertopright )
    Plots.savefig(p, "Plots/RelativeAnalyticalErrorPerCavs.pdf")
#

p2 = plot(heatmap(dep_array[2:end], [2:max_cav;], abs.(transpose(data_array))[:,2:end] .<5))
plot!(xlabel="Dephasing Rate", ylabel="Number of Cavities" , yticks=[2:2:max_cav;])
# Plots.savefig(p2, "Plots/DephasingErrorHeatmapLessthan5.pdf")

p3 = plot(heatmap(dep_array[1:end], [2:max_cav;], abs.(transpose(err_array))[:,1:end]))
plot!(xlabel="Dephasing Rate", ylabel="Number of Cavities" , yticks=[2:2:max_cav;])
# Plots.savefig(p2, "Plots/DephasingErrorHeatmapLessthan5.pdf")

## alt graph for error
# contour(dep_array[1:end], [2:max_cav;], abs.(transpose(err_array))[:,1:end], clabels=true)