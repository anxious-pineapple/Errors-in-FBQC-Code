#
    # Code generates the graph of distinguishability vs dephasing rate for a given number of cavities
    # New graph is generated and saving in /Plots for each number of cavities
    # Inputs : no_cavs rest is default
    # if compare_cavs true generates plot comparing cavs as well
#

using ITensors
using .MPOFuncs
using Plots

gamma = 1
no_cavs_list = [10, 12, 15]
dep_list = [0:0.0025:0.02;]
dt = 0.01
t_final = 10

compare_cavs = true
g34_all = Dict()
num_ineal_all = Dict()

for no_cavs in no_cavs_list 

    g34 = [;]
    num_ideal_list = [;]

    for dep in dep_list
        println(dep)
        # ti = now()
        mpo, sites = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)
        double2, double_sites2 = MPOFuncs.n_copy_mpo(mpo, sites, 2)
        for i=no_cavs:-1:1
            print(i, " , ")
            double2 = MPOFuncs.swap_ij!(double2, double_sites2, i, 2*i-1)
            MPOFuncs.beamsplitter_nextsite!(double2, double_sites2, 2*i-1)
        end
        push!(g34, MPOFuncs.g_34_new(double2, double_sites2, no_cavs))
        # println("This round took ", now()-ti)

        gf, eival = MPOFuncs.g2(gamma, dep, t_final, dt, 2; reverse=false)
        eival = eival/sum(eival)
        push!(num_ideal_list, 0.5 * ( 1 - (sum(eival[1:no_cavs].^2)/(sum(eival[1:no_cavs])^2))) )

    end

    # println("This took total ", now()-t)

    abs_ideal_list = [2*i/(1+4*i) for i in dep_list]

    p1 = plot( dep_list , real.(g34), seriestype="scatter", label="Numerical")
    plot!( dep_list , abs_ideal_list, label="Abs Ideal")
    plot!( dep_list , num_ideal_list, label="Num Ideal")
    plot!(xlabel="Dephasing Rate", ylabel="Distinguishability")
    Plots.savefig(p1, "Plots/DistinguishabilityVsCav_" * string(no_cavs) * "cavs.pdf")

    p2 = plot( dep_list , 100*abs.(real.(g34) .- abs_ideal_list)./abs_ideal_list, seriestype="scatter", label="Error (%)")
    plot!( dep_list , 100*abs.(real.(num_ideal_list) .- abs_ideal_list)./abs_ideal_list, label="Ideal Error (%)")
    plot!(xlabel="Dephasing Rate", ylabel="Relative Error")
    Plots.savefig(p2, "Plots/DistinguishabilityVsCav_RelativeErr_" * string(no_cavs) * ".pdf")

    if compare_cavs
        g34_all[no_cavs] = g34
        num_ineal_all[no_cavs] = num_ideal_list
    end
end


if compare_cavs
    abs_ideal_list = [2*i/(1+4*i) for i in dep_list]
    p3 = plot(dep_list, abs_ideal_list, label = "Abs Ideal")
    for no_cavs in no_cavs_list
        plot!(dep_list, real.(g34_all[no_cavs]), label = string(no_cavs) * "cavs")
    end
    plot!(xlabel="Dephasing Rate", ylabel="Distinguishability")
    Plots.savefig(p3, "Plots/DistinguishabilityVsCav_AllCavs.pdf")

    p4 = plot()
    for no_cavs in no_cavs_list
        plot!(dep_list, 100*abs.(g34_all[no_cavs] .- abs_ideal_list)./abs_ideal_list, label = string(no_cavs) * "cavs")
    end
    plot!(xlabel="Dephasing Rate", ylabel="Relative Error")
    Plots.savefig(p4, "Plots/DistinguishabilityVsCav_RelativeErr_AllCavs.pdf")
    # num_ineal_all[no_cavs] = num_ideal_list
end