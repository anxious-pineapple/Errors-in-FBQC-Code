using ITensors, Integrals, BenchmarkTools, Plots, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, Metal
using Dates
import .MPOFuncs
using JLD2

gamma = 1
no_cavs = 12
dt = 0.01
t_final = 10

g34_12 = [;]
t = now()


for dep in 0:0.0025:0.02
    println(dep)
    ti = now()
    mpo, sites = MPOFuncs.cash_karppe_evolve(no_cavs, dep, gamma, dt, t_final)
    double2, double_sites2 = MPOFuncs.n_copy_mpo(mpo, sites, 2)
    for i=no_cavs:-1:1
        print(i, " , ")
        double2 = MPOFuncs.swap_ij!(double2, double_sites2, i, 2*i-1)
        MPOFuncs.beamsplitter_nextsite!(double2, double_sites2, 2*i-1)
    end
    push!(g34_12, MPOFuncs.g_34_new(double2, double_sites2, no_cavs))
    println("This round took ", now()-ti)
end

println("This took total ", now()-t)

ideal_list = [2*i/(1+4*i) for i=0:0.0025:0.02]
plot( [0:0.0025:0.02;] , real.(g34_12))
plot!( [0:0.0025:0.02;] , ideal_list, label="")
plot!( [0:0.0025:0.02;] , ideal_list, seriestype="scatter", label="Ideal")

plot( [0:0.0025:0.02;] , 100*abs.(real.(g34_12) .- ideal_list)./ideal_list)
plot!(ylimits=(5,10.5))

jldsave("Data/g34_12_ComparingBeamsplitter.jld2"; g34_12)






println("Now for 15 cavities")
no_cavs = 15
g34_15 = [;]
t = now()


for dep in 0:0.0025:0.02
    println(dep)
    ti = now()
    mpo, sites = MPOFuncs.cash_karppe_evolve(no_cavs, dep, gamma, dt, t_final)
    double2, double_sites2 = MPOFuncs.n_copy_mpo(mpo, sites, 2)
    for i=no_cavs:-1:1
        print(i, " , ")
        double2 = MPOFuncs.swap_ij!(double2, double_sites2, i, 2*i-1)
        MPOFuncs.beamsplitter_nextsite!(double2, double_sites2, 2*i-1)
    end
    push!(g34_15, MPOFuncs.g_34_new(double2, double_sites2, no_cavs))
    println("This round took ", now()-ti)
end

println("This took total ", now()-t)

ideal_list = [2*i/(1+4*i) for i=0:0.0025:0.02]
plot( [0:0.0025:0.02;] , real.(g34_15))
plot!( [0:0.0025:0.02;] , ideal_list, label="")
plot!( [0:0.0025:0.02;] , ideal_list, seriestype="scatter", label="Ideal")

plot( [0:0.0025:0.02;] , 100*abs.(real.(g34_15) .- ideal_list)./ideal_list)
plot!(ylimits=(5,10.5))

jldsave("Data/g34_15_ComparingBeamsplitter.jld2"; g34_15)


f = jldopen("Data/g34_10_ComparingBeamsplitter.jld2", "r")
g34_10 = f["g34_10"]
close(f)

p = plot([0:0.0025:0.02;] , real.(g34_10), label="10 cav")
plot!([0:0.0025:0.02;] , real.(g34_12), label="12 cav")
plot!([0:0.0025:0.02;][1:length(g34_15)] , real.(g34_15),  label="15 cav")
plot!([0:0.0025:0.02;] , ideal_list, label="ideal")
plot!(xlabel="Dephasing", ylabel="g34")
savefig(p, "Plots/10_plus_cavs.pdf")


p2 = plot([0:0.0025:0.02;] , 100*abs.(real.(g34_10) .- ideal_list)./ideal_list, label="10 cav")
plot!([0:0.0025:0.02;] , 100*abs.(real.(g34_12) .- ideal_list)./ideal_list, label="12 cav")
plot!([0:0.0025:0.02;][1:length(g34_15)] , 100*abs.(real.(g34_15) .- ideal_list[1:length(g34_15)])./ideal_list[1:length(g34_15)],  label="15 cav")
plot!(ylims=(5,10.5))
plot!(xlabel="Dephasing", ylabel="Error (%)", title="Relative error")
savefig(p2, "Plots/Relative_10_plus_cavs.pdf")



dep = 0.01
no_cavs = 10
t = now()
mpo, sites = MPOFuncs.cash_karppe_evolve(no_cavs, dep, gamma, dt, t_final)
println("This took ", now()-t)


println(sum(ITensors.expect(mpo, sites)), " capture for tolerance 1e-4")