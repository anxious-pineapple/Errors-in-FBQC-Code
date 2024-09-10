using ITensors, Integrals, BenchmarkTools, Plots, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, Metal
using Dates
import MPOFuncs
using JLD2
using CurveFit

gamma = 1
no_cavs = 3
dt = 0.01
t_final = 10
dep = 0.02

# if can get global list to work for modules and initialising them etc, then evolve_test2 might be faster

# 1, try lowerin dt and see if graph gets better , also compare with % absorbed, (more absorbed = better distingushability graph?)
# 2, try lowerin tolerance 
# 3, try lowerin g2's time scale 

# we will conclude which has the bigger effect

# Secondly it would be good to show this with something numerical to back it up

#



# g_34_list = [;]
g_34 = [;]
absorbed= [;]
# we want to test if  inscreasing absorbed photon by lowering tolerandce/ dt if error decreases
# gvf, eigenvals = MPOFuncs.g2(gamma, dep, t_final, dt, no_cavs; reverse=false) 

# @btime 
mpo2, sites2, eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt*2, t_final)



@show sum(eigenvals[1:30]./sum(eigenvals))
push!(absorbed, sum(eigenvals[1:no_cavs]./sum(eigenvals)))

@show  sum(ITensors.expect(mpo2, sites2))
@show absorbed

double2, double_sites2 = MPOFuncs.n_copy_mpo(mpo2, sites2, 2)
for i=no_cavs:-1:1
    print(i, " , ")
    double2 = MPOFuncs.swap_ij!(double2, double_sites2, i, 2*i-1)
    MPOFuncs.beamsplitter_nextsite!(double2, double_sites2, 2*i-1)
end
push!(g_34, MPOFuncs.g_34_new(double2, double_sites2, no_cavs))
push!(absorbed, sum(ITensors.expect(mpo2, sites2)))
# @show sum(ITensors.expect(mpo2, sites2))

ideal_valg34 = 2*dep/(1+4*dep)

@show g_34[1]-ideal_valg34


@show (100*(g_34[1]-ideal_valg34)/ideal_valg34)
@show absorbed[2]

plot(100 .*(real(g_34).-ideal_valg34)./ideal_valg34, label="g34 error")
plot!((absorbed[2:end].-absorbed[1]), label="Absorbed error")


# ITensors.orthogonalize!(mpo2, 1; cutoff = 1e-10)





#can make a 3d plot, dephasing, no_cavs and relative error that i should ideally get


no_cavs = 3
dep = 0.05
gf, eival = MPOFuncs.g2(gamma, dep, t_final, dt, no_cavs; reverse=false)


no_cavs = 15
eival = eival/sum(eival)
ideal = 1 - sum(eival.^2)
err  = sum(eival.^2) - (sum(eival[1:no_cavs].^2)/(sum(eival[1:no_cavs])^2))
relerr = 100 * err/ideal