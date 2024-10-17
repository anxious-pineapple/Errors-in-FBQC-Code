import ITensors
using .MPOFuncs
using BenchmarkTools


gamma = 1
no_cavs = 2
dt = 0.01
t_final = 10
dep = 0.002

# spzeros(10,10)

mpo_i, sites_i, eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)
# signal_mpo, signal_sites = MPOFuncs.n_copy_mpo(mpo_i, sites_i, 4)

sites_i

ancilla_site_i = siteinds("Qudit", no_cavs, dim=3)
list_1 = repeat(["Ground",],no_cavs)
list_1[1] = "Excite1"
list_1[end] = "Excite1"
ancilla_i = MPO(ancilla_site_i, list_1)

@show issparse(op("Ground", ancilla_site_i[1]))




@btime ancilla_i = beamsplitter!($ancilla_i, $ancilla_site_i, 1,10)

# @btime ancilla_i = 
MPOFuncs.beamsplitter!(ancilla_i, ancilla_site_i, 1,10)


#it looks like for far ranged beamsplitters time evolution is the way

