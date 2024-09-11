using ITensors
using .MPOFuncs
using Plots
using BenchmarkTools

gamma = 1.0
no_cavs = 4
dt = 0.01
t_final = 10.0
dep = 0.002


function peps_expect(peps_array, sites_i)
 # here assuming list, to write for whole array
    list_exp = []
    for i in eachindex(sites_i)
        temp_list = deepcopy(peps_array)
        @show MPOFuncs.trace(temp_list, sites_i)
        temp_list[i] = op("n", sites_i[i])' * temp_list[i]
        temp_list[i] = replaceprime(temp_list[i], 2=>1)
        append!(list_exp, MPOFuncs.trace(temp_list, sites_i))
        @show MPOFuncs.trace(temp_list, sites_i)
    end
    return list_exp

end

MPOFuncs.trace(peps[1,:], sites_i)
plot(peps_expect(peps[1,:], sites_i); seriestype=:"scatter")
plot(peps_expect(mpo_i, sites_i); seriestype=:"scatter")

mpo_i, sites_i, eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)

sites_i = ITensors.siteinds("Qudit", no_cavs; dim=4)
input_list = repeat(["Ground",],no_cavs)
input_list[1] = "Excite1"
# input_list[2] = "Excite1"
mpo_i = MPO(sites_i, input_list)
plot(ITensors.expect(mpo_i, sites_i); seriestype=:"scatter")

ancilla_sites = siteinds("Qudit", no_cavs, dim=3)
ancilla_mpo = MPO(ancilla_sites, repeat(["Ground",],no_cavs))
plot!(ITensors.expect(ancilla_mpo, ancilla_sites); seriestype=:"scatter")

mpo_i[1]
replaceprime!(mpo_i[1], 1=>3)

peps = Array{ITensor}(undef, 2, no_cavs)
# do i need a peps sites array as well?
for i=1:no_cavs
    peps[1, i] = mpo_i[i]
    peps[2, i] = ancilla_mpo[i]
end

peps[1,2] , peps[2,2] = MPOFuncs.beamsplitter_peps_svd(peps[1,2] , peps[2,2], sites_i[2], ancilla_sites[2])

for i=1:no_cavs
    mult_placeholder = peps[2,i] * delta(ancilla_sites[i], ancilla_sites[i]')
    peps[1,i] *= mult_placeholder
end 

for i=1:no_cavs
    if i!=no_cavs
        ind4 = commoninds(peps[1,i], peps[1,i+1])
        peps[1,i] *= combiner(ind4)
        peps[1,i+1] *= combiner(ind4)
    end
end

peps[1,2]

plot(ITensors.expect(peps[1,:], sites_i); seriestype=:"scatter")

cutoff = 1E-10

index_1 = 2

site_list = sites_i
op_1 = ((op("A",site_list[index_1]) * op("Adag",site_list[index_1+1])) + (op("A",site_list[index_1+1]) * op("Adag",site_list[index_1])))
H_ = exp((-im/4) * pi * op_1)

test_mpo = MPO(H_, [site_list[index_1],site_list[index_1+1]])
bs1, bs2 = qr(H_, (site_list[index_1],site_list[index_1]'))
bs1
bs1' * mpo_i[2]
#easier to use apply since preserves MPO data type and doesnt change to iTensor type
#requires iTensor be applied to an MPO in that order particularly hence double dagger
H3 = conj(swapprime( apply( H_, swapprime(conj(H2), 1,0); cutoff ), 1,0))

MPO_i[:] = H3
# MPO_i /= trace(MPO_i, site_list)
# return nothing







@btime mpo_i = MPOFuncs.beamsplitter!($mpo_i, $sites_i, 2,3)
mpo_i = MPO(sites_i, input_list)
@btime mpo_i = MPOFuncs.beamsplitter_nextsite!($mpo_i, $sites_i, 2)
mpo_i = MPO(sites_i, input_list)
# @btime mpo_i = MPOFuncs.beamsplitter_nextsite_svded!($mpo_i, $sites_i, 2)

signal_mpo, signal_sites = MPOFuncs.n_copy_mpo(mpo_i, sites_i, 4)
plot(ITensors.expect(signal_mpo, signal_sites); seriestype=:"scatter")
for i=4:-1:1
    for j=no_cavs:-1:1
        signal_mpo = MPOFuncs.swap_ij!(signal_mpo, signal_sites, (no_cavs*(i-1))+j, (no_cavs*(i-1))+j + ((4-i)*(j-1)))
    end
end
@show signal_sites

