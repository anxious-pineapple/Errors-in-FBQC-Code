#
# In this code im making a bell state in dual rail encoding
# However since all the cavities are in the same MPO, this becomes increasingly data intensive 
# with higher cavity numbers (very high entangling across the MPO chain)
# Switched to a peps between the signal and ancilla 'sites' done in a subsequent code.
#


using ITensors
import MPOFuncs
using Plots

gamma = 1.0
no_cavs = 2
dt = 0.01
t_final = 10.0
dep = 0.002


mpo_i, sites_i, eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)
# signal_mpo, signal_sites = MPOFuncs.n_copy_mpo(mpo_i, sites_i, 4)

ancilla_site_i = siteinds("Qudit", no_cavs, dim=3)
input_list = repeat(["Ground",],no_cavs)
input_list[1] = "Excite1"
ancilla_i = MPO(ancilla_site_i, repeat(["Ground",],no_cavs))
# MPOFuncs.beamsplitter_nextsite!(ancilla_i, ancilla_site_i, 4)
# plot(ITensors.expect(ancilla_i, ancilla_site_i); seriestype=:"scatter")
# ancilla_i = MPOFuncs.swap_ij!(ancilla_i, ancilla_site_i, 1, 5)


# ancilla, ancilla_sites = MPOFuncs.n_copy_mpo(ancilla_i, ancilla_site_i, 4)

sys, sys_sites = MPOFuncs.join(mpo_i, sites_i, ancilla_i, ancilla_site_i)
plot(ITensors.expect(sys, sys_sites); seriestype=:"scatter")

# swap the ancillas in the mpo chain
for i=no_cavs:-1:1
    sys = MPOFuncs.swap_ij!(sys, sys_sites, i, 2i-1 )
end

plot(ITensors.expect(sys, sys_sites)[1:end]; seriestype=:"scatter")

sys_full, sys_sites_full = MPOFuncs.n_copy_mpo(sys, sys_sites, 4)
plot(real.(ITensors.expect(sys_full, sys_sites_full)); seriestype=:"scatter")
plot(ITensors.linkdims(sys_full); seriestype=:"scatter")

#apply beamsplitters
for i=1:4
    for j=1:no_cavs
        sys_full = MPOFuncs.beamsplitter_nextsite!(sys_full, sys_sites_full, (2*no_cavs*(i-1))+(2j-1))
    end
end
plot(real.(ITensors.expect(sys_full, sys_sites_full)); seriestype=:"scatter")
ITensors.truncate!(sys_full; cutoff= 1e-10)
plot(ITensors.linkdims(sys_full); seriestype=:"scatter")

# swap ancillas back ?
# confused on best possible way to do this 


for i=1:4
    for j=1:no_cavs
        println(i,",",j)
        sys_full = MPOFuncs.swap_ij!(sys_full, sys_sites_full, (no_cavs*(i-1))+ j +1 , (no_cavs*8) )
        # println( (2*no_cavs*(i-1))+2j,",", (no_cavs*(3+i)) +j )
    end
end

# takes like 10 minutes to run the swap

# Check the way you have constructed the beamsplitter 
# other thing we can do, 2d mpos and not linear 
# 3rd thing back propogate throught the last really bad beamsplitter and project




# complicated beamsplitter 
for j=1:no_cavs
    sys_full = MPOFuncs.beamsplitter!(sys_full, sys_sites_full, 4no_cavs+j, 5no_cavs + j)
    sys_full = MPOFuncs.beamsplitter!(sys_full, sys_sites_full, (6no_cavs) + j, (7no_cavs) + j)
end
plot(real.(ITensors.expect(sys_full, sys_sites_full)); seriestype=:"scatter")
plot(ITensors.linkdims(sys_full))

for j=1:no_cavs
    @show (j)
    sys_full = MPOFuncs.beamsplitter!(sys_full, sys_sites_full, 4no_cavs+j, 7no_cavs + j)
    @show ("mid")
    sys_full = MPOFuncs.beamsplitter!(sys_full, sys_sites_full, 5no_cavs+j, 6no_cavs + j)
    plot(ITensors.linkdims(sys_full))
end
plot(real.(ITensors.expect(sys_full, sys_sites_full)); seriestype=:"scatter")
plot(ITensors.linkdims(sys_full))
# measure out |1100> state


p=plot(rand(30); label="test", legendfontsize=10)
length(signal_mpo)