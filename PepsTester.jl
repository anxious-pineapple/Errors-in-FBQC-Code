using ITensors
import MPOFuncs
using Plots
using ITensorsVisualization
no_cavs = 4


site_1 = siteinds("Qudit", no_cavs, dim=3)
mpo_1 = MPO(site_1, repeat(["Ground",],no_cavs))
site_2 = siteinds("Qudit", no_cavs, dim=4)
mpo_2 = MPO(site_2, repeat(["Ground",],no_cavs))

peps = Array{ITensor}(undef, 2, no_cavs)
# peps[1,1] = mpo_1[1]
# peps

for i=1:no_cavs
    #now to multiply then svd then store 
    mult_term = mpo_1[i] * mpo_2[i]
    inds3 = inds(mpo_1[i])
    U,S,V = svd( mult_term, inds3, cutoff=1E-8)
    peps[1,i] = U
    peps[2,i] = S*V
end

@visualise  mpo_1




# bunch of nonsense comments
#right back at you