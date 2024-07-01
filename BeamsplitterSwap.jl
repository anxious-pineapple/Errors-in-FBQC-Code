using MPOFuncs, ITensors, Integrals, BenchmarkTools, Plots, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, Metal

no_cavs = 9
chain_sites = siteinds("Qudit", no_cavs+1, dim=3)
chain = MPO(chain_sites, ["Excite1","Ground", "Excite1", "Ground", "ExciteMax","Excite1","Ground", "Excite1", "Ground", "ExciteMax"])
double_mpo, double_sites = MPOFuncs.n_copy_mpo(chain, chain_sites, 2)

@show ITensors.expect(chain, chain_sites)
@show ITensors.expect(double_mpo, double_sites)

# swap_ij!(double_mpo, double_sites, 5, 10)
# @show ITensors.expect(double_mpo, double_sites)
# swap_ij!(double_mpo, double_sites, 4, 8)
# @show ITensors.expect(double_mpo, double_sites)
# swap_ij!(double_mpo, double_sites, 3, 6)
# @show ITensors.expect(double_mpo, double_sites)
# swap_ij!(double_mpo, double_sites, 2, 4)
# @show ITensors.expect(double_mpo, double_sites)

#swap like modes next to each other 
for i=no_cavs+1:-1:2
    println(i)
    swap_ij!(double_mpo, double_sites, i, 2*i-1)
    MPOFuncs.beamsplitter_nextsite!(double_mpo, double_sites, 2*i-1)
    println(dims.(double_mpo))
end

#need to compare the old and new beamsplitter

@show (dims.(double_mpo))
@show ITensors.expect(double_mpo, double_sites)
@time MPOFuncs.beamsplitter_nextsite!(chain, chain_sites, 1)
@show ITensors.expect(chain, chain_sites)


# function swap_nextsite!(mpo_i, sites, i)
#     #Function swaps the i-th and i+1-th sites in the mpo
#     mpo_contrac = mpo_i[i] * mpo_i[i+1]
#     beg_ind = uniqueinds(mpo_i[i+1],mpo_i[i])
#     length(mpo_i) < i+2 ? nothing : deleteat!(beg_ind, findall(x->x==commonind(mpo_i[i+1], mpo_i[i+2]),beg_ind))
#     i == 1 ? nothing : push!(beg_ind, commonind(mpo_i[i], mpo_i[i-1]))
#     U,S,V = svd(mpo_contrac, beg_ind)
#     mpo_i[i] = U
#     mpo_i[i+1] = S * V

#     place_holder = sites[i]
#     sites[i] = sites[i+1]
#     sites[i+1] = place_holder
#     # return mpo_i, sites
# end

# function swap_ij!(mpo_i, sites, i, j)
# # Assuming j > i and both in length
#     for k=i:j-1
#         swap_nextsite!(mpo_i, sites, k)
#     end
# end
