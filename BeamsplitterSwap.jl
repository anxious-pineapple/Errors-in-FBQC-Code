using .MPOFuncs, ITensors, Integrals, BenchmarkTools, Plots, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, Metal

no_cavs = 7
chain_sites = siteinds("Qudit", no_cavs+1, dim=3)
chain = MPO(chain_sites, ["Excite1","Ground", "Excite1", "Ground", "ExciteMax","Excite1","Ground", "ExciteMax"])
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
println(double_sites)
#swap like modes next to each other 
@time for i=no_cavs+1:-1:2
    println(i)
    swap_ij!(double_mpo, double_sites, i, 2*i-1)
    MPOFuncs.beamsplitter_nextsite!(double_mpo, double_sites, 2*i-1)
    println(double_sites)
    println(dims.(double_mpo))
end

#need to compare the old and new beamsplitter

@show (dims.(double_mpo))
@show (dim.(chain))
@show ITensors.expect(double_mpo, double_sites)
@time MPOFuncs.beamsplitter_nextsite!(chain, chain_sites, 1)
@show ITensors.expect(chain, chain_sites)