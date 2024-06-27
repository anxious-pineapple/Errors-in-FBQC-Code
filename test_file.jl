using MPOFuncs, ITensors, Integrals, BenchmarkTools, Plots, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, Metal, JLD2

# g34_list10 =  [;]
gamma = 1
no_cavs = 2
dt = 0.02
t_final = 10
dep=0.0125
println(dep)
@time mpo, sites = MPOFuncs.cash_karppe_evolve(no_cavs, dep, gamma, dt, t_final)
println(MPOFuncs.trace(mpo,sites))
