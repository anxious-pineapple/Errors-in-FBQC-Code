using ITensors, Integrals, BenchmarkTools, Plots, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, Metal
using Dates
import MPOFuncs
using JLD2

gamma = 1
no_cavs = 4
dt = 0.01
t_final = 10
dep = 0.02

@btime mpo, sites = MPOFuncs.cash_karppe_evolve(no_cavs, dep, gamma, dt, t_final)
@btime mpo2, sites2 = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)


@show ITensors.expect(mpo, sites)


@show ITensors.expect(mpo2, sites2)


MPOFuncs.test_list
MPOFuncs.init3()