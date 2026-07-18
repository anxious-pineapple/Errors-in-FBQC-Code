###
# Code runs the time evolution output of a three level atom and correlation functions
# See Tiurev et al., “Fidelity of Time-Bin-Entangled Multiphoton States from a Quantum Emitter.”
# For protocol details on order of flips
# Disclaimer the pi-pulse and stregth are probably not accurate/well implemented
# But this was just a test to see if first and second photon outputted have any corellation
# They do not!
###

using QuantumOptics
using Plots
using LinearAlgebra

# 3 is the excited state, 2 is the ground state, 1 is the non-excitable other ground state
b = NLevelBasis(3)
ψ₀ = (nlevelstate(b, 2) + nlevelstate(b, 1))/√2

# Define the operators
c = transition(b, 2, 3)
cdag = transition(b, 3, 2)
sx = transition(b, 2, 3) + transition(b, 3, 2)
flip_X = transition(b, 2, 1) + transition(b, 1, 2)
proj_ground = transition(b, 2, 2)
proj_excite = transition(b, 3, 3)
proj_1 = transition(b, 1, 1)
const J = [transition(b, 2, 3)]
const Jdagger = [transition(b, 3, 2)]

g0 = 5
h0 = 5
flip_t = π/2h0           # Width of ground state flip pulse
width = √π/g0            # Width of the gaussian pulse (if used)

tmax = 5
dt = 0.1
tspan = [0:dt:2tmax;]


### Function Block
    function calc_pops(t, ρ)
        p1 = real(expect(proj_ground, ρ))
        p2 = real(expect(proj_excite, ρ))
        p0 = real(expect(proj_1, ρ))
        return p1, p2, p0, ρ
    end

    function h(t)
        if t < tmax && t > tmax-flip_t
            return h0
        else
            return 0
        end
    end

    function g(t)
        t > tmax ? t1 = t - tmax : t1 = t
        if t1 <= π/(2*g0)
            return(g0)
            # return(0)
        else
            return(0)
        end
        # return g0 * exp(-(t1 -5dt - width/2)^2/(width^2))
    end

    ##  Plot the pulse shapes and the durations
    # plot(g.(tspan))
    # plot!(h.(tspan))

    function f(t,rho)
        H = g(t)*sx + h(t)*flip_X + identityoperator(b)
        return H, J, Jdagger
    end
###


tout, ρₜ = timeevolution.master_dynamic(tspan, ψ₀, f ;fout=calc_pops)

#
    p = plot([p[1] for p in ρₜ])
    plot!([p[2] for p in ρₜ])
    plot!([p[3] for p in ρₜ])
    display(p)
#

corel_g = zeros(length(tspan), length(tspan))

# Constructing the correlation matrix since inbuilt function cant handle a time dependent Hamiltonian
for i in eachindex(tspan)
    # println(i)
    j=i
    corel_g[i,j] = real(tr(cdag * c * ρₜ[i][end]))
    for j in i+1:length(tspan)

        rho_in = c*ρₜ[i][end]
        tout, rho_end = timeevolution.master_dynamic([tspan[i]:dt:tspan[j];], rho_in, f ;fout=calc_pops)
        corel_g[i,j] = real(tr(cdag * rho_end[end][end]))
    end
end
        

corel_g = transpose(corel_g) + corel_g - Diagonal(corel_g)
corel_g = corel_g ./ corel_g[1,1]
heatmap(corel_g)