## This code is to test out TEBD for MPOs having cascaded master equations



using ITensors
import MPOFuncs

no_cavs = 1

sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
input_list = repeat(["Ground",],no_cavs+1)
input_list[1] = "Excite1"
sys = MPS(sys_sites, input_list)

sys = MPS(sys_sites, [2,1]) 

# ============ Add parts of Lindbladian ============
# first without swap gates it slightly long term operators
gates = ITensor[]
for j in 1:(N - 1)
s1 = sys_sites[j]
s2 = sys_sites[j + 1]
hj =
    op("Sz", s1) * op("Sz", s2) +
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2)
Gj = exp(-im * tau / 2 * hj)
push!(gates, Gj)
end


# ============ evolve with gates ============



# ============ measure ============




@show sys

ground_state = MPS()