using QuantumOptics
using Plots

plot(rand(10), fmt = :png)
n = 0.9 # Pumping strength
κ = 1 # Decay rate

Ncutoff = 20 # Maximum photon number
T = [0:0.1:10;]
basis = FockBasis(20)
a = destroy(basis)
at = create(basis)
n = number(basis)
H = η*(a+at)
J = [sqrt(κ)*a]
Ψ₀ = fockstate(basis, 10)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)
tout, ρt_master = timeevolution.master(T, ρ₀, H, J)

plot(T, real(expect(n, ρt_master)) , fmt = :png)