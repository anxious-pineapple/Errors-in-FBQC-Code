###
# Code tests out creation of gvi functions based on paper https://arxiv.org/abs/1902.09833
# Returns eigenvalues and various gvi functions
###

using QuantumOptics, LinearAlgebra, Interpolations, Plots

gamma = 1 
deph = 0.1
t_fin = 20
dt = 0.01
no_cavs = 10
Δ = 0.0     # emitter detuning from the rotating frame


basis = SpinBasis(1//2)
a = sigmam(basis)
at = sigmap(basis)
H = Δ * at * a
# deph is the pure dephasing rate γ* of the coherence ρ_eg (total coherence decay γ/2 + deph).
# D[√κ σz] damps ρ_eg at 2κ since σz² = I, so κ = deph/2. Populations are unaffected.
J = [sqrt(gamma)*a, sqrt(deph/2)*sigmaz(basis)]

t_list = [0:dt:t_fin;]
t_size = length(t_list)
corel_m = zeros(ComplexF64, t_size, t_size)

ρ₀ = spinup(basis) ⊗ dagger(spinup(basis))

tout, ρt_master = timeevolution.master(t_list, ρ₀, H, J; dt=dt)
for i=1:t_size-1
    rhot = ρt_master[i]
    corel_m[i,i:end] = timecorrelations.correlation(t_list[1:(t_size-i+1)], rhot, H, J, at, a)
end
# the loop never reaches the last diagonal entry: it is ⟨σ₊σ₋⟩(t_fin), the population left in the emitter
# (setting it to 0 made the matrix non-positive, giving one spurious negative eigenvalue ~ -population·dt/3)
corel_m[t_size,t_size] = real(expect(at*a, ρt_master[end]))
# Row i holds ⟨σ₊(tⱼ)σ₋(tᵢ)⟩ for j ≥ i. Since ⟨σ₊(s)σ₋(t)⟩* = ⟨σ₊(t)σ₋(s)⟩ the lower triangle is the adjoint
corel_m = corel_m + corel_m' - Diagonal(diag(corel_m))

# 4th-order cumulative integral on the grid (exact for cubics, one-sided at the ends)
function cumul_int(y)
    n = length(y); I = zeros(eltype(y), n)
    for k in 1:n-1
        s = k == 1   ? 9y[1] + 19y[2] - 5y[3] + y[4] :
            k == n-1 ? 9y[n] + 19y[n-1] - 5y[n-2] + y[n-3] :
                       -y[k-1] + 13y[k] + 13y[k+1] - y[k+2]
        I[k+1] = I[k] + s * dt / 24
    end
    return I
end

# Weights w with Σₖ wₖ yₖ = cumul_int(y)[end], i.e. the same quadrature rule applied to the whole interval
function quad_weights(n, dt)
    w = zeros(n)
    for k in 1:n-1
        idx, c = k == 1   ? (1:4, (9, 19, -5, 1)) :
                 k == n-1 ? ((n:-1:n-3), (9, 19, -5, 1)) :
                            (k-1:k+2, (-1, 13, 13, -1))
        w[idx] .+= c .* dt ./ 24
    end
    return w
end

# The modes solve the integral eigenproblem ∫ C(t,t') v(t') dt' = n v(t) with ∫|v|²dt = 1.
# Discretising with quadrature weights w gives Σₖ C(t,tₖ) wₖ v(tₖ) = n v(t); substituting x = √w·v makes it
# Hermitian: (√w C √w) x = n x, with ∫|v|² = Σₖ wₖ|vₖ|² = |x|² = 1. Using cumul_int's own weights makes
# the modes orthonormal under exactly the integral the recursion below uses. Plain eigen(C)·dt corresponds to
# wₖ = dt everywhere, which over-weights the end points (the emitter is brightest at t = 0) by ~dt/2.
sw = sqrt.(quad_weights(t_size, dt))

# eigenvalues are the photon numbers nᵢ (Σnᵢ ≈ 1), columns of eigens the mode functions vᵢ(t) (arbitrary phase)
eigenvals, eigens = eigen(Hermitian(sw .* corel_m .* sw'); sortby=-)
eigens ./= sw
# photon numbers are ≥ 0; any remaining negatives are solver noise (~1e-8), so set them to 0
eigenvals = max.(eigenvals, 0.0)

eigenvals_trunc = eigenvals[1:100]/sum(eigenvals[1:100])
S = sum([-i*log(i) for i in eigenvals_trunc])
bond_dim  = exp(S)

# columns of eigens are the mode functions vᵢ(t) on the grid, each normalised so ∫|vᵢ|²dt = 1
v_0 = [cubic_spline_interpolation(0:dt:t_fin, eigens[:,i]; extrapolation_bc=0) for i in 1:no_cavs]
plot(abs.(v_0[1].(0:dt:t_fin)))


# Virtual-cavity chain, solved in closed form.
# With uⱼ = vⱼ^(j-1) the mode reaching cavity j, Nⱼ(t) = ∫₀ᵗ |uⱼ|² and gⱼ = -uⱼ*/√Nⱼ, the cavity obeys
#   dαᵢⱼ/dt = -½|gⱼ|²αᵢⱼ - gⱼ vᵢ^(j-1),  αᵢⱼ(0) = 0,   output vᵢ^(j) = vᵢ^(j-1) + gⱼ* αᵢⱼ
# (this convention makes cavity j absorb uⱼ completely). The ODE integrates exactly
# (the decay factor is √(Nⱼ(s)/Nⱼ(t))) to
#   αᵢⱼ(t) = ∫₀ᵗ uⱼ* vᵢ^(j-1) / √Nⱼ(t),   vᵢ^(j) = vᵢ^(j-1) - uⱼ ∫₀ᵗ uⱼ* vᵢ^(j-1) / Nⱼ(t)
# i.e. Gram-Schmidt done locally in time, with no 1/√t anywhere in the recursion.
#
# t = 0: uⱼ ~ t^(j-1), so gⱼ ~ ∓√(2j-1)/√t diverges (physical: the cavity must catch the leading edge).
# We store Gⱼ = √t·gⱼ, which is finite, and gv_f[j](t) = Gⱼ(t)/√t is ±Inf at t = 0.
# The early-time t^(j-1) behaviour of higher modes is below what the discretised eigenvectors resolve,
# so mode j is only resolved from t_res[j], where it has carried ε_res of its photon (Nⱼ > ε_res).
# Before that, later modes are set to 0 and gⱼ continues as the c/√t form matched at t_res[j].
ε_res = 1e-10

V = copy(eigens[:, 1:no_cavs])      # V[:,i] holds vᵢ^(j) on the grid, updated as each cavity j is added
G = zeros(ComplexF64, t_size, no_cavs)   # Gⱼ = √t·gⱼ on the grid
t_res = zeros(no_cavs)
alpha = Dict()
gv_f = [;]
for j=1:no_cavs
    u = V[:, j]
    N = cumul_int(abs2.(u))
    k0 = findfirst(N .> ε_res); r = k0:t_size
    t_res[j] = t_list[k0]
    G[r, j] = -conj.(u[r]) ./ sqrt.(N[r]) .* sqrt.(t_list[r])
    G[1:k0-1, j] .= G[k0, j]
    for i=j+1:no_cavs
        ## overlap of mode i with mode j up to time t
        ## But with alpha starting from zero is Photons of cavity i in j
        intermed_photon = cumul_int(conj.(u) .* V[:, i])
        # Getting alpha from the paper :
        # a_ij = zeros(ComplexF64, t_size)
        # a_ij[r] = intermed_photon[r] ./ sqrt.(N[r])
        # alpha[[i,j]] = cubic_spline_interpolation(0:dt:t_fin, a_ij; extrapolation_bc=Line())
        V[r, i] .-= u[r] .* intermed_photon[r] ./ N[r]
        V[1:k0-1, i] .= 0.0
    end
    G_spl = cubic_spline_interpolation(0:dt:t_fin, G[:, j]; extrapolation_bc=Line())
    push!(gv_f, t -> G_spl(t) / sqrt(t))
end
# usual
# return gv_f
# only for test
return gv_f, eigenvals


########################################################

# As used in the modudule
using MPOFuncs

g,e_vals = g2(gamma, deph, t_fin, dt, no_cavs)
sum(e_vals[1:10])
1-sum(e_vals[1:200])
########################################################

# Comparison plot: old equal-weight modes (eigen(C)·dt) vs the quadrature-weighted modes (eigens above), modes 1-3.
# Run by hand after the main part (uses corel_m, eigens, eigenvals, t_list, dt).
# Saved as Plots/modes_old_vs_new_weights.png. Rows: full window, zoom on t ≤ 0.15 (every grid point), new - old.
ev_old, V_old = eigen(Hermitian(corel_m); sortby=-)
ev_old .*= dt; V_old ./= sqrt(dt)
# remove each mode's arbitrary phase (make it real), then match the sign of new to old
realify(v) = (k = argmax(abs.(v)); real.(v .* conj(v[k]) ./ abs(v[k])))
v_old = [realify(V_old[:, i]) for i in 1:3]
v_new = [(r = realify(eigens[:, i]); sign(sum(r .* v_old[i])) * r) for i in 1:3]
c_old, c_new = colorant"#2a78d6", colorant"#eb6834"
z = t_list .<= 0.15
ps = []
for i in 1:3
    p = plot(t_list, v_old[i]; lw=2, color=c_old, label="old: equal weights, n=$(round(ev_old[i], digits=4))",
             title="Mode $i", xlabel="t (1/γ)", ylabel="v(t)")
    plot!(p, t_list, v_new[i]; lw=2, ls=:dash, color=c_new, label="new: quadrature weights, n=$(round(eigenvals[i], digits=4))")
    push!(ps, p)
end
for i in 1:3
    p = plot(t_list[z], v_old[i][z]; lw=2, color=c_old, marker=:circle, ms=6.5, msw=1.5, msc=:white, label="old",
             title="Mode $i, t ≤ 0.15 (every grid point)", xlabel="t (1/γ)", ylabel="v(t)")
    plot!(p, t_list[z], v_new[i][z]; lw=2, ls=:dash, color=c_new, marker=:utriangle, ms=7.5, msw=1.5, msc=:white, label="new")
    push!(ps, p)
end
for i in 1:3
    d = v_new[i] .- v_old[i]
    p = plot(t_list, d; lw=2, color=:gray30, label="", title="Mode $i: new - old", xlabel="t (1/γ)", ylabel="difference")
    scatter!(p, t_list[1:4], d[1:4]; color=c_new, marker=:utriangle, ms=7.5, msw=1.5, msc=:white, label="first 4 grid points")
    hline!(p, [0]; color=:gray30, lw=0.8, label="")
    push!(ps, p)
end
fig = plot(ps...; layout=(3, 3), size=(1500, 1150), left_margin=6Plots.mm, bottom_margin=5Plots.mm,
           plot_title="Temporal modes: equal weights (old) vs 4th-order quadrature weights (new), dt = $dt")
savefig(fig, joinpath(@__DIR__, "..", "Plots", "modes_old_vs_new_weights.png"))
