module MPOFuncs
using ITensors: @OpName_str, @SiteType_str   # no longer exported by ITensors ≥ 0.8

using ITensors, ITensorMPS, Integrals, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, SparseArrays
export beamsplitter!, swap_nextsite!, swap_ij!, trace, measure_corel, measure_a, g_34, g2, cash_karppe, capture_photon
# export ITensors, beamsplitter!, drho, trace, measure_corel, measure_a, g_34, g2, cash_karppe, cash_karppe_evolve

#= ===================================== EVOLUTION FUNCTIONS: OVERVIEW =====================================
Pipeline: g2 -> gᵢ(t) couplings  ->  drho gives dρ/dt  ->  cash_karppe takes one adaptive step
          ->  capture_photon loops to t_final and traces out the emitter.

1. Couplings to the virtual cavities
   g2                      Emitter correlation ⟨σ₊(t)σ₋(t')⟩ (QuantumOptics) -> eigenmodes vᵢ(t) -> closed-form
                           cavity couplings gᵢ(t). Returns (gv_f, eigenvals).                       IN USE
   cumul_int, quad_weights 4th-order running integral + matching quadrature weights (modes orthonormal
                           under the same integral). Helpers for g2.                                IN USE
   g2 (old, ODE-based)     Previous version incl. a_, vᵢⁱ⁻¹.                                         COMMENTED OUT

2. Right-hand side dρ/dt   (all: dephasing D[√κ σz] with κ = deph/2, same as g2)
   drho                    Non-Hermitian form Kρ + ρK† + L₀ρL₀† + κσzρσz, K written out by hand.     IN USE
     pre_term, post_term,  Build K, K†, L₀/L₀† as MPOs. Emitter term is "A",1,"Adag",1 = c c† (not c†c):
     Lo                    adds -γρ, which the per-step renormalisation removes.                    IN USE

3. Time stepping
   CK_a, CK_b, CK_c, CK_cꜛ, k_gen   Cash-Karp tableau and the six stages.
   cash_karppe             Accept/reject step: accept if error ≤ tol, else retry smaller (6 RHS calls per
                           accepted step). Returns (ρ_new, dt_taken, dt_next).                      IN USE

4. Full evolution (emitter excited -> photon state in the virtual cavities)
   capture_photon            drho, g2 on dt/5 grid. Returns (mpo, sites, eigenvals).  IN USE BY EVERY SCRIPT
                             Steps with cash_karppe (tol 1e-4, max step 10·dt), ends exactly at t_final. kwargs:
                               progress=true, progress_every=1.0   progress bar
                               cutoff=1e-12        truncate! ρ each step (nothing = never); ~5x faster than
                                                   never at 4 cavities, result changes ~1e-4
                               dt_start=1e-5       first step size tried (small, so the first step isn't rejected)

5. Progress display
   print_progress          One-line bar redrawn in place: t/t_final, dt, χ, elapsed, ETA.
   fmt_secs                Seconds -> "42s" / "3m05s".
========================================================================================================== =#


##### Defining States Needed #####
#-------------------------------------------------------------------------

#note: using sparse matrices seems liek it might be faster 
#but most documentation says its not unless for super specefic cases that are probably not se relevant atm
#eg 0.05% dense or something

function ITensors.op(::OpName"Ground" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[1,1] = 1
    return mat
end
function ITensors.op(::OpName"Ground" , ::SiteType"Qubit" )
    mat = zeros(2, 2)
    mat[1,1] = 1
    return mat
end
function ITensors.op(::OpName"Excite1" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[2,2] = 1
    return mat
end
function ITensors.op(::OpName"Excite1" , ::SiteType"Qubit")
    mat = zeros(2, 2)
    mat[2,2] = 1
    return mat
end
function ITensors.op(::OpName"ExciteMax" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[d,d] = 1
    return mat
end
function ITensors.op(::OpName"Sz" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[1,1] = 1
    mat[2,2] = -1
    return mat
end

function n_copy_mpo(mpo_i, sites, n)
    #Assuming qudit
    l = length(mpo_i)
    new_inds = siteinds("Qudit", n*l, dim=dim(sites[1]))
    I_mpo = MPO(l*n)
    I_op = OpSum()
    for i=1:l
        for j=1:n
            ind = i + (l*(j-1))
            I_mpo[ind] = deepcopy(mpo_i[i])
            replaceind!(I_mpo[ind], sites[i], new_inds[ind])
            replaceind!(I_mpo[ind], sites[i]', new_inds[ind]')
            I_op += "I",ind
        end
    end
    #applying I to change up link indices
    I_op = MPO(I_op, new_inds)
    I_mpo = apply(I_op, I_mpo)
    I_mpo /= tr(I_mpo)
    #normalize!(I_mpo)
    return I_mpo, new_inds
end

# code to join two MPOs
function join(mpo_1, sites1, mpo_2, sites2)
    #Assuming qudit
    #creates new mpo in order given in input
    l1 = length(mpo_1)
    l2 = length(mpo_2)

    new_inds = siteinds("Qudit", l1+l2, dim=dim(sites1[1]))
    I_mpo = MPO(l1+l2)

    I_op = OpSum()
    I_op += "I",l1,"I",(l1+1)
    I_op = MPO(I_op, new_inds)

    for i=1:max(l1,l2)
        if (i > l1)
            I_mpo[l1+i] = deepcopy(mpo_2[i])
            replaceind!(I_mpo[l1+i], sites2[i], new_inds[l1+i])
            replaceind!(I_mpo[l1+i], sites2[i]', new_inds[l1+i]')
        elseif (i > l2)
            I_mpo[i] = deepcopy(mpo_1[i])
            replaceind!(I_mpo[i], sites1[i], new_inds[i])
            replaceind!(I_mpo[i], sites1[i]', new_inds[i]')
        else
            I_mpo[i] = deepcopy(mpo_1[i])
            I_mpo[l1+i] = deepcopy(mpo_2[i])
            replaceind!(I_mpo[i], sites1[i], new_inds[i])
            replaceind!(I_mpo[i], sites1[i]', new_inds[i]')
            replaceind!(I_mpo[l1+i], sites2[i], new_inds[l1+i])
            replaceind!(I_mpo[l1+i], sites2[i]', new_inds[l1+i]')
        end
    end

    #applying I to change up link indices
    I_mpo = apply(I_op, I_mpo)
    truncate!(I_mpo; cutoff= 1e-12)

    return I_mpo, new_inds
end

##### Defining operations on the MPO #####
#-------------------------------------------------------------------------

# notes for beamsplitter application 
# 1. Cannot exponentiate OpSum
# 2. Cannot apply unequal length MPOs to each other
# 3. making Itensor then multiplying to Identity MPO then applying works, 
#    but slower than just applying Itensor directly




function beamsplitter!(MPO_i, site_list, index_1, index_2)
    # more relvant for sites not next to each other
    cutoff = 1E-12
    op_1 = ((op("A",site_list[index_1]) * op("Adag",site_list[index_2])) + (op("A",site_list[index_2]) * op("Adag",site_list[index_1])))
    H_ = exp((-im/4) * pi * op_1)
    # for i=1:length(site_list)
    #     (i==index_1 || i==index_2) ? nothing : H_*= op("I",site_list[i])
    # end
    # H_ = denseblocks(H_)
    # @show H_
    Iden = MPO(site_list, "I")
    H_ = apply(H_, Iden; cutoff=cutoff)
    @show typeof(H_)
    #above step is supposed to SVD the H_ to make it an MPO not itensor
    H2 = apply(H_, MPO_i; cutoff=cutoff)
    #easier to use apply since preserves MPO data type and doesnt change to iTensor type
    #requires iTensor be applied to an MPO in that order particularly hence double dagger
    H3 = apply(H2, swapprime(conj(H_), 1,0); cutoff=cutoff)

    MPO_i[:] = H3
    # MPO_i /= trace(MPO_i, site_list)
    # return nothing
end

function beamsplitter_nextsite!(MPO_i, site_list, index_1)
    cutoff = 1E-12
    
    op_1 = ((op("A",site_list[index_1]) * op("Adag",site_list[index_1+1])) + (op("A",site_list[index_1+1]) * op("Adag",site_list[index_1])))
    H_ = exp((-im/4) * pi * op_1)
    H2 = apply(H_, MPO_i; cutoff=cutoff)
    #easier to use apply since preserves MPO data type and doesnt change to iTensor type
    #requires iTensor be applied to an MPO in that order particularly hence double dagger
    H3 = conj(swapprime( apply( H_, swapprime(conj(H2), 1,0); cutoff ), 1,0))

    MPO_i[:] = H3
    # MPO_i /= trace(MPO_i, site_list)
    # return nothing
end

function beamsplitter_peps_tensor(tens1, tens2, site1, site2)
    # here applying itensor first, svd later
    #interlayer beamsplitter
    inds3 = uniqueinds(tens1, tens2)
    # @show inds(tens1)
    # @show inds(tens2)
    # @show inds3
    g = pi/((8)^0.5)
    bs_op = ((op("A",site1) * op("Adag",site2)) + (op("A",site2) * op("Adag",site1)))
    bs_op +=  ((op("I",site2)*op("N", site1)) - (op("N", site2)*op("I",site1)) )
    bs_op += -(2)^0.5 * (op("I",site2)*(op("N", site1)) + (op("N", site2))*op("I",site1))
    bs_op = exp((-im * g) * bs_op)

    # bs_op = exp((-im/4) * pi * bs_op)
    # bs_op2 = exp((im/4) * pi * bs_op)
    # bs_optop = replaceprime(bs_op, 1, 2)
    # @show inds(bs_op)
    mult_term = tens1 * tens2
    mult_term = bs_op' * mult_term
    mult_term = replaceprime(mult_term, 2=>1)

    mult_term = mult_term' * swapprime(conj(bs_op), 1, 0)
    mult_term = replaceprime(mult_term, 2=>1)
    mult_term = swapprime(conj(mult_term), 1, 0)
    # @show inds(mult_term)
    u, s, v = svd(mult_term, inds3; cutoff=1E-12)
    tens1_new = u
    tens2_new = s*v
    # @show inds(tens1_new)
    # @show inds(tens2_new)
    inds4 = commoninds(tens1_new, tens2_new)
    if length(inds4) > 1
        comb_op = combiner(inds4)
        tens1_new = tens1_new * comb_op
        tens2_new = tens2_new * comb_op
        println("hi")
    end

    return tens1_new, tens2_new
end

function excite_site(tens1, site1)
    a_op = op("Adag",site1)
    mult_term = tens1 * a_op'
    @show mult_term
    mult_term = replaceprime(mult_term, 2=>1)

    mult_term = mult_term' * swapprime(conj(op("Adag", site1)), 1, 0)
    @show mult_term
    mult_term = replaceprime(mult_term, 2=>1)
    @show "Chak de"
    return mult_term
end


function beamsplitter_peps_svd(tens1, tens2, site1, site2)
    # # here svd on beamsplitter first, then apply

    # inds3 = uniqueinds(tens1, tens2)
    bs_op = ((op("A",site1) * op("Adag",site2)) + (op("A",site2) * op("Adag",site1)))
    bs_op = exp((-im/4) * pi * bs_op)

    q, r = qr(bs_op, (site1, site1'))
    bs_op1 = q 
    bs_op2 = r
    @show q
    @show r


    tens1_new = bs_op1' * tens1
    tens1_new = replaceprime(tens1_new, 2=>1)
    tens1_new = tens1_new' * swapprime(conj(bs_op1), 1,0)
    tens1_new = replaceprime(tens1_new, 2=>1)
    tens2_new = bs_op2' * tens2
    tens2_new = replaceprime(tens2_new, 2=>1)
    tens2_new = tens2_new' * swapprime(conj(bs_op2), 1,0)
    tens2_new = replaceprime(tens2_new, 2=>1)

    @show tens1_new
    @show tens2_new

    inds4 = commoninds(tens1_new, tens2_new)
    comb_op = combiner(inds4)

    @show comb_op

    tens1_new = tens1_new * comb_op
    tens2_new = tens2_new * comb_op


    return tens1_new, tens2_new
end


function swap_nextsite!(mpo_i, sites, i)
    #Function swaps the i-th and i+1-th sites in the mpo

    #archived function
    # mpo_contrac = mpo_i[i] * mpo_i[i+1]
    # beg_ind = uniqueinds(mpo_i[i+1],mpo_i[i])
    # length(mpo_i) < i+2 ? nothing : deleteat!(beg_ind, findall(x->x==commonind(mpo_i[i+1], mpo_i[i+2]),beg_ind))
    # i == 1 ? nothing : push!(beg_ind, commonind(mpo_i[i], mpo_i[i-1]))
    # U,S,V = svd(mpo_contrac, beg_ind; cutoff=1E-12)
    # # U,S,V = svd(mpo_contrac, beg_ind)
    # mpo_i[i] = U
    # mpo_i[i+1] = S * V

    mpo_new = swapbondsites(mpo_i, i)

    place_holder = sites[i]
    sites[i] = sites[i+1]
    sites[i+1] = place_holder
    return mpo_new
end

function swap_ij!(mpo_i, sites, i, j)
    # Assuming j > i and both in length

    # for k=i:j-1
    #     swap_nextsite!(mpo_i, sites, k)
    # end
    mpo_new = truncate!(movesite(mpo_i, Pair(i,j)),site_range=i:j, cutoff=1E-12)
    place_holder = sites[i]
    deleteat!(sites, i)
    insert!(sites, j, place_holder)

    return mpo_new
end

function trace(MPO_i, sites_i)
    tr_ = 1.0
    for i in eachindex(sites_i)
        tr_ *= delta(sites_i[i], sites_i[i]') * MPO_i[i]
    end
    return tr_[1]
end

function expect(MPO_i, sites_i)
    list_exp = []
    for i in eachindex(sites_i)
        append!(list_exp, trace(apply( op("n", sites_i[i]), MPO_i), sites_i))
    end
    return list_exp
end

#measures <a1 a2 rho adag1 adag2> at sites 1 and 2
function measure_corel(mpo_i, sites, site1, site2)
    AA_op = OpSum()
    AA_op += "A",site1,"A",site2
    AA_op = MPO(AA_op, sites)
    AdAd_op = swapprime(conj(AA_op), 1,0)
    val = trace(apply(AA_op, apply(mpo_i, AdAd_op)), sites)
    return val
end 

#measures <a1 a2 rho adag1 adag2> at sites 1 and 2
function measure_a(mpo_i, sites, site1)
    A_op = OpSum()
    A_op += "A",site1
    A_op = MPO(A_op, sites)
    # println(A_op)
    Ad_op = swapprime(conj(A_op), 1,0)
    val = trace(apply(A_op, apply(mpo_i, Ad_op)), sites)
    return val
end 

function g_34(mpo_i, sites, no_cavs)
    # measuring G_34 t, tau integrated
    G_34 = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for i=1:no_cavs
        for j=1:no_cavs
            #if v^2 integrates out to 1
            term = measure_corel(mpo_i, sites, i, no_cavs+j)
            #println(ITensors.expect(double, double_sites))
            #println(term)
            G_34 += 0.5 * term
        end
        denom1 += measure_a(mpo_i, sites, i)
        #println(ITensors.expect(mpo_i, sites))
        denom2 += measure_a(mpo_i, sites, no_cavs+i)
    end
    denom =0.5 * denom1 * denom2

    return G_34/denom
end

function g_34_new(mpo_i, sites, no_cavs)
    # measuring G_34 t, tau integrated
    G_34 = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for i=1:no_cavs
        for j=1:no_cavs
            #if v^2 integrates out to 1
            # println(2*i-1, " ", 2*j)
            term = measure_corel(mpo_i, sites, 2*i-1, 2*j)
            G_34 += 0.5 * term
        end
        denom1 += measure_a(mpo_i, sites, 2*i-1)
        denom2 += measure_a(mpo_i, sites, 2*i)
    end
    denom =0.5 * denom1 * denom2
    return G_34/denom
end

#= Old implementation (ODE-based, Kiilerich & Mølmer supp. mat. Eqs. S10-S12), kept for reference.
   Replaced by the closed-form g2 below. Known issues: √deph σz dephasing convention, real modes only,
   regularised g(0), and poor convergence in dt for modes ≥ 2 caused by the 1/√t singularity in the ODEs.
function g2(gamma, deph, t_fin, dt, no_cavs; reverse=false)

    basis = SpinBasis(1//2)
    a = sigmam(basis)
    at = sigmap(basis)
    H =  identityoperator(basis)
    J = [sqrt(gamma)*a, sqrt(deph)*sigmaz(basis)]

    t_list = [0:dt:t_fin;]
    t_size = length(t_list)
    # corel_m = Array{ComplexF64}(undef, t_size, t_size)
    # corel_m *= 0
    corel_m = fill(0.0, (t_size,t_size))

    ρ₀ = spinup(basis) ⊗ dagger(spinup(basis))

    tout, ρt_master = timeevolution.master(t_list, ρ₀, H, J; dt=dt)
    for i=1:t_size-1
        rhot = ρt_master[i]
        corel_m[i,i:end] = timecorrelations.correlation(t_list[1:(t_size-i+1)], rhot, H, J, at, a)
    end
    corel_m[t_size,t_size] = 0.0
    corel_m = -diagm((diag(corel_m))) + corel_m + conj(corel_m)'
    corel_m = real(corel_m)
    # println(any(isnan.(corel_m)))
    eigenvals, eigens = eigen(corel_m; sortby=-)

    v_0 = [cubic_spline_interpolation(0:dt:t_fin, eigens[:,i]; extrapolation_bc=0) for i in 1:no_cavs]
    gv = Dict()
    alpha = Dict()
    gv_f = [;]

    function a_(a, p, t)
        i, j = deepcopy(p)
        da = -(v_0[i](t)*gv_f[j](t))-(a * 0.5 * gv_f[j](t)^2)
        for k=1:j-1
            da -= gv_f[j](t) * gv_f[k](t) * alpha[[i,k]](t)
        end
        return da
    end

    function vᵢⁱ⁻¹(i,t)
        v = v_0[i](t)
        for k=1:i-1
            v += gv_f[k](t) * alpha[[i,k]](t)
        end
        return v
    end

    # function cum_int(vᵢⁱ⁻¹, i, t)

    #     sol.u == 0 ? (return 10^-8) : (return sqrt(sol.u))
    #     return sol.u
    # end

    for i=1:Int(no_cavs)
        for j=1:Int(i-1)
            # a0 = 0
            # tspan = (0, t_fin)
            prob = ODEProblem(a_, 0, (0,t_fin), [i,j])
            sol = solve(prob, saveat=dt)
            alpha[[i,j]] = cubic_spline_interpolation(0:dt:t_fin, sol.u; extrapolation_bc=Line())
        end
        cum_int_list = [0.0,]
        for t=dt:dt:t_fin;
            f(u, p) =  conj(vᵢⁱ⁻¹(p, u)) * vᵢⁱ⁻¹(p, u)
            domain = (t-dt, t) # (lb, ub)
            
            prob = IntegralProblem(f, domain, i)
            sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
            push!(cum_int_list, cum_int_list[end]+sol.u)
        end
        cum_int_list = sqrt.(cum_int_list)
        cum_int_list[1] = cum_int_list[2]
        # println(vᵢⁱ⁻¹(1,0))
        cum_int_extrap = cubic_spline_interpolation(0:dt:t_fin, cum_int_list;extrapolation_bc=Line())
        gᵥ(t) = - conj(vᵢⁱ⁻¹(i,t))/cum_int_extrap(t)
        push!(gv_f, gᵥ)
    end
    # usual
    # return gv_f
    # only for test
    return gv_f, eigenvals
end
=#

# 4th-order cumulative integral on a uniform grid of step dt (exact for cubics, one-sided at the ends)
#For g2
function cumul_int(y, dt)
    n = length(y); I = zeros(eltype(y), n)
    for k in 1:n-1
        s = k == 1   ? 9y[1] + 19y[2] - 5y[3] + y[4] :
            k == n-1 ? 9y[n] + 19y[n-1] - 5y[n-2] + y[n-3] :
                       -y[k-1] + 13y[k] + 13y[k+1] - y[k+2]
        I[k+1] = I[k] + s * dt / 24
    end
    return I
end

# Weights w with Σₖ wₖ yₖ = cumul_int(y, dt)[end], i.e. the same quadrature rule applied to the whole interval
#For g2
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

# Virtual-cavity couplings gᵥᵢ(t) for the output modes of a decaying, dephased emitter
# (Kiilerich & Mølmer, arXiv:1902.09833 and its supp. mat.). Returns gv_f (functions of t, complex)
# and eigenvals (photon number nᵢ in each mode, Σnᵢ ≈ 1).
#   deph: pure dephasing rate of the emitter coherence (jump operator √(deph/2) σz, matching drho).
#   Δ:    emitter detuning. Δ ≠ 0 also needs a Δ·c†c term in drho's Hamiltonian (not implemented there).
#   ε_res: mode j is only resolved once it has carried ε_res of its photon (see comments below).
function g2(gamma, deph, t_fin, dt, no_cavs; reverse=false, Δ=0.0, ε_res=1e-10)

    basis = SpinBasis(1//2)
    a = sigmam(basis)
    at = sigmap(basis)
    H = Δ * at * a
    # D[√κ σz] damps the coherence at 2κ since σz² = I, so κ = deph/2 gives a dephasing rate of deph
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
    corel_m[t_size,t_size] = real(QuantumOptics.expect(at*a, ρt_master[end]))   # MPOFuncs has its own expect
    # Row i holds ⟨σ₊(tⱼ)σ₋(tᵢ)⟩ for j ≥ i. Since ⟨σ₊(s)σ₋(t)⟩* = ⟨σ₊(t)σ₋(s)⟩ the lower triangle is the adjoint
    corel_m = corel_m + corel_m' - Diagonal(diag(corel_m))
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

    # Virtual-cavity chain, solved in closed form.
    # With uⱼ = vⱼ^(j-1) the mode reaching cavity j, Nⱼ(t) = ∫₀ᵗ |uⱼ|² and gⱼ = -uⱼ*/√Nⱼ (S11), the cavity obeys
    #   dαᵢⱼ/dt = -½|gⱼ|²αᵢⱼ - gⱼ vᵢ^(j-1),  αᵢⱼ(0) = 0,   output vᵢ^(j) = vᵢ^(j-1) + gⱼ* αᵢⱼ   (S10, S12)
    # which integrates exactly (the decay factor is √(Nⱼ(s)/Nⱼ(t))) to
    #   αᵢⱼ(t) = ∫₀ᵗ uⱼ* vᵢ^(j-1) / √Nⱼ(t),   vᵢ^(j) = vᵢ^(j-1) - uⱼ ∫₀ᵗ uⱼ* vᵢ^(j-1) / Nⱼ(t)
    # i.e. Gram-Schmidt done locally in time, with no 1/√t anywhere in the recursion.
    #
    # t = 0: uⱼ ~ t^(j-1), so gⱼ ~ ∓√(2j-1)/√t diverges. We store Gⱼ = √t·gⱼ, which is finite, and return
    # gv_f[j](t) = Gⱼ(t)/√max(t, t_min) with t_min = dt/2, so the MPO evolution (which starts at t = 0) sees a
    # finite coupling. Cavity 1 is therefore under-coupled for t < t_min, letting through ~t_min of the leading edge.
    # The early-time t^(j-1) behaviour of higher modes is below what the discretised eigenvectors resolve,
    # so mode j is only resolved from t_res[j], where it has carried ε_res of its photon (Nⱼ > ε_res).
    # Before that, later modes are set to 0 and gⱼ continues as the c/√t form matched at t_res[j].
    t_min = dt / 2
    V = copy(eigens[:, 1:no_cavs])           # V[:,i] holds vᵢ^(j), updated as each cavity j is added
    G = zeros(ComplexF64, t_size, no_cavs)   # Gⱼ = √t·gⱼ on the grid
    gv_f = [;]
    for j=1:no_cavs
        u = V[:, j]
        N = cumul_int(abs2.(u), dt)
        k0 = findfirst(N .> ε_res)
        k0 === nothing && error("g2: mode $j never exceeds ε_res = $ε_res (too many cavities for this data, or ε_res ≤ 0)")
        r = k0:t_size
        G[r, j] = -conj.(u[r]) ./ sqrt.(N[r]) .* sqrt.(t_list[r])
        G[1:k0-1, j] .= G[k0, j]
        for i=j+1:no_cavs
            overlap = cumul_int(conj.(u) .* V[:, i], dt)     # ∫₀ᵗ uⱼ* vᵢ^(j-1)
            V[r, i] .-= u[r] .* overlap[r] ./ N[r]
            V[1:k0-1, i] .= 0.0
        end
        G_spl = cubic_spline_interpolation(0:dt:t_fin, G[:, j]; extrapolation_bc=Line())
        push!(gv_f, t -> G_spl(t) / sqrt(max(t, t_min)))
    end
    return gv_f, eigenvals
end

# pre = K = -iH - ½L₀†L₀ - ½κ and post = K†, so dρ/dt = Kρ + ρK† + L₀ρL₀† + κσzρσz. For the cascade
#   K = -½[γc†c + Σᵢ|gᵢ|²aᵢ†aᵢ + 2√γ Σᵢ gᵢ* aᵢ†c + 2 Σ_{i>j} gᵢ* gⱼ aᵢ†aⱼ + κ]
# (g is complex, so the coefficients of raising operators aᵢ† carry gᵢ*)
function pre_term(sites, g_f, rg, dep, t)
    no_cavs = length(sites) - 1
    pre = OpSum()
    pre += (rg^2),"A",1,"Adag",1
    pre += dep,"I",1
    for i=1:no_cavs
        pre += (2*rg*conj(g_f[i](t))),"A",1,"Adag",i+1
        pre += abs2(g_f[i](t)),"Adag",i+1,"A",i+1
        for j=1:i-1
            pre += (2*g_f[j](t)*conj(g_f[i](t))),"Adag",i+1,"A",j+1
        end
    end
    pre = -0.5 * MPO(pre, sites)
    return pre
end

function post_term(sites, g_f, rg, dep, t)
    no_cavs = length(sites) - 1
    post = OpSum()
    post += (rg^2),"A",1,"Adag",1
    post += dep,"I",1
    for i=1:no_cavs
        post += (2*rg*g_f[i](t)),"Adag",1,"A",i+1
        post += abs2(g_f[i](t)),"Adag",i+1,"A",i+1
        for j=i+1:no_cavs
            post += (2*g_f[j](t)*conj(g_f[i](t))),"Adag",i+1, "A",j+1
        end
    end
    post = -0.5 * MPO(post, sites)
    return post
end

function Lo(sites, g_f, rg, t)
    no_cavs = length(sites) - 1
    L_0 = OpSum()
    L_0 += rg,"A",1
    for i=2:no_cavs+1
        L_0 += g_f[i-1](t),"A",i
    end
    L_0 = MPO(L_0, sites)
    L_0d = swapprime(conj(deepcopy(L_0)), 1,0)
    return L_0, L_0d
end

#func calc drho/dt
function drho(rho, p, t)
    #drho(sites, rho, t)
    #(H_int * rho) + (rho * H_int) + sum(Ld rho L)
    sites, gv_f, rg, deph = p
    cut_off=1E-12
    algo="naive"
    no_cavs = length(sites) - 1
    κ = deph / 2    # dephasing D[√κ σz]: the emitter coherence dephases at rate deph (same as g2)

    pre = pre_term(sites, gv_f, rg, κ, t)
    post = post_term(sites, gv_f, rg, κ, t)
    L_0, L_0d = Lo(sites, gv_f, rg, t)

    pre_rho = apply(pre, rho; alg=algo, cutoff=cut_off)
    post_rho = apply(rho, post; alg=algo, cutoff=cut_off)

    drho_ = pre_rho + post_rho
    # L₀ρL₀† as one exact product, truncated once at the end, so the result stays Hermitian
    drho_ += apply(apply(L_0 , rho; alg=algo, truncate=false), L_0d; alg=algo, cutoff=cut_off)

    if deph!= 0
        L_1 = op("Sz",sites[1])
        drho_ += κ*conj(swapprime( apply( L_1, swapprime(conj(apply(L_1, rho; cutoff=cut_off)), 1,0) ; cutoff=cut_off), 1,0))
    end

    # the sum of the terms has ~2-3x the bond dimension of ρ (MPO + only truncates at 1e-15), which makes every stage
    # sum in cash_karppe as expensive as a drho call; truncating here halves it (4 cavities: 17 -> 9, change ~1e-6)
    truncate!(drho_; cutoff=cut_off)
    return drho_
end

# Cash-Karp tableau: nodes a, stage weights b, 5th-order weights c, embedded 4th-order weights cꜛ
const CK_a = [0, 0.2, 0.3, 0.6, 1, 7/8]
const CK_b = [[1,],[1/5,],[3/40, 9/40], [3/10, -9/10, 6/5], [-11/54, 5/2, -70/27, 35/27],
    [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]
const CK_c = [37/378, 0, 250/621, 125/594, 0, 512/1771]
const CK_cꜛ = [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]

# the six Cash-Karp stages kᵢ = dt·dρ/dt(ρ + Σⱼ bᵢⱼkⱼ, t + aᵢdt)
function k_gen(sys, d_rho, t, dt, a, b ; p=nothing)
    # p = (sites, gv_f, rg, deph)
    k_list = [ ; ]
    kₙ = dt *  d_rho(sys, p, t)
    push!( k_list, kₙ)
    for i=2:6
        sys⁻ = sys + sum(k_list .* b[i])
        t⁻ = t + a[i]*dt
        kₙ = dt *  d_rho(sys⁻, p, t⁻)
        push!( k_list, kₙ)
    end
    return k_list
end

# Adaptive Cash-Karp step with accept/reject: the step is accepted only if its error estimate ‖ρ₅ - ρ₄‖ (5th- minus
# embedded 4th-order result) is ≤ tol, otherwise it is retried with a smaller dt. 6 RHS calls per accepted step.
# The next step is proposed as dt·0.9·(tol/err)^(1/5), limited to 5x growth and max_dt.
# Returns (ρ_new, dt_taken, dt_next).
# (Replaced the earlier version, which estimated the error at dt, then always recomputed the step at the new dt
#  without checking it: 12 RHS calls per step. ~2x slower for the same accuracy.)
function cash_karppe(sys, d_rho, t, dt; tol=1e-4, p_=nothing, max_dt=Inf, safety=0.9)
    wsum(w, ks) = sum(w[i] * ks[i] for i in eachindex(w) if w[i] != 0)
    while true
        k_s = k_gen(sys, d_rho, t, dt, CK_a, CK_b, p=p_)
        err = norm(wsum(CK_cꜛ .- CK_c, k_s))
        if err <= tol
            dt_next = min(dt * min(5.0, safety * (tol/err)^0.2), max_dt)
            return sys + wsum(CK_c, k_s), dt, dt_next
        end
        dt *= max(0.1, safety * (tol/err)^0.25)
        dt < 1e-12 && error("cash_karppe: step size underflow at t = $t (error $err > tol $tol)")
    end
end

# cutoff: rho is compressed with truncate! after every step (nothing = never). The Cash-Karp update adds MPOs
# without truncating, so otherwise the bond dimension keeps growing with numerical noise (4 cavities at t = 1:
# χ = 45, of which ~14 survive cutoff 1e-12).
# Steps with the accept/reject cash_karppe (tol 1e-4, max step 10·dt) and lands exactly on t_final.
# dt_start: size of the first step tried. The couplings diverge as 1/√t near t = 0, so a first try of dt gets
# rejected a few times (2-3 retries of 6 RHS calls); a small start avoids that, and the step can grow 5x per step.

function capture_photon(no_cavs, depha, gamma, dt, t_final; progress=true, progress_every=1.0, cutoff=1e-12,
                        dt_start=1e-5)
    g_f, eigenvals = g2(gamma, depha, t_final, dt/5, no_cavs)

    sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
    input_list = repeat(["Ground",],no_cavs+1)
    input_list[1] = "Excite1"
    sys = MPO(sys_sites, input_list)

    t = 0.0
    expect_list=[]
    # t_list = [t,]
    t_start = time()
    last_print = -Inf
    p_ = (sys_sites, g_f, sqrt(gamma), depha)
    h = dt_start      # proposed size of the next step

    while t<t_final
        sys_, dt_, h = cash_karppe(sys, drho, t, min(h, t_final - t); tol=1E-4, p_=p_, max_dt=dt*10)
        t += dt_
        sys = sys_/trace(sys_, sys_sites)
        isnothing(cutoff) || truncate!(sys; cutoff=cutoff)

        #progress bar
        if progress && (time() - last_print >= progress_every || t >= t_final)
            print_progress(t, t_final, dt_, maxlinkdim(sys), t_start)
            last_print = time()
        end
    end
    progress && println()

    copy_sys = MPO(no_cavs)
    copy_sys[1] = sys[1] * delta(sys_sites[1], sys_sites[1]') * sys[2]
    for i=2:no_cavs
        copy_sys[i] = deepcopy(sys[i+1]) 
    end
    copy_sys /= trace(copy_sys, sys_sites[2:end])

    return copy_sys, sys_sites[2:end], eigenvals
    # , t_list
end

fmt_secs(s) = s < 60 ? "$(round(Int, s))s" : "$(div(round(Int, s), 60))m$(lpad(rem(round(Int, s), 60), 2, '0'))s"

# One-line progress bar for the time evolution, redrawn in place (\r), so it works in the terminal and in Jupyter.
# The ETA assumes the remaining simulated time runs at the average speed so far. It is optimistic, since steps
# get slower as the bond dimension of rho grows.
function print_progress(t, t_final, dt, chi, t_start; width=30)
    frac = clamp(t / t_final, 0, 1)
    n = round(Int, frac * width)
    elapsed = time() - t_start
    eta = frac > 0 ? fmt_secs(elapsed * (1 - frac) / frac) : "?"
    print("\r[", "="^n, " "^(width - n), "] t = ", round(min(t, t_final), digits=2), "/", t_final,
          "  dt = ", round(dt, sigdigits=2), "  χ = ", chi, "  elapsed ", fmt_secs(elapsed), "  ETA ~", eta, "    ")
    flush(stdout)
end

end