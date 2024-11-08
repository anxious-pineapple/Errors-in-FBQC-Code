module MPOFuncs

using ITensors, Integrals, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations, SparseArrays
export beamsplitter!, swap_nextsite!, swap_ij!, drho, trace, measure_corel, measure_a, g_34, g2, cash_karppe, cash_karppe_evolve
# export ITensors, beamsplitter!, drho, trace, measure_corel, measure_a, g_34, g2, cash_karppe, cash_karppe_evolve


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
    truncate!(I_mpo; cutoff= 1e-10)

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
    cutoff = 1E-10
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
    cutoff = 1E-10
    
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
    u, s, v = svd(mult_term, inds3; cutoff=1E-8)
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
    mpo_new = truncate!(movesite(mpo_i, Pair(i,j)),site_range=i:j, cutoff=1E-10)
    place_holder = sites[i]
    deleteat!(sites, i)
    insert!(sites, j, place_holder)

    return mpo_new
end

#func calc drho/dt
function drho(rho, p, t)
    #drho(sites, rho, t)
    #(H_int * rho) + (rho * H_int) + sum(Ld rho L)
    sites, gv_f, rg, deph = p
    cut_off=1E-10
    algo="naive"
    no_cavs = length(sites) - 1

    H = OpSum()
    H += "I",1
    L_0 = OpSum()
    L_0 += rg,"A",1

    for i=2:no_cavs+1
        L_0 += gv_f[i-1](t),"A",i
        H += (0.5im*rg*gv_f[i-1](t)),"Adag",1,"A",i
        H -= (0.5im*rg*gv_f[i-1](t)),"A",1,"Adag",i

        for j=2:i-1
            H += (0.5im*gv_f[j-1](t)*gv_f[i-1](t)),"Adag",j,"A",i
            H -= (0.5im*gv_f[j-1](t)*gv_f[i-1](t)),"A",j,"Adag",i
        end
    end

    H = MPO(H, sites; cutoff=cut_off)
    H_rho = apply(H, rho; alg=algo, cutoff=cut_off)
    
    drho_ = -1im*(H_rho - swapprime(conj(deepcopy(H_rho)), 1,0))

    #L dagger
    L_0 = MPO(L_0, sites; cutoff=cut_off)
    L_0d = swapprime(conj(deepcopy(L_0)), 1,0)
    LdL = 0.5 * apply(L_0d , L_0; alg=algo, cutoff=cut_off)

    drho_ += apply(apply(L_0 , rho; alg=algo, cutoff=cut_off), L_0d; alg=algo, cutoff=cut_off)
    LdL_rho = apply(LdL, rho; alg=algo, cutoff=cut_off)
    drho_ -= LdL_rho
    drho_ -= swapprime(conj(deepcopy(LdL_rho)), 1,0)

    if deph!= 0
        L_1 = op("Sz",sites[1])
        drho_ += deph*conj(swapprime( apply( L_1, swapprime(conj(apply(L_1, rho; cutoff=cut_off)), 1,0) ; cutoff=cut_off), 1,0))
        drho_ -= deph*rho
    end

    return drho_
end

function create_MPO(no_cavs, depha, gamma, dt, t_final; sites_provided = [;], reverse=false)

    #creating initial state and sites
    sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
    if length(sites_provided)!=0
        sys_sites = sites_provided
    end
    input_list = repeat(["Ground",],no_cavs+1)
    input_list[1] = "Excite1"
    sys = MPO(sys_sites, input_list)

    #Generating G2
    #gamma = 1
    g_f = g2_(gamma, depha, t_final, dt/2, no_cavs)
    
    println("genfunctions")
    
    expect_list = Array{Float64}(undef, Int(t_final/dt + 1), no_cavs+1)
    expect_list *= 0
    # #ldl_list = [;]
    j = 1
    
    """
    #evolving atom-cavity system
    for i=0:dt:t_final
        d_rho , ldl = drho(sys_sites, sys, g_f, sqrt(gamma); t=i, deph=depha)
        sys +=  dt * d_rho
        sys /= tr(sys)

        push!(ldl_list, ldl)
        expect_list[j,:] = real(ITensors.expect(sys, sys_sites))
        j += 1
        #println(i)
    end
    """
    #bonddim = [;]

    for i=0:dt:t_final
        d_rho1 = drho2(sys_sites, sys, g_f, sqrt(gamma); t=i, deph=depha)
        sys1 = sys + (d_rho1*dt/2)
        #sys1 /=tr(sys1)
        d_rho2 = drho2(sys_sites, sys1, g_f, sqrt(gamma); t=(i+(dt/2)), deph=depha)
        sys2 = sys + (d_rho2*dt/2)
        #sys2 /=tr(sys2)
        d_rho3  = drho2(sys_sites, sys2, g_f, sqrt(gamma); t=(i+(dt/2)), deph=depha)
        sys3 = sys + (d_rho3*dt)
        #sys3 /=tr(sys3)
        d_rho4  = drho2(sys_sites, sys3, g_f, sqrt(gamma); t=(i+dt), deph=depha)
        sys +=  (dt * (d_rho1 + (2*d_rho2) + (2*d_rho3) + d_rho4)/6)
        sys /= trace(sys, sys_sites)

        #push!(ldl_list, (ldl+2*ldl2+2*ldl3+ldl4)/6)
        #println("max bond dim at time: ", maximum(linkdims(sys)), " ", i)
        # #append!(bonddim, maximum(linkdims(sys)))
        expect_list[j,:] = real(ITensors.expect(sys, sys_sites))
        j += 1
    end
    
    #trace out atom
    copy_sys = MPO(no_cavs)
    copy_sys[1] = sys[1] * delta(sys_sites[1], sys_sites[1]') * sys[2]
    for i=2:no_cavs
        copy_sys[i] = deepcopy(sys[i+1]) 
    end
    copy_sys /= trace(copy_sys, sys_sites[2:end])

    return copy_sys, sys_sites[2:end], expect_list
    #return sys, sys_sites
    #, g_f
    # return expect_list
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

#returns weighted d rho acc some adaptive step algo by cash and karp with fifth order RK4
function cash_karppe(sys, sys_sites, d_rho, t, dt; tol=0.1, p_ = nothing, max_dt=nothing)

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
    # yn

    a = [0, 0.2, 0.3, 0.6, 1, 7/8]
    b = [[1,],[1/5,],[3/40, 9/40], [3/10, -9/10, 6/5], [-11/54, 5/2, -70/27, 35/27], 
        [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]
    c = [37/378, 0, 250/621, 125/594, 0, 512/1771]
    cꜛ = [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]

    k_s = k_gen(sys, d_rho, t, dt, a, b, p=p_)
    # println("Step1")
    new_sys = sys + sum(c.*k_s)
    new_sysꜛ = sys + sum(cꜛ.*k_s)
    error = abs(norm(new_sysꜛ- new_sys))

    new_dt = dt * (tol/error)^0.2
    # new_dt > dt ? new_dt = dt : nothing
    # println(new_dt)
    isnothing(max_dt) ? nothing : ((new_dt>max_dt) && (new_dt=max_dt; true) )
    ks_new = k_gen(sys, d_rho, t, new_dt, a, b, p=p_)
    # print(ks_new)
    better_sys = sys + sum(c.*ks_new)

    return better_sys, new_dt
    # return sys, new_dt
end

function cash_karppe_evolve(no_cavs, depha, gamma, dt, t_final)
    g_f = g2(gamma, depha, t_final, dt/2, no_cavs)

    sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
    input_list = repeat(["Ground",],no_cavs+1)
    input_list[1] = "Excite1"
    sys = MPO(sys_sites, input_list)

    t = 0.0
    expect_list=[]
    # t_list = [t,]

    while t<t_final
        sys_, dt_ = cash_karppe(sys, sys_sites, drho, t, dt; tol=1E-4, p_= (sys_sites, g_f, sqrt(gamma), depha), max_dt = dt*10)
        t += dt_
        sys = sys_/trace(sys_, sys_sites)
        # push!(expect_list, ITensors.expect(sys, sys_sites))
        # push!(t_list, t)
    end

    copy_sys = MPO(no_cavs)
    copy_sys[1] = sys[1] * delta(sys_sites[1], sys_sites[1]') * sys[2]
    for i=2:no_cavs
        copy_sys[i] = deepcopy(sys[i+1]) 
    end
    copy_sys /= trace(copy_sys, sys_sites[2:end])

    return copy_sys, sys_sites[2:end]
    # , expect_list
    # , t_list
end

#
    ###########################################################
    #test area 

    # cda = [;]
    # cad = [;]
    # ada = [;]
    # ada_pre = Dict()
    # ada_post = Dict()

    # function init_0(cda, cad, ada, ada_pre, ada_post)
    #     cda = [;]
    #     cad = [;]
    #     ada = [;]
    #     ada_pre = Dict()
    #     ada_post = Dict()

    #     return cda, cad, ada, ada_pre, ada_post
    # end
#

function init(par_list, sites, rg)
    cda, cad, ada, ada_pre, ada_post = par_list
    no_cavs = length(sites) - 1
    for i=1:no_cavs
        term_cda = OpSum()
        term_cda += rg,"Adag",1,"A",i+1
        push!(cda, MPO(term_cda,sites))
        term_cad = OpSum()
        term_cad += rg,"A",1,"Adag",i+1
        push!(cad, MPO(term_cad, sites))
        term_ada = OpSum()
        term_ada += "N",i+1
        push!(ada, MPO(term_ada, sites))
        for j=1:i-1
            # key is dagger index, then non dagger index
            term_pre = OpSum()
            term_pre += "Adag",i+1,"A",j+1
            ada_pre[(i,j)] = MPO(term_pre, sites)
            term_post = OpSum()
            term_post += "A",i+1,"Adag",j+1
            ada_post[(j,i)] = MPO(term_post, sites)
        end
    end
    par_list = [cda, cad, ada, ada_pre, ada_post]
    return par_list
end

function pre_term2(par_list, sites, g_f, rg, dep, t)
    cda, cad, ada, ada_pre, ada_post = par_list
    no_cavs = length(sites) - 1
    pre = OpSum()
    pre += dep,"I",1
    pre += (rg^2),"N",1
    pre = MPO(pre, sites)
    for i=1:no_cavs
        pre += cad[i] * 2 * g_f[i](t)
        pre += ada[i] * g_f[i](t)^2
        for j=1:i-1
            pre += ada_pre[(i,j)] * 2 * g_f[j](t) * g_f[i](t)
        end
    end
    return (-0.5 * pre)
end

function post_term2(par_list, sites, g_f, rg, dep, t)
    cda, cad, ada, ada_pre, ada_post = par_list
    no_cavs = length(sites) - 1
    post = OpSum()
    post += dep,"I",1
    post += (rg^2),"N",1
    post = MPO(post, sites)
    for i=1:no_cavs
        post += cda[i] * 2 * g_f[i](t)
        post += ada[i] * g_f[i](t)^2
        for j=1:i-1
            post += ada_post[(j,i)] * 2 * g_f[j](t) * g_f[i](t)
        end
    end
    return (-0.5 * post)    
end

#func calc drho/dt
function drho_test2(rho, p, t)
    #drho(sites, rho, t)
    #(H_int * rho) + (rho * H_int) + sum(Ld rho L)
    par_list, sites, gv_f, rg, deph = p
    cut_off=1E-10
    algo="naive"
    no_cavs = length(sites) - 1


    pre = pre_term2(par_list, sites, gv_f, rg, deph, t)
    post = post_term2(par_list, sites, gv_f, rg, deph, t)
    L_0, L_0d = Lo(sites, gv_f, rg, t)

    pre_rho = apply(pre, rho; alg=algo, cutoff=cut_off)
    post_rho = apply(rho, post; alg=algo, cutoff=cut_off)
    
    drho_ = pre_rho + post_rho
    drho_ += apply(apply(L_0 , rho; alg=algo, cutoff=cut_off), L_0d; alg=algo, cutoff=cut_off)

    if deph!= 0
        L_1 = op("Sz",sites[1])
        drho_ += deph*conj(swapprime( apply( L_1, swapprime(conj(apply(L_1, rho; cutoff=cut_off)), 1,0) ; cutoff=cut_off), 1,0))
    end

    return drho_
end 

function cash_karppe_evolve_test2(no_cavs, depha, gamma, dt, t_final)

    cda = [;]
    cad = [;]
    ada = [;]
    ada_pre = Dict()
    ada_post = Dict()

    par_list = [cda, cad, ada, ada_pre, ada_post]
    # cda, cad, ada, ada_pre, ada_post = init_0(cda, cad, ada, ada_pre, ada_post)

    println(length(cda))

    g_f = g2(gamma, depha, t_final, dt/2, no_cavs)

    sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
    input_list = repeat(["Ground",],no_cavs+1)
    input_list[1] = "Excite1"
    sys = MPO(sys_sites, input_list)

    t = 0.0
    expect_list=[]
    # t_list = [t,]
    par_list = init(par_list, sys_sites, gamma^0.5)
    println(length(cda))
    while t<t_final
        sys_, dt_ = cash_karppe(sys, sys_sites, drho_test2, t, dt; tol=1E-4, p_= (par_list, sys_sites, g_f, sqrt(gamma), depha), max_dt = dt*10)
        t += dt_
        sys = sys_/trace(sys_, sys_sites)
        push!(expect_list, ITensors.expect(sys, sys_sites))
        # push!(t_list, t)
    end

    copy_sys = MPO(no_cavs)
    copy_sys[1] = sys[1] * delta(sys_sites[1], sys_sites[1]') * sys[2]
    for i=2:no_cavs
        copy_sys[i] = deepcopy(sys[i+1]) 
    end
    copy_sys /= trace(copy_sys, sys_sites[2:end])

    return copy_sys, sys_sites[2:end], expect_list
    # , t_list
end

##############################################################################

function pre_term(sites, g_f, rg, dep, t)
    no_cavs = length(sites) - 1
    pre = OpSum()
    pre += (rg^2),"A",1,"Adag",1
    pre += dep,"I",1
    for i=1:no_cavs
        pre += (2*rg*g_f[i](t)),"A",1,"Adag",i+1
        pre += (g_f[i](t)^2),"Adag",i+1,"A",i+1
        for j=1:i-1
            pre += (2*g_f[j](t)*g_f[i](t)),"Adag",i+1,"A",j+1
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
        post += (g_f[i](t)^2),"Adag",i+1,"A",i+1
        for j=i+1:no_cavs
            post += (2*g_f[j](t)*g_f[i](t)),"Adag",i+1, "A",j+1
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
function drho_test(rho, p, t)
    #drho(sites, rho, t)
    #(H_int * rho) + (rho * H_int) + sum(Ld rho L)
    sites, gv_f, rg, deph = p
    cut_off=1E-10
    algo="naive"
    no_cavs = length(sites) - 1


    pre = pre_term(sites, gv_f, rg, deph, t)
    post = post_term(sites, gv_f, rg, deph, t)
    L_0, L_0d = Lo(sites, gv_f, rg, t)

    pre_rho = apply(pre, rho; alg=algo, cutoff=cut_off)
    post_rho = apply(rho, post; alg=algo, cutoff=cut_off)
    
    drho_ = pre_rho + post_rho
    drho_ += apply(apply(L_0 , rho; alg=algo, cutoff=cut_off), L_0d; alg=algo, cutoff=cut_off)

    if deph!= 0
        L_1 = op("Sz",sites[1])
        drho_ += deph*conj(swapprime( apply( L_1, swapprime(conj(apply(L_1, rho; cutoff=cut_off)), 1,0) ; cutoff=cut_off), 1,0))
    end

    return drho_
end

function cash_karppe_evolve_test(no_cavs, depha, gamma, dt, t_final)
    g_f, eigenvals = g2(gamma, depha, t_final, dt/5, no_cavs)

    sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
    input_list = repeat(["Ground",],no_cavs+1)
    input_list[1] = "Excite1"
    sys = MPO(sys_sites, input_list)

    t = 0.0
    expect_list=[]
    # t_list = [t,]

    while t<t_final
        sys_, dt_ = cash_karppe(sys, sys_sites, drho_test, t, dt; tol=1E-4, p_= (sys_sites, g_f, sqrt(gamma), depha), max_dt = dt*10)
        t += dt_
        sys = sys_/trace(sys_, sys_sites)
        # push!(expect_list, ITensors.expect(sys, sys_sites))
        # push!(t_list, t)
    end

    copy_sys = MPO(no_cavs)
    copy_sys[1] = sys[1] * delta(sys_sites[1], sys_sites[1]') * sys[2]
    for i=2:no_cavs
        copy_sys[i] = deepcopy(sys[i+1]) 
    end
    copy_sys /= trace(copy_sys, sys_sites[2:end])

    return copy_sys, sys_sites[2:end], eigenvals
    # , t_list
end

end