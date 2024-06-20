#Importing necessary Packages
using ITensors
using JLD2
using Plots
using QuantumOptics
using LinearAlgebra
using Interpolations
using DifferentialEquations

##  Functions

#settign up states

function ITensors.op(::OpName"Ground" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[1,1] = 1
    return mat
end

function ITensors.op(::OpName"Excite1" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[2,2] = 1
    return mat
end

function ITensors.op(::OpName"Sz" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[1,1] = 1
    mat[2,2] = -1
    return mat
end

function beamsplitter!(MPO_i, index_1, index_2, site_list; cutoff=1E-10)
    op_1 = ((op("A",site_list[index_1]) * op("Adag",site_list[index_2])) + (op("A",site_list[index_2]) * op("Adag",site_list[index_1])))
    H_ = exp((-im/4) * pi * op_1)
    H2 = apply(H_, MPO_i; cutoff=cutoff)
    #easier to use apply since preserves MPO data type and doesnt change to iTensor type
    #requires iTensor be applied to an MPO in that order particularly hence double dagger
    H3 = conj(swapprime( apply( H_, swapprime(conj(H2), 1,0) ; cutoff=cutoff), 1,0))
    MPO_i[:] = H3
    MPO_i /= trace(MPO_i, site_list)
    return nothing
end

function n_copy_mpo(n, mpo_i, sites)
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
    I_mpo /= trace(I_mpo, new_inds)
    return I_mpo, new_inds
end


#func calc drho/dt
function drho(sites, rho, gv_f, rg; t=0, deph = 0, cut_off=1E-10, algo="naive")

    #drho(sites, rho, t)
    #(H_int * rho) + (rho * H_int) + sum(Ld rho L)
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


function trace(MPO_i, sites_i)
    tr_ = 1.0
    for i in 1:length(sites_i)
        tr_ *= delta(sites_i[i], sites_i[i]') * MPO_i[i]
    end
    return tr_[1]
end

function ITensors.expect(MPO_i, sites_i)
    list_exp = []
    for i in 1:length(sites_i)
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
            G_34 += 0.5 * term
        end
        denom1 += measure_a(mpo_i, sites, i)
        denom2 += measure_a(mpo_i, sites, no_cavs+i)
    end
    denom =0.5 * denom1 * denom2

    return G_34/denom
end


function create_MPO(no_cavs, depha, gamma, dt, t_final)

    #creating initial state and sites
    sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
    input_list = repeat(["Ground",],no_cavs+1)
    input_list[1] = "Excite1"
    sys = MPO(sys_sites, input_list)

    #Generating G2
    #gamma = 1
    g_f, eval_ = g2(gamma, depha, t_final, dt/2, no_cavs)
    println("genfunctions")

    for i=0:dt:t_final
        d_rho1 = drho(sys_sites, sys, g_f, sqrt(gamma); t=i, deph=depha)
        sys1 = sys + (d_rho1*dt/2)
        #sys1 /=tr(sys1)
        d_rho2 = drho(sys_sites, sys1, g_f, sqrt(gamma); t=(i+(dt/2)), deph=depha)
        sys2 = sys + (d_rho2*dt/2)
        #sys2 /=tr(sys2)
        d_rho3  = drho(sys_sites, sys2, g_f, sqrt(gamma); t=(i+(dt/2)), deph=depha)
        sys3 = sys + (d_rho3*dt)
        #sys3 /=tr(sys3)
        d_rho4  = drho(sys_sites, sys3, g_f, sqrt(gamma); t=(i+dt), deph=depha)
        sys +=  (dt * (d_rho1 + (2*d_rho2) + (2*d_rho3) + d_rho4)/6)
        sys /= trace(sys, sys_sites)

    end
    
    #trace out atom
    copy_sys = MPO(no_cavs)
    copy_sys[1] = sys[1] * delta(sys_sites[1], sys_sites[1]') * sys[2]
    for i=2:no_cavs
        copy_sys[i] = deepcopy(sys[i+1]) 
    end
    copy_sys /= trace(copy_sys, sys_sites[2:end])

    return copy_sys, sys_sites[2:end]
end

#make a g2 function, outputs the cavity functions
function g2(gamma, deph, t_fin, dt, no_cavs)

    
    basis = SpinBasis(1//2)
    a = sigmam(basis)
    at = sigmap(basis)
    H =  identityoperator(basis)
    J = [sqrt(gamma)*a, sqrt(deph)*sigmaz(basis)]

    t_list = [0:dt:t_fin;]
    t_size = length(t_list)
    corel_m = Array{ComplexF64}(undef, t_size, t_size)
    corel_m *= 0
    
    ρ₀ = spinup(basis) ⊗ dagger(spinup(basis))
    
    tout, ρt_master = timeevolution.master(t_list, ρ₀, H, J; dt=dt)
    for i=1:t_size-1
        rhot = ρt_master[i]
        corel_m[i,i:end] = timecorrelations.correlation(t_list[1:(t_size-i+1)], rhot, H, J, at, a)
    end
    corel_m[t_size,t_size] = 0.0
    corel_m = -diagm((diag(corel_m))) + corel_m + conj(corel_m)'
    corel_m = real(corel_m)
    eigenvals, eigens = eigen(corel_m; sortby=-)

    v_0 = [cubic_spline_interpolation(0:dt:t_fin, eigens[:,i]; extrapolation_bc=0) for i in 1:no_cavs]
    
    gv = Dict()
    alpha = Dict()
    gv_f = [;]
    for i=1:Int(no_cavs)
        for j=1:Int(i-1)
            a0 = 0
            function a_(a, p, t)
                da = -(v_0[i](t)*gv_f[j](t))-(a * 0.5 * gv_f[j](t)^2)
                for k=1:j-1
                    da -= gv_f[j](t) * gv_f[k](t) * alpha[[i,k]](t)
                end
                return da
            end
            tspan = (0.0, t_fin)
            prob = ODEProblem(a_, a0, tspan)
            sol = solve(prob)
            alpha[[i,j]] = interpolate(sol.t,sol.u,FritschCarlsonMonotonicInterpolation())
        end
        v_i = deepcopy(v_0[i])
        for k=1:Int(i-1)
            v_i += (gv_f[k].(0:dt:t_fin) .* alpha[[i,k]].(0:dt:t_fin))
        end
        norm = dt * cumsum(v_i.^2)
        
        gv[i] = -v_i./sqrt.(norm)
        #i > 1 ? gv[i][1] = 0 : nothing
        extrap = cubic_spline_interpolation(0:dt:t_fin, gv[i]; extrapolation_bc=0)
        push!(gv_f, extrap)
    end
    
    return gv_f, eigenvals
end



Threads.nthreads()

# running the loop
g34_list5 =  [;]
gamma = 1
no_cavs = 5
dt = 0.02
t_final = 10
for dep=0.0025:0.0025:0.0025
    println("deph: ",dep)
    mpo, sites = create_MPO(no_cavs, dep, gamma, dt, t_final)
    double, double_sites = n_copy_mpo(2, mpo, sites)
    for i=1:no_cavs
        beamsplitter!(double, i, i+no_cavs, double_sites)
    end
    push!(g34_list5, g_34(double, double_sites, no_cavs))
end


plot([0,0.0025,],[0,real(g34_list5[1])])

f = jldopen("g5.jld2", "r")
g34_list5_old = f["g34_list5"]
close(f)
f = jldopen("g8.jld2", "r")
g34_list8 = f["g34_list8"]
close(f)
plot([0:0.0025:0.03;][1:11], [real(g34_list8)]; seriestype="scatter", label="Simulated 8")
plot!([0:0.0025:0.03;][1:11], [real(g34_list8)]; label="")

plot!([0:0.0025:0.03;], [real(g34_list5)]; seriestype="scatter", label="Simulated 5")
plot!([0:0.0025:0.03;], [real(g34_list5)]; label="")
plot!([0:0.0025:0.02;], [real(g34_list10)]; seriestype="scatter", label="Simulated 10")
plot!([0:0.0025:0.02;], [real(g34_list10)]; label="")

#plot!([0:0.0025:0.02;], [real(g34_list2)], label="")
plot!([0:0.0025:0.03;], [2*i/(1+4*i) for i=0:0.0025:0.03], label="ideal")
xlabel!("Dephasing")
ylabel!("Distinguishability")

jldsave("g5_pt2.jld2"; g34_list5)

op("Excite1", siteinds("Qudit", 11, dim=3)[2])

no_cavs = 3
sys_sites = siteinds("Qudit", no_cavs+1, dim=3)
input_list = repeat(["Ground",],no_cavs+1)
input_list[1] = "Excite1"
sys = MPO(sys_sites, input_list)