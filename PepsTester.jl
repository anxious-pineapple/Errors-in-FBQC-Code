using ITensors
using .MPOFuncs
using Plots
using BenchmarkTools
using JLD2

gamma = 1.0
no_cavs = 10
dt = 0.01
t_final = 10.0
dep = 0.0

prob_list = [;]
fidel_list = [;]
stabxx_list = [;]
stabzz_list = [;]

function peps_expect(peps_array, peps_sites)
    # here assuming list, to write for whole array
    exp_array = Array{Float64}(undef, size(peps))
    for i in eachindex(peps_sites)
        println(i)
        peps_deep = deepcopy(peps_array)
        peps_deep[i] = op("n", peps_sites[i])' * peps_deep[i]
        peps_deep[i] = replaceprime(peps_deep[i], 2=>1)
        exp_array[i] = peps_trace(peps_deep, peps_sites)
    end
    return exp_array
end

function peps_trace(peps_array, peps_sites)
    #with the assumption its only a 2 layer peps
    trace_val = 1.0
    len = length(peps_sites[1,:])
    for i=1:len
        a = peps_array[1,i] * delta(peps_sites[1,i], peps_sites[1,i]')
        b = peps_array[2,i] * delta(peps_sites[2,i], peps_sites[2,i]')
        c = a * b
        @show (length(inds(c)) > 4) 
        trace_val = trace_val * c
        @show inds(trace_val)
    end
    return real(trace_val[1])
end

function peps_flatten!(peps_array, peps_sites; layer=1)
    #with the assumption its only a 2 layer peps
    # if not layer 1 or 2 , flatten peps for both layers

    len = length(peps_sites[1,:])
    if layer in [1,2]
        for i=1:len-1
            ind4 = commoninds(peps_array[layer,i], peps_array[layer,i+1])
            comb4 = combiner(ind4)
            peps_array[layer,i] *= comb4
            peps_array[layer,i+1] *= comb4
        end
    else
        for i=1:len-1
            ind4 = commoninds(peps_array[1,i], peps_array[1,i+1])
            ind5 = commoninds(peps_array[2,i], peps_array[2,i+1])
            comb4 = combiner(ind4)
            comb5 = combiner(ind5)
            peps_array[1,i] *= comb4
            peps_array[1,i+1] *= comb4
            peps_array[2,i] *= comb5
            peps_array[2,i+1] *= comb5
        end
    end
    return nothing
end

function peps_zipper(signal_mpo, ancilla_mpo, signal_sites, ancilla_sites)
    len = length(signal_sites)
    peps = Array{ITensor}(undef, 2, len)
    peps_sites = Array{Index{Int64}}(undef, 2, Int(len))
    for i=1:len
        peps[1, i] = signal_mpo[i]
        peps[2, i] = ancilla_mpo[i]
        peps_sites[1, i] = signal_sites[i]
        peps_sites[2, i] = ancilla_sites[i]
    end
    return peps, peps_sites
end

function beamsplitter_peps_tensor_linear!(peps, peps_sites, layer, site_i, site_j)
    # here applying itensor first, svd later
    # wuthin layer beamsplitters, max 4 distance
    # site i and site j are the number of the sites and not the actual site objects
    # bs_list = Array{ITensor}(undef, site_j-site_i+1)
    bs_list = [;]
    # @show length(bs_list)
    g = pi/((8)^0.5)
    bs_op = ((op("A",peps_sites[layer,site_i]) * op("Adag", peps_sites[layer, site_j])) + (op("A",peps_sites[layer,site_j]) * op("Adag",peps_sites[layer,site_i])))
    bs_op +=  (op("I",peps_sites[layer,site_j])*(op("N", peps_sites[layer, site_i])) - (op("N", peps_sites[layer, site_j]))*op("I",peps_sites[layer,site_i]))
    bs_op += -(2)^0.5 * (op("I",peps_sites[layer,site_j])*(op("N", peps_sites[layer, site_i])) + (op("N", peps_sites[layer, site_j]))*op("I",peps_sites[layer,site_i]))
    bs_op = exp((-im * g) * bs_op)

    
    
    # bs_op = exp((-im/4) * pi * bs_op)
    for i=site_i+1:site_j-1
        bs_op = bs_op * op("I", peps_sites[layer,i])
    end

    for i=site_i:site_j-1
        inds3 = uniqueinds(peps[layer, i], peps[layer, i+1])
        i == site_i ? nothing : setdiff!(inds3, commoninds(peps[layer, i], peps[layer, i-1]))
        i == site_i ? nothing : append!(inds3 , commoninds(bs_list[end], bs_op))
        u,s,v = svd(bs_op, inds3 ; cutoff = 1e-5)
        push!(bs_list, u)
        bs_op = s*v
        # println(diag(Array(s, inds(s))))
    end
    push!(bs_list, bs_op)
    # @show length(bs_list)
    for i=site_i:site_j
        # peps[layer, i] 
        temp_tens = bs_list[i-site_i+1]' * peps[layer, i]
        temp_tens = replaceprime(temp_tens, 2=>1)
        #below line is cause problemm
        temp_tens = temp_tens' * swapprime(conj(bs_list[i-site_i+1]), 1, 0)
        temp_tens = replaceprime(temp_tens, 1=>0)
        # @show inds(temp_tens)
        peps[layer, i] = replaceprime(temp_tens, 2=>1)
    # commoninds(peps[layer, 1], peps[layer, 2])
    end
    peps_flatten!(peps, peps_sites; layer=layer)
    return nothing
end

function beamsplitter_peps_tensor_nonlinear!(peps, peps_sites, layer, site_i, site_j)
    # here applying itensor first, svd later
    # wuthin layer beamsplitters, max 4 distance
    # site i and site j are the number of the sites and not the actual site objects
    # bs_list = Array{ITensor}(undef, site_j-site_i+1)
    bs_list = [;]
    # @show length(bs_list)
    g = pi/((8)^0.5)
    bs_op = ((op("A",peps_sites[layer,site_i]) * op("Adag", peps_sites[layer, site_j])) + (op("A",peps_sites[layer,site_j]) * op("Adag",peps_sites[layer,site_i])))
    bs_op +=  (op("I",peps_sites[layer,site_j])*(op("N", peps_sites[layer, site_i])) - (op("N", peps_sites[layer, site_j]))*op("I",peps_sites[layer,site_i]))
    bs_op += -(2)^0.5 * (op("I",peps_sites[layer,site_j])*(op("N", peps_sites[layer, site_i])) + (op("N", peps_sites[layer, site_j]))*op("I",peps_sites[layer,site_i]))
    bs_op = exp((-im * g) * bs_op)

    
    # bs_op = exp((-im/4) * pi * bs_op)

    i = site_i
    inds3 = uniqueinds(peps[layer, i], peps[layer, site_j])
    # i == site_i ? nothing : setdiff!(inds3, commoninds(peps[layer, i], peps[layer, i-1]))
    # i == site_i ? nothing : append!(inds3 , commoninds(bs_list[end], bs_op))
    u,s,v = svd(bs_op, inds3 ; cutoff = 1e-5)
    push!(bs_list, u)
    bs_op = s*v
    # println(diag(Array(s, inds(s))))
    push!(bs_list, bs_op)
    # @show length(bs_list)
    # for i in [site_i,site_j]
        # peps[layer, i] 
    temp_tens = bs_list[1]' * peps[layer, site_i]
    temp_tens = replaceprime(temp_tens, 2=>1)
    temp_tens = temp_tens' * swapprime(conj(bs_list[1]), 1, 0)
    temp_tens = replaceprime(temp_tens, 1=>0)
    peps[layer, site_i] = replaceprime(temp_tens, 2=>1)

    #site_j
    temp_tens = bs_list[2]' * peps[layer, site_j]
    temp_tens = replaceprime(temp_tens, 2=>1)
    #below line is cause problemm
    temp_tens = temp_tens' * swapprime(conj(bs_list[2]), 1, 0)
    temp_tens = replaceprime(temp_tens, 1=>0)
    # @show inds(temp_tens)
    peps[layer, site_j] = replaceprime(temp_tens, 2=>1)
    comb = combiner(commoninds(peps[layer, site_i], peps[layer, site_j]))
    peps[layer,site_i] *= comb
    peps[layer,site_j] *= comb
    # end
    # peps_flatten(peps, peps_sites; layer=layer)
    return nothing
end

function detect_1100_anymode( site_list )
    #returns an MPO that detects 1100 in any mode
    no_cavs = Int(length(site_list) / 4)
    mps_list = [;]
    loc_1 = 1
    loc_2 = 2
    proj_list = repeat(["Ground",],length(site_list))
    for i=0:no_cavs-1
        for j=0:no_cavs-1
            # println(i,j)
            proj_list[4*i + loc_1] = "Excite1"
            proj_list[4*j + loc_2] = "Excite1"
            push!(mps_list, MPO(site_list, proj_list))
            proj_list[4*i + loc_1] = "Ground"
            proj_list[4*j + loc_2] = "Ground"
        end
    end
    all_mps = sum(mps_list)
    return all_mps
end

function peps_project_out!(peps_array, peps_sites, project_mpo, layer)
    # assuming 2 layer peps
    # assuming we're projecting out using an MPO (not MPS)
    # output is the peps mum in layer specified
    # also assumes length of peps_array is 4*no_cavs
    comb3 = 1
    no_cavs =  Int(length(peps_array[1,:])/4)
    if project_mpo == 0
        for i=1:(no_cavs*4)
            peps_array[layer,i] = peps_array[layer,i] * delta(peps_sites[layer,i], peps_sites[layer,i]') 
        end
    else 
        for i=1:(no_cavs*4)
            a = peps_array[layer,i] * project_mpo[i] 
            a *= comb3
            if i==(no_cavs*4)
                nothing
            else
                comb3_1 = commonind(a, peps_array[layer,i+1])
                comb3_2 = commonind(a, project_mpo[i+1])
                comb3 = combiner([comb3_1, comb3_2])
                a *= comb3
            end
            peps_array[layer,i] = a
        end
    end
    return nothing
end

function peps_trace_mum(peps_array)
    # assuming 2 layer peps
    # also assumed site inds have been traced out
    before_node = 1
    no_cavs =  Int(length(peps_array[1,:])/4)
    for i=0:no_cavs-1
        println(i)
        i1, i4 = 4i+1, 4i+4
        sequ = ITensors.optimal_contraction_sequence([peps_array[1,i1:i4];peps_array[2,i1:i4]])
        block = contract([peps_array[1,i1:i4];peps_array[2,i1:i4]]; sequence=sequ)
        before_node *= block 
    end
    return before_node
end

function ideal_bell(sites_array)
    # Making ideal bell state
    proj_list = repeat([1,],length(sites_array))
    proj_list[1] = 2
    proj_list[3] = 2
    term_1 = MPS(sites_array, proj_list)
    proj_list = repeat([1,],length(sites_array))
    proj_list[2] = 2
    proj_list[4] = 2
    term_2 = MPS(sites_array, proj_list)
    bell_mps = (term_1 - term_2) / sqrt(2)
    bell_mpo = [;]
    for i in bell_mps
        push!(bell_mpo, i * i')
    end
    for i =1:length(sites_array)-1
        comb_bell = combiner(commoninds(bell_mpo[i], bell_mpo[i+1]))
        bell_mpo[i] *= comb_bell
        bell_mpo[i+1] *= comb_bell
    end
    return bell_mpo
end

function stabalizer_gen(site_array)
    
    no_cavs = Int(length(site_array)/4)

    X_op12 = OpSum()
    X_op34 = OpSum()
    Z_op12 = OpSum()
    Z_op34 = OpSum()

    for i=0:no_cavs-1
        #
        X_op12 += "A",4i+1,"Adag",4i+2
        X_op12 += "Adag",4i+1,"A",4i+2
        #
        X_op34 += "A",4i+3,"Adag",4i+4
        X_op34 += "Adag",4i+3,"A",4i+4
        #
        Z_op12 += "Adag",4i+1,"A",4i+1
        Z_op12 -= "Adag",4i+2,"A",4i+2
        #
        Z_op34 += "Adag",4i+3,"A",4i+3
        Z_op34 -= "Adag",4i+4,"A",4i+4
    end

    X_op12 = MPO(X_op12, site_array)
    X_op34 = MPO(X_op34, site_array)
    Z_op12 = MPO(Z_op12, site_array)
    Z_op34 = MPO(Z_op34, site_array)

    # Stab_1 = apply(X_op12 , Z_op34)
    # Stab_2 = apply(Z_op12 , X_op34)

    Stab_3 = -1 * apply(X_op12 , X_op34)
    Stab_4 = apply(Z_op12 , Z_op34)

    return Stab_3, Stab_4
end


for dep in [0.02, ]
    println("dep is  ",dep)
    mpo_i, sites_i, eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)
    # sites_i = ITensors.siteinds("Qudit", no_cavs; dim=5)
    # input_list = repeat(["Ground",],no_cavs)
    # input_list[1] = "Excite1" 
    # mpo_i = MPO(sites_i, input_list)
    signal_mpo, signal_sites = MPOFuncs.n_copy_mpo(mpo_i, sites_i, 4)
    # below bloack rearranges MPO to keep all similar modes together
    # in essence we only have to deal with 4 sites of a mode at a time
    for i=4:-1:1
        for j=no_cavs:-1:1
            signal_mpo = MPOFuncs.swap_ij!(signal_mpo, signal_sites, (no_cavs*(i-1))+j, (no_cavs*(i-1))+j + ((4-i)*(j-1)))
        end
    end
    ancilla_sites = siteinds("Qudit", length(signal_mpo), dim=5)
    ancilla_mpo = MPO(ancilla_sites, repeat(["Ground",],length(signal_mpo)))

    peps, peps_sites = peps_zipper(signal_mpo, ancilla_mpo, signal_sites, ancilla_sites)
    println("peps state 0")
    # peps_expect(peps, peps_sites)
    # peps_trace(peps, peps_sites)

    # Below block applies beamsplitter across signal and ancilla sites
    for i=1:length(signal_mpo)
        peps[1,i] , peps[2,i] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i], peps[2,i], peps_sites[1,i], peps_sites[2,i])
    end

    #Below block applies the fourway beamsplitter
    for i=0:no_cavs-1
        println(i)
        i1, i2, i3, i4 = 4i+1, 4i+2, 4i+3, 4i+4
        beamsplitter_peps_tensor_linear!(peps, peps_sites, 2, i1, i2)
        beamsplitter_peps_tensor_linear!(peps, peps_sites, 2, i3, i4)
        beamsplitter_peps_tensor_nonlinear!(peps, peps_sites, 2, i1,i3)
        beamsplitter_peps_tensor_nonlinear!(peps, peps_sites, 2, i2, i4)
    end
    println("peps state 1")
    # add project out layer 2 on particular MPO
    detect_mpo = detect_1100_anymode(peps_sites[2,:])
    peps_project_out!(peps, peps_sites, detect_mpo, 2)

    println("peps state end")
    jldsave("MPOFuncs/Data/peps_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2" ; peps)
    jldsave("MPOFuncs/Data/peps_sites_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2" ; peps_sites)
end

#comment 
    # peps_copy = deepcopy(peps)
    # peps_project_out!(peps_copy, peps_sites, 0, 1)
    # probab_after_detec = peps_trace_mum(peps_copy)[1]
    # push!(prob_list, probab_after_detec)

    # peps_copy = deepcopy(peps)
    # ideal_bell_mpo = ideal_bell(peps_sites[1,:])
    # peps_project_out!(peps_copy, peps_sites, ideal_bell_mpo, 1)
    # fidel = (peps_trace_mum(peps_copy)[1])/probab_after_detec
    # push!(fidel_list, fidel)

    # stabxx, stabzz = stabalizer_gen(peps_sites[1,:])
    # peps_copy = deepcopy(peps)
    # peps_project_out!(peps_copy, peps_sites, stabxx, 1)
    # stabxx_measure = (peps_trace_mum(peps_copy)[1])/probab_after_detec
    # push!(stabxx_list, stabxx_measure)
    # #
    # peps_copy = deepcopy(peps)
    # peps_project_out!(peps_copy, peps_sites, stabzz, 1)
    # stabzz_measure = (peps_trace_mum(peps_copy)[1])/probab_after_detec
    # push!(stabzz_list, stabzz_measure)
#

for dep in [0, 0.005, 0.01, 0.02]
    println("dep is  ",dep)
    f = jldopen("MPOFuncs/Data/peps_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps = f["peps"]
    close(f)
    f = jldopen("MPOFuncs/Data/peps_sites_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)

    # stabxx, stabzz = stabalizer_gen(peps_sites[1,:])
    println("peps state 0")
    peps_project_out!(peps, peps_sites, 0, 1)
    println("peps state 1")
    probab_after_detec = peps_trace_mum(peps)[1]
    println("peps state 2")
    push!(prob_list, probab_after_detec)
end

for dep in [0.005, ]
    println("dep is  ",dep)
    # println("MPOFuncs/Data/peps_sites_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2")
    f = load("MPOFuncs/Data/peps_sites_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "peps_sites")
    peps_sites = f
    # close(f)

    carryover = 1

    for i=1:no_cavs
        println("no_cavs is  ",i)   
        f = jldopen("MPOFuncs/Data/peps_part" * string(i) * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
        peps_part = f["peps_part"]
        close(f)

        t1 = time()
        for j=1:4
            peps_part[1, j] *= delta(peps_sites[1, 4*(i-1)+j], peps_sites[1, 4*(i-1)+j]')
        end
        println("Contraction took ", time()-t1)
        t1 = time()
        sequ = ITensors.optimal_contraction_sequence([peps_part[1,1:4];peps_part[2,1:4]])
        println("Finding optimal took ", time()-t1)
        t1 = time()
        carryover *= contract([peps_part[1,1:4];peps_part[2,1:4]]; sequence=sequ)
        println("Contraction took ", time()-t1)
    end
    push!(prob_list, carryover)
end

for dep in [ 0.005, ]
    f = jldopen("MPOFuncs/Data/peps_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps = f["peps"]
    close(f)
    f = jldopen("MPOFuncs/Data/peps_sites_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)

    ideal_bell_mpo = ideal_bell(peps_sites[1,:])
    peps_project_out!(peps, peps_sites, ideal_bell_mpo, 1)
    fidel = (peps_trace_mum(peps)[1])
    push!(fidel_list, fidel)
end



for dep in [0, 0.005, 0.01, 0.02]
    f = jldopen("MPOFuncs/Data/peps_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps = f["peps"]
    close(f)
    f = jldopen("MPOFuncs/Data/peps_sites_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)

    stabxx, stabzz = stabalizer_gen(peps_sites[1,:])
    peps_project_out!(peps, peps_sites, stabxx, 1)
    stabxx_measure = (peps_trace_mum(peps)[1])
    push!(stabxx_list, stabxx_measure)
end
# println("This xx took total ", time()-t)
# plot(real(stabxx_list))

# t = time()

for dep in [0, 0.005, 0.01, 0.02]
    f = jldopen("MPOFuncs/Data/peps_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps = f["peps"]
    close(f)
    f = jldopen("MPOFuncs/Data/peps_sites_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)

    stabxx, stabzz = stabalizer_gen(peps_sites[1,:])
    peps_project_out!(peps, peps_sites, stabzz, 1)
    stabzz_measure = (peps_trace_mum(peps)[1])
    push!(stabzz_list, stabzz_measure)
end
# println("This zz took total ", time()-t)
plot(real(stabzz_list))



plot(real(prob_list))
plot!(real(prob_list), seriestype="scatter", label="Probability")

plot(real(fidel_list./prob_list))
plot!(32 .*real(fidel_list))
plot!(real(fidel_list./prob_list), seriestype="scatter", label="Fidelity")

plot(real(stabxx_list./prob_list))
plot!(real(stabxx_list./prob_list), seriestype="scatter", label="Stab XX")
plot!(real(stabzz_list./prob_list))
plot!(real(stabzz_list./prob_list), seriestype="scatter", label="Stab ZZ")
# stabzz_list
plot!(0 .* real(prob_list))
100 .* (1 .- (32 .* real(prob_list)))
# ######################### this is to show correct matrix is produced form hamiltonian ###########################
    # layer = 1
    # site_i = 1
    # site_j = 2
    # g = pi/((8)^0.5)
    # bs_op = ((op("A",peps_sites[layer,site_i]) * op("Adag", peps_sites[layer, site_j])) + (op("A",peps_sites[layer,site_j]) * op("Adag",peps_sites[layer,site_i])))
    # bs_op +=  (op("I",peps_sites[layer,site_j])*(op("N", peps_sites[layer, site_i])) - (op("N", peps_sites[layer, site_j]))*op("I",peps_sites[layer,site_i]))
    # bs_op += -(2)^0.5 * (op("I",peps_sites[layer,site_j])*(op("N", peps_sites[layer, site_i])) + (op("N", peps_sites[layer, site_j]))*op("I",peps_sites[layer,site_i]))
    # bs_op = exp((-im * g) * bs_op)

    # comb = combiner(peps_sites[layer, site_i], peps_sites[layer, site_j])
    # comb2 = combiner(peps_sites[layer, site_i]', peps_sites[layer, site_j]') 
    # bs_op = bs_op * comb * comb2
    # bs_op
    # real.(Array(bs_op, inds(bs_op)))

# a = [2,3,3,2,2,2]
# jldsave("MPOFuncs/Data/peps_test.jld2" ; a)

dep = 0.02
mpo_i, sites_i, eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)
sum(eigenvals)
ex = expect(mpo_i, sites_i)

1-sum(ex)