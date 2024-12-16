using ITensors
using .MPOFuncs
using Plots
using BenchmarkTools
using JLD2

gamma = 1.0
no_cavs = 5
dt = 0.01
t_final = 10.0
dep = 0.014

prob_list = Dict()
fidel_list = Dict()
stabxx_list = Dict()
stabzz_list = Dict()
stabyy_list = Dict()
err_IX_list = Dict()
err_XI_list = Dict()
err_IZ_list = Dict()
err_ZI_list = Dict()
err_XZ_list = Dict()
err_ZX_list = Dict()


function peps_expect(peps_array, peps_sites)
    # here assuming list, to write for whole array
    exp_array = Array{Float64}(undef, size(peps_array))
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

    len = length(peps_array[1,:])
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


    # peps_flatten!(peps, peps_sites; layer=layer)

    comb = combiner(commoninds(peps[layer, site_i], peps[layer, site_j]))
    peps[layer,site_i] *= comb
    peps[layer,site_j] *= comb

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
    # inds3 = uniqueinds(peps[layer, i], peps[layer, site_j])
    inds3 = [peps_sites[layer, site_i], peps_sites[layer, site_i]']
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

function error_bell( err_type, sites_array)
    bell_mpo = ideal_bell(sites_array)
    X12, X34, Z12, Z34 = op_gen(sites_array)
    if err_type[1] == 'X' && err_type[2] == 'X'
        err_mpo = apply(X12, X34)
    elseif err_type[1] == 'Z' && err_type[2] == 'Z'
        err_mpo = apply(Z12, Z34)
    elseif err_type[1] == 'X' && err_type[2] == 'Z'
        err_mpo = apply(X12, Z34)
    elseif err_type[1] == 'Z' && err_type[2] == 'X'
        err_mpo = apply(Z12, X34)
    elseif err_type[1] == 'I' && err_type[2] == 'X'
        err_mpo = X34
    elseif err_type[1] == 'X' && err_type[2] == 'I'
        err_mpo = X12
    elseif err_type[1] == 'I' && err_type[2] == 'Z'
        err_mpo = Z34
    elseif err_type[1] == 'Z' && err_type[2] == 'I'
        err_mpo = Z12
    end
    comm_ind = 1
    for i=1:length(sites_array)
        temp_term = err_mpo[i]' * bell_mpo[i]
        temp_term = replaceprime(temp_term, 2=>1)
        temp_term = temp_term' * err_mpo[i]
        temp_term = replaceprime(temp_term, 1=>0)
        temp_term = replaceprime(temp_term, 2=>1)
        temp_term *= comm_ind
        if i == length(sites_array)
            nothing
        else
            comm_ind1 = commonind(bell_mpo[i], bell_mpo[i+1])
            comm_ind2 = commonind(err_mpo[i], err_mpo[i+1])
            comm_ind = combiner([comm_ind1, comm_ind2, comm_ind2'])
            temp_term *= comm_ind
        end
        bell_mpo[i] = temp_term
    end
    # bell_mpo = apply( err_mpo, apply(bell_mpo, err_mpo))
    return bell_mpo
end

function op_gen(site_array)

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

    return X_op12, X_op34, Z_op12, Z_op34
end

function stabalizer_gen(site_array)

    X_op12, X_op34, Z_op12, Z_op34 = op_gen(site_array)
    Stab_3 = -1 * apply(X_op12 , X_op34)
    Stab_4 = -1 * apply(Z_op12 , Z_op34)
    Stab_5 = apply( apply(Z_op12, X_op12) , apply(Z_op34, X_op34) )

    return Stab_3, Stab_4, Stab_5
end

dep_list = [0:0.0025:0.02;]

for dep in dep_list
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
            println(i,j)
            signal_mpo = MPOFuncs.swap_ij!(signal_mpo, signal_sites, (no_cavs*(i-1))+j, (no_cavs*(i-1))+j + ((4-i)*(j-1)))
        end
    end
    ancilla_sites = siteinds("Qudit", length(signal_mpo), dim=5)
    ancilla_mpo = MPO(ancilla_sites, repeat(["Ground",],length(signal_mpo)))

    peps, peps_sites = peps_zipper(signal_mpo, ancilla_mpo, signal_sites, ancilla_sites)
    println("peps state 0")

    detect_mpo = detect_1100_anymode(peps_sites[2,:])
    comb3 = 1

    jldsave("Data/peps_sites_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2" ; peps_sites)

    for i=0:no_cavs-1

        ti = time()
        println("at cav no ", i+1)

        i1, i2, i3, i4 = 4i+1, 4i+2, 4i+3, 4i+4
        println(i1, i2, i3, i4)

        peps[1,i1] , peps[2,i1] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i1] , peps[2,i1], peps_sites[1,i1], peps_sites[2,i1])
        peps[1,i2] , peps[2,i2] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i2] , peps[2,i2], peps_sites[1,i2], peps_sites[2,i2])
        peps[1,i3] , peps[2,i3] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i3] , peps[2,i3], peps_sites[1,i3], peps_sites[2,i3])
        peps[1,i4] , peps[2,i4] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i4] , peps[2,i4], peps_sites[1,i4], peps_sites[2,i4])

        beamsplitter_peps_tensor_linear!(peps, peps_sites, 2, i1, i2)
        beamsplitter_peps_tensor_linear!(peps, peps_sites, 2, i3, i4)
        beamsplitter_peps_tensor_nonlinear!(peps, peps_sites, 2, i1, i3)
        beamsplitter_peps_tensor_nonlinear!(peps, peps_sites, 2, i2, i4)

        println("peps state 1 | time ", time()-ti)
        ti = time()
        # comb3 = 1
        for j=1:4
            ind = 4i + j
            a = peps[2,ind] * detect_mpo[ind] 
            a *= comb3
            if ind==(no_cavs*4)
                nothing
            else
                comb3_1 = commonind(a, peps[2,ind+1])
                comb3_2 = commonind(a, detect_mpo[ind+1])
                comb3 = combiner([comb3_1, comb3_2])
                a *= comb3
            end
            peps[2,ind] = a
        end

        # # peps[2,2]
        println("peps state 2 | time ", time()-ti)
        ti = time()

        # # finding optimal THEN contract vs JUST contract ?????
        sequ = ITensors.optimal_contraction_sequence(peps[2,i1:i4])
        block = contract(peps[2,i1:i4]; sequence = sequ)  
        # peps_top = peps_subblock[1,i1:i4]   
        
        println("peps state 3 | time ", time()-ti)


        
        save("Data/peps_dep" * string(trunc( Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i+1)) * ".jld2" , "peps_top", peps[1,i1:i4], "bottom", block)
        
        
        for loc in [i1,i2,i3,i4]
            peps[1,loc] = ITensor(0)
            peps[2,loc] = ITensor(0)
        end

    end
end


for dep in dep_list
    println("dep is  ",dep)
    trace_val = 1.0
    f = jldopen("Data/peps_sites_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int,dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        # peps_top = load("Data/peps_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2", "peps_top")
        for j=1:4
            peps_top[j] *= delta(peps_sites[1, 4*(i-1)+j], peps_sites[1, 4*(i-1)+j]')
        end
        # bottom = load("Data/peps_dep" * string(Int(dep*1000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)

        # println("block ", inds(block))
        trace_val *= block

    end
    prob_list[dep] = trace_val[1]

    # fidel below
    trace_val = 1.0
    ideal_bell_mpo = ideal_bell(peps_sites[1,:])
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        peps_top[1] *= trace_val
        for j=1:4
            peps_top[j] *= ideal_bell_mpo[4(i-1) + j]
        end
        # peps_top[1]
        # peps_top
        bottom = load("Data/peps_dep" * string(trunc(Int,dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        # [peps_top ; bottom]
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val = block
    end
    fidel_list[dep] = trace_val[1]
end

for dep in dep_list

    println("dep is  ",dep)

    f = jldopen("Data/peps_sites_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)

    # stabxx below
    stabxx, stabzz, stabyy = stabalizer_gen(peps_sites[1,:])
    # stabzz
    # stabxx = -1 * apply(X_op12 , X_op34)
    # stabzz =  apply(Z_op12 , Z_op34)
    # err_IX = X_op34
    # err_XI = X_op12
    # err_IZ = Z_op34
    # err_ZI = Z_op12
    # err_XZ = apply(X_op12 , Z_op34)
    # err_ZX = apply(Z_op12 , X_op34)



    # stabxx below
    println("stabxx")
    trace_val = 1.0
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= stabxx[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    stabxx_list[dep] = trace_val[1]

    # stabzz below
    println("stabzz")
    trace_val = 1.0
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= stabzz[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    stabzz_list[dep] = trace_val[1]

    # stabyy below
    println("stabyy")
    trace_val = 1.0
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= stabyy[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    stabyy_list[dep] = trace_val[1]

end


pl= [real.(prob_list[i]) for i in dep_list]

plot(dep_list, pl, label="prob_list", seriestype="scatter")
p = plot(dep_list.*100, [real.(fidel_list[i]) for i in dep_list]./pl, size=(800,500))
plot!(dep_list.*100, [real.(fidel_list[i]) for i in dep_list]./pl, seriestype="scatter")
plot!(legend=false)
plot!(xlabel = "Depolarizing rate (%)", ylabel = "Fidelity of Bell State") 
Plots.pdf(p, "bellstate_fidel.pdf")
#  label="fidel_list")
p2 = plot(dep_list.*100, [real.(stabxx_list[i]) for i in dep_list]./pl, label="XX Stab measure")
plot!(dep_list.*100, [real.(stabxx_list[i]) for i in dep_list]./pl, seriestype="scatter", label="")
plot!(dep_list.*100, -[real.(stabzz_list[i]) for i in dep_list]./pl, label="ZZ Stab measure")
plot!(dep_list.*100, -[real.(stabzz_list[i]) for i in dep_list]./pl, seriestype="scatter", label="")
plot!(dep_list.*100, -[real.(stabyy_list[i]) for i in dep_list]./pl, label="YY Stab measure")
plot!(dep_list.*100, -[real.(stabyy_list[i]) for i in dep_list]./pl, seriestype="scatter", label="")
plot!(xlabel = "Depolarizing rate (%)", ylabel = "Stabalizer Measurements", size=(800,500))
Plots.pdf(p2, "bellstate_stabmeasure.pdf")

plot(dep_list, 4 .* [real.(fidel_list[i]) for i in dep_list], label="fidel_list")
plot!(dep_list, pl .+ [real.(stabxx_list[i]) for i in dep_list] .- [real.(stabzz_list[i]) for i in dep_list] .- [real.(stabyy_list[i]) for i in dep_list], label="stab_sum", ylimits=(0.1, 0.125))


#
    dep = 0.01
    println("dep is  ",dep)

    f = jldopen("Data/peps_sites_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)

    # stabxx below
    stabxx, stabzz, stabyy = stabalizer_gen(peps_sites[1,:])
    println(linkdims(stabxx))
    println(linkdims(stabzz))
    println(linkdims(stabyy))
#

for dep in dep_list

    println("dep is  ",dep)
    f = jldopen("Data/peps_sites_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * ".jld2", "r")
    peps_sites = f["peps_sites"]
    close(f)

    # # IX_error below
    trace_val = 1.0
    IX_mpo = error_bell("IX", peps_sites[1,:])
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= IX_mpo[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    err_IX_list[dep] = trace_val[1]
    # # trace_val[1]*32

    # # XI_error below
    trace_val = 1.0
    XI_mpo = error_bell("XI", peps_sites[1,:])
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= XI_mpo[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    err_XI_list[dep] = trace_val[1]


    # IZ_error below
    trace_val = 1.0
    IZ_mpo = error_bell("IZ", peps_sites[1,:])
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= IZ_mpo[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    err_IZ_list[dep] = trace_val[1]

    # ZI_error below
    trace_val = 1.0
    ZI_mpo = error_bell("ZI", peps_sites[1,:])
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= ZI_mpo[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    err_ZI_list[dep] = trace_val[1]


    # XZ_error below
    trace_val = 1.0
    XZ_mpo = error_bell("XZ", peps_sites[1,:])
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= XZ_mpo[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    err_XZ_list[dep] = trace_val[1]


    # ZX_error below
    trace_val = 1.0
    ZX_mpo = error_bell("ZX", peps_sites[1,:])
    for i=1:no_cavs
        peps_top = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "peps_top")
        for j=1:4
            peps_top[j] *= ZX_mpo[4(i-1) + j]
        end
        bottom = load("Data/peps_dep" * string(trunc(Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i)) * ".jld2" , "bottom")
        sequ = ITensors.optimal_contraction_sequence([peps_top ; bottom])
        block = contract([peps_top ; bottom]; sequence=sequ)
        # println("block ", inds(block))
        trace_val *= block
    end
    err_ZX_list[dep] = trace_val[1]
end


plot(dep_list, [real.(err_IX_list[i]) for i in dep_list]./pl, label="err_IX_list")
plot!(dep_list, [real.(err_XI_list[i]) for i in dep_list]./pl, label="err_XI_list")
plot!(dep_list, [real.(err_IZ_list[i]) for i in dep_list]./pl, label="err_IZ_list")
plot!(dep_list, [real.(err_ZI_list[i]) for i in dep_list]./pl, label="err_ZI_list")
plot!(dep_list, [real.(err_XZ_list[i]) for i in dep_list]./pl, label="err_XZ_list")
plot!(dep_list, [real.(err_ZX_list[i]) for i in dep_list]./pl, label="err_ZX_list")



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