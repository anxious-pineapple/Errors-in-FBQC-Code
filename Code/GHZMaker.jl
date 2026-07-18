using ITensors
using .MPOFuncs
using Plots
using BenchmarkTools
using JLD2

gamma = 1.0
no_cavs = 3
dt = 0.01
t_final = 10.0
dep = 0.014


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

function detect_10( site_list )
    # 
    mps_list = [;]
    proj_list = repeat(["Ground",],length(site_list))
    # for i in [1:length(site_list);]
    #     println(i)
    proj_list[1] = "Excite1"
    # push!(mps_list, MPO(site_list, proj_list))
        # proj_list[i] = "Ground"
    # end
    # all_mps = sum(mps_list)
    all_mps = MPO(site_list, proj_list)
    return all_mps
end

function detect_10_anymode( site_list )
    # 
    mps_list = [;]
    proj_list = repeat(["Ground",],length(site_list))
    for i in 1:no_cavs
        for j in 1:no_cavs
            for k in 1:no_cavs
                proj_list[6*(i-1) + 1] = "Excite1"
                proj_list[6*(j-1) + 3] = "Excite1"
                proj_list[6*(k-1) + 5] = "Excite1"
                push!(mps_list, MPO(site_list, proj_list))
                proj_list[6*(i-1) + 1] = "Ground"
                proj_list[6*(j-1) + 3] = "Ground"
                proj_list[6*(k-1) + 5] = "Ground"
            end
        end
    end
    all_mps = sum(mps_list)
    return all_mps
end

function detect_W_anymode( site_list )
    # 
    mps_list = [;]
    proj_list = repeat(["Ground",],length(site_list))
    for i in 1:length(site_list)
        proj_list[i] = "Excite1"
        push!(mps_list, MPO(site_list, proj_list))
        proj_list[i] = "Ground"
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


# mpo_i, sites_i, eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)
sites_i = ITensors.siteinds("Qudit", no_cavs; dim=4)
input_list = repeat(["Ground",],no_cavs)
input_list[1] = "Excite1" 
mpo_i = MPO(sites_i, input_list)
signal_mpo, signal_sites = MPOFuncs.n_copy_mpo(mpo_i, sites_i, 6)

for i=6:-1:1
    for j=no_cavs:-1:1
        println(i,j)
        signal_mpo = MPOFuncs.swap_ij!(signal_mpo, signal_sites, (no_cavs*(i-1))+j, (no_cavs*(i-1))+j + ((6-i)*(j-1)))
    end
end
MPOFuncs.expect(signal_mpo, signal_sites)
# ideally dim 3 but in case 4
ancilla_sites = siteinds("Qudit", length(signal_mpo), dim=4)
# ancilla_mpo = [;]
# for i in ancilla_sites
#     push!(ancilla_mpo, op(i, "Ground"))
# end

# # ancilla_mpo[4]
# # state(ancilla_sites[1], 1 ) * state(ancilla_sites[1]' , 1 )
ancilla_mpo = MPO(ancilla_sites, repeat(["Ground",],length(signal_mpo)))

peps, peps_sites = peps_zipper(signal_mpo, ancilla_mpo, signal_sites, ancilla_sites)

detect_mpo = detect_10_anymode(peps_sites[2,:]) 
truncate!(detect_mpo, cutoff=1e-5)
plot(linkdims(detect_mpo))

detect_mpo = detect_W_anymode(peps_sites[2,:]) 
plot(linkdims(detect_mpo))
truncate!(detect_mpo, cutoff=1e-5)
comb=1



for i in 0:no_cavs-1
    # i = 1
    println(i)

    i1, i2, i3, i4, i5, i6 = 6i+1, 6i+2, 6i+3, 6i+4, 6i+5, 6i+6
    i_list = [i1, i2, i3, i4, i5, i6]

    beamsplitter_peps_tensor_linear!(peps, peps_sites, 1, i1, i2)
    beamsplitter_peps_tensor_linear!(peps, peps_sites, 1, i3, i4)
    beamsplitter_peps_tensor_linear!(peps, peps_sites, 1, i5, i6)

    println("linear done")
    peps[1,i1] , peps[2,i1] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i1] , peps[2,i1], peps_sites[1,i1], peps_sites[2,i1])
    peps[1,i2] , peps[2,i2] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i2] , peps[2,i2], peps_sites[1,i2], peps_sites[2,i2])
    peps[1,i3] , peps[2,i3] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i3] , peps[2,i3], peps_sites[1,i3], peps_sites[2,i3])
    peps[1,i4] , peps[2,i4] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i4] , peps[2,i4], peps_sites[1,i4], peps_sites[2,i4])
    peps[1,i5] , peps[2,i5] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i5] , peps[2,i5], peps_sites[1,i5], peps_sites[2,i5])
    peps[1,i6] , peps[2,i6] = MPOFuncs.beamsplitter_peps_tensor(peps[1,i6] , peps[2,i6], peps_sites[1,i6], peps_sites[2,i6])
    println("cross done")
    beamsplitter_peps_tensor_linear!(peps, peps_sites, 2, i2, i3)
    beamsplitter_peps_tensor_linear!(peps, peps_sites, 2, i4, i5)
    beamsplitter_peps_tensor_nonlinear!(peps, peps_sites, 2, i6, i1)
    println("nonlinear done")

    for i in i_list
        peps[2,i] = peps[2,i] * detect_mpo[i] * comb
        peps[1,i] = peps[1,i] * peps[2,i]
    end

    save("Data/peps_ghz_dep" * string(trunc( Int, dep*10000)) * "no_cavs" * string(Int(no_cavs)) * "part_" * string(Int(i+1)) * ".jld2" , "peps_top", peps[1,i1:i6])
    peps[1,i1:i6] = repeat([ITensor(0),],6)
end
##
    j = 1
    j+=1
    j
    commoninds(peps[2,j-1], peps[2, j])
    peps[2,2]


    detect_11_23 = detect_10(peps_sites[2,2:3])
    detect_11_45 = detect_10(peps_sites[2,4:5])
    detect_11_61 = detect_10([peps_sites[2,6],peps_sites[2,1]])

    linkdims(detect_11_61)
    commoninds(peps[2,5], peps[2, 4])
    commoninds(ancilla_mpo[5], ancilla_mpo[6])

    # collapse 23
    term2 = detect_11_23[1] * peps[2, i2]
    term3 = detect_11_23[2] * peps[2, i3]
    term23 = term2 * term3


    # collapse 45
    term4 = detect_11_45[1] * peps[2, i4]
    term5 = detect_11_45[2] * peps[2, i5]
    term45 = term4 * term5

    # collapse 61
    term6 = detect_11_61[1] * peps[2, i6]
    term1 = detect_11_61[2] * peps[2, i1]
    term61 = term6 * term1
##
#save mode n 

ghz_mps = (MPS(peps_sites[1,:], append!([i%2==0 ? 1 : 2 for i=1:6], repeat([1,], (no_cavs-1)*6)) )
+ MPS(peps_sites[1,:], append!([i%2==0 ? 2 : 1 for i=1:6], repeat([1,], (no_cavs-1)*6))) )/sqrt(2)
expect(ghz_mps, "N")
ghz_mpo = [i*i' for i in ghz_mps]


tr = 1
for i in 1:6*no_cavs
    println(i)
    tr *= peps[1,i] * ghz_mpo[i] * peps[2,i]
end
# tr *= term61
tr[1] * 32 * 8

# trace
tr = 1
i = 1
i += 1
tr *= peps[1,i] * delta(peps_sites[1,i], peps_sites[1,i]')
tr *= term61
tr[1]

1/(32*8)
# expectation value


