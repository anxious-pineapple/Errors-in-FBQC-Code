
import PauliStrings as ps 
using PrettyTables
using Plots
using DataFrames
using GLM
using LinearAlgebra, Statistics, StatsBase
using Base.Threads
using CSV
# Initialise
num_sites = 3
Stab_gen_list = []
Stab_list = []

# Create stabilizer generators
push!(Stab_gen_list, ps.Operator(num_sites))
Stab_gen_list[1] +=  "X",1,"Z",2
for i in 2:num_sites-1
    push!(Stab_gen_list, ps.Operator(num_sites))
    Stab_gen_list[i] +=  "Z",i-1,"X",i,"Z",i+1
end
push!(Stab_gen_list, ps.Operator(num_sites))
Stab_gen_list[num_sites] += "Z",num_sites-1,"X",num_sites
# Stab_gen_list

# Create stabilizer list
for i in 0:2^num_sites-1
    push!(Stab_list, ps.Operator(num_sites))
    Stab_list[end] += repeat("I",num_sites)
    bin_num = bitstring(i)[end-num_sites+1:end]
    for j in 1:num_sites
        if bin_num[j] == '1' 
            Stab_list[end] *= Stab_gen_list[j]
        end
    end
end
# Stab_list
Stab_list_string = [ps.op_to_strings(i)[2][1] for i in Stab_list]

# All possible errors in pauli error map with Just Z errors
## Note, Is a string!!!!
error_list_string = ps.op_to_strings(ps.all_z(num_sites))[2]
error_list = [ps.Operator(i) for i in error_list_string]

# Construct matrix of error coeffs
error_coeffs = zeros(Int,2^num_sites, 2^num_sites)
Threads.@threads for i in eachindex(Stab_list_string)
    Threads.@threads for j in eachindex(error_list_string)
        coeff = 1
        # println(i)
        opA = Stab_list_string[i]
        opB = error_list_string[j]
        for z_pos in findall(x->x=='Z', opB)
            if opA[z_pos] in ['Z', '1']
                coeff *= 1
            else
                coeff *= -1
            end
                
        end    
        error_coeffs[i,j] = coeff
    end
end
error_coeffs

γ = 0.0
η = 0.9
# ζ = ∑ αi^2 βi^2
ζ = 0.9

# Construct vectors of expectation value of stabalisers
stab_exp_list = zeros(2^num_sites, 1)
for i in eachindex(Stab_list_string)
    stab_exp = [1.0 1.0 1.0 1.0]
    for j in Stab_list_string[i]
        if j == '1'
            stab_exp *= (exp(-2*abs(γ)^2) * (η + (1-η)*2*abs(γ)^2)) * 0.5 * [1.0 1.0 1.0 1.0 ; 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 ; 1.0 -1.0 -1.0 1.0]
        elseif j == 'X'
            a = 2 * (1-η) * abs(γ)^2
            b = η * ζ
            stab_exp *= exp(-2*abs(γ)^2) * 0.5 * [a a a a ; b -b b -b ; b b -b -b ; a -a -a a]
        elseif j == 'Y'
            stab_exp *= (1im * η * ζ * exp(-2*abs(γ)^2)) * 0.5 * [0.0 0.0 0.0 0.0 ; 1.0 -1.0 1.0 -1.0 ; -1.0 -1.0 1.0 1.0 ; 0.0 0.0 0.0 0.0]
        elseif j == 'Z'
            stab_exp *= (η * exp(-2*abs(γ)^2)) * 0.5 * [1.0 1.0 1.0 1.0 ; 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 ; -1.0 1.0 1.0 -1.0]
        end
    end
    stab_exp *= [1.0 0.0 0.0 0.0]' 
    # println(stab_exp)
    stab_exp_list[i] = stab_exp[1]
end
# stab_exp_list        
    
#normalise stabilizer expectation values by identity
stab_exp_list = abs.(stab_exp_list / stab_exp_list[1])

errorrate_ = (error_coeffs^-1 * stab_exp_list)
# error_list_string
errorrate = errorrate_[repeat([1],2^num_sites) + parse.(Int,replace.(replace.(error_list_string, "1"=>"0"),"Z"=>"1");base=2)]
# parse.(Int,replace.(error_list_string, "Z"=>"0");base=2)



using NLsolve

#num_sites
n = num_sites

#simulated error rate
# errorrate = rand(0.0:0.01:1, 2^n)
# errorrate = errorrate ./ sum(errorrate)
# error_string = [bitstring(i)[end-n+1:end] for i in 0:2^n-1]
# sum(errorrate)
# prob_dict= Dict{}


error_dict = Dict{Any, Any}()
for i in (collect(Iterators.product([[1:4 for _ in 1:n-1] ; [1:4 for _ in 1:1]]...)))
    # print("$(i) ")
    # prob_dict[i] = prod([p[j,i[j]] for j in 1:n])
    # println(xor(convert_to_bin(i)...))
    error_dict[i] = xor(convert_to_bin(i)...)
end

# 1 = no error
# 2 = Z err
# 3 = X err
# 4 = Y err
# 5 = ZZ err starting from that point
function convert_to_bin(x)
    list_bin = []
    for i in eachindex(x)
        if x[i] == 1
            push!(list_bin, 0) 
        elseif x[i] == 2
            push!(list_bin, 2^(n-i))
        elseif x[i] == 3
            if i != 1 && i!= n
                push!(list_bin, 2^(n-i+1) + 2^(n-i-1))
            elseif i == 1
                push!(list_bin, 2^(n-i-1))
            elseif i == n
                push!(list_bin, 2^(n-i+1))
            end
        elseif x[i] == 4
            if i != 1 && i!=n
                push!(list_bin, 2^(n-i+1) + 2^(n-i-1) + 2^(n-i))
            elseif i == 1
                push!(list_bin, 2^(n-i-1) + 2^(n-i))
            elseif i == n
                push!(list_bin, 2^(n-i+1) + 2^(n-i))
            end
        elseif x[i] == 5
            if i != n
                push!(list_bin, 2^(n-i) + 2^(n-i-1))
            end
        end
    end
    return list_bin
end
# convert_to_bin([5,5,5])
###
    # function convert_to_bin2(x)
    #     list_bin = []
    #     # zero_list = repeat("0",n)
    #     for i in eachindex(x)
    #         if x[i] == 1
    #             # push!(list_bin, 0) 
    #             push!(list_bin, bitstring(0)[end-n+1:end])
    #             # zero_list[i] = "1"
    #         elseif x[i] == 2
    #             push!(list_bin, bitstring(2^(n-i))[end-n+1:end])
    #         elseif x[i] == 3
    #             if i != 1 || i != n
    #                 push!(list_bin, bitstring(2^(n-i+1) + 2^(n-i-1))[end-n+1:end])
    #             elseif i == 1
    #                 push!(list_bin, bitstring(2^(n-i-1))[end-n+1:end])
    #             elseif i == n
    #                 push!(list_bin, bitstring(2^(n-i+1))[end-n+1:end])
    #             end
    #         elseif x[i] == 4
    #             if i != 1 || n
    #                 push!(list_bin, bitstring(2^(n-i+1) + 2^(n-i-1) + 2^(n-i))[end-n+1:end])
    #             elseif i == 1
    #                 push!(list_bin, bitstring(2^(n-i-1) + 2^(n-i))[end-n+1:end])
    #             elseif i == n
    #                 push!(list_bin, bitstring(2^(n-i+1) + 2^(n-i))[end-n+1:end])
    #             end
    #         elseif x[i] == 5
    #             if i != n
    #                 push!(list_bin, bitstring(2^(n-i) + 2^(n-i-1))[end-n+1:end])
    #             end
    #         end
    #     end
    #     return list_bin
    # end
###

err_dict_invert = Dict{valtype(error_dict), Vector{keytype(error_dict)}}()
for (k, v) in error_dict
    push!(get!(() -> valtype(err_dict_invert)[], err_dict_invert, v), k)
end
err_dict_invert
err_dict_invert[0]

p0 = rand(n,5)
p0[1:n,5] = repeat([0.0],n) # Initial guess for the last column
for i in 1:n
    p0[i,:] = p0[i,:] ./ sum(p0[i,:]) # Normalization constraint
end
p0

function prob_product_map(p, list_)
    sum_term = 0.0
    #list_ is a list of tuples
    for i in list_ 
        prod_term = 1.0
        for j in eachindex(i)
            println("i: $i, j: $j")
            prod_term *= p[j, i[j]]
        end
        sum_term += prod_term
    end
    return sum_term
end

function f!(F , p, error_dict_invert, errorrate)
    # [(x[1]+3)*(x[2]^3-7)+18,
    # sin(x[2]*exp(x[1])-1)]
    F = [
        
        # sum(p, dims = 2) - repeat([1.0],n) ; # Normalization constraint
        [prob_product_map(p,error_dict_invert[j-1]) - errorrate[j] for j in 1:2^n];

    ]
end

sol = nlsolve((F,p0) -> f!(F, p0, err_dict_invert, errorrate), p0)
p0
sol.zero


