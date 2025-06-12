import PauliStrings as ps
using PrettyTables
using Plots

# Initialise
num_sites = 5
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
for i in eachindex(Stab_list)
    for j in eachindex(error_list)
        coeff = 1
        opA = Stab_list[i]
        opB = error_list[j]
        if ps.op_to_strings(opA*opB) == ps.op_to_strings(opB*opA)
            coeff = 1
        else
            coeff = -1
        end
        error_coeffs[i,j] = coeff
    end
end
# error_coeffs

γ = 0.5
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

errorrate = (error_coeffs^-1 * stab_exp_list)
# can check 
sum(errorrate)

errors_dict = Dict(zip(error_list_string, errorrate))
# pretty_table(errors_dict, sortkeys=true)
pretty_table([error_list_string errorrate], header=["Error", "Probability"], title="Pauli error map")

#if in descending order
sortedindices = sortperm(errorrate, rev=true, dims=1)
sorted_errorrate = errorrate[sortedindices]
sorted_error_list_string = error_list_string[sortedindices]
pretty_table([sorted_error_list_string sorted_errorrate], header=["Error", "Probability"], title="Pauli error map (sorted)")

d = Dict([(1,1),(2,3)])
[x==2 && d[x] for x in findall(x->true , d)]


for i in Dict([(1,1),(2,3)])
    println(i[1]==2)
end
count(i->i==2, [2,3,4])

typeof(zip(error_list_string, errorrate))


marginalprob_list = []
for i in 1:num_sites
    marginalprob = 0.0
    println("site ",i)
    for j in eachindex(error_list_string)
        if error_list_string[j][i] == 'Z'
            println(error_list_string[j])
            marginalprob += errorrate[j]
        end
    end
    push!(marginalprob_list, marginalprob)
end
marginalprob_list

expected_indepprob = []
for i in error_list_string
    indep_prob = 1.0
    for j in 1:num_sites
        if i[j] == 'Z'
            indep_prob *= marginalprob_list[j]
        else
            indep_prob *= (1 - marginalprob_list[j])
        end
    end
    push!(expected_indepprob, indep_prob)
end
# expected_indepprob

plot(errorrate) 
plot!(expected_indepprob, linestyle=:dash) 
pstat = sum((errorrate .- expected_indepprob).^2 ./ expected_indepprob )

using Distributions
p_value = ccdf(Chisq(2^num_sites - 1 - num_sites), pstat)
# p_value > > 0.05 hence all sites are independent

n_errors_list(n) = [x for x in error_list_string if count(y -> y == 'Z', x)==n ]

errorstring_2_indices = findall(x -> count(y -> y == 'Z', x) == 2, error_list_string)
for i in errorstring_2_indices
    obs_prob = errorrate[i]
    error_locale = findall(x -> x =='Z', error_list_string[i])
    indep_prob = prod([ j in error_locale ? marginalprob_list[j] : (1-marginalprob_list[j]) for j in 1:num_sites])
    println(error_list_string[i])
    println("correlated fraction ",(obs_prob-indep_prob)/obs_prob)
    println("independent fraction ",indep_prob/obs_prob)
end

count(x -> x == 'Z', error_list_string[1])