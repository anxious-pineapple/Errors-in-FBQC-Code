# Code calculates expectation values of stabaliser measurements in a chain graph 
# state, and probabilities of error for a Z error model


import PauliStrings as ps 
using PrettyTables
using Plots
using DataFrames
using GLM
using LinearAlgebra, Statistics, StatsBase
using Base.Threads
using CSV
# Initialise
num_sites = 12;
Stab_gen_list = [];
Stab_list = [];

# Create stabilizer generators
push!(Stab_gen_list, ps.Operator(num_sites));
Stab_gen_list[1] +=  "X",1,"Z",2;
for i in 2:num_sites-1
    push!(Stab_gen_list, ps.Operator(num_sites))
    Stab_gen_list[i] +=  "Z",i-1,"X",i,"Z",i+1
end
push!(Stab_gen_list, ps.Operator(num_sites));
Stab_gen_list[num_sites] += "Z",num_sites-1,"X",num_sites;
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
Stab_list_string = [ps.op_to_strings(i)[2][1] for i in Stab_list];

# All possible errors in pauli error map with Just Z errors
## Note, Is a string!!!!
error_list_string = ps.op_to_strings(ps.all_z(num_sites))[2];
error_list = [ps.Operator(i) for i in error_list_string];

# Construct matrix of error coeffs
error_coeffs = zeros(Int,2^num_sites, 2^num_sites);
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
# error_coeffs

γ = 0.5;
η = 0.98;
# ζ = ∑ αi^2 βi^2
ζ = 0.9;

# Construct vectors of expectation value of stabalisers
stab_exp_list = zeros(2^num_sites, 1);
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
stab_exp_list = abs.(stab_exp_list / stab_exp_list[1]);

errorrate = (error_coeffs^-1 * stab_exp_list);
# can check 
# sum(errorrate) == 1

#if in descending order
sortedindices = sortperm(errorrate, rev=true, dims=1);
sorted_errorrate = errorrate[sortedindices];
sorted_error_list_string = error_list_string[sortedindices];
pretty_table([sorted_error_list_string sorted_errorrate cumsum(sorted_errorrate[:,1])], header=["Error", "Probability", "Cumulative"], title="Pauli error map (sorted)")

plot(count.( x -> x == 'Z', sorted_error_list_string));
plot!(cumsum(sorted_errorrate[:,1]).<0.999) 
findfirst(x -> x >0.999, cumsum(sorted_errorrate[:,1]) )
println(sorted_error_list_string[1:1068])

###
    # marginalprob_list = []
    # for i in 1:num_sites
    #     marginalprob = 0.0
    #     # println("site ",i)
    #     for j in eachindex(error_list_string)
    #         if error_list_string[j][i] == 'Z'
    #             # println(error_list_string[j])
    #             marginalprob += errorrate[j]
    #         end
    #     end
    #     push!(marginalprob_list, marginalprob)
    # end
    # marginalprob_list

    # expected_indepprob = []
    # for i in error_list_string
    #     indep_prob = 1.0
    #     for j in 1:num_sites
    #         if i[j] == 'Z'
    #             indep_prob *= marginalprob_list[j]
    #         else
    #             indep_prob *= (1 - marginalprob_list[j])
    #         end
    #     end
    #     push!(expected_indepprob, indep_prob)
    # end
    # expected_indepprob

    # plot(errorrate[2:end], seriestype=scatter)
    # plot!(errorrate[2:end]) 
    # plot!(expected_indepprob[2:end], seriestype=scatter) 
    # plot!(expected_indepprob[2:end], linestyle=:dash) 

    # higher_than_expected = []
    # h_del = []
    # lower_than_expected = []
    # percents = []
    # for i in eachindex(expected_indepprob)
    #     push!(percents, abs((expected_indepprob[i] - errorrate[i])))
    #     if expected_indepprob[i] < errorrate[i]
    #         push!(higher_than_expected, error_list_string[i])
    #         push!(h_del, errorrate[i] - expected_indepprob[i])
    #     else
    #         push!(lower_than_expected, error_list_string[i])
    #     end
    # end

    # sortedindices2 = sortperm(h_del, rev=true)
    # plot(percents[2:end])
    # println("Higher than indep:")
    # [println(i) for i in higher_than_expected[sortedindices2]];
    # println("Lower than indep:")
    # [println(i) for i in lower_than_expected];


    # pstat = sum((errorrate .- expected_indepprob).^2 ./ expected_indepprob )

    # using Distributions
    # p_value = ccdf(Chisq(2^num_sites - 1 - num_sites), pstat)
    # # p_value > > 0.05 hence all sites are independent

    # n_errors_list(n) = [x for x in error_list_string if count(y -> y == 'Z', x)==n ]

    # errorstring_2_indices = findall(x -> count(y -> y == 'Z', x) == 2, error_list_string)
    # for i in errorstring_2_indices
    #     obs_prob = errorrate[i]
    #     error_locale = findall(x -> x =='Z', error_list_string[i])
    #     indep_prob = prod([ j in error_locale ? marginalprob_list[j] : (1-marginalprob_list[j]) for j in 1:num_sites])
    #     println(error_list_string[i])
    #     println("correlated fraction ",(obs_prob-indep_prob)/obs_prob)
    #     println("independent fraction ",indep_prob/obs_prob)
    # end

    # count(x -> x == 'Z', error_list_string[1])
###

# ============= GLM Block ============= #

# Full model with everything
error_matrix = zeros(Float64, length(error_list_string) , num_sites);
for i in eachindex(error_list_string)
    for j in 1:num_sites
        error_list_string[i][j] == 'Z' ? error_matrix[i,j] = 1.0 : nothing
    end
end
df_data = DataFrame();
df_data.errorrate = (errorrate[:,1]);
#Adding indep probs
for i in 1:num_sites
    colname = "pos$i"
    df_data[!, Symbol(colname)] = error_matrix[:, i]
    for j in 1:i-1
        colname2 =  "pos$(i)_$(j)"
        df_data[!, Symbol(colname2)] = error_matrix[:, i] .* error_matrix[:, j]
        for k in 1:j-1
            colname3 = "pos$(i)_$(j)_$(k)"
            df_data[!, Symbol(colname3)] = error_matrix[:, i] .* error_matrix[:, j] .* error_matrix[:, k]
        end
    end
end
predictor_names = names(df_data)[2:end];
sort!(predictor_names, by = x -> length(split(x[4:end],"_")));
all_terms = [Term(Symbol(name)) for name in predictor_names];
single_terms =  [Term(Symbol("pos$i")) for i in 1:num_sites];
adjacent_terms = [Term(Symbol("pos$(i)_$(i-1)")) for i in num_sites:-1:2];
xerror_terms = [Term(Symbol("pos$(i)_$(i-2)")) for i in num_sites:-1:3] ;
yerror_terms = [[Term(Symbol("pos$(i)_$(i-1)_$(i-2)")) for i in num_sites:-1:3] ; [Term(Symbol("pos$(2)_$(1)")), Term(Symbol("pos$(num_sites)_$(num_sites-1)"))] ];
# Build the formula
full_model = glm(Term(:errorrate) ~ sum(all_terms), df_data, Normal(), LogLink());
all_error_model = glm(Term(:errorrate) ~ sum([single_terms; xerror_terms; yerror_terms]), df_data, Normal(), LogLink());
testmod = glm(Term(:errorrate) ~ sum([ xerror_terms; single_terms; adjacent_terms; yerror_terms]), df_data, Normal(), LogLink());


# Full model with only reliable weights
error_list_string_2 = sorted_error_list_string[1:findfirst(x -> x >0.99, cumsum(sorted_errorrate[:,1]) )];
error_matrix = zeros(Float64, length(error_list_string_2) , num_sites);
for i in eachindex(error_list_string_2)
    for j in 1:num_sites
        error_list_string_2[i][j] == 'Z' ? error_matrix[i,j] = 1.0 : nothing
    end
end
df_data2 = DataFrame();
df_data2.errorrate = (sorted_errorrate[:,1])[1:length(error_list_string_2)];

#Adding indep probs
for i in 1:num_sites
    colname = "pos$i"
    df_data2[!, Symbol(colname)] = error_matrix[:, i]
    for j in 1:i-1
        colname2 =  "pos$(i)_$(j)"
        df_data2[!, Symbol(colname2)] = error_matrix[:, i] .* error_matrix[:, j]
        for k in 1:j-1
            colname3 = "pos$(i)_$(j)_$(k)"
            df_data2[!, Symbol(colname3)] = error_matrix[:, i] .* error_matrix[:, j] .* error_matrix[:, k]
            for l in 1:k-1
                colname4 = "pos$(i)_$(j)_$(k)_$(l)"
                df_data2[!, Symbol(colname4)] = error_matrix[:, i] .* error_matrix[:, j] .* error_matrix[:, k] .* error_matrix[:, l]
                for m in 1:l-1
                    colname5 = "pos$(i)_$(j)_$(k)_$(l)_$(m)"
                    df_data2[!, Symbol(colname5)] = error_matrix[:, i] .* error_matrix[:, j] .* error_matrix[:, k] .* error_matrix[:, l] .* error_matrix[:, m]
                end
            end
        end
    end
end


# predictor_names = names(df_data)[2:end];
# sort!(predictor_names, by = x -> length(split(x[4:end],"_")));
# all_terms = [Term(Symbol(name)) for name in predictor_names];
# single_terms =  [Term(Symbol("pos$i")) for i in 1:num_sites];
# adjacent_terms = [Term(Symbol("pos$(i)_$(i-1)")) for i in num_sites:-1:2];
# xerror_terms = [Term(Symbol("pos$(i)_$(i-2)")) for i in num_sites:-1:3] ;
# yerror_terms = [[Term(Symbol("pos$(i)_$(i-1)_$(i-2)")) for i in num_sites:-1:3] ; [Term(Symbol("pos$(2)_$(1)")), Term(Symbol("pos$(num_sites)_$(num_sites-1)"))] ];

# Build the formula
full_model_limited = glm(Term(:errorrate) ~ sum(all_terms), df_data2, Normal(), LogLink());
single_pos_model_ltd = glm(Term(:errorrate) ~ sum(single_terms), df_data, Normal(), LogLink());
# just_xmodel = glm(Term(:errorrate) ~ sum([single_terms; xerror_terms]), df_data, Normal(), LogLink())
# just_ymodel = glm(Term(:errorrate) ~ sum([single_terms; yerror_terms]), df_data, Normal(), LogLink())
all_error_model_ltd = glm(Term(:errorrate) ~ sum([single_terms; xerror_terms; yerror_terms]), df_data2, Normal(), LogLink());
testmod_ltd = glm(Term(:errorrate) ~ sum([ xerror_terms; single_terms; adjacent_terms; yerror_terms]), df_data2, Normal(), LogLink());
aic.([full_model, full_model_limited, all_error_model, testmod])
bic.([full_model, single_pos_model_ltd,all_error_model, testmod])
# bic.([full_model, single_pos_model, just_xmodel, just_ymodel, all_error_model])
# deviance.([full_model, single_pos_model, just_xmodel, just_ymodel, all_error_model, testmod])
deviance.([full_model, all_error_model, testmod])
deviance.([full_model_limited, all_error_model_ltd, testmod_ltd])
aic.([full_model, full_model_limited])
lrtest(single_pos_model, full_model)
lrtest(single_pos_model, all_error_model)
lrtest(all_error_model, testmod)
lrtest(all_error_model, full_model)
lrtest( testmod, full_model)

residuals(full_model, :pearson)

model1 = all_error_model;
model2 = all_error_model_ltd;
model3 = testmod_ltd;
observed = response(model1) ; # or model1.model.rr.y
fitted_vals = fitted(model1);
fitted_vals2 = fitted(model2);
fitted_vals3 = fitted(model3);
observed2 = response(model2);
# Standardized Pearson residuals
std_pearson = (observed - fitted_vals) ./ sqrt.(fitted_vals);
# plot(sort!(abs.(std_pearson) , rev=true))

# max_val = maximum((observed - fitted_vals))
range1 , range2 = 1,35...;
# range2
sort_ind_fit = sortperm(rel_error3, rev=true);
rel_error1 = (abs.(observed - fitted_vals)./observed);
rel_error2 = (abs.(observed2 - fitted_vals2)./observed2);
rel_error3 = (abs.(observed - fitted_vals3)./observed);
plot((rel_error1[sort_ind_fit])[range1:range2], label="model 1") 
plot!(sort!(abs.(observed2-fitted_vals2)./observed2, rev=true), label="model 2")
plot!(rel_error2[sort_ind_fit][range1:range2], label="model 2") ;
plot!(rel_error3[sort_ind_fit][range1:range2], label="model 3") 
# plot!(rel_error2[range1:range2], label="model 2") ;
# plot!(rel_error3[range1:range2], label="model 3") ;
# plot((abs.(observed2[sort_ind_fit] - fitted_vals2[sort_ind_fit])./observed2[sort_ind_fit])[range1:range2] , label="model 2");
# plot!((abs.(observed3[sort_ind_fit] - fitted_vals3[sort_ind_fit])./observed3[sort_ind_fit])[range1:range2] , label="model 3")
# plot(observed[sort_ind_fit])
findfirst(x -> x == max_val, observed - fitted_vals)
(observed[sort_ind_fit])[80:100]
no_of_Z = [count(x -> x == 'Z', i) for i in (sorted_error_list_string[sort_ind_fit])[1:500]]
mean(rel_error1[sort_ind_fit][range1:range2])
mean(rel_error2[sort_ind_fit][range1:range2])
mean(rel_error3[sort_ind_fit][range1:range2])
plot(sort(no_of_Z))
fitted_vals[15]
sorted_error_list_string[sort_ind_fit][1:35]
plot(observed[sort_ind_fit][range1:range2]);
plot!(fitted_vals[sort_ind_fit][range1:range2], label="fitted vals");
plot!(observed2[sort_ind_fit][range1:range2], label="observed2");
plot!(fitted_vals2[sort_ind_fit][range1:range2], label="fitted vals2");
plot!(fitted_vals3[sort_ind_fit][range1:range2], label="fitted vals3")

mean(rel_error1)
mean(rel_error2)
mean(rel_error3)

df_coefs = DataFrame(coeftable(full_model_limited))[1:end, [1,2,5]];
filter(row-> length(split(row["Name"][4:end],"_")) > 0, filter(row -> row["Pr(>|z|)"] < 1e-9, df_coefs))
plot(sort!(df_coefs[:,3]))
# pretty_table(df_coefs)
# full_model
# show(filter(row -> row["Pr(>|z|)"] < 1e-9, df_coefs) , allrows=true, )

# df_data[2,1]
# marginalprob_list
marginal_probs = []
for i in df_coefs[!,"Name"]
    # println(i)
    prob = 0
    for j in eachindex(errorrate)
        df_data[j,i] == 1.0 ? prob += df_data[j,"errorrate"] : nothing
    end
    push!(marginal_probs, prob)
end 
df_coefs[!,"MarginalProb"] = marginal_probs

# independent_probs = Dict()
indepenprob_list = []
# independent_probs

for i in df_coefs[!,"Name"]
    # println(i)
    indepprob = 0
    pos = split(i[4:end], "_")
    if length(pos) == 1
        indepprob = df_coefs[!,"MarginalProb"][findfirst(x -> x == i, df_coefs[!,"Name"])]
        # independent_probs[i] = indepprob
    elseif length(pos) == 2
        indepprob = df_coefs[!,"MarginalProb"][findfirst(x -> x == i, df_coefs[!,"Name"])]
        prob1 = indepenprob_list[parse(Int,pos[1])]
        prob2 = indepenprob_list[parse(Int,pos[2])]
        indepprob -= prob1 * prob2
        # independent_probs[i] = indepprob
    elseif length(pos) == 3
        indepprob = df_coefs[!,"MarginalProb"][findfirst(x -> x == i, df_coefs[!,"Name"])]
        prob1 = indepenprob_list[parse(Int,pos[1])]
        prob2 = indepenprob_list[parse(Int,pos[2])]
        prob3 = indepenprob_list[parse(Int,pos[3])]
        prob1_2 = df_coefs[!,"MarginalProb"][findfirst(x -> x == "pos$(pos[1])_$(pos[2])", df_coefs[!,"Name"])]
        prob2_3 = df_coefs[!,"MarginalProb"][findfirst(x -> x == "pos$(pos[2])_$(pos[3])", df_coefs[!,"Name"])]
        prob1_3 = df_coefs[!,"MarginalProb"][findfirst(x -> x == "pos$(pos[1])_$(pos[3])", df_coefs[!,"Name"])]
        indepprob -= prob1_2 * prob3
        indepprob -= prob2_3 * prob1
        indepprob -= prob1_3 * prob2
        indepprob += 2 * prob1 * prob2 * prob3
        # independent_probs[i] = indepprob
    end
    push!(indepenprob_list, indepprob)
    # println("Independent prob for $i: $indepprob")
end 
# parse(Int,"3")
# independent_probs
# indepenprob_list
df_coefs[!,"IndepProb"] = indepenprob_list

show(filter(row-> length(split(row["Name"][4:end],"_")) > 0 , filter(row -> row["Pr(>|z|)"] < 1e-9, df_coefs)[!,[1,2,4,5]]), allrows=true)
# CSV.write("SignificantCorrelation_N$num_sites.csv", filter(row -> row["Pr(>|z|)"] < 1e-9, df_coefs))
# show(filter(row -> row["Pr(>|z|)"] < 1e-9, df_coefs), allrows=true)

# significant_terms = [coeftable(full_model).cols[4][i+1] < 1e-9 ? [all_terms[i], coeftable(full_model).cols[1][i+1]] : nothing for i in eachindex(all_terms)] 
# filter!(x-> x !== nothing, significant_terms, 2)
# significant_terms[;][1]
# pretty_table( [all_terms coeftable(full_model).cols[1][2:end] coeftable(full_model).cols[4][2:end]] , header=["Term", "Coefficient"], title="Full model coefficients") 
# sum(errorrate)