
import PauliStrings as ps 
using PrettyTables, Plots, DataFrames, GLM, LinearAlgebra, Statistics, StatsBase, Base.Threads, CSV, Symbolics
using BenchmarkTools


# Initialise
num_sites = 6;
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
# error_list_string
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
error_coeffs;

γ = 0.3;
η = 0.9;
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

errorrate_ = (error_coeffs^-1 * stab_exp_list);
# error_list_string;
# errorrate = errorrate_[repeat([1],2^num_sites) + parse.(Int,replace.(replace.(error_list_string, "1"=>"0"),"Z"=>"1");base=2)]
# parse.(Int,replace.(error_list_string, "Z"=>"0");base=2)

using Optim
using LinearAlgebra
using StatsBase

# Define the error model
struct ErrorModel
    n_sites::Int
    observed_probs::Vector{Float64}  # Length 2^n_sites
    
    function ErrorModel(n_sites::Int, observed_probs::Vector{Float64})
        @assert length(observed_probs) == 2^n_sites "observed_probs must have length 2^n_sites"
        @assert abs(sum(observed_probs) - 1.0) < 1e-10 "observed_probs must sum to 1"
        new(n_sites, observed_probs)
    end
end

# # Convert binary string index to actual binary vector
# function index_to_binary(idx::Int, n_bits::Int)
#     # idx is 1-based, convert to 0-based for binary representation
#     binary_val = idx - 1
#     return [(binary_val >> i) & 1 for i in 0:(n_bits-1)]
# end

# Apply error effects to a binary string
function apply_error_effects(original::Vector{Int}, site::Int, error_type::Int, n_sites::Int)
    result = copy(original)
    
    if error_type == 1  # Flip only the site
        result[site] = 1 - result[site]
    elseif error_type == 2  # Flip neighbors only
        # Left neighbor
        if site > 1
            result[site-1] = 1 - result[site-1]
        end
        # Right neighbor
        if site < n_sites
            result[site+1] = 1 - result[site+1]
        end
    elseif error_type == 3  # Flip site and neighbors
        # The site itself
        result[site] = 1 - result[site]
        # Left neighbor
        if site > 1
            result[site-1] = 1 - result[site-1]
        end
        # Right neighbor  
        if site < n_sites
            result[site+1] = 1 - result[site+1]
        end
    end
    
    return result
end

# Convert binary vector back to index
function binary_to_index(binary_vec::Vector{Int})
    return sum(binary_vec[i] * 2^(i-1) for i in 1:length(binary_vec)) + 1
end

# Calculate theoretical probabilities given error parameters
function calculate_theoretical_probs(p_matrix, model::ErrorModel)
    n_sites = model.n_sites
    n_strings = 2^n_sites
    
    # assumes p_matrix is a 3Xn_sites matrix
    # p_matrix = (symmetric ? repeat(params, outer = (1,2)) : params)
    
    # Initialize probability array
    theoretical_probs = zeros(Float64, n_strings)* p_matrix[1,1]
    
    # Iterate over all possible error configurations
    # For each site, we have 4 possibilities: no error, type 1, type 2, type 3
    for error_config in 0:(4^n_sites - 1)
        # Decode error configuration
        current_config = error_config
        error_prob = 1.0
        
        # Start with the target state (all zeros)
        current_state = zeros(Int, n_sites) 
        
        for site in 1:n_sites
            error_type = current_config % 4
            current_config = current_config ÷ 4

            if error_type == 0  # No error
                error_prob *= (1.0 - p_matrix[1, site] - p_matrix[2, site] - p_matrix[3, site])
            elseif error_type == 1  # Type 1 error
                error_prob *= p_matrix[1, site]
                current_state = apply_error_effects(current_state, site, 1, n_sites)
            elseif error_type == 2  # Type 2 error
                error_prob *= p_matrix[2, site]
                current_state = apply_error_effects(current_state, site, 2, n_sites)
            elseif error_type == 3  # Type 3 error
                error_prob *= p_matrix[3, site]
                current_state = apply_error_effects(current_state, site, 3, n_sites)
            end
        end
        
        # Add this probability to the corresponding final state
        final_index = binary_to_index(current_state)
        # println
        theoretical_probs[final_index] += error_prob
    end
    
    return theoretical_probs
end

# Negative log-likelihood function
function neg_log_likelihood(params::Any, model::ErrorModel, symmetric::Bool, theoretical_probs_formula, p)
    # Check parameter constraints: all probabilities >= 0 and sum <= 1 for each site
    
    n_sites = model.n_sites
    p_matrix = (symmetric ? reshape(params, (3, Int(ceil(n_sites/2)))) : reshape(params, (3, n_sites)))
    

    for site in 1:(symmetric ? Int(ceil(n_sites/2)) : n_sites)
        if any(p_matrix[:, site] .< 0) || sum(p_matrix[:, site]) > 1.0
            return Inf64  # Return a large value if constraints are violated
        end
    end
    
    # Calculate theoretical probabilities
    theoretical_probs = []
    nll = zeros(length(model.observed_probs))
    p_subs = symmetric ? hcat(p_matrix, (floor(n_sites/2)>1 ? reverse((isodd(n_sites) ? p_matrix[:,1:end-1] : p_matrix), dims=2) : p_matrix[:,1] ) ) : p_matrix
    # theoretical_probs = Symbolics.substitute(theoretical_probs_formula, Dict(p => p_subs))
    # println("p_subs: ", p_subs, " ", size(p_subs))
    Threads.@threads for i in 1:length(model.observed_probs)
        theory_probs_i = Symbolics.value(Symbolics.substitute(theoretical_probs_formula[i], Dict(p => p_subs))) 
        # theory_probs_i = theoretical_probs[i]
        typeof(theory_probs_i) != Float64 && continue
        #     theory_probs_i = Symbolics.value(Symbolics.substitute(theory_probs_i, Dict(p => p_subs)))
        #     println("Hii")
        # end
        # println(typeof(theory_probs_i)== Float64)
    #     push!(theoretical_probs, theory_probs_i)
        if model.observed_probs[i] > 0
            if theory_probs_i <= 0
                return Inf64
            end
            # nll -= model.observed_probs[i] * log(theory_probs_i)
            nll[i] =  -model.observed_probs[i] * log(theory_probs_i)

        end
    end
    # println("theoretical_probs: ", theoretical_probs, " ", size(theoretical_probs))
    # theoretical_probs2 = calculate_theoretical_probs((symmetric ? hcat(p_matrix, (floor(n_sites/2)>1 ? reverse((isodd(n_sites) ? p_matrix[:,1:end-1] : p_matrix), dims=2) : p_matrix[:,1] ) ) : p_matrix), model)
    # println(all(isapprox.(theoretical_probs, theoretical_probs2)))
    # Calculate negative log-likelihood
    # nll = 0.0
    # for i in 1:length(model.observed_probs)
    #     if model.observed_probs[i] > 0
    #         if theoretical_probs[i] <= 0
    #             return Inf
    #         end
    #         nll -= model.observed_probs[i] * log(theoretical_probs[i])
    #     end
    # end
    # println(typeof(sum(nll)))
    return sum(nll)
end

# Estimate parameters using MLE

function estimate_parameters(model::ErrorModel; initial_guess::Union{Vector{Float64}, Nothing} = nothing, symmetric::Bool = true)

    n_sites = model.n_sites 
    if symmetric
        #symmetric case: 3 * n_sites/2 parameters
        if initial_guess === nothing
            initial_guess = fill(0.01, ( 3 * Int(ceil(n_sites/2)) ) )
        end
        # Define bounds
        lower_bounds = zeros(3 * Int(ceil(n_sites/2)))
        upper_bounds = ones(3 * Int(ceil(n_sites/2)))
    else
        # Symmetric case: only 3 parameters (p1, p2, p3)
        if initial_guess === nothing
            initial_guess = fill(0.01, (3* n_sites))
        end
        # Optimize with constraints
        lower_bounds = zeros(3 * Int((n_sites)))
        upper_bounds = ones(3 * Int((n_sites)))
    end

    # p = reshape(Symbolics.variables(:p, 1:(3*n_sites)), (3, n_sites))
    @variables p[3,1:n_sites]
    theorylist = calculate_theoretical_probs(p, model::ErrorModel)
    # p_matrix = rand(3,n_sites);
    # theoretical_probs = Symbolics.substitute(theorylist, Dict(p => (symmetric ? repeat(p_matrix, outer =(1,2)) : p_matrix)))
    # theoretical_probs2 = calculate_theoretical_probs(p_matrix, model)
    # println(all(isapprox.(theoretical_probs, theoretical_probs2)))
    println("Hi")
    result = optimize(params -> neg_log_likelihood(params, model, symmetric, theorylist, p), lower_bounds, upper_bounds, initial_guess, Fminbox()
        ,  Optim.Options(show_trace = true, show_every = 1
        # , f_reltol=0.01
        )
        )
    estimated_params = Optim.minimizer(result)
    return estimated_params, result
end

plot(cumsum(sort(errorrate_[:,1], rev=true)) , marker=:circle)
size(errorrate_)
# utoff_index

model = ErrorModel(num_sites, errorrate_[:,1]);
# Estimate parameters
estimated_params, result = estimate_parameters(model, symmetric=false)
estimated_params2, result2 = estimate_parameters(model, symmetric=true)
param_result = reshape(estimated_params2, (3, Int(ceil(num_sites/2))))
# param_result = estimated_params2;
plot();
plot!(param_result[1,:], seriestype = :line, marker=:circle);
plot!(param_result[2,:], seriestype = :line, marker=:circle);
plot!(param_result[3,:], seriestype = :line, marker=:circle)
plot!(estimated_params[1,1:Int(ceil(num_sites/2))], seriestype = :scatter);
plot!(estimated_params[2,1:Int(ceil(num_sites/2))], seriestype = :scatter);
plot!(estimated_params[3,1:Int(ceil(num_sites/2))], seriestype = :scatter);
plot!(title= "num sites = $num_sites")
# plot!(hcat(estimated_params2, (estimated_params2[:,1]))[1,:] , seriestype = :scatter);
# plot!(hcat(estimated_params2, (estimated_params2[:,1]))[2,:] , seriestype = :scatter);
# plot!(hcat(estimated_params2, (estimated_params2[:,1]))[3,:] , seriestype = :scatter)
# plot!(hcat(estimated_params2, reverse(estimated_params2, dims=2))[2,:], seriestype = :scatter);
# plot!(hcat(estimated_params2, reverse(estimated_params2, dims=2))[3,:], seriestype = :scatter)
param_result


# Create symbolic matrix



# theorylist[1]
# n_sites = 3
# p = reshape(Symbolics.variables(:p, 1:(3*n_sites)), (3, n_sites))
# func_list = []
# p[1,1]
# function testerr(p_)
#     return [ p_[1,1] , p_[2,1]*p_[1,1]]
# end
# y = testerr(p)
# p = rand(4,5)
# y[1]
# rand()
# Symbolics.substitute(theorylist, Dict(p => rand(3,3)))


# test_var =  fill(0.0, ( 3 , Int(ceil(6/2)) ) )
# test_var[1,1] = 0.1
# test_var[2,1] = 0.2
# test_var[3,1] = 0.3
# test_var[1,2] = 0.4
# test_var[2,2] = 0.5
# test_var[3,2] = 0.6
# test_var[1,3] = 0.7
# test_var[2,3] = 0.8
# test_var[3,3] = 0.9
# # test_var = repeat(test_var, outer = (1,2))
# reverse(test_var[:,1:2], dims=2)
# [test_var ; reverse(test_var, dims=2)]
# hcat(test_var, reverse(test_var, dims=2))
# test_var[:,1:2]
# repeat(test_var, outer = (1,2))

# @variables p[3,1:num_sites]
# theorylist = calculate_theoretical_probs(p, model::ErrorModel)
# p_rand = rand(3,6)
# @btime Symbolics.substitute(theorylist, Dict(p => p_rand))
# @btime for i in eachindex(theorylist)
#     Symbolics.substitute(theorylist[i], Dict(p => p_rand))
# end
# Symbolics.substitute(theorylist[8], Dict(p => rand(3,6)))
sum([1,2,3])