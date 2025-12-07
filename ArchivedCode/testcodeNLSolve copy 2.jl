
import PauliStrings as ps 
using PrettyTables
using Plots
using DataFrames
using GLM
using LinearAlgebra, Statistics, StatsBase
using Base.Threads
using CSV
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

γ = 0.2;
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
function calculate_theoretical_probs(params::Vector{Float64}, model::ErrorModel)
    n_sites = model.n_sites
    n_strings = 2^n_sites
    
    # Reshape parameters: [p1_1, p2_1, p3_1, p1_2, p2_2, p3_2, ...]
    p_matrix = reshape(params, (3, n_sites))
    
    # Initialize probability array
    theoretical_probs = zeros(Float64, n_strings)
    
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
            
            p_site = site
            # site > ceil(n_sites/2) ? p_site = n_sites - site + 1 : p_site = site

            if error_type == 0  # No error
                error_prob *= (1.0 - p_matrix[1, p_site] - p_matrix[2, p_site] - p_matrix[3, p_site])
            elseif error_type == 1  # Type 1 error
                error_prob *= p_matrix[1, p_site]
                current_state = apply_error_effects(current_state, site, 1, n_sites)
            elseif error_type == 2  # Type 2 error
                error_prob *= p_matrix[2, p_site]
                current_state = apply_error_effects(current_state, site, 2, n_sites)
            elseif error_type == 3  # Type 3 error
                error_prob *= p_matrix[3, p_site]
                current_state = apply_error_effects(current_state, site, 3, n_sites)
            end
        end
        
        # Add this probability to the corresponding final state
        final_index = binary_to_index(current_state)
        theoretical_probs[final_index] += error_prob
    end
    
    return theoretical_probs
end

# Negative log-likelihood function
function neg_log_likelihood(params::Vector{Float64}, model::ErrorModel)
    # Check parameter constraints: all probabilities >= 0 and sum <= 1 for each site
    n_sites = model.n_sites
    p_matrix = reshape(params, (3, n_sites))
    
    for site in 1:n_sites
        p_site = site
        # p_site = site > ceil(n_sites/2) ? n_sites - site + 1 : site
        if any(p_matrix[:, p_site] .< 0) || sum(p_matrix[:, p_site]) > 1.0
            return Inf
        end
    end
    
    # Calculate theoretical probabilities
    theoretical_probs = calculate_theoretical_probs(params, model)
    
    # Calculate negative log-likelihood
    nll = 0.0
    for i in 1:length(model.observed_probs)
        if model.observed_probs[i] > 0
            if theoretical_probs[i] <= 0
                return Inf
            end
            nll -= model.observed_probs[i] * log(theoretical_probs[i])
        end
    end
    
    return nll
end

# Estimate parameters using MLE
function estimate_parameters(model::ErrorModel; initial_guess::Union{Vector{Float64}, Nothing} = nothing, symmetric::Bool = true)
    
    n_sites = model.n_sites
    
    if symmetric
        # Symmetric case: only 3 parameters (p1, p2, p3)
        if initial_guess === nothing
            initial_guess = [0.01, 0.01, 0.01]  # Small initial values
        end
        
        # Define symmetric negative log-likelihood
        function symmetric_nll(symmetric_params::Vector{Float64})
            # Expand symmetric parameters to full parameter vector
            full_params = repeat(symmetric_params, n_sites)
            return neg_log_likelihood(full_params, model)
        end
        
        # Optimize with constraints
        lower_bounds = [0.0, 0.0, 0.0]
        upper_bounds = [1.0, 1.0, 1.0]
        
        result = optimize(symmetric_nll, lower_bounds, upper_bounds, initial_guess, Fminbox(LBFGS()))
        
        # Expand result to full parameter vector
        estimated_params = repeat(Optim.minimizer(result), n_sites)
        
    else
        # Non-symmetric case: 3 * n_sites parameters
        if initial_guess === nothing
            initial_guess = fill(0.01, Int(3 * n_sites))
        end
        
        # Define bounds
        lower_bounds = zeros(3 * Int(ceil(n_sites)))
        upper_bounds = ones(3 * Int(ceil(n_sites)))
        
        result = optimize(params -> neg_log_likelihood(params, model), lower_bounds, upper_bounds, initial_guess, Fminbox(LBFGS()))
        
        estimated_params = Optim.minimizer(result)
    end
    
    return estimated_params, result
end

# Example usage and testing
# function example_usage()
# println("=== Binary Error MLE Estimation Example ===")

# Create a synthetic dataset for testing
# n_sites = 4  # Start small for testing

# True parameters (symmetric case)
# true_p1, true_p2, true_p3 = 0.05, 0.03, 0.02
# true_params = repeat([true_p1, true_p2, true_p3], n_sites)

# Generate synthetic observed probabilities
# println("Generating synthetic data with parameters:")
# println("p1 = $true_p1, p2 = $true_p2, p3 = $true_p3")

# Create model with true parameters to generate data
# dummy_probs = ones(2^n_sites) / 2^n_sites  # Placeholder
# error_list_string
# errorrate_
model = ErrorModel(num_sites, errorrate_[:,1]);
# true_theoretical_probs = calculate_theoretical_probs(true_params, temp_model)

# Create the actual model with synthetic data
# model = ErrorModel(n_sites, true_theoretical_probs)



# Estimate parameters
# println("\nEstimating parameters...")
estimated_params, result = estimate_parameters(model, symmetric=false)

# println("\nResults:")
# println("Convergence: ", Optim.converged(result))
# println("Estimated parameters: ", estimated_params[1:3])
# println("True parameters: ", [true_p1, true_p2, true_p3])
# println("Final objective value: ", Optim.minimum(result))

param_result = reshape(estimated_params, (3, Int(ceil(num_sites))))
plot();
plot!(param_result[1,:]);
plot!(param_result[2,:]);
plot!(param_result[3,:])
# return model, estimated_params, result
# end

