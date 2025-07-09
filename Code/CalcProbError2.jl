# THis version witha difeerent error model, all paulis and not just Z

# using Optim, LinearAlgebra

# function maxent_pauli_reconstruction(A, b; initial_guess=nothing, tol=1e-8)
#     """
#     Solve for probability distribution p using Maximum Entropy principle
    
#     Args:
#         A: 2^n × 4^n matrix relating Pauli errors to stabilizer expectations
#         b: 2^n vector of measured stabilizer expectation values
#         initial_guess: optional starting point for optimization
#         tol: convergence tolerance
    
#     Returns:
#         p: 4^n probability vector over Pauli errors
#     """
    
#     n_stabilizers, n_paulis = size(A)
    
#     # Initial guess - uniform distribution if not provided
#     if initial_guess === nothing
#         p0 = ones(n_paulis) / n_paulis
#     else
#         p0 = initial_guess
#     end
    
#     # Objective function: negative entropy (we minimize)
#     function objective(p)
#         # Add small epsilon to avoid log(0)
#         entropy = sum(p[i] * log(p[i] + 1e-12) for i in 1:length(p) if p[i] > 1e-12)
#         return entropy
#     end
    
#     # Constraint function combining all equality constraints
#     function constraints!(c, p)
#         # Stabilizer constraints: A*p = b
#         c[1:n_stabilizers] = A * p .- b
#         # Normalization constraint: sum(p) = 1
#         c[n_stabilizers + 1] = sum(p) - 1.0
#         return c
#     end
    
#     # Set up optimization problem
#     n_constraints = n_stabilizers + 1
#     c = zeros(n_constraints)
    
#     # Box constraints: 0 ≤ p_i ≤ 1
#     lower_bounds = zeros(n_paulis)
#     upper_bounds = ones(n_paulis)
    
#     # Solve using interior point method
#     result = Optim.optimize(
#         objective,
#         (c, p) -> constraints!(c, p),
#         lower_bounds,
#         upper_bounds,
#         p0,
#         IPNewton(),
#         Optim.Options(
#             g_tol=tol,
#             iterations=10000,
#             show_trace=false
#         )
#     )
    
#     p_opt = Optim.minimizer(result)
    
#     # Check convergence
#     if !Optim.converged(result)
#         @warn "Optimization did not converge"
#     end
    
#     # Verify constraints are satisfied
#     constraint_residual = norm(A * p_opt - b)
#     normalization_error = abs(sum(p_opt) - 1.0)
    
#     println("Constraint residual: $(constraint_residual)")
#     println("Normalization error: $(normalization_error)")
#     println("Final entropy: $(-sum(p_opt[i] * log(p_opt[i] + 1e-12) for i in 1:length(p_opt) if p_opt[i] > 1e-12))")
    
#     return p_opt, result
# end

# # Alternative implementation using NLopt for better constraint handling
# using NLopt

# function maxent_pauli_reconstruction_nlopt(A, b; initial_guess=nothing)
#     """
#     Alternative implementation using NLopt for better constraint handling
#     """
    
#     n_stabilizers, n_paulis = size(A)
    
#     # Initial guess
#     if initial_guess === nothing
#         p0 = ones(n_paulis) / n_paulis
#     else
#         p0 = initial_guess
#     end
    
#     # Set up NLopt problem
#     opt = Opt(:LD_SLSQP, n_paulis)  # Sequential Least Squares Programming
    
#     # Bounds: 0 ≤ p_i ≤ 1
#     lower_bounds!(opt, zeros(n_paulis))
#     upper_bounds!(opt, ones(n_paulis))
    
#     # Objective: minimize negative entropy
#     function obj_func(p, grad)
#         if length(grad) > 0
#             # Gradient of negative entropy: -(log(p) + 1)
#             for i in 1:length(p)
#                 grad[i] = -(log(p[i] + 1e-12) + 1.0)
#             end
#         end
#         return -sum(p[i] * log(p[i] + 1e-12) for i in 1:length(p) if p[i] > 1e-12)
#     end
    
#     min_objective!(opt, obj_func)
    
#     # Equality constraints: A*p = b
#     for i in 1:n_stabilizers
#         function constraint_i(p, grad)
#             if length(grad) > 0
#                 grad[:] = A[i, :]
#             end
#             return dot(A[i, :], p) - b[i]
#         end
#         equality_constraint!(opt, constraint_i, 1e-8)
#     end
    
#     # Normalization constraint: sum(p) = 1
#     function norm_constraint(p, grad)
#         if length(grad) > 0
#             grad[:] .= 1.0
#         end
#         return sum(p) - 1.0
#     end
#     equality_constraint!(opt, norm_constraint, 1e-8)
    
#     # Set tolerance and max evaluations
#     xtol_rel!(opt, 1e-10)
#     maxeval!(opt, 10000)
    
#     # Optimize
#     (minf, p_opt, ret) = optimize(opt, p0)
    
#     println("Optimization result: $ret")
#     println("Final objective value: $minf")
#     println("Constraint residual: $(norm(A * p_opt - b))")
#     println("Normalization error: $(abs(sum(p_opt) - 1.0))")
    
#     return p_opt, ret
# end

# # Example usage:
# # Assuming you have your A matrix and b vector
# # p_solution, result = maxent_pauli_reconstruction(A, b)

# # For better performance with large problems, use the NLopt version:
# # p_solution, status = maxent_pauli_reconstruction_nlopt(A, b)

#  # Example number of qubits

import PauliStrings as ps
using PrettyTables
using Plots
using Combinatorics, LinearAlgebra

# Initialise
num_sites = 8
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


γ = 0.0
#0 and 0.1
η = 0.9
# 0.9 and 0.95 and 0.99
# ζ = ∑ αi^2 βi^2
ζ = 0.9
# 0.9 and 0.95 and 0.99

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

A = zeros(2^num_sites, 4^num_sites)  # 2^n × 4^n matrix
#b is stab_exp_list  
# 2^n expectation values

# Generate all possible error combinations
arr_values = ['1','X','Y','Z']
error_list_string = join.(vec(collect(Iterators.product(Iterators.repeated(arr_values, num_sites)...))))

for i in eachindex(Stab_list_string)
    for j in eachindex(error_list_string)
        coeff = 1
        for k in 1:num_sites
            if error_list_string[j][k] == '1' || error_list_string[j][k] == Stab_list_string[i][k] || Stab_list_string[i][k] == '1'
                coeff *= 1
            else
                coeff *= -1
            end
        end
        A[i,j] += coeff
    end
end
# Stab_list_string[1][1] == '1'
# any(A .== 0)
# Check if there are any zeros in the matrix
# A

#Maximum liklihood estimation of the error probabilities
# p = A^T * (A * A^T)^(-1) * b
p = Transpose(A) * (A * Transpose(A))^(-1) * stab_exp_list

sortedindices = sortperm(p, rev=true, dims=1)
sorted_errorrate = p[sortedindices]
sorted_error_list_string = error_list_string[sortedindices]
pretty_table([sorted_error_list_string[128:200] sorted_errorrate[128:200]], header=["Error", "Probability"], title="Pauli error map (sorted)", display_size = (-1, -1))

# dict_ghzerr= Dict(zip(sorted_error_list_string, sorted_errorrate))

###
    # unique_errorrates = unique(sorted_errorrate)
    # lowest_weight_err = []
    # for i in unique_errorrates
    #     # print("prob :",i)
    #     prob_indices_list = findall(x -> x == i, sorted_errorrate[:,1])
    #     error_list = sorted_error_list_string[prob_indices_list]
    #     sort!(error_list, lt = (x, y) -> count(z -> z == '1',x) > count(z -> z == '1',y))
    #     push!(lowest_weight_err, error_list[1])
    # end
    # pretty_table([lowest_weight_err unique_errorrates], header=["Lowest weight error", "Probability"], title="Lowest weight errors")
###

# Find lowest weight errors for each probability
nonstab_errorrates = sorted_errorrate[1:2^num_sites:end]
lowest_weight_err2 = []
for i in 0:(2^num_sites)-1
    start_ind = 1 + (i * 2^num_sites)
    error_list = sorted_error_list_string[start_ind:(i+1)*2^num_sites]
    sort!(error_list, lt = (x, y) -> count(z -> z == '1',x) > count(z -> z == '1',y))
    # push!(lowest_weight_err2, error_list[1])
    if count(z -> z == '1', error_list[1]) == count(z -> z == '1', error_list[2])
        push!(lowest_weight_err2, error_list[1:2])
    else
        push!(lowest_weight_err2, error_list[1])
    end
end
pretty_table([lowest_weight_err2 2^num_sites.*nonstab_errorrates], header=["Lowest weight error", "Probability"], title="Lowest weight errors", display_size = (-1, -1))

#for single error 
marginal_error = []
for i in lowest_weight_err2
    prob = 0
    non_one = findall(x -> x != '1', i)
    for j in eachindex(error_list_string)
        err = error_list_string[j]
        if err[non_one] == i[non_one]
            # println(err[non_one])
            prob += p[j]
        end
    end
    push!(marginal_error, prob)
end

sort_ind2 = sortperm(lowest_weight_err2, rev=true, lt = (x, y) -> count(z -> z == '1',x) < count(z -> z == '1',y))
pxpypz = zeros(marginal_error)

pretty_table([lowest_weight_err2[sort_ind2] marginal_error[sort_ind2]], header=["Lowest weight error", "Marginal probability"], title="Marginal probabilities of lowest weight errors")
# lowest_weight_err2[sortperm(lowest_weight_err2, rev=true, lt = (x, y) -> count(z -> z == '1',x) < count(z -> z == '1',y))]