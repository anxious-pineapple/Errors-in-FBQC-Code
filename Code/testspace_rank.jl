using LinearAlgebra

px = 0.01
py = 0.005
pz = 0.007
po = 1 - (px + py + pz)

function get_gate(pauli_errs)
    px, py, pz = pauli_errs
    po = 1 - (px + py + pz)
    rand_no = rand()
    # println(rand_no)
    if rand_no < po
        return "I"
    elseif rand_no < po + px
        return "X"
    elseif rand_no < po + px + py
        return "Y"
    else
        return "Z"
    end
    
end

gate = get_gate((px, py, pz))
println(gate)

function stab_val(stab_list, pauli_errs)
    exp_val = 1.0
    for gate in stab_list
        if gate == "I"
            nothing
        else
            err = get_gate(pauli_errs)
            if err == "I"
                nothing
            else
                err == gate ? exp_val *= 1 : exp_val *= -1
            end
        end
    end
    return exp_val
end


function stab_exp_val(stab_list, pauli_errs; num_trials=5000000)
    exp_val = 0.0
    for _ in 1:num_trials
        exp_val += stab_val(stab_list, pauli_errs)
    end
    return exp_val / num_trials
end

println(stab_exp_val(["Z", "X", "Z"], (px, py, pz)))
println(stab_exp_val(["Z", "Y", "Y", "Z"], (px, py, pz)))
println(stab_exp_val(["Z","X", "I","X", "Z"], (px, py, pz)))

n = 7
A = zeros(Int, 3n, 3n)
I_diag = zeros(Int, 3n, 3n)

A
rank(A)

for i in 0:n-1 
    i_1 = i - 1 < 0 ? i - 1 + n : i - 1
    i_2 = i - 2 < 0 ? i - 2 + n : i - 2
    i_p1 = i + 1 > n-1 ? i + 1 - n : i + 1
    i_p2 = i + 2 > n-1 ? i + 2 - n : i + 2

    println([i_2, i_1, i, i_p1, i_p2])

    A[3i+1 ,3i+1:3i+3 ] = [ 1 0 0]    
    A[3i+2 ,3i+1:3i+3 ] = [ 0 1 0]
    A[3i+3 ,3i+1:3i+3 ] = [ 0 0 0]
    # A[3i+3 ,3i+1:3i+3 ] = [ 1 -1 -1]

    A[3i+1 ,3i_1+1:3i_1+3 ] = [ 0 0 1]    
    A[3i+2 ,3i_1+1:3i_1+3 ] = [ 0 0 1]
    A[3i+3 ,3i_1+1:3i_1+3 ] = [ 1 0 0]
    # A[3i+3 ,3i_1+1:3i_1+3 ] = [1 -1 1 -1]

    A[3i+1 ,3i_p1+1:3i_p1+3 ] = [ 0 0 1]    
    A[3i+2 ,3i_p1+1:3i_p1+3 ] = [ 0 1 0]
    A[3i+3 ,3i_p1+1:3i_p1+3 ] = [ 1 0 0]
    # A[3i+3 ,3i_p1+1:3i_p1+3 ] = [1 -1 1 -1]

    A[3i+3 ,3i_2+1:3i_2+3 ] = [ 0 0 1]    
    # A[3i+3 ,3i_2+1:3i_2+3 ] = [1 -1 -1 1]

    A[3i+2 ,3i_p2+1:3i_p2+3 ] = [ 0 0 1]
    A[3i+3 ,3i_p2+1:3i_p2+3 ] = [ 0 0 1]
    # A[4i+3 ,4i_p2+1:4i_p2+3 ] = [1 -1 -1 1]

    I_diag[3i+1:3i+3 ,3i+1:3i+3 ] = ones(Int, 3, 3) - I(3)
end 

I_diag

rank(A)
A
b = repeat([0.78, 0.72, 0.68], n)
unlog = exp.(inv(A) * log.(b))
err = (ones(3n) - unlog)/2
inv(I_diag) * err


(ones(3n) - exp.(inv(A) * log.(b))) ./2

test_err = repeat([0.01, 0.02, 0.04], n)
test_err_t = 1 .- (2 .* I_diag * test_err)
exp.(A * log.(test_err_t))

log(exp(1))


exp.(b)

inv(A - I_diag) * b
b
rank(I_diag -A)
# A[:, end] = repeat([-1, 1, 1], n)
# A^-1
inv(A) * b
inv((1 .-A)./2) * b
inv(A) * (1 .- b)./2
inv( (1 .- A)./2) * (1 .- b)/2

b = repeat([0.01, 0.01, 0.01], n);
b = 1   .-     b            
inv(-I_diag + A) * b
b = (1 .- b)/2;
p = reshape(inv(A) * b, (3, n))
p;
# sum(p, dims=1)[1]
# p ./ sum(p, dims=1)[1]


b = [-1 1 1; 1 -1 1; 1 1 -1]
[1 0 0; 1 9 1; 1 1 2] * [1 3 4]'

function periodic_cond(ind, n)
    if ind < 1
        return ind + n
    elseif ind > n
        return ind - n
    else
        return ind
    end
end

#################################
# Linear chain

#at least 8 long
n = 8
A = zeros(Int, 3n, 3n)
I_diag = zeros(Int, 3n, 3n)
for i in 1:n-1

    #block 1
    # i 2 to n-1
    if i == 1
        A[i, 3(i-1)+1:3i] = [0 1 1]
        A[i, 3i+1:3(i+1)] = [0 0 1]
        A[n, end-5:end] = [0 0 1 0 1 1]
    else 
        A[i, 3(i-1)+1:3i] = [1 0 0]
        A[i, 3(i-2)+1:3(i-1)] = [0 0 1]
        A[i, 3i+1:3(i+1)] = [0 0 1]
    end

    #block 2
    A[n+i, 3(i-1)+1:3i] = [0 1 0]
    A[n+i, 3i+1:3(i+1)] = [0 1 0]
    i+2<=n ? A[n+i, 3(i+1)+1:3(i+2)] = [0 0 1] : nothing
    i-1 > 0 ? A[n+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing

    #block 3
    if i == (n-1)
        nothing
    else
        if i == 1
            A[2n-1+i, 3(i-1)+1:3i] = [0 1 1]
        else
            A[2n-1+i, 3(i-1)+1:3i] = [1 0 0]
        end
        # A[2n-1+i, 3i+1:3(i+1)] = [1 0 0]
        if i == n-2
            A[2n-1+i, 3(i+1)+1:3(i+2)] = [0 1 1]
        else
            A[2n-1+i, 3(i+1)+1:3(i+2)] = [1 0 0]
        end
        i+3<= n ? A[2n-1+i, 3(i+2)+1:3(i+3)] = [0 0 1] : nothing
        i-1 > 0 ? A[2n-1+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing
    end

    I_diag[3i-2:3i ,3i-2:3i ] = ones(Int, 3, 3) - I(3)
end 
i = n
I_diag[3i-2:3i ,3i-2:3i ] = ones(Int, 3, 3) - I(3)
I_diag = I_diag[2:end-1, 2:end-1]
# Y X Y Z
A[3n-2,1:12] = [0 1 0 1 0 0 0 1 0 0 0 1]
A = A[1:end-2, 1:end .!= 3n-2][:,2:end]

b = repeat([0.78, 0.72, 0.68], n)
unlog = exp.(inv(A) * log.(b))
err = (ones(3n) - unlog)/2
inv(I_diag) * err

