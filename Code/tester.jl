using Optim

n = 2
neg_entropy(x) = sum(x .* log.(x .+ 1e-12))
res  = Optim.optimize(f, [0.5, 0.5])

res.minimizer
res.minimum

A = rand([1,-1],2^n,4^n)
b = ones(2^n)

# for i in CartesianIndex(A)
#     println(i)
# end




function con_c!(c, x)
    c[1:2^n] =  A*x - b      # 1st constraint
    c[2^n + 1] =   sum(x) - 1 # 2nd constraint
    c
end
function con_Jac!(J, x)    # Jacobian of the constraint
    for i in CartesianIndices(A)
        J[i[1], i[2]] = A[i[1], i[2]]
    end
    J[2^n + 1, 1:2^n] .= 1.0   # Jacobian of the normalization constraint
    # J[2^n + 1, 2^n + 1] = 0.0 
    J
end
function con_Hess!(h, x, λ)        # Hessian of the constraint
end

dfc= TwiceDifferentiableConstraints(con_c!, con_Jac!, con_Hess!, zeros(2^n), ones(2^n), zeros(2^n + 1), zeros(2^n + 1))

x0 = ones(4^n)./(4^n)
res = Optim.optimize(neg_entropy, dfc , x0, IPNewton())


[0,10] .* [1,2]