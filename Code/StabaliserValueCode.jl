using Plots, LinearAlgebra
# γ, η, ζ = 0.1, 0.9, 0.9

function create_exp_dict3L(γ1, η1, ζ1)
    exp_Dict3l = Dict()
    # in order 00->I, 01->X, 10->Z, 11->Y
    # [_ _ _ _] in order is <0|Op|0> <0|Op|1> <1|Op|0> <1|Op|1>
    a = (2*γ1^2*(1-η1))
    b = η1*ζ1
    exp_Dict3l[BitVector([0, 0])] = [1 0 0 1] * (a + η1)
    exp_Dict3l[BitVector([0, 1])] = [a b b a]
    exp_Dict3l[BitVector([1, 1])] = [0 -1 1 0] * η1 * ζ1 * 1im
    exp_Dict3l[BitVector([1, 0])] = [1 0 0 -1] * η1
    return exp_Dict3l
end

function create_exp_dict4L(γ1, η1, D1)
    exp_Dict4l = Dict()
    # in order 00->I, 01->X, 10->Z, 11->Y
    # [_ _ _ _] in order is 00 01 10 11
    a = η1 + ((1-η1) * γ1^2)
    b = (2*η1*sqrt(D1*(1-D1))) + ((1-η1) * γ1^2)
    exp_Dict4l[BitVector([0, 0])] = [a b b a]
    exp_Dict4l[BitVector([0, 1])] = [b a a b]
    exp_Dict4l[BitVector([1, 1])] = [0 -1 1 0] * η1 * (2*D1 - 1) * 1im
    exp_Dict4l[BitVector([1, 0])] = [1 0 0 -1] * η1 * (2*D1 - 1)
    return exp_Dict4l
end
# create_exp_dict4L(0.1, 0.9, 0.9)

function mat_return(rel_exp)
    #rel_exp is relevant expectation vals
    mat = [rel_exp[1] rel_exp[1] rel_exp[1] rel_exp[1] ;
    rel_exp[3] -rel_exp[3] rel_exp[3] -rel_exp[3];
    rel_exp[2]  rel_exp[2] -rel_exp[2] -rel_exp[2];
    rel_exp[4] -rel_exp[4] -rel_exp[4] rel_exp[4]]
    return mat/2
end

function stab_exp(StabBitString, expsDict)
    l = length(StabBitString)
    prod = [1 1 1 1]
    for i in 1:l÷2
        prod *= mat_return(expsDict[StabBitString[2i-1:2i]])
        # println(i, " prod ", prod)
    end
    prod *= transpose([1 0 0 0])
    # println("prod size :", size(prod))
    return prod[1,1]
end

function stabGen(n)
    stab_gen = []
    stabs = [[],[],[]]
    base_list = BitVector(append!([1,0,0,1,1,0],repeat([0,],2(n-2))))
    #x = 1, z = 2 y = 3, i = 0
    push!(stab_gen, base_list[3:end])
    push!(stabs[1], base_list[3:end])
    for i in 1:n-1
        base_list = base_list >> 2
        push!(stab_gen, base_list[3:end])
        push!(stabs[1], base_list[3:end])
        push!(stabs[2], stab_gen[end] .⊻ stab_gen[end-1])
        i>1 ? push!(stabs[3], stab_gen[end] .⊻ stab_gen[end-2]) : nothing
    end
    stabs = vcat(stabs...)
    push!(stabs, stab_gen[1] .⊻ stab_gen[2] .⊻ stab_gen[3])
    if isodd(n)
        stab_gen = stab_gen[1:end-1]
        push!(stab_gen, stab_gen[2] .⊻ stab_gen[4] )
    end
    return stab_gen, stabs
end

function stabGen2(n)
    stab_gen = []
    stabs = [[],[],[]]
    base_list = BitVector(append!([1,0,0,1,1,0],repeat([0,],2(n-2))))
    #x = 1, z = 2 y = 3, i = 0
    push!(stab_gen, base_list[3:end])
    push!(stabs[1], base_list[3:end])
    for i in 1:2
        base_list = base_list >> 2
        push!(stab_gen, base_list[3:end])
        push!(stabs[1], base_list[3:end])
        push!(stabs[2], stab_gen[end] .⊻ stab_gen[end-1])
    end
    push!(stabs[3], stab_gen[end] .⊻ stab_gen[end-2])
    push!(stabs[3], stab_gen[end] .⊻ stab_gen[end-1] .⊻ stab_gen[end-2])
    stabs = vcat(stabs...)
    # push!(stabs, stab_gen[1] .⊻ stab_gen[2] .⊻ stab_gen[3])
    return stab_gen, stabs
end

function stabGen3(n)
    stab_gen = []
    stabs = []
    base_list = BitVector(append!([1,0,0,1,1,0],repeat([0,],2(n-2))))
    #x = 1, z = 2 y = 3, i = 0
    push!(stab_gen, base_list[3:end])
    push!(stabs, base_list[3:end])
    for i in 1:3
        base_list = base_list >> 2
        push!(stab_gen, base_list[3:end])
        push!(stabs, base_list[3:end])
        # push!(stabs[2], stab_gen[end] .⊻ stab_gen[end-1])
    end
    # push!(stabs[3], stab_gen[end] .⊻ stab_gen[end-2])
    # push!(stabs[3], stab_gen[end] .⊻ stab_gen[end-1] .⊻ stab_gen[end-2])
    push!(stabs, stab_gen[1] .⊻ stab_gen[2])
    # println(stabs)
    # stabs = vcat(stabs...)
    # push!(stabs, stab_gen[1] .⊻ stab_gen[2] .⊻ stab_gen[3])
    return stab_gen, stabs
end


function return_errors_redu3(γ, η, ζ)
    exp_Dict3l = create_exp_dict3L(γ, η, ζ);
    println("n ", n)
    norm_I = stab_exp(BitVector(repeat([0,],2n)), exp_Dict3l);
    stabVal = [];
    for stab in stabGen2(n)[2]
        push!(stabVal, abs(stab_exp(stab, exp_Dict3l)))
    end
    stabVal/= norm_I;
    A = [ 1 1 1 0 0 0 0 ; 1 1 0 0 0 1 0; 0 2 0 0 0 0 1; 0 1 1 1 0 0 0 ; 1 1 0 1 1 0 0 ; 1 1 1 0 0 0 1; 0 1 1 0 1 1 0];
    I_diag = [0 0 0 1 0 0; 0 0 0 0 1 1; 1 0 0 0 0 0 ; 0 1 0 0 0 1; 0 0 1 0 0 1; 0 1 0 0 1 0; ]; 
    unlog = exp.(inv(A) * log.(stabVal));
    err = (ones(6) - unlog[1:end-1])./2 ;
    error_mat = inv(I_diag) * err
    # x1, y1, z1, z2 , z3
    return error_mat, norm_I, unlog
end

function return_errors_redu4(γ, η, D)
    exp_Dict4l = create_exp_dict4L(γ, η, D);
    # println("n ", n)
    norm_I = stab_exp(BitVector(repeat([0,],2n)), exp_Dict4l);
    stabVal = [];
    for stab in stabGen(n)[2]
        push!(stabVal, abs(stab_exp(stab, exp_Dict4l)))
    end
    stabVal/= norm_I;
    # stabs XZIII, ZXZII, IZXZI, IIZXZ, YYZII
    # A = [1 0 0 0 0; 0 0 1 1 0; 1 0 0 0 1; 0 0 0 2 0;  0 1 0 1 0];
    # I_diag = [0 1 1 0 0; 1 0 1 0 0; 1 1 0 0 0 ; 0 0 0 1 0; 0 0 0 0 1]; 
    # println(stabVal)
    A = zeros(Int, 3n, 3n);
    I_diag = zeros(Int, 3n, 3n);
    for i in 1:n-1
        #block 1
        # i 2 to n-1
        # -------ZXZ------ centred at zXz
        A[i, 3(i-1)+1:3i] = [1 0 0]
        A[i, 3i+1:3(i+1)] = [0 0 1]
        if i == 1
            A[n, end-5:end] = [1 1 0 1 0 0]
        else
            A[i, 3(i-2)+1:3(i-1)] = [0 0 1]
        end

        #block 2
        # zYyz centred at Y
        A[n+i, 3(i-1)+1:3i] = [0 1 0]
        A[n+i, 3i+1:3(i+1)] = [0 1 0]
        i+2<=n ? A[n+i, 3(i+1)+1:3(i+2)] = [0 0 1] : nothing
        i-1 >= 1 ? A[n+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing

        #block 3
        # zXixz centred at X
        if i == (n-1)
            nothing
        else
            A[2n-1+i, 3(i-1)+1:3i] = [1 0 0]
            A[2n-1+i, 3(i+1)+1:3(i+2)] = [1 0 0]
            i+3 <= n ? A[2n-1+i, 3(i+2)+1:3(i+3)] = [0 0 1] : nothing
            # i+3 == n ? A[2n-1+i, 3(i+2)+1:3(i+3)] = [1 1 0] : nothing
            i-1 >= 1 ? A[2n-1+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing
            # i-1 == 1 ? A[2n-1+i, 3(i-2)+1:3(i-1)] = [1 1 0] : nothing
        end

        I_diag[3i-2:3i ,3i-2:3i ] = ones(Int, 3, 3) - I(3)
    end 

    I_diag[3n-2:3n ,3n-2:3n ] = ones(Int, 3, 3) - I(3);
    I_diag = I_diag[1:end .!= 4, 1:end .!= 4];
    I_diag = I_diag[1:end .!= end-3, 1:end .!= end-3];
    # I_diag
    # Y X Y Z
    A[3n-2,1:12] = [0 1 0 1 0 0 0 1 0 0 0 1];
    for i in 1:3n
        if A[i,6] == 1
            A[i, 4:6] = [1 1 0]
        end
        if A[i,end-3] == 1
            A[i, end-5:end-3] = [1 1 0]
        end
    end
    # A
    A = A[:,1:end .!= end-3] ;
    A = A[:, 1:end .!= 6];
    A = A[1:end-2, :]

    # println(stabVal)
    unlog = exp.(inv(A) * log.(stabVal));
    err = (ones(3n-2) - unlog)/2 ;
    error_mat = inv(I_diag) * err
    insert!(error_mat, 6, 0)
    insert!(error_mat, 3n-3, 0)
    return reshape(error_mat, 3, :)', norm_I, stabVal
end

function return_errors(γ, η, ζ, lev, bound_cond)
    if lev == 3
        exp_Dict3l = create_exp_dict3L(γ, η, ζ);
    else #lev == 4
        exp_Dict3l = create_exp_dict4L(γ, η, ζ);
    end
    norm = stab_exp(BitVector(repeat([0,],2n)), exp_Dict3l);
    stabVal = [];
    for stab in stabGen(n)[2]
        push!(stabVal, abs(stab_exp(stab, exp_Dict3l)))
    end
    stabVal/= norm ;

    A = zeros(Int, 3n, 3n);
    I_diag = zeros(Int, 3n, 3n);
    for i in 1:n-1

        #block 1
        # i 2 to n-1
        # -------ZXZ------ centred at zXz
        if i == 1
            A[i, 3(i-1)+1:3(i+1)] = [1 0 0 0 0 1]
            A[n, end-5:end] = [0 0 1 1 0 0]
        else 
            A[i, 3(i-1)+1:3i] = [1 0 0]
            A[i, 3(i-2)+1:3(i-1)] = [0 0 1]
            A[i, 3i+1:3(i+1)] = [0 0 1]
        end

        #block 2
        # zYyz centred at Y

        A[n+i, 3(i-1)+1:3i] = [0 1 0]
        A[n+i, 3i+1:3(i+1)] = [0 1 0]
        i+2<=n ? A[n+i, 3(i+1)+1:3(i+2)] = [0 0 1] : nothing
        i-1 > 0 ? A[n+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing

        #block 3
        # zXixz centred at X
        if i == (n-1)
            nothing
        else
            A[2n-1+i, 3(i-1)+1:3i] = [1 0 0]
            # if i == 1
            #     A[2n-1+i, 3(i-1)+1:3i] = [0 1 1]
            # else
            #     A[2n-1+i, 3(i-1)+1:3i] = [1 0 0]
            # end
            # A[2n-1+i, 3i+1:3(i+1)] = [1 0 0]
            A[2n-1+i, 3(i+1)+1:3(i+2)] = [1 0 0]
            # if i == n-2
            #     A[2n-1+i, 3(i+1)+1:3(i+2)] = [0 1 1]
            # else
            #     A[2n-1+i, 3(i+1)+1:3(i+2)] = [1 0 0]
            # end
            i+3<= n ? A[2n-1+i, 3(i+2)+1:3(i+3)] = [0 0 1] : nothing
            i-1 > 0 ? A[2n-1+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing
        end

        I_diag[3i-2:3i ,3i-2:3i ] = ones(Int, 3, 3) - I(3)
    end 
    I_diag[3n-2:3n ,3n-2:3n ] = ones(Int, 3, 3) - I(3);
    # I_diag = I_diag[2:end-1, 2:end-1];
    # Y X Y Z
    A[3n-2,1:12] = [0 1 0 1 0 0 0 1 0 0 0 1];
    bound_cond == 'X' ? a = [1, 3n-2] : a = [6, 3n-3]
    # rank(A) == 3n-2 
    for i in 1:3n
        if bound_cond == 'X'
            A[i,1] == 1 ? A[i,1:3] = [0 1 1] : nothing
            A[i, end-2] == 1 ? A[i, end-2:end] = [0 1 1] : nothing
        else #bound_cond == 'Z'
            A[i,6] == 1 ? A[i,4:6] = [1 1 0] : nothing
            A[i, end-3] == 1 ? A[i, end-5:end-3] = [1 1 0] : nothing
        end
    end
    bound_cond == 'X' ? A = (A[:,1:end .!= end-2])[:, 1:end .!= 1] : A = (A[:,1:end .!= end-3])[:, 1:end .!= 6]
    bound_cond == 'X' ? I_diag = I_diag[2:end-1, 2:end-1] : I_diag = (I_diag[1:end .!= end-3,1:end .!= end-3])[1:end .!= 6, 1:end .!= 6]    
    # print(I_diag)
    A = A[1:end-2, :]
    unlog = exp.(inv(A) * log.(stabVal));
    err = (ones(3n-2) - unlog)/2 ;

    error_mat = reshape(insert!(insert!(inv(I_diag) * err, a[1], 0), a[2], 0), (3, :));
    print(stabVal)
    return error_mat'
end

function return_errors_only4test(γ, η, ζ)
    exp_Dict3l = create_exp_dict4L(γ, η, ζ);
    norm = stab_exp(BitVector(repeat([0,],2n)), exp_Dict3l);
    stabVal = [];
    for stab in stabGen(n)[1]
        push!(stabVal, abs(stab_exp(stab, exp_Dict3l)))
    end
    stabVal/= norm ;
    print("Stabval :", stabVal)

    A = zeros(Int, n, n);
    # I_diag = I(n);
    for i in 1:n-1

        #block 1
        # i 2 to n-1
        # -------ZXZ------ centred at zXz
        if i == 1
            A[i, 1:2] = [0 1]
            A[n, end-1:end] = [1 0]
        else 
            A[i, (i-1):i+1] = [1 0 1]
        end

    end 
    if isodd(n)
        A[end,end-1:end] = [0,0]
        A[end, 1:5] = [1,0,0,0,1]
    end
    unlog = exp.(inv(A) * log.(stabVal));
    err = (ones(n) - unlog)/2 ;

    # error_mat = reshape(insert!(insert!(inv(I_diag) * err, a[1], 0), a[2], 0), (3, :));

    return err
end

γ, η, ζ = (0.5,1,1)
n = 7
η_list = 0.7:0.01:0.999
z_list = []
x_list = []
for η in η_list
    push!(z_list, return_errors(γ, η, 1, 3, 'X')[4,3])
    push!(x_list, return_errors(γ, η, 1, 4, 'Z')[4,1])
end
plot!(η_list, z_list, label="3L Z error γ = $γ", marker=:circle);
plot!(η_list, x_list, label="4L X error γ = $γ", marker=:star)


plot(z_list - x_list)

all(x_list .== 0 )
all(z_list .== 0 )
plot!(yscale=:log10, xscale=:log10)
plot!(ylabel= "Probability", xlabel = "η", title = "Comparing main emmitter errors for diff γ ")
