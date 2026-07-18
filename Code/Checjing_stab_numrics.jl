
function create_exp_dict4L_new9march(γ1, η1, D1)
    exp_Dict4l = Dict()
    # in order 00->I, 01->X, 10->Z, 11->Y
    # [_ _ _ _] in order is 00 01 10 11
    a = (1-η1) * γ1^2
    k = (2*sqrt(D1*(1-D1)))
    exp_Dict4l[BitVector([0, 0])] = [1 k k 1] * (η1 + a)
    exp_Dict4l[BitVector([0, 1])] = [ (a+(k*η1)) (η1+(k*a)) (η1+(k*a)) (a+(k*η1))]
    exp_Dict4l[BitVector([1, 1])] = [0 -1 1 0] * η1 * (2*D1 - 1) * 1im
    exp_Dict4l[BitVector([1, 0])] = [1 0 0 -1] * η1 * (2*D1 - 1) 
    return exp_Dict4l
end

function stab_exp(StabBitString, expsDict)
    # Returns expectation value of operator
    # Op given by StabBitString
    # expsDict is dict of ops on the qubit basis

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

function mat_return(rel_exp)
    #rel_exp is relevant expectation vals
    #returns the matrix form of <MPS|Op|MPS>
    mat = [rel_exp[1] rel_exp[1] rel_exp[1] rel_exp[1] ;
    rel_exp[2] -rel_exp[2] rel_exp[2] -rel_exp[2];
    rel_exp[3]  rel_exp[3] -rel_exp[3] -rel_exp[3];
    rel_exp[4] -rel_exp[4] -rel_exp[4] rel_exp[4]]
    return mat/2
end

function stabGen_march13(n)
    stab_gen = []
    stabs = [[],[],[]]
    # add ZXZIIII.... as the base 
    base_list = BitVector(append!([1,0,0,1,1,0],repeat([0,],2(n-2))))
    #x = 1, z = 2 y = 3, i = 0
    push!(stab_gen, base_list[3:end])
    push!(stabs[1], base_list[3:end])
    for i in 1:n-1
        base_list = base_list >> 2
        push!(stab_gen, base_list[3:end])
        push!(stabs[1], base_list[3:end])
        i>1 ? push!(stabs[2], stab_gen[end] .⊻ stab_gen[end-1] .⊻ stab_gen[end-2]) : nothing
        i>1 ? push!(stabs[3], stab_gen[end] .⊻ stab_gen[end-2]) : nothing
    end
    stabs = vcat(stabs...)
    push!(stabs, stab_gen[1] .⊻ stab_gen[2] .⊻ stab_gen[end-1] .⊻ stab_gen[end])
    push!(stabs, stab_gen[3] .⊻ stab_gen[2])
    return stabs
    # stab_gen, 
end


# exp_Dict3l = create_exp_dict4L(γ, η, ζ);
γ, η, ζ = 0.3, 0.7, 0.9;
exp_Dict3l = create_exp_dict4L_new9march(γ, η, ζ);
n = 11;
norm = stab_exp(BitVector(repeat([0,],2n)), exp_Dict3l);
stabVal = [];
for stab in stabGen_march13(n)
    push!(stabVal, abs(stab_exp(stab, exp_Dict3l)))
end
stabVal/= norm ;
stabVal;

I_mat =  mat_return(exp_Dict3l[BitVector([0, 0])]);
Z_mat =  mat_return(exp_Dict3l[BitVector([1, 0])]);
X_mat =  mat_return(exp_Dict3l[BitVector([0, 1])]);
Y_mat =  mat_return(exp_Dict3l[BitVector([1, 1])]);
#matrix of 11 qubits
(I_mat/I_mat[1]);

# 2 - 3 - 6
I_mat^2 * Z_mat * Y_mat * X_mat * Y_mat * Z_mat * I_mat^4 

# [[1, 1,1,1];]' *
-real([1 1 1 1;] * I_mat^3 * Z_mat * Y_mat * X_mat * Y_mat * Z_mat * I_mat^3 * [1 0 0 0;]' /norm  )[1]  ≈ real(stabVal[n+3])

I_mat * I_mat^2
I_mat^2 * I_mat
I_mat^4
size([[1, 0,0,0];]')

X_mat 
Z_mat * X_mat * Z_mat ./ (4 * Z_mat[1]^2)


even_bulk = [stabvals[2*num_chain+2]/stabvals[4] , stabvals[3]/sqrt(stabvals[2*num_chain+1])]
odd_bulk = [stabvals[2*num_chain+1]/stabvals[3],  stabvals[2]/sqrt(stabvals[2*num_chain])]

yo = sqrt(zyyz_stabs[2] * yxyz_stab/(zyyz_stabs[1] * even_bulk[1]))/even_bulk[2]
insert!(odd_bulk, 2, yo)

ye = sqrt(zyyz_stabs[2] * zyyz_stabs[1]* even_bulk[1]/(yxyz_stab ))/odd_bulk[2]
insert!(even_bulk, 2, ye)

bound_1 = [stabvals[1]/(even_bulk[3]), zyyz_stabs[1]*sqrt(even_bulk[1]/(even_bulk[2]*even_bulk[3]))/odd_bulk[3] , odd_bulk[3]*sqrt(even_bulk[1]*even_bulk[2]/even_bulk[3])]
bound_2 = [sqrt(even_bulk[1]*even_bulk[3]/even_bulk[2]) , sqrt(even_bulk[2]*even_bulk[3]/even_bulk[1])]
