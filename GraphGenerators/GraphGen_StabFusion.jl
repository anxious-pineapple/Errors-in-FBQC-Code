#
    # This code is to generate the pauli error rates as perceived in cluster graph states produced by 
    # 3 level and 4 level atoms, both on the states on their own as well as after fusion.
    # There are also Pluto notebooks with sliders for visualising, but this is the core code for documentation
#

using Plots
using LinearAlgebra


function stab_to_pauli_pl(stab_list)
    # Converts stabaliser measurements to pauli errors, returns in order of stabalisers + po
    # But convention would be X, Y, Z, I

	temp_probs = (1 - stab_list[1] - stab_list[2] - stab_list[3]) .* [1.0,1.0,1.0]
	temp_probs += 2*stab_list
	push!(temp_probs, (1+sum(stab_list)))
	return temp_probs/4
end

function pauli_err_fusion3(γ1, η1, ζ1)
    # Calculate thes stabaliser measurements of the input and output states
    # for a 3 level system undergoing fusion (Ref obsidian)
    # Then the corresponding pauli errors
    # Then pauli errors induced by fusion, and returns the same in an array

    N_1inv = ( η1 + (2 * (1-η1) * γ1^2) )^(-2)
    stab_in =  [ (ζ1*η1^2 + (4* (1-η1)^2 *γ1^4))*N_1inv ,
                η1^2 * (ζ1) * N_1inv,
                η1^2*N_1inv
    ]

    N_Tinv =  ( (32* (1-η1)^2 *γ1^4) + (16 * (1-η1) * η1 * γ1^2) + (η1^2))^(-1)
    stab_out =  [ (ζ1^2*η1^4*(N_Tinv) + (4* (1-η1)^2 *γ1^4))*N_1inv ,
                η1^4 * (ζ1^2) * N_1inv*N_Tinv,
                η1^4*N_1inv*N_Tinv
    ]


    px, py, pz, po = stab_to_pauli_pl(stab_in)
    p_oz = po + pz
    m_oz = po - pz
    p_xy = px + py
    m_xy = px - py
    Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    P_oz = Po + Pz
    M_oz = Po - Pz
    P_xy = Px + Py
    M_xy = Px - Py

    po_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) + ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    pz_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) - ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    px_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) + ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    py_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) - ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    return (0.5 .* [px_, py_, pz_, po_])
end

function pauli_err_fusion4(γ1, η1, D1)
    # Calculate thes stabaliser measurements of the input and output states
    # for a 4 level system undergoing fusion (Ref obsidian/one note)
    # Then the corresponding pauli errors
    # Then pauli errors induced by fusion, and returns the same in an array

	γ_2 = γ1^2
	
	η_tilde = (1-η1) * γ_2
	
	k = sqrt(4*D1*(1-D1))
	N_2 = 1 + k^2
	
	
	B = (2*η1*η_tilde)
	ApC =  η1^2 + (4*B) + (2* B^2)
	
	αpϵ = (ApC * N_2^2) + (4*k^2 *η1^2) + (8*k*N_2*B)
	δpχ = (η1^2 * N_2^2) + (4*k^2 * ApC) + (8*k*N_2*B)
	
	
	XX = η_tilde^2 + (η1^2 * δpχ / αpϵ) + (4*η1*η_tilde * B/αpϵ)
	ZZ = (η1^4 * (N_2-2)^2)/αpϵ
	YY = ZZ

	stab_out = [XX, YY, ZZ]
	
	XX_i =  (η_tilde^2 + η1^2) + (4*k*η1*η_tilde/N_2)
	ZZ_i = η1^2 * ( 1 - k^2)/N_2
	stab_in = [XX_i, ZZ_i, ZZ_i]
	
    px, py, pz, po = stab_to_pauli_pl(stab_in)
    p_oz = po + pz
    m_oz = po - pz
    p_xy = px + py
    m_xy = px - py
    Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    P_oz = Po + Pz
    M_oz = Po - Pz
    P_xy = Px + Py
    M_xy = Px - Py

    po_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) + ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    pz_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) - ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    px_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) + ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    py_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) - ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    return (0.5 .* [px_, py_, pz_, po_])
    # , stab_in, stab_out
end

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

function mat_return(rel_exp)
    #rel_exp is relevant expectation vals
    #returns the matrix form of <MPS|Op|MPS>
    mat = [rel_exp[1] rel_exp[1] rel_exp[1] rel_exp[1] ;
    rel_exp[3] -rel_exp[3] rel_exp[3] -rel_exp[3];
    rel_exp[2]  rel_exp[2] -rel_exp[2] -rel_exp[2];
    rel_exp[4] -rel_exp[4] -rel_exp[4] rel_exp[4]]
    return mat/2
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

function stabGen(n)
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
        push!(stabs[2], stab_gen[end] .⊻ stab_gen[end-1])
        i>1 ? push!(stabs[3], stab_gen[end] .⊻ stab_gen[end-2]) : nothing
    end
    stabs = vcat(stabs...)
    push!(stabs, stab_gen[1] .⊻ stab_gen[2] .⊻ stab_gen[3])
    return stabs
    # stab_gen, 
end

function return_errors(γ, η, ζ, lev, bound_cond='X', n=10)
    #Returns the pauli error rates on each qubit seperately for a chain of n qubits
    # As a n x 3 matrix, row = qubit, col = X, Y, Z error rates

    # A is the matrix connecting the stabaliser measurements to the log of the error rates
    # Boundary conditions set either X or Z on the ends as zero
    # The various stabs used are in the blocks : 
    # n ZXZ along the chain
    # n-1 zYyz along the chain
    # n-2 zXixz along the chain
    # And the last stab is YXYZ on the start

    # Generally boundary conditions : 3=X 4=Z

    # To note, since in the 4 level there is only X errors, there is possibly a more efficient way to do this 
    # but this is general for both 3 and 4 level

    if lev == 3
        exp_Dict3l = create_exp_dict3L(γ, η, ζ);
    else #lev == 4
        exp_Dict3l = create_exp_dict4L(γ, η, ζ);
    end
    norm = stab_exp(BitVector(repeat([0,],2n)), exp_Dict3l);
    stabVal = [];
    for stab in stabGen(n)
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
            A[2n-1+i, 3(i+1)+1:3(i+2)] = [1 0 0]
            i+3<= n ? A[2n-1+i, 3(i+2)+1:3(i+3)] = [0 0 1] : nothing
            i-1 > 0 ? A[2n-1+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing
        end

        I_diag[3i-2:3i ,3i-2:3i ] = ones(Int, 3, 3) - I(3)
    end 
    I_diag[3n-2:3n ,3n-2:3n ] = ones(Int, 3, 3) - I(3);
    # I_diag = I_diag[2:end-1, 2:end-1];
    # Y X Y Z
    A[3n-2,1:12] = [0 1 0 1 0 0 0 1 0 0 0 1];
    bound_cond == 'X' ? a = [1, 3n-2] : a = [6, 3n-3] # where to add zeros in error list
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

    A = A[1:end-2, :]
    unlog = exp.(inv(A) * log.(stabVal));
    err = (ones(3n-2) - unlog)/2 ;

    error_mat = reshape(insert!(insert!(inv(I_diag) * err, a[1], 0), a[2], 0), (3, :));
    return error_mat'
end

plot(return_errors(0.2, 0.9, 0.8, 3, 'X', 7))


# 3 Level : Calculating variation of Z error with ζ 
    ζ_list = [0.7:0.01:1.0;]
    η_1 = 1
    γ_1 = 0.2
    p = Plots.plot(ζ_list, [pauli_err_fusion3(γ_1, η_1, ζ)[3] for ζ in ζ_list], label="γ=0.2, η=1.0", xlabel="ζ", ylabel="Pauli Z Error Rate", title="3 Level Atom Fusion Pauli Z Error vs ζ", legend=:topright)
    η_2 = 0.99
    γ_2 = 0.2
    Plots.plot!(ζ_list, [pauli_err_fusion3(γ_2, η_2, ζ)[3] for ζ in ζ_list], label="γ=0.2, η=0.99", marker=(:circle,4))
    η_3 = 0.95
    γ_3 = 0.2
    Plots.plot!(ζ_list, [pauli_err_fusion3(γ_3, η_3, ζ)[3] for ζ in ζ_list], label="γ=0.2, η=0.95", marker=(:diamond,4))
# Similarly can do other errors/parameters



return_errors(0.1, 0.9, 0.9, 3, 'X')
return_errors(0.1, 0.9, 0.9, 4, 'Z')



