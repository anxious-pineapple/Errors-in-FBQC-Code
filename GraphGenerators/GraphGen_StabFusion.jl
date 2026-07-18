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


    # px, py, pz, po = stab_to_pauli_pl(stab_in)
    # p_oz = po + pz
    # m_oz = po - pz
    # p_xy = px + py
    # m_xy = px - py
    # Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    # P_oz = Po + Pz
    # M_oz = Po - Pz
    # P_xy = Px + Py
    # M_xy = Px - Py

    # po_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) + ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    # pz_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) - ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    # px_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) + ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    # py_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) - ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    # return (0.5 .* [px_, py_, pz_, po_])

    px, py, pz, po = stab_to_pauli_pl(stab_in)
    a_var = (po + pz)^2 + (px+py)^2
    b_var = 2*(po + pz)*(px + py)
    c_var = (po - pz)^2 + (px-py)^2
    d_var = 2*(po - pz)*(px - py)
    Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    P_oz = Po + Pz
    M_oz = Po - Pz
    P_xy = Px + Py
    M_xy = Px - Py

    p_fuse_oz = (a_var*P_oz - b_var*P_xy)/(a_var - b_var)
    m_fuse_oz = (c_var*M_oz - d_var*M_xy)/(c_var^2 - d_var^2)

    p_fuse_xy = (a_var*P_xy - b_var*P_oz)/(a_var - b_var)
    m_fuse_xy = (c_var*M_xy - d_var*M_oz)/(c_var^2 - d_var^2)

    po_ = (p_fuse_oz + m_fuse_oz)/2
    pz_ = (p_fuse_oz - m_fuse_oz)/2
    px_ = (p_fuse_xy + m_fuse_xy)/2
    py_ = (p_fuse_xy - m_fuse_xy)/2
    return ([px_, py_, pz_, po_])

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
	
    # px, py, pz, po = stab_to_pauli_pl(stab_in)
    # p_oz = po + pz
    # m_oz = po - pz
    # p_xy = px + py
    # m_xy = px - py
    # Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    # P_oz = Po + Pz
    # M_oz = Po - Pz
    # P_xy = Px + Py
    # M_xy = Px - Py

    # po_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) + ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    # pz_ = ((p_oz*P_oz - p_xy*P_xy)/(p_oz^2 - p_xy^2)) - ((m_oz*M_oz - m_xy*M_xy)/(m_oz^2 - m_xy^2))
    # px_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) + ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    # py_ = ((p_oz*P_xy - p_xy*P_oz)/(p_oz^2 - p_xy^2)) - ((m_oz*M_xy - m_xy*M_oz)/(m_oz^2 - m_xy^2))
    # return (0.5 .* [px_, py_, pz_, po_])


    px, py, pz, po = stab_to_pauli_pl(stab_in)
    a_var = (po + pz)^2 + (px+py)^2
    b_var = 2*(po + pz)*(px + py)
    c_var = (po - pz)^2 + (px-py)^2
    d_var = 2*(po - pz)*(px - py)
    Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    P_oz = Po + Pz
    M_oz = Po - Pz
    P_xy = Px + Py
    M_xy = Px - Py

    p_fuse_oz = (a_var*P_oz - b_var*P_xy)/(a_var - b_var)
    m_fuse_oz = (c_var*M_oz - d_var*M_xy)/(c_var^2 - d_var^2)

    p_fuse_xy = (a_var*P_xy - b_var*P_oz)/(a_var - b_var)
    m_fuse_xy = (c_var*M_xy - d_var*M_oz)/(c_var^2 - d_var^2)

    po_ = (p_fuse_oz + m_fuse_oz)/2
    pz_ = (p_fuse_oz - m_fuse_oz)/2
    px_ = (p_fuse_xy + m_fuse_xy)/2
    py_ = (p_fuse_xy - m_fuse_xy)/2
    return ([px_, py_, pz_, po_])
    # , stab_in, stab_out
end

function pauli_err_fusion4_new(γ1, η1, D1)
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
    a_var = (po + pz)^2 + (px+py)^2
    b_var = 2*(po + pz)*(px + py)
    c_var = (po - pz)^2 + (px-py)^2
    d_var = 2*(po - pz)*(px - py)
    Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    P_oz = Po + Pz
    M_oz = Po - Pz
    P_xy = Px + Py
    M_xy = Px - Py

    p_fuse_oz = (a_var*P_oz - b_var*P_xy)/(a_var - b_var)
    m_fuse_oz = (c_var*M_oz - d_var*M_xy)/(c_var^2 - d_var^2)

    p_fuse_xy = (a_var*P_xy - b_var*P_oz)/(a_var - b_var)
    m_fuse_xy = (c_var*M_xy - d_var*M_oz)/(c_var^2 - d_var^2)

    po_ = (p_fuse_oz + m_fuse_oz)/2
    pz_ = (p_fuse_oz - m_fuse_oz)/2
    px_ = (p_fuse_xy + m_fuse_xy)/2
    py_ = (p_fuse_xy - m_fuse_xy)/2
    return ([px_, py_, pz_, po_])
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

#Acc new states
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

function mat_return(rel_exp)
    #rel_exp is relevant expectation vals
    #returns the matrix form of <MPS|Op|MPS>
    mat = [rel_exp[1] rel_exp[1] rel_exp[1] rel_exp[1] ;
    rel_exp[2] -rel_exp[2] rel_exp[2] -rel_exp[2];
    rel_exp[3]  rel_exp[3] -rel_exp[3] -rel_exp[3];
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
        # exp_Dict3l = create_exp_dict4L(γ, η, ζ);
        exp_Dict3l = create_exp_dict4L_new9march(γ, η, ζ);
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
    return error_mat', stabVal,unlog
end

function return_errors_coppymarch13(γ, η, ζ, lev, bound_cond='X', n=10)
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
        # exp_Dict3l = create_exp_dict4L(γ, η, ζ);
        exp_Dict3l = create_exp_dict4L_new9march(γ, η, ζ);
    end
    norm = stab_exp(BitVector(repeat([0,],2n)), exp_Dict3l);
    stabVal = [];
    for stab in stabGen_march13(n)
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
        # zYxyz centred at Y
        if i == (n-1)
            nothing
        else
            # Intended local Pauli pattern (up to boundary truncation): Z_{i-1} Y_i X_{i+1} Y_{i+2} Z_{i+3}
            A[n+i, 3(i-1)+1:3i] = [0 1 0]          # Y on qubit i
            A[n+i, 3i+1:3(i+1)] = [1 0 0]          # X on qubit i+1
            A[n+i, 3(i+1)+1:3(i+2)] = [0 1 0]      # Y on qubit i+2
            i+3<=n ? A[n+i, 3(i+2)+1:3(i+3)] = [0 0 1] : nothing  # Z on qubit i+3 (if present)
            i-1 > 0 ? A[n+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing
        end

        #block 3
        # zXixz centred at X
        if i == (n-1)
            nothing
        else
            A[2n-2+i, 3(i-1)+1:3i] = [1 0 0]
            A[2n-2+i, 3(i+1)+1:3(i+2)] = [1 0 0]
            i+3<= n ? A[2n-2+i, 3(i+2)+1:3(i+3)] = [0 0 1] : nothing
            i-1 > 0 ? A[2n-2+i, 3(i-2)+1:3(i-1)] = [0 0 1] : nothing
        end

        I_diag[3i-2:3i ,3i-2:3i ] = ones(Int, 3, 3) - I(3)
    end 
    I_diag[3n-2:3n ,3n-2:3n ] = ones(Int, 3, 3) - I(3);
    # I_diag = I_diag[2:end-1, 2:end-1];
    # Y Y Z ..... Z Y Y
    A[3n-3,1:12] = [0 1 0 0 1 0 0 0 1 0 0 0];
    A[3n-3,end-8:end] = [0 0 1 0 1 0 0 1 0];
    # Z Y Y Z
    A[3n-2,1:12] = [0 0 1 0 1 0 0 1 0 0 0 1];

    bound_cond == 'X' ? a = [1, 3n-2] : a = [6, 3n-3] # where to add zeros in error list
    println("Rank, ", rank(A) == 3n-2)
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

    println(rank(A))
    A = A[1:end-2, :]
    unlog = exp.(inv(A) * log.(stabVal));
    err = (ones(3n-2) - unlog)./2 ;

    error_mat = reshape(insert!(insert!(inv(I_diag) * err, a[1], 0), a[2], 0), (3, :));
    return error_mat', stabVal,unlog, err, norm
end


function pauli_err_fusion4_2(γ, η, D, V)
	γ_2 = γ^2
	
	η_t = (1-η) * γ_2
	
	k = sqrt(4*D*(1-D))
	N_2 = 1 + k^2
	D_t_2 = 1 - k^2
	
	trace = (η_t + (0.5*η))^2
	trace_term2 = η^2 * (1 + (V*( (D_t_2/N_2)^2 - 1 )))/8
	trace = trace - trace_term2
	
	XX = 4*k*η* ((k*η) + (η_t*(N_2^0.5))) * ((η_t + (0.5*η))^2 + η^2(V-1)/8)/N_2^2
	XX += (V* η^4 * D_t_2^2) / (8 * N_2^2)
	XX /= trace 
	XX += η_t^2

	
	ZZ = (η^4 * D_t_2^2)/(8 * N_2^2 * trace)
	YY = ZZ * V

	stab_out = [XX, YY, ZZ]
	
	XX_i =  (η_t^2 + η^2) + (4*k*η*η_t/N_2)
	ZZ_i = η^2 * D_t_2/N_2
	stab_in = [XX_i, ZZ_i, ZZ_i]
	
	px, py, pz, po = stab_to_pauli_pl(stab_in)
    a_var = (po + pz)^2 + (px+py)^2
    b_var = 2*(po + pz)*(px + py)
    c_var = (po - pz)^2 + (px-py)^2
    d_var = 2*(po - pz)*(px - py)
    Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
    P_oz = Po + Pz
    M_oz = Po - Pz
    P_xy = Px + Py
    M_xy = Px - Py

    p_fuse_oz = (a_var*P_oz - b_var*P_xy)/(a_var - b_var)
    m_fuse_oz = (c_var*M_oz - d_var*M_xy)/(c_var^2 - d_var^2)

    p_fuse_xy = (a_var*P_xy - b_var*P_oz)/(a_var - b_var)
    m_fuse_xy = (c_var*M_xy - d_var*M_oz)/(c_var^2 - d_var^2)

    po_ = (p_fuse_oz + m_fuse_oz)/2
    pz_ = (p_fuse_oz - m_fuse_oz)/2
    px_ = (p_fuse_xy + m_fuse_xy)/2
    py_ = (p_fuse_xy - m_fuse_xy)/2
    return ([px_, py_, pz_, po_], stab_in, stab_out) 
end

function pauli_err_fusion4_pluto(γ, η, D, V)
	γ_2 = γ^2
	
	η_t = (1-η) * γ_2
	
	k = sqrt(4*D*(1-D))
	N_2 = 1 + k^2
	D_t_2 = 1 - k^2
	
	trace = (η_t + (0.5*η))^2
	trace_term2 = η^2 * (1 + (V*( (D_t_2/N_2)^2 - 1 )))/8
	trace = trace - trace_term2

	II = (η + η_t)^2
	
	XX = 4*k*η* ((k*η) + (η_t*(N_2^0.5))) * ((η_t + (0.5*η))^2 + (η^2 * (V-1)/8))/N_2^2
	XX += (V* η^4 * D_t_2^2) / (8 * N_2^2)
	XX /= trace 
	XX += η_t^2
	ZZ = (η^4 * D_t_2^2)/(8 * N_2^2 * trace)
	YY = ZZ * V
	stab_out = [XX, YY, ZZ]/II
	
	XX_i =  (η_t^2) + (η^2) + (4*k*η*η_t/N_2)
	ZZ_i = η^2 * D_t_2/N_2
	stab_in = [XX_i, ZZ_i, ZZ_i]/II
	
	px, py, pz, po = stab_to_pauli_pl(stab_in)
	# po = 1 - px - py - pz
    p_oz = po + pz
	m_oz = po - pz
	p_xy = px + py
	m_xy = px - py

	
    Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
	
    P_oz = Po + Pz
    M_oz = Po - Pz
    P_xy = Px + Py
    M_xy = Px - Py

    p_fuse_oz = (P_oz - (2*p_oz*p_xy))/(p_oz - p_xy)^2
	m_fuse_oz_num = (M_oz*(m_oz+m_xy)^2) - (2*m_oz*m_xy*(M_oz+M_xy))
	m_fuse_oz_denom = (m_oz^2 - m_xy^2)^2
	m_fuse_oz = m_fuse_oz_num/m_fuse_oz_denom
	
	p_fuse_xy = (P_xy - (2*p_oz*p_xy))/((p_oz - p_xy)^2)
	m_fuse_xy_num = (M_xy*(m_oz+m_xy)^2) - (2*m_oz*m_xy*(M_oz+M_xy))
	m_fuse_xy_denom = (m_oz^2 - (m_xy^2))^2
	m_fuse_xy = m_fuse_xy_num/m_fuse_xy_denom
	
	po_ = (p_fuse_oz + m_fuse_oz)/2
	pz_ = (p_fuse_oz - m_fuse_oz)/2
	px_ = (p_fuse_xy + m_fuse_xy)/2
	py_ = (p_fuse_xy - m_fuse_xy)/2

	plus_oz = (p_oz^2) + (p_xy^2)
	minus_oz = (m_oz^2) + (m_xy^2)
	plus_xy = 2*p_oz*p_xy
	minus_xy = 2*m_xy*m_oz

	trial_o = (plus_oz+minus_oz)/2
	trial_z = (plus_oz-minus_oz)/2
	trial_x = (plus_xy+minus_xy)/2
	trial_y = (plus_xy-minus_xy)/2

	matrix_m = [trial_o trial_z trial_y trial_x;
            trial_z trial_o trial_x trial_y;
            trial_y trial_x trial_o trial_z;
            trial_x trial_y trial_z trial_o]

	out = inv(matrix_m) * [Px Py Pz Po]'
	
    return ([px_, py_, pz_, po_], stab_in, stab_out, [trial_x, trial_y, trial_z, trial_o], out, matrix_m) 
end
# plot(return_errors(0.2, 0.9, 0.8, 4, 'Z', 7))

pauli_err_fusion4_pluto(0.1, 0.99, 0.99, 0.99)[3]


2^(-408000)

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
    η_3 = 0.95
    γ_3 = 0.1
    Plots.plot!(ζ_list, [pauli_err_fusion3(γ_3, η_3, ζ)[3] for ζ in ζ_list], label="γ=0.1, η=0.95", marker=(:utriangle,4))
# Similarly can do other errors/parameters

num_chain = 31
d = 0.95
i = 0.8
# for i in 0.6:0.05:1
γ, η, D = 0.1, i, d 
er, stabvals, unlog, err, norm_ii = return_errors_coppymarch13(0.1, i, d, 4, 'Z', num_chain);
er2, stabvals2, unlog2 = return_errors(0.1, i, d, 3, 'X', num_chain);

plot(er2[:,1]);
plot!(er2[:,2]);
plot!(er2[:,3])
plot(stabvals2)

 η_t = (1-η) * γ^2
k = sqrt(4*D*(1-D))
b = η + (k*η_t)
a = η_t + (k*η)
b_a = b + (a* k^((num_chain-1)/2))
b_a2 = b^2 + (a^2 * k^((num_chain-3)/2))
k_1 = 1 + k^((num_chain+1)/ 2)
#checking unlog 

b_a/sqrt(b_a2)
# even chain
# px_ = b/(η + η_t)
# py_ = η * (2*D - 1) / (η + η_t)
# pz_ = py_
# px_1 = px_
# py_1 = py_ * sqrt(px_)
# pz_1 = py_1
# px_2 = sqrt(px_)
# py_2 = py_/sqrt(px_)

#odd chain
px_even = b/(η + η_t)
py_even = η * (2*D - 1) / (η + η_t)
pz_even = py_even * b_a / sqrt(k_1 * b_a2)
px_odd = b_a2 / ((η + η_t)*b_a)
py_odd = py_even * sqrt(b_a2)/b_a
pz_odd = py_even / sqrt(k_1)

px_1o = sqrt(b_a2/k_1)/(η + η_t)
py_1o = py_even * sqrt(px_even * sqrt(b_a2) / b_a)
pz_1o = py_1o / k_1^0.25
px_2o = sqrt(px_even) * sqrt(b_a/sqrt(k_1 * b_a2))
py_2o = py_even/sqrt(px_even) * sqrt(b_a/sqrt(k_1 * b_a2))



unlog2[1]≈ px_1o # should be px
(unlog2[2] - py_1o )/unlog2[2]# should be py
unlog2[3] ≈ pz_1o # should be pz
unlog2[4] ≈ px_2o # should be px
unlog2[5] ≈ py_2o # should be py
unlog2[6] ≈ px_odd # should be px
unlog2[7] ≈ py_odd # should be py
unlog2[8] ≈ pz_odd # should be pz
unlog2[9] ≈ px_even # should be px
unlog2[10] ≈ py_even # should be py
unlog2[11] ≈ pz_even # should be pz
unlog2[11:6:end]

plot(unlog2[8:3:end-4])

zxz_analytic = (2D -1)^2 * η^2 * b/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^3)
zxz_analytic_odd(N) = (2D -1)^2 * η^2 * (b + (a*k^((N-1)/2)))/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^3)
zxixz_analytic = (2D -1)^2 * η^2 * b^2/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^4)
zxixz_analytic_odd(N) = (2D -1)^2 * η^2 * (b^2 + (a^2 * k^((N-3)/2)))/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^4)
zyyz_analytic = (2D -1)^4 * η^4 / ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^4)
xz_analytic = (2D -1) * η * b/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^2)
xz_analytic_odd(N) = (2D -1) * η * (b + a * k^((N-1)/2))/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^2)
yxyz_analytic = (2D -1)^3 * η^3 * b/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^4)
xixz_analytic = (2D -1) * η * b^2/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^3)
xixz_analytic_odd(N) = (2D -1) * η * (b^2 + a^2 * k^((N-3)/2))/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^3)
yyz_analytic = (2D -1)^3 * η^3/ ((1 + (num_chain%2)*k^((num_chain+1) ÷ 2)) * (η + η_t)^3)

# iii_analytic
7 ÷ 2
(η + η_t)^num_chain  * (1 + (num_chain%2)*k^((num_chain+1) ÷ 2))
norm_ii

# unlog
# stabvals
# err
zxz_stabs = stabvals[1:num_chain]

zyxyz_stabs = stabvals[num_chain+1:2*num_chain-2]
zyyz_stabs = stabvals2[num_chain+1:2*num_chain-1]
zxixz_stabs = stabvals[2*num_chain-1:3*num_chain-4]
zxixz_stabs2 = stabvals2[2*num_chain:3*num_chain-3]
yxyz_stab = stabvals2[end]

zxz_analytic
zxz_stabs[2]
zxz_analytic_odd(num_chain)
zxz_stabs[3]

num_chain%2 == 0 ? xz_analytic ≈ zxz_stabs[1] : xz_analytic_odd(num_chain) ≈ zxz_stabs[1]

zxixz_analytic ≈ zxixz_stabs[2]
zxixz_analytic_odd(num_chain) ≈ zxixz_stabs[5]

num_chain%2 == 0 ? xixz_analytic ≈ zxixz_stabs[1] : xixz_analytic_odd(num_chain) ≈ zxixz_stabs[1]

zyyz_analytic ≈ zyyz_stabs[2]
yyz_analytic ≈ zyyz_stabs[1]
yxyz_analytic ≈ yxyz_stab

# 23 + 22 + 21 + 1
# length(stabvals)

even_bulk = [stabvals[2*num_chain+2]/stabvals[4] , sqrt((stabvals[num_chain+2] * stabvals[2*num_chain] * stabvals[3]) / (stabvals[2]^2 * stabvals[2*num_chain+1])) , stabvals[3]/sqrt(stabvals[2*num_chain+1])]
odd_bulk = [stabvals[2*num_chain+1]/stabvals[3],  sqrt((stabvals[num_chain+3] * stabvals[2*num_chain+1] * stabvals[2]) / (stabvals[3]^2 * stabvals[2*num_chain])), stabvals[2]/sqrt(stabvals[2*num_chain])]
bound_1 = [stabvals[1]/(even_bulk[3]), (stabvals[num_chain+1] / (odd_bulk[2] * even_bulk[3])) * sqrt(even_bulk[2]/(even_bulk[1]*even_bulk[3])) , odd_bulk[3]*sqrt(even_bulk[1]*even_bulk[2]/even_bulk[3])]
bound_2 = [sqrt(even_bulk[1]*even_bulk[2]/even_bulk[3]) , sqrt(even_bulk[2]*even_bulk[3]/even_bulk[1])]

isapprox((inv(ones(Int, 3, 3) - I(3)) * (1 .- even_bulk)./2), er2[4,:])
# even_bulk .- (1 .- 2((ones(Int, 3, 3) - I(3)) * er2[4,:]))
(100 .* ((inv(ones(Int, 3, 3) - I(3)) * (1 .- even_bulk)./2) .- er2[4,:]) ./ er2[4,:])
# er2[4,:]

# rec_unlog_even = (1 .- 2((ones(Int, 3, 3) - I(3)) * er2[4,:]))
# rec_unlog_odd = (1 .- 2((ones(Int, 3, 3) - I(3)) * er2[5,:]))
# isapprox( (rec_unlog_odd[3] * rec_unlog_even[2] * rec_unlog_odd[1] * rec_unlog_even[2] * rec_unlog_odd[3]) , stabvals[num_chain+                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    2])

(isapprox((inv(ones(Int, 3, 3) - I(3)) * (1 .- odd_bulk)./2), er2[5,:]))
println(100 .* ((inv(ones(Int, 3, 3) - I(3)) * (1 .- odd_bulk)./2) .- er2[5,:]) ./ er2[5,:])
# er2[5,:]

(isapprox((inv(ones(Int, 3, 3) - I(3)) * (1 .- bound_1)./2), er2[1,:]))
println(100 .* ((inv(ones(Int, 3, 3) - I(3)) * (1 .- bound_1)./2) .- er2[1,:]) ./ er2[1,:] )
# er2[1,:]

(isapprox((inv(ones(Int, 2, 2) - I(2)) * (1 .- bound_2)./2), er2[2,1:end-1]))
println(100 .* ((inv(ones(Int, 2, 2) - I(2)) * (1 .- bound_2)./2) .- er2[2,1:end-1]) ./ er2[2,1:end-1])
# er2[2,:]
# end

plot(stabvals[1:num_chain])
isapprox.([er[1,2];er[3:end-2,2];er[end, 2]], er[3,3], rtol=sqrt(eps(er[3,3])))
eps(stabvals[2])
plot(er[:,2])
plot(er2[:,1])
er2[4:2:end-2,1]
er2[3:2:end-2,1]
plot(unlog2)
plot!(unlog)
(unlog[6:end-5][2:3:end][2:end] .* unlog[6:end-5][5:3:end][1:12]) .- (unlog2[6:end-5][2:3:end][2:end] .* unlog2[6:end-5][5:3:end][1:12])
[1,2,3,4][1:2:end]
# even case≈
# px_bulk = stabvals[25]/stabvals[2]
# pz_bulk = stabvals[2]/sqrt(stabvals[25])
# py_bulk = sqrt(stabvals[14] * stabvals[25])/stabvals[2]

# py2_bound = stabvals[13]/(pz_bulk * py_bulk)
# px2_bound = stabvals[end]/(pz_bulk * py_bulk^2)
# px1_bound = (stabvals[24]/(px_bulk*pz_bulk))

# px_bulk = (1- px_bulk)/2
# py_bulk = (1- py_bulk)/2
# pz_bulk = (1- pz_bulk)/2
# px2_bound = (1- px2_bound)/2
# py2_bound = (1- py2_bound)/2
# px1_bound = (1- px1_bound)/2

# odd case
px_bulk_even = stabvals[2*num_chain+1]/stabvals[2]
px_bulk_odd = stabvals[2*num_chain+2]/stabvals[3]
pz_bulk_odd = stabvals[2]/sqrt(stabvals[2*num_chain+1])
pz_bulk_even = stabvals[3]/sqrt(stabvals[2*num_chain+2])
py_bulk_even = pz_bulk_even/px_bulk_even
py_bulk_odd = (stabvals[num_chain+2] * px_bulk_even)/(pz_bulk_odd * pz_bulk_even^2)

py1_bound = (stabvals[num_chain+1] * px_bulk_even)/(pz_bulk_even * pz_bulk_odd)
px1_bound = stabvals[1]/pz_bulk_even

py1_bound * py_bulk_even * pz_bulk_odd  - stabvals[num_chain+1]
isapprox(py1_bound * px_bulk_even * py_bulk_odd * pz_bulk_even, stabvals[end], rtol = 0.01)
((py1_bound * px_bulk_even * py_bulk_odd * pz_bulk_even)- stabvals[end])/stabvals[end]

unlog[1:3] - [px1_bound, py1_bound, pz_bulk_odd]
unlog[4:5] - [px_bulk_even, py_bulk_even]
(unlog[6:8] ./ [px_bulk_odd, py_bulk_odd, pz_bulk_odd]) .* (unlog[9:11] ./ [px_bulk_even, py_bulk_even, pz_bulk_even])

px_bulk_even = (1- px_bulk_even)/2
py_bulk_even = (1- py_bulk_even)/2
pz_bulk_even = (1- pz_bulk_even)/2
px_bulk_odd = (1- px_bulk_odd)/2
py_bulk_odd = (1- py_bulk_odd)/2
pz_bulk_odd = (1- pz_bulk_odd)/2
py1_bound = (1- py1_bound)/2
px1_bound = (1- px1_bound)/2

stabvals[3]/stabvals[5]

inv(ones(Int, 3, 3) - I(3)) * [px1_bound, py1_bound, pz_bulk_odd]
er[1,:]
inv(ones(Int, 2, 2) - I(2)) * [px_bulk_even, py_bulk_even]
er[2,:]
sum(inv(ones(Int, 3, 3) - I(3)) * [px_bulk_odd, py_bulk_odd, pz_bulk_odd])- sum(er[7,:])
sum(inv(ones(Int, 3, 3) - I(3)) * [px_bulk_even, py_bulk_even, pz_bulk_even])-sum(er[8,:])


plot(stabvals[1:12])
plot(er[1:end,1])
er[:,3]
indices = [1; 3:17-2; 17]
@views vals = er[:,1][indices]
# Compare all elements to the first one within 1% relative tolerance
return all(isapprox.(vals, vals[1], rtol=0.01))
er[1:2,2]
plot(er[2:2:end,2], label="Y")
er[:,3][indices]
er[2:2:end,2]
plot(er[:,3][indices])


# Testing zig zag in 4 levl odd/even

    # n must range from 4 onwards
    err_x_mean = []
    min_l = []
    max_l = []
    delta_odd = []

    η = 0.9
    γ = 0.1
    D = 0.9

    n1 = 5
    n2 = 100
    for n in n1:n2
        println(n)
        errs = return_errors(γ, η, D, 4, 'Z', n)[:,1]
        push!(err_x_mean, sum(errs)/n)
        push!(min_l, minimum(errs))
        push!(max_l, maximum(errs))
        if isodd(n)
            push!(delta_odd, (sum(errs[1:2:end])/(0.5*n + 0.5)) - (sum(errs[2:2:end])/(0.5*(n-1))))
        end
    end

    # err_x_mean
    n_list  = [n1:n2;][1:2:end]
    plot()
    plot((n_list), abs.(err_x_mean[1:2:end] .- err_x_mean[end]), linestyle=:solid, label="mean")
    
    
    plot!(n_list, min_l[1:2:end] .- min_l[end], label="min", linestyle=:dash, marker=(:circle,4))
    plot!(n_list, max_l[1:2:end] .- max_l[end], label="max", linestyle=:dash, xlabel="n", ylabel="Pauli X Error Rate", title="4 Level Atom Zig-Zag Chain (η=$η, γ=$γ, D=$D)", marker=(:diamond,4))
    plot!( yscale=:log10)
    
    
    plot(n1:2:n2, delta_odd)

    rel_diff = 100 .* (max_l - min_l) ./ min_l
    plot(rel_diff[1:2:end])
    plot(rel_diff)
    rel_diff[10:15]
    # may be 3% diff of the max and min after 15
###############################################



γ, η, D, V = (0.1, 0.99, 0.99, 0.99)
γ_2 = γ^2

η_t = (1-η) * γ_2

k = sqrt(4*D*(1-D))
N_2 = 1 + k^2
D_t_2 = 1 - k^2


trace = (η_t + (0.5*η))^2
trace_term2 = η^2 * (1 + (V*( (D_t_2/N_2)^2 - 1 )))/8
trace = trace - trace_term2
	
XX = 4*k*η* ((k*η) + (η_t*(N_2^0.5))) * ((η_t + (0.5*η))^2 + (η^2 * (V-1)/8))/N_2^2
XX += (V* η^4 * D_t_2^2) / (8 * N_2^2)
XX /= trace 
XX += η_t^2


ZZ = (η^4 * D_t_2^2)/(8 * N_2^2 * trace)
YY = ZZ * V

stab_out = [XX, YY, ZZ]

XX_i =  (η_t^2 + η^2) + (4*k*η*η_t/N_2)
ZZ_i = η^2 * D_t_2/N_2
stab_in = [XX_i, ZZ_i, ZZ_i]

px, py, pz, po = stab_to_pauli_pl(stab_in)
sum([px, py, pz, po])
# po = 1 - px - py - pz
p_oz = po + pz
m_oz = po - pz
p_xy = px + py
m_xy = px - py
# a_var = ((po + pz)^2) + ((px+py)^2)
# b_var = 2*(po + pz)*(px + py)
# c_var = ((po - pz)^2) + ((px-py)^2)
# d_var = 2*(po - pz)*(px - py)

Px, Py, Pz, Po = stab_to_pauli_pl(stab_out)
sum([Px, Py, Pz, Po])
P_oz = Po + Pz
M_oz = Po - Pz
P_xy = Px + Py
M_xy = Px - Py

p_fuse_oz = (P_oz - (2*p_oz*p_xy))/(p_oz - p_xy)^2
m_fuse_oz_num = (M_oz*(m_oz+m_xy)^2) - (2*m_oz*m_xy*(M_oz+M_xy))
m_fuse_oz_denom = (m_oz^2 - m_xy^2)^2
m_fuse_oz = m_fuse_oz_num/m_fuse_oz_denom

p_fuse_xy = (P_xy - (2*p_oz*p_xy))/(p_oz - p_xy)^2
m_fuse_xy_num = (M_xy*(m_oz+m_xy)^2) - (2*m_oz*m_xy*(M_oz+M_xy))
m_fuse_xy_denom = (m_oz^2 - m_xy^2)^2
m_fuse_xy = m_fuse_xy_num/m_fuse_xy_denom

po_ = (p_fuse_oz + m_fuse_oz)/2
pz_ = (p_fuse_oz - m_fuse_oz)/2
px_ = (p_fuse_xy + m_fuse_xy)/2
py_ = (p_fuse_xy - m_fuse_xy)/2

plus_oz = p_oz^2 + p_xy^2
minus_oz = m_oz^2 + m_xy^2
plus_xy = 2*p_oz*p_xy
minus_xy = 2*m_xy*m_oz

rho_after_fusion = 0.5 .* [plus_oz minus_oz 0 0;
                          minus_oz plus_oz 0 0;
                          0 0 plus_xy minus_xy;
                          0 0 minus_xy plus_xy]

rho_actual_af_fusion = 0.5 .* [P_oz M_oz 0 0;
                                  M_oz P_oz 0 0;
                                  0 0 P_xy M_xy;
                                  0 0 M_xy P_xy]
sum(diag(rho_after_fusion))
sum(diag(rho_actual_af_fusion))
eigenvals, eigenvecs = eigen(rho_after_fusion)
eigenvals_a, eigenvecs_a = eigen(rho_actual_af_fusion)
trial_o = (plus_oz+minus_oz)/2
trial_z = (plus_oz-minus_oz)/2
trial_x = (plus_xy+minus_xy)/2
trial_y = (plus_xy-minus_xy)/2

[trial_x, trial_y, trial_z, trial_o]

matrix_m = [po pz py px;
            pz po px py;
            py px po pz;
            px py pz po]

out = inv(matrix_m) * [Px Py Pz Po]'


[px_, py_, pz_, po_]


Po
matrix_m * out
[Px Py Pz Po]

# sum([px_, py_, pz_, po_])
# return ([px_, py_, pz_, po_], stab_in, stab_out) 
# end


X = [0 1; 1 0]
kron(X,X)
