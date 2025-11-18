using Plots
using LaTeXStrings
using Printf
# gr()
plotly()
γ = 0.9
ζ = 0.8
η = 0.99


function stab_to_pauli_pl(stab_list)
	temp_probs = (1 - stab_list[1] - stab_list[2] - stab_list[3]) .* [1.0,1.0,1.0]
	temp_probs += 2*stab_list
	push!(temp_probs, (1+sum(stab_list)))
	return temp_probs/4
end

function out_pauli_pl(in_pauli)
	t_8inv = sum(in_pauli.^2) + (2*sum(in_pauli[3:4])*sum(in_pauli[1:2])) + 2*(prod(in_pauli[3:4])+prod(in_pauli[1:2]))
	p1, p2, p3, p4 = in_pauli
	theory_probs = [ (p1+p2)*(p3+p4) + (p1-p2)*(p3-p4) ,
		(p1+p2)*(p3+p4) - (p1-p2)*(p3-p4), 
		2*(p1*p2 + p3*p4) , 
		sum(in_pauli.^2) ]
	return theory_probs * t_8inv
end


function pauli_err_fusion(γ, ζ, η)
    N_1inv = ( η + (2 * (1-η) * γ^2) )^(-2)
    stab_in =  [ (ζ*η^2 + (4* (1-η)^2 *γ^4))*N_1inv ,
                η^2 * (ζ) * N_1inv,
                η^2*N_1inv
    ]

    N_Tinv =  ( (32* (1-η)^2 *γ^4) + (16 * (1-η) * η * γ^2) + (η^2))^(-1)
    stab_out =  [ (ζ^2*η^4*(N_Tinv) + (4* (1-η)^2 *γ^4))*N_1inv ,
                η^4 * (ζ^2) * N_1inv*N_Tinv,
                η^4*N_1inv*N_Tinv
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

fusion_errs99 = []
fusion_errs95 = []
fusion_errs9 = []

γ = 0.2
zeta_list = 0.7:0.01:1.0
for ζ in zeta_list
    push!(fusion_errs99, pauli_err_fusion(γ, ζ, 0.99))
    push!(fusion_errs95, pauli_err_fusion(γ, ζ, 0.95))
    push!(fusion_errs9, pauli_err_fusion(γ, ζ, 0.9))
end

fusion_errs9 = stack(fusion_errs9)'
fusion_errs95 = stack(fusion_errs95)'
fusion_errs99 = stack(fusion_errs99)'

fusion_errs9
plot(zeta_list, fusion_errs9[:,1], label="η=0.9", marker=(:circle, 4), xlabel="ζ", ylims = (0, maximum(fusion_errs9[:,1])+0.01), ylabel="Px", legend=:topleft,  title="Fusion: Pauli X Error Rate vs ζ");
plot!(zeta_list, fusion_errs95[:,1], label="η=0.95", marker=(:circle, 4));
plot!(zeta_list, fusion_errs99[:,1], label="η=0.99", marker=(:circle, 4));

yticks = collect(minimum(fusion_errs9[:,1]):0.000005:maximum(fusion_errs9[:,1]))
yticklabels = [ @sprintf("%.4E",x) for x in yticks ]

plot!(zeta_list, fusion_errs9[:,1]; marker=(:circle, 1), frame=:box,
    inset=bbox(0.6,0.15,0.35,0.35),
    subplot=2, label=nothing,
    yticks=(yticks,yticklabels))



plot(zeta_list, fusion_errs9[:,2], label="η=0.9", marker=(:circle, 4), xlabel="ζ", ylims = (0, maximum(fusion_errs9[:,1])+0.01), ylabel="Py", legend=:topleft,  title="Fusion: Pauli Y Error Rate vs ζ");
plot!(zeta_list, fusion_errs95[:,2], label="η=0.95", marker=(:circle, 4));
plot!(zeta_list, fusion_errs99[:,2], label="η=0.99", marker=(:circle, 4));

yticks = collect(minimum(fusion_errs9[:,2]):0.000005:maximum(fusion_errs9[:,2]))
yticklabels = [ @sprintf("%.4E",x) for x in yticks ]

plot!(zeta_list, fusion_errs9[:,2]; marker=(:circle, 1), frame=:box,
    inset=bbox(0.6,0.15,0.35,0.35),
    subplot=2, label=nothing,
    yticks=(yticks,yticklabels))
    

plot(zeta_list, fusion_errs9[:,3], label="η=0.9", marker=(:circle, 3), xlabel="ζ", ylims = (0, maximum(fusion_errs9[:,3])+0.01), ylabel="Pz", legend=:topright,  title="Fusion: Pauli Z Error Rate vs ζ, γ=$γ");
plot!(zeta_list, fusion_errs95[:,3], label="η=0.95", marker=(:circle, 3));
plot!(zeta_list, fusion_errs99[:,3], label="η=0.99", marker=(:circle, 3))

# yticks = collect(minimum(fusion_errs9[:,3]):0.000005:maximum(fusion_errs9[:,3]))
# yticklabels = [ @sprintf("%.4E",x) for x in yticks ]

plot!(zeta_list, fusion_errs9[:,3]; marker=(:circle, 1), frame=:box,
    inset=bbox(0.6,0.15,0.35,0.35),
    subplot=2, label=nothing,
    yticks=(yticks,yticklabels))




################################
ζ = 0.9
γ = 0.2

fusion_errs_eta = []
eta_list = 0.4:0.01:1.0
for η in eta_list
    push!(fusion_errs_eta, pauli_err_fusion(γ, ζ, η))
end
fusion_errs_eta = stack(fusion_errs_eta)'

plot(eta_list, fusion_errs_eta[:,1], label="Px", marker=(:circle, 3), xlabel="η", ylabel = "Probability", title="Fusion: Pauli Error Rates vs η, ζ=$ζ, γ=$γ");
plot!(eta_list, fusion_errs_eta[:,2], label="Py", marker=(:square, 3));
plot!(eta_list, fusion_errs_eta[:,3], label="Pz", marker=(:star, 3))