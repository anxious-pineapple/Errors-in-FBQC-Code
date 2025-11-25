
function fusion_err(γ, η, D)
    trace_2 = η^2 * ( 0.25 + (D*(1-D)))
    trace_2 += 2 * (1-η)^2 * γ^4
    trace_2 += 4 * η * (1-η) * γ^2 * sqrt(D*(1-D)) 
    B = 4 * η * (1-η) * γ^2 * sqrt(D*(1-D)) 
    B += 2 * (1-η)^2 * γ^4
    B += η^2 * sqrt(D*(1-D))
    B *= 2
    c = η + ((1-η) * γ^2)
    d = (2*η*sqrt(D*(1-D))) + ((1-η) * γ^2)
    e = η*((2*D)-1)
    A_C = η^2 * ( 0.25 -  (D*(1-D)))

    II_in = c^2 + d^2
    # stab_in =  [ c^2 + d^2 ,
    #             e^2,
    #             e^2
    # ]
    # stab_in /= II_in

    # X, Y, Z
    II = 1 + (d/c)^2 + (2*d*B/(trace_2*c))
    # stab_out =  [ (1 + (d/c)^2 + (2*d*B/(trace_2*c))) ,
    #             (e^2 * A_C/ trace_2),
    #             (e^2 * A_C/ trace_2)
    # ]
    # stab_out /= II


    fusion_errx = 0.5 - ((0.5*II_in * A_C)/(trace_2 * II))
    return fusion_errx
end

γ = 0.0;
η_list = [0:0.01:0.3;];
D_list = [0.00001:0.01:0.3;];

f(η,D) = fusion_err(γ, 1-η, 1-D)

using PlotlyJS
# gr()
γ = 0.0;
gam0 = surface(x=η_list, y=D_list, z=[fusion_err(γ, 1-η, 1-D) for η in η_list, D in D_list], opacity=0.5, 
name="γ=$γ")
γ = 0.5;
gam1 = surface(x=η_list, y=D_list, z=[fusion_err(γ, 1-η, 1-D) for η in η_list, D in D_list], opacity=0.5, name="γ=$γ")
plot([gam0, gam1], Layout(
        scene=attr(
            xaxis=attr(title="1-η"),
            yaxis=attr(title="1-D"),
            zaxis=attr(title="Prob")
        ),
        title="3D Surface Plot"
    ))



using Plots
γ = 0.1;
η = 1;
Plots.plot(D_list, [fusion_err(γ, η, 1-D) for D in D_list], label="γ=$γ, η=$η", xlabel="1-D", ylabel="Prob", legend=:topleft, marker=:circle)
η = 0.9;
Plots.plot!(D_list, [fusion_err(γ, η, 1-D) for D in D_list], label="γ=$γ, η=$η", marker=:star)
η = 0.8;
Plots.plot!(D_list, [fusion_err(γ, η, 1-D) for D in D_list], label="γ=$γ, η=$η", marker=:diamond)
η = 0.7;
Plots.plot!(D_list, [fusion_err(γ, η, 1-D) for D in D_list], label="γ=$γ, η=$η", marker=:utriangle)
# Plots.plot!(yscale=:log10)
# Plots.plot!(xscale=:log10)
Plots.plot!(xscale=:linear)
Plots.plot!(title="Fusion X error vs D : 4 level sys")


γ = 0;
D = 1;
Plots.plot(η_list, [fusion_err(γ, 1-η, D) for η in η_list], label="γ=$γ, D=$D", xlabel="1-η", ylabel="Prob", legend=:topleft, marker=:circle)
D = 0.99;
Plots.plot!(η_list, [fusion_err(γ, 1-η, D) for η in η_list], label="γ=$γ, D=$D", marker=:star)
D = 0.95;
Plots.plot!(η_list, [fusion_err(γ, 1-η, D) for η in η_list], label="γ=$γ, D=$D", marker=:diamond)
D = 0.9;
Plots.plot!(η_list, [fusion_err(γ, 1-η, D) for η in η_list], label="γ=$γ, D=$D", marker=:utriangle)
# Plots.plot!(yscale=:log10)
Plots.plot!(title="Fusion X error vs η : 4 level sys")