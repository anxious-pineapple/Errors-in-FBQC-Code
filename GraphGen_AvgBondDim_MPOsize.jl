using ITensors, Plots
import MPOFuncs
using CurveFit

gamma = 1
dt = 0.01
t_final = 10
dep = 0.02

##  Checking how bond dim changes (is it a sublinear increase) at a fixed Dephasing

avg_bond_dim = [;]
mpo_size = [;]
avg_bond_dim_doubl = [;]
mpo_size_doubl = [;]

for no_cavs in 2:15

    println("No cavs ",  no_cavs)
    mpo, sites , eigenvals = MPOFuncs.cash_karppe_evolve_test(no_cavs, dep, gamma, dt, t_final)
    truncate!(mpo; cutoff=1e-10)
    link_dim_vec = ITensors.linkdims(mpo)
    push!(avg_bond_dim, sum(link_dim_vec)/no_cavs)
    push!(mpo_size, sum( ( push!(link_dim_vec, 1) .* append!([1,],link_dim_vec[1:end-1])) ) )

    double, double_sites = MPOFuncs.n_copy_mpo(mpo, sites, 2)
    for i=no_cavs:-1:1
        print(i, " , ")
        double = MPOFuncs.swap_ij!(double, double_sites, i, 2*i-1)
        MPOFuncs.beamsplitter_nextsite!(double, double_sites, 2*i-1)
    end
    truncate!(double; cutoff=1e-10)
    link_dim_vec2 = ITensors.linkdims(double)
    push!(avg_bond_dim_doubl, sum(link_dim_vec2)/no_cavs)
    push!(mpo_size_doubl, sum( ( push!(link_dim_vec2, 1) .* append!([1,],link_dim_vec2[1:end-1])) ) )
end



p=plot(avg_bond_dim, label="before bmsplt")
plot!(avg_bond_dim_doubl./2, label="after")

a,b = linear_fit([2:15;], avg_bond_dim)
plot!(a .+ b*[2:15;] ;label="", line=:dash)
a2 ,b2 = linear_fit([2:15;], avg_bond_dim_doubl./2)
plot!(a2 .+ b2*[2:15;] ; label="", line=:dash)
plot!(xlabel="Number of cavities", ylabel="Average bond dimension")
Plots.pdf(p, "bond_dim_linearPlot.pdf")

p2= plot(mpo_size./[2:15;])
plot!(mpo_size_doubl./(2*[2:15;]) )

a,b = linear_fit([2:15;], mpo_size./[2:15;])
plot!(a .+ b*[2:15;] ;label="", line=:dash)
a2 ,b2 = linear_fit([2:15;], mpo_size_doubl./(2*[2:15;]))
plot!(a2 .+ b2*[2:15;] ; label="", line=:dash)
plot!(xlabel="Number of cavities", ylabel="MPO size ")
Plots.pdf(p2, "mposize_linearPlot.pdf")