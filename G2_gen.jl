using Plots
using QuantumOptics
using LinearAlgebra
using Interpolations
using DifferentialEquations



no_cavs = 4
gamma = 1
deph = 0.01
dt = 0.01
t_fin = 10


basis = SpinBasis(1//2)
a = sigmam(basis)
at = sigmap(basis)
H =  identityoperator(basis)
J = [sqrt(gamma)*a, sqrt(deph)*sigmaz(basis)]

t_list = [0:dt:t_fin;]
t_size = length(t_list)
corel_m = Array{ComplexF64}(undef, t_size, t_size)
corel_m *= 0

ρ₀ = spinup(basis) ⊗ dagger(spinup(basis))

tout, ρt_master = timeevolution.master(t_list, ρ₀, H, J; dt=dt)
for i=1:t_size-1
    rhot = ρt_master[i]
    corel_m[i,i:end] = timecorrelations.correlation(t_list[1:(t_size-i+1)], rhot, H, J, at, a)
end
corel_m[t_size,t_size] = 0.0
corel_m = -diagm((diag(corel_m))) + corel_m + conj(corel_m)'
corel_m = real(corel_m)
eigenvals, eigens = eigen(corel_m; sortby=-)

v_0 = [cubic_spline_interpolation(0:dt:t_fin, eigens[:,i]; extrapolation_bc=0) for i in 1:no_cavs]

gv = Dict()
alpha = Dict()
gv_f = [;]
for i=1:Int(no_cavs)
    for j=1:Int(i-1)
        a0 = 0
        function a_(a, p, t)
            da = -(v_0[i](t)*gv_f[j](t))-(a * 0.5 * gv_f[j](t)^2)
            for k=1:j-1
                da -= gv_f[j](t) * gv_f[k](t) * alpha[[i,k]](t)
            end
            return da
        end
        tspan = (0.0, t_fin)
        prob = ODEProblem(a_, a0, tspan)
        sol = solve(prob)
        alpha[[i,j]] = interpolate(sol.t,sol.u,FritschCarlsonMonotonicInterpolation())
    end
    v_i = deepcopy(v_0[i])
    for k=1:Int(i-1)
        v_i += (gv_f[k].(0:dt:t_fin) .* alpha[[i,k]].(0:dt:t_fin))
    end
    norm = dt * cumsum(v_i.^2)
    
    gv[i] = -v_i./sqrt.(norm)
    #i > 1 ? gv[i][1] = 0 : nothing
    extrap = cubic_spline_interpolation(0:dt:t_fin, gv[i]; extrapolation_bc=0)
    push!(gv_f, extrap)
end
