using Integrals, QuantumOptics, LinearAlgebra, Interpolations, DifferentialEquations

gamma = 1 
deph = 0.01
t_fin = 10
dt = 0.01
no_cavs = 10


basis = SpinBasis(1//2)
a = sigmam(basis)
at = sigmap(basis)
H =  identityoperator(basis)
J = [sqrt(gamma)*a, sqrt(deph)*sigmaz(basis)]

t_list = [0:dt:t_fin;]
t_size = length(t_list)
# corel_m = Array{ComplexF64}(undef, t_size, t_size)
# corel_m *= 0
corel_m = fill(0.0, (t_size,t_size))

ρ₀ = spinup(basis) ⊗ dagger(spinup(basis))

tout, ρt_master = timeevolution.master(t_list, ρ₀, H, J; dt=dt)
for i=1:t_size-1
    rhot = ρt_master[i]
    corel_m[i,i:end] = timecorrelations.correlation(t_list[1:(t_size-i+1)], rhot, H, J, at, a)
end
corel_m[t_size,t_size] = 0.0
corel_m = -diagm((diag(corel_m))) + corel_m + conj(corel_m)'
corel_m = real(corel_m)
# println(any(isnan.(corel_m)))
eigenvals, eigens = eigen(corel_m; sortby=-)

v_0 = [cubic_spline_interpolation(0:dt:t_fin, eigens[:,i]; extrapolation_bc=0) for i in 1:no_cavs]
gv = Dict()
alpha = Dict()
gv_f = [;]

function a_(a, p, t)
    i, j = deepcopy(p)
    da = -(v_0[i](t)*gv_f[j](t))-(a * 0.5 * gv_f[j](t)^2)
    for k=1:j-1
        da -= gv_f[j](t) * gv_f[k](t) * alpha[[i,k]](t)
    end
    return da
end

function vᵢⁱ⁻¹(i,t)
    v = v_0[i](t)
    for k=1:i-1
        v += gv_f[k](t) * alpha[[i,k]](t)
    end
    return v
end

# function cum_int(vᵢⁱ⁻¹, i, t)

#     sol.u == 0 ? (return 10^-8) : (return sqrt(sol.u))
#     return sol.u
# end


plot(v_0[1].(0:dt:t_fin))
f(u, p) =  u + 2

i = 1
j = 1
for i=1:Int(no_cavs)
    println(i)
    for j=1:Int(i-1)
        # a0 = 0
        # tspan = (0, t_fin)
        prob = ODEProblem(a_, 0, (0,t_fin), [i,j])
        sol = solve(prob, saveat=dt)
        alpha[[i,j]] = cubic_spline_interpolation(0:dt:t_fin, sol.u; extrapolation_bc=Line())
    end
    cum_int_list = [0.0,]
    for t=dt:dt:t_fin
        # f(u, p) =  u + 2
        f(u, p) = conj(vᵢⁱ⁻¹(p, u)) * vᵢⁱ⁻¹(p, u)
        
        domain = (t-dt, t) # (lb, ub)
        # println(typeof(domain))
        prob = IntegralProblem(f, domain, i)
        # prob = IntegralProblem((u,p) -> u+2, domain)
        sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        push!(cum_int_list, cum_int_list[end]+sol.u)
    end
    cum_int_list = sqrt.(cum_int_list)
    cum_int_list[1] = cum_int_list[2]
    # println(vᵢⁱ⁻¹(1,0))
    cum_int_extrap = cubic_spline_interpolation(0:dt:t_fin, cum_int_list;extrapolation_bc=Line())
    gᵥ(t) = - conj(vᵢⁱ⁻¹(i,t))/cum_int_extrap(t)
    push!(gv_f, gᵥ)


end
# usual
# return gv_f
# only for test
return gv_f, eigenvals

# status
