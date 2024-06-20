using ITensors,  Plots,  DifferentialEquations

function ITensors.op(::OpName"Ground" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[1,1] = 1
    return mat
end

function ITensors.op(::OpName"Excite1" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[2,2] = 1
    return mat
end

function ITensors.op(::OpName"Sz" , ::SiteType"Qudit" , d::Int)
    mat = zeros(d, d)
    mat[1,1] = 1
    mat[2,2] = -1
    return mat
end

no_cavs = 1
sites = siteinds("Qudit", no_cavs+1, dim=3)
input_list = repeat(["Ground",],no_cavs+1)
input_list[1] = "Excite1"
a = MPO(sites, input_list)
T = randn(3,3)
a = ITensor(T,sites)
b = MPO(sites, input_list)

# a/b

function ITensors.:*(a::MPO, b::MPO)
    if length(a) == length(b)
        return apply(a, b)
    else
        throw("Diff size")
    end
end

function ITensors.:/(a::MPO, b::MPO)
    if length(a) == length(b)
        return a/tr(b)
    else
        throw("Diff size")
    end
end

# place_holder_func(a) = Base.oneunit(a)

#similar and one unit
function Base.oneunit(a::Any)
    # println(a)
    if typeof(a) == MPO
        mpo_sites = siteinds(a)
        site_list = Index{Int64}[;]
        for i in mpo_sites
            plev(i[1]) == 0 ? push!(site_list, i[1]) : push!(site_list, i[2])
        end
        iden_mpo = MPO(site_list, "I")
        return iden_mpo
    else
        nothing
        # return place_holder_func(a)
    end
end

function dmpo_test(rho, p, t)
    # mpo_sites = siteinds(rho)
    # z = MPO(mpo_sites, "Sz")
    return 10^(-4) .* deepcopy(rho)
end

function Base.length(a::ITensor)
    return size(a)[1]
end

function NDTensors.EmptyNumber(::Int64)
    return 0
end

function Base.similar(::Nothing, a::Type{Float64})
    return nothing
end

@show (size(a)[1])
prob2 = ODEProblem(dmpo_test, T, (0.0,5))
sol = solve(prob2)