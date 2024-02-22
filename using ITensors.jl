using ITensors
ITensors.enable_debug_checks()

N = 3

#see if i can append to this list to make dif dims ie atom and cavities
s = siteinds( "Electron" , 6)

function ITensors.op(::OpName"P1", ::SiteType"Electron", d::Int=3)
    o = zeros(d, d)
    o[1, 1] = 1
    return o
  end

  ampo = OpSum()
for n=1:N
  ampo += "ground",n
end

System_0 = MPO(sites, "zero")

@show op("P1", s, 1)