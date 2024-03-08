using ITensors
function ITensors.op(a::OpName"P1", b::SiteType"Boson", d::Int)
  o = zeros(d, d)
  o[1, 1] = 1
  return o
end

d=5
s = siteind("Boson",dim=d)
mpo= op("P1",s)
println(mpo)
