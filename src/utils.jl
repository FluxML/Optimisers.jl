
mapvalue(f, x...) = map(f, x...)
mapvalue(f, x::Dict, ys...) = Dict(k => f(v, (get(y, k, nothing) for y in ys)...) for (k,v) in x)

# without theses, tuples are returned instead of NamedTuples
mapvalue(f, x::NamedTuple{Ks}, y::Tangent{<:Any,<:NamedTuple}) where {Ks} = 
  NamedTuple{Ks}((f(v, y[k]) for (k,v) in pairs(x)))

mapkey(f, x::NamedTuple{Ks}) where Ks = NamedTuple{Ks}(map(f, Ks))
mapkey(f, x::Dict) = Dict(k => f(k) for k in keys(x))
mapkey(f, x::Tuple) = ntuple(i -> f(i), length(x))
mapkey(f, x::AbstractArray) = [f(i) for i=1:length(x)]

foreachvalue(f, x...) = foreach(f, x...)

foreachvalue(f, x::Dict, ys...) = foreach(pairs(x)) do (k, v)
  f(v, (get(y, k, nothing) for y in ys)...)
end

