
mapvalue(f, x...) = map(f, x...)
mapvalue(f, x::Dict, ys...) = Dict(k => f(v, (get(y, k, nothing) for y in ys)...) for (k,v) in x)

mapkey(f, x::NamedTuple{Ks}) where Ks = NamedTuple{Ks}(map(f, Ks))
mapkey(f, x::Dict) = Dict(k => f(k) for k in keys(x))
mapkey(f, x::Tuple) = ntuple(i -> f(i), length(x))
mapkey(f, x::AbstractArray) = [f(i) for i=1:length(x)]

foreachvalue(f, x...) = foreach(f, x...)

foreachvalue(f, x::Dict, ys...) = foreach(pairs(x)) do (k, v)
  f(v, (get(y, k, nothing) for y in ys)...)
end

ofeltype(x, y) = convert(float(eltype(x)), y)

_eps(T::Type{<:AbstractFloat}, e) = T(e)
# catch complex and integers
_eps(T::Type{<:Number}, e) = _eps(real(float(T)), e) 
# avoid small e being rounded to zero
_eps(T::Type{Float16}, e) = e == 0 ? T(0) : max(T(1e-7), T(e))
