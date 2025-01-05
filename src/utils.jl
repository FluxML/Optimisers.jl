
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

"""
    _eps(Type{T}, val)

Mostly this produces `real(T)(val)`, so that `_eps(Float32, 1e-8) === 1f-8` will
convert the Float64 parameter epsilon to work nicely with Float32 parameter arrays.

But for Float16, it imposes a minimum of `Float16(1e-7)`, unless `val==0`.
This is basically a hack to increase the default epsilon, to help many optimisers avoid NaN.
"""
_eps(T::Type{<:Number}, e) = real(float(T))(e) 
_eps(T::Type{Float16}, e) = e == 0 ? T(0) : max(T(1e-7), T(e))
