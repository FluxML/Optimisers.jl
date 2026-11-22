module OptimisersForwardDiffExt

using Optimisers: Optimisers, _eltype_for
using ForwardDiff: Dual

# Dual numbers take the precision of the model's array, also when nested:
Optimisers._retype(::Type{Dual{T, V, N}}, ::Type{Y}) where {T, V, N, Y <: Real} = Dual{T, _eltype_for(Y, V), N}

Optimisers._precision(::Type{Dual{T, V, N}}) where {T, V, N} = Optimisers._precision(V)

end
