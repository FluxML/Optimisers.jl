
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, unthunk
const NoT = NoTangent()

"""
    destructure(model) -> vector, reconstructor

Copies all [`trainable`](@ref Optimisers.trainable), [`isnumeric`](@ref Optimisers.isnumeric) parameters in the model
to a vector, and returns also a function which reverses this transformation.
Differentiable.

# Example
```jldoctest
julia> v, re = destructure((x=[1.0, 2.0], y=(sin, [3.0 + 4.0im])))
(ComplexF64[1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 4.0im], Restructure(NamedTuple, ..., 3))

julia> re([3, 5, 7+11im])
(x = [3.0, 5.0], y = (sin, ComplexF64[7.0 + 11.0im]))
```

If `model` contains various number types, they are promoted to make `vector`,
and are usually restored by `Restructure`. Such restoration follows the rules 
of `ChainRulesCore.ProjectTo`, and thus will restore floating point precision,
but will permit more exotic numbers like `ForwardDiff.Dual`.

If `model` contains only GPU arrays, then `vector` will also live on the GPU.
At present, a mixture of GPU and ordinary CPU arrays is undefined behaviour.
"""
function destructure(x)
  flat, off, len = _flatten(x)
  flat, Restructure(x, off, len)
end

"""
    Restructure(Model, ..., length)

This is what [`destructure`](@ref Optimisers.destructure) returns, and `re(p)` will re-build the model with
new parameters from vector `p`. If the model is callable, then `re(x, p) == re(p)(x)`.

# Example
```julia-repl
julia> using Flux, Optimisers

julia> _, re = destructure(Dense([1 2; 3 4], [0, 0], sigmoid))
([1, 3, 2, 4, 0, 0], Restructure(Dense, ..., 6))

julia> m = re(-4:1)
Dense(2, 2, σ)      # 6 parameters

julia> m([0.2, 0.3]) ≈ re([0.2, 0.3], -4:1)
true
```
"""
struct Restructure{T,S}
  model::T
  offsets::S
  length::Int
end
(re::Restructure)(flat::AbstractVector) = _rebuild(re.model, re.offsets, flat, re.length)
(re::Restructure)(x, flat::AbstractVector) = re(flat)(x)
Base.show(io::IO, re::Restructure{T}) where T = print(io, "Restructure(", T.name.name, ", ..., ", re.length, ")")
Base.length(re::Restructure) = re.length

# This flattens a model, and returns a web of offsets for later use:
function _flatten(x)
  isnumeric(x) && return vcat(_vec(x)), 0, length(x)  # trivial case
  arrays = AbstractVector[]
  len = Ref(0)
  off = fmap(x; exclude = isnumeric, walk = TrainableStructWalk()) do y
    push!(arrays, _vec(y))
    o = len[]
    len[] = o + length(y)
    o
  end
  isempty(arrays) && return Bool[], off, 0
  return reduce(vcat, arrays), off, len[]
end

struct TrainableStructWalk <: AbstractWalk end

(::TrainableStructWalk)(recurse, x) = mapvalue(recurse, _trainable(x))

_vec(x::Number) = LinRange(x,x,1)
_vec(x::AbstractArray) = vec(x)

function ChainRulesCore.rrule(::typeof(_flatten), x)
  flat, off, len = _flatten(x)
  _maybewarn()
  _flatten_back((dflat, _, _)) = (NoT, _rebuild(x, off, unthunk(dflat), len; walk = _Tangent_biwalk(), prune = NoT))
  (flat, off, len), _flatten_back
end

# This reconstructs either a model like x, or a gradient for it:
function _rebuild(x, off, flat::AbstractVector, len = length(flat); walk = _Trainable_biwalk(), kw...)
  len == length(flat) || throw(DimensionMismatch("Rebuild expected a vector of length $len, got $(length(flat))"))
  fmap(x, off; exclude = isnumeric, walk, kw...) do y, o
    _getat(y, o, flat)
  end
end

_getat(y::Number, o::Int, flat::AbstractVector) = ProjectTo(y)(flat[o + 1])
_getat(y::AbstractArray, o::Int, flat::AbstractVector) =
  ProjectTo(y)(reshape(flat[o .+ (1:length(y))], axes(y)))  # ProjectTo is just correcting eltypes

struct _Trainable_biwalk <: AbstractWalk end

function (::_Trainable_biwalk)(f, x, aux)
  ch, re = functor(typeof(x), x)
  au, _ = functor(typeof(x), aux)
  _trainmap(f, ch, _trainable(x), au) |> re
end

function _trainmap(f, ch, tr, aux)
  map(ch, tr, aux) do c, t, a  # isnothing(t) indicates non-trainable field, safe given isnumeric(c)
    isnothing(t) ? c : f(t, a)
  end
end

struct _Tangent_biwalk <: AbstractWalk end

function (::_Tangent_biwalk)(f, x, aux)  # use with prune = NoT
  ch, re = functor(typeof(x), x)
  au, _ = functor(typeof(x), aux)
  y = _trainmap(f, ch, _trainable(x), au)
  y isa Tuple{} && return NoT
  p = ProjectTo(x)
  if p isa ProjectTo  # e.g. Array, NamedTuple
    p(y)
  else  # p === identity for unknown structs
    Tangent{typeof(x), typeof(y)}(y)
  end
end

function ChainRulesCore.rrule(::typeof(_rebuild), x, off, flat, len; kw...)
  _rebuild_back(dx) = (NoT, NoT, NoT, _grad!(x, unthunk(dx), off, _zero(flat)), NoT)
  _rebuild(x, off, flat, len; kw...), _rebuild_back
end

_zero(x) = map!(zero, similar(x, float(eltype(x))), x)  # mutable zero array for _grad!
ChainRulesCore.@non_differentiable _zero(x)

# This is the gradient of model reconstruction, accumulating duplicates:
function _grad!(x, dx, off, flat::AbstractVector)
  x′, _ = functor(typeof(x), x)
  dx′, _ = functor(typeof(x), base(dx))
  off′, _ = functor(typeof(x), off)
  for (xᵢ, dxᵢ, oᵢ) in zip(x′, dx′, off′)
    flat = _grad!(xᵢ, dxᵢ, oᵢ, flat)
  end
  flat
end
function _grad!(x, dx, off::Integer, flat::AbstractVector{T}) where T
  dx_un = unthunk(dx)
  T2 = promote_type(T, eltype(dx_un))
  if T != T2  # then we must widen the type
    flat = copyto!(similar(flat, T2), flat)
  end
  @views flat[off .+ (1:length(x))] .+= vec(dx_un)  # must visit all tied nodes
  flat
end
_grad!(x, dx::Zero, off, flat::AbstractVector) = flat
_grad!(x, dx::Zero, off::Integer, flat::AbstractVector) = flat  # ambiguity

# These are only needed for 2nd derivatives:
function ChainRulesCore.rrule(::typeof(_grad!), x, dx, off, flat)
  @warn "second derivatives of Restructure may not work yet, sorry!" maxlog=3
  _grad_back(dflat) = (NoT, NoT, _rebuild(x, off, unthunk(dflat); walk = _Tangent_biwalk(), prune = NoT), NoT, NoT)
  _grad!(x, dx, off, flat), _grad_back
end
base(dx::Tangent{<:Tangent}) = backing(dx).backing  # might be needed for gradient(gradient(destructure))
base(dx::Tangent{Any, <:NamedTuple{(:backing,)}}) = base(backing(dx).backing)  # Zygote version
_maybewarn() = nothing
function ChainRulesCore.rrule(::typeof(_maybewarn))
  @warn "second derivatives of destructure may not work yet, sorry!" maxlog=3
  nothing, _ -> (NoT,)
end

