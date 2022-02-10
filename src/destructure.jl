
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, unthunk
const NoT = NoTangent()

"""
    destructure(model) -> vector, reconstructor

Copies all [`trainable`](@ref), [`isnumeric`](@ref) parameters in the model
to a vector, and returns also a function which reverses this transformation.
Differentiable.

# Example
```jldoctest
julia> v, re = destructure((x=[1.0, 2.0], y=(sin, [3 + 4im])))
(ComplexF64[1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 4.0im], Restructure(NamedTuple, ..., 3))

julia> re([3, 5-im, 7+11im])
(x = [3.0, 5.0], y = (sin, ComplexF64[7.0 + 11.0im]))
```
"""
function destructure(x)
  flat, off, len = _flatten(x)
  flat, Restructure(x, off, len)
end

"""
    Restructure(Model, ..., length)

This is what [`destructure`](@ref) returns, and `re(p)` will re-build the model with
new parameters from vector `p`. If the model is callable, then `re(x, p) == re(p)(x)`.

# Example
```julia
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
  isnumeric(x) && return vcat(vec(x)), 0, length(x)  # trivial case
  arrays = AbstractVector[]
  len = Ref(0)
  off = fmap(x; exclude = isnumeric, walk = (f, z) -> map(f, _trainable(z))) do y
    push!(arrays, vec(y))
    o = len[]
    len[] = o + length(y)
    o
  end
  reduce(vcat, arrays), off, len[]
end

function ChainRulesCore.rrule(::typeof(_flatten), x)
  flat, off, len = _flatten(x)
  _flatten_back((dflat, _, _)) = (NoT, _rebuild(x, off, dflat, len; walk = _Tangent_biwalk, prune = NoT))
  (flat, off, len), _flatten_back
end

# This reconstructs either a model like x, or a gradient for it:
function _rebuild(x, off, flat::AbstractVector, len = length(flat); walk = _trainable_biwalk, kw...)
  len == length(flat) || throw(DimensionMismatch("Rebuild expected a vector of length $len, got $(length(flat))"))
  fmap(x, off; exclude = isnumeric, walk, kw...) do y, o
    _getat(y, o, flat)
  end
end

_getat(y::Number, o::Int, flat::AbstractVector) = ProjectTo(y)(flat[o + 1])
_getat(y::AbstractArray, o::Int, flat::AbstractVector) =
  ProjectTo(y)(reshape(flat[o .+ (1:length(y))], axes(y)))  # ProjectTo is just correcting eltypes

function _trainable_biwalk(f, x, aux)
  ch, re = functor(typeof(x), x)
  au, _ = functor(typeof(x), aux)
  _trainmap(f, ch, _trainable(x), au) |> re
end

function _trainmap(f, ch, tr, aux)
  map(ch, tr, aux) do c, t, a  # isnothing(t) indicates non-trainable field, safe given isnumeric(c)??
    isnothing(t) ? c : f(t, a)
  end
end

function _Tangent_biwalk(f, x, aux)  # use with prune = NoT
  ch, re = functor(typeof(x), x)
  au, _ = functor(typeof(x), aux)
  y = _trainmap(f, ch, _trainable(x), au)
  y isa Tuple{} && return NoT
  Tangent{typeof(x), typeof(y)}(y)
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
  dx′, _ = functor(typeof(x), dx)
  off′, _ = functor(typeof(x), off)
  foreach((xᵢ, dxᵢ, oᵢ) -> _grad!(xᵢ, dxᵢ, oᵢ, flat), x′, dx′, off′)
  flat
end
function _grad!(x, dx, off::Integer, flat::AbstractVector)
  @views flat[off .+ (1:length(x))] .+= dx  # must visit all tied nodes
  flat
end
_grad!(x, dx::Zero, off, flat::AbstractVector) = nothing
_grad!(x, dx::Zero, off::Integer, flat::AbstractVector) = nothing  # ambiguity

