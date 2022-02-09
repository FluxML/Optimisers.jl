
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo
const NoT = NoTangent()

"""
    destructure(model) -> vector, reconstructor

Copies all [`trainable`](@ref), [`isnumeric`](@ref) parameters in the model
to a vector, and returns also a function which reverses this transformation.
Differentiable.

# Example
```jldoctest
julia> v, re = destructure((x=[1.0, 2.0], y=(sin, [3.0])))
([1.0, 2.0, 3.0], Restructure(NamedTuple, ..., 3))

julia> re([10,20,30])
(x = [10.0, 20.0], y = (sin, [30.0]))
```
"""
function destructure(x)
  flat, off, len = _flatten(x)
  flat, Restructure(x, off, len)
end

"""
    Restructure(Model, ..., length)

This is what [`destructure`](@ref) returns, and `re(p)` will re-build the model with
new parameters from vector `p`. If the model is callable, then `re(x, p)` .

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
(re::Restructure)(flat::AbstractVector) = _rebuild(re.model, re.offsets, flat; len = re.length)
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
  _flatten_back((dflat, _)) = (NoT, _rebuild(x, off, dflat; walk = _Tangent_biwalk, prune = NoT, len))
  (flat, off, len), _flatten_back
end

# This reconstructs either a model like x, or a gradient for it:
function _rebuild(x, off, flat::AbstractVector; len, walk = _trainable_biwalk, kw...)
  len == length(flat) || error("wrong length")
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

function ChainRulesCore.rrule(::typeof(_rebuild), x, off, flat; len)
  dflat = map!(zero, similar(flat, float(eltype(flat))), flat)
  _rebuild_back(dx) = (NoT, NoT, NoT, _accumulate!(x, dx, off, dflat))
  _rebuild(x, off, flat; len), _rebuild_back
end

# This is the gradient of model reconstruction, accumulating duplicates:
function _accumulate!(x, dx, off, flat::AbstractVector)
  x′, _ = functor(typeof(x), x)
  dx′, _ = functor(typeof(x), dx)
  off′, _ = functor(typeof(x), off)
  foreach((xᵢ, dxᵢ, oᵢ) -> _accumulate!(xᵢ, dxᵢ, oᵢ, flat), x′, dx′, off′)
  flat
end
function _accumulate!(x, dx, off::Integer, flat::AbstractVector)
  @views flat[off .+ (1:length(x))] .+= dx  # must visit all tied nodes
  flat
end
_accumulate!(x, dx::Zero, off, flat::AbstractVector) = nothing
_accumulate!(x, dx::Zero, off::Integer, flat::AbstractVector) = nothing  # ambiguity

