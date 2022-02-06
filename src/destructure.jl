
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo
const NoT = NoTangent()

"""
    destructure([T], model) -> vector, reconstructor

Copies all [`trainable`](@ref), [`isnumeric`](@ref) parameters in the model
to a `Vector{T}`, and returns also a function which reverses this transformation.
Differentiable.
"""
function destructure(::Type{T}, x) where T
  flat, off = alpha!(x, T[])
  len = length(flat)
  # flat, newflat -> beta(x, off, newflat; len)
  flat, Restucture(x, off, len)
end

struct Restucture{T,S}
  model::T
  offsets::S
  length::Int
end
(re::Restucture)(flat) = beta(re.model, re.offsets, flat; len = re.length)
Base.show(io::IO, re::Restucture{T}) where T = print(io, "Restructure(", T.name.name, ", ..., ", re.length, ")")

# This flattens a model, and returns a web of offsets for later use:
function alpha!(x, flat::AbstractVector)
  isempty(flat) || error("this won't work")
  isnumeric(x) && return append!(flat, x), 0  # trivial case
  off = fmap(x; exclude = isnumeric, walk = (f, z) -> map(f, _trainable(z))) do y
    append!(flat, y)
    length(flat) - length(y)
  end
  flat, off
end

function ChainRulesCore.rrule(::typeof(alpha!), x, flat)
  flat′, off = alpha!(x, flat)
  len = length(flat′)
  alpha_back((dflat, _)) = (NoT, beta(x, off, dflat; walk = _Tangent_biwalk, prune = NoT, len), NoT)
  (flat′, off), alpha_back
end

# This reconstructs either a model like x, or a gradient for it:
function beta(x, off, flat::AbstractVector; len, walk = _trainable_biwalk, kw...)
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
  trainmap(f, ch, _trainable(x), au) |> re
end

function trainmap(f, ch, tr, aux)
  map(ch, tr, aux) do c, t, a
    isnothing(t) ? c : f(t, a)
  end
end

function _Tangent_biwalk(f, x, aux)  # use with prune = true
  ch, re = functor(typeof(x), x)
  au, _ = functor(typeof(x), aux)
  y = trainmap(f, ch, _trainable(x), au)
  y isa Tuple{} && return NoT
  Tangent{typeof(x), typeof(y)}(y)
end
# _Tangent_biwalk(f, x::Tuple{}, aux) = NoT

function ChainRulesCore.rrule(::typeof(beta), x, off, flat; len)
  dflat = map!(zero, similar(flat, float(eltype(flat))), flat)
  beta_back(dx) = (NoT, NoT, NoT, gamma!(x, dx, off, dflat))
  beta(x, off, flat; len), beta_back
end

# This is the gradient of model reconstruction, accumulating duplicates:
function gamma!(x, dx, off, flat::AbstractVector)
  x′, _ = functor(typeof(x), x)
  dx′, _ = functor(typeof(x), dx)
  off′, _ = functor(typeof(x), off)
  foreach((xᵢ, dxᵢ, oᵢ) -> gamma!(xᵢ, dxᵢ, oᵢ, flat), x′, dx′, off′)
  flat
end
function gamma!(x, dx, off::Integer, flat::AbstractVector)
  @views flat[off .+ (1:length(x))] .+= dx  # must visit all tied nodes, hence no fmap.
  flat
end
gamma!(x, dx::Zero, off, flat::AbstractVector) = nothing
gamma!(x, dx::Zero, off::Integer, flat::AbstractVector) = nothing  # ambiguity

# Least importantly, this infers the eltype if one is not given:
destructure(x) = destructure(omega(x), x)
function omega(x)
  T = Bool
  fmap(x; exclude = isnumeric, walk = (f, z) -> foreach(f, _trainable(z))) do y
    T = promote_type(T, eltype(y))
  end
  T
end
ChainRulesCore.@non_differentiable omega(::Any)
