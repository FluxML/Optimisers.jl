
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
  flat, off, len, ntied = _flatten(x)
  flat, Restructure(x, off, len, ntied)
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
  ntied::Int
end
Restructure(model, offsets, length::Int) = Restructure(model, offsets, length, 0)
(re::Restructure)(flat::AbstractVector) = _rebuild(re.model, re.offsets, flat, re.length, re.ntied)
(re::Restructure)(x, flat::AbstractVector) = re(flat)(x)
Base.show(io::IO, re::Restructure{T}) where T = print(io, "Restructure(", T.name.name, ", ..., ", re.length, ")")
Base.length(re::Restructure) = re.length

# The offsets mirror _trainable(x): an Int at each parameter array, `nothing` at each non-trainable child.
# An object reached more than once, by the rule of Functors' cache, is wrapped as Tied, numbered k:
struct Tied{O}
  k::Int
  offsets::O
end

# Used by trainables:
struct TrainableStructWalk <: AbstractWalk end

(::TrainableStructWalk)(recurse, x) = mapvalue(recurse, _trainable(x))

###
### flatten
###

function _flatten(x)
  counts = IdDict{Any,Int}()
  _count!(counts, x)
  plan = _Plan(counts, IdDict{Any,Any}(), AbstractArray[], Ref(0), Ref(0))
  off = _plan!(plan, x)
  len, ntied = plan.len[], plan.ntied[]
  isempty(plan.leaves) && return Bool[], off, 0, ntied
  T = mapreduce(eltype, promote_type, plan.leaves)
  flat = similar(vec(first(plan.leaves)), T, len)
  o = 0
  for y in plan.leaves
    copyto!(flat, o + 1, vec(y), 1, length(y))
    o += length(y)
  end
  flat, off, len, ntied
end

_values(x::AbstractDict) = values(x)
_values(x) = x

function _count!(counts::IdDict{Any,Int}, x)
  if _usecache(x)
    n = counts[x] = get(counts, x, 0) + 1
    n > 1 && return
  end
  isnumeric(x) && return
  for c in _values(_trainable(x))
    c === nothing || _count!(counts, c)
  end
end

struct _Plan
  counts::IdDict{Any,Int}
  seen::IdDict{Any,Any}
  leaves::Vector{AbstractArray}
  len::Base.RefValue{Int}
  ntied::Base.RefValue{Int}
end

function _plan!(p::_Plan, x)
  if _usecache(x) && p.counts[x] > 1
    haskey(p.seen, x) && return p.seen[x]
    k = p.ntied[] += 1
    return p.seen[x] = Tied(k, _plan_node!(p, x))
  end
  _plan_node!(p, x)
end

function _plan_node!(p::_Plan, x)
  if isnumeric(x)
    o = p.len[]
    push!(p.leaves, x)
    p.len[] = o + length(x)
    return o
  end
  mapvalue(c -> c === nothing ? nothing : _plan!(p, c), _trainable(x))
end

function ChainRulesCore.rrule(::typeof(_flatten), x)
  flat, off, len, ntied = _flatten(x)
  _maybewarn()
  _flatten_back((dflat, _, _, _)) = (NoT, _tangent(x, off, unthunk(dflat), ntied))
  (flat, off, len, ntied), _flatten_back
end

###
### rebuild
###

# Nodes whose offsets are a Tuple or NamedTuple are unrolled using the type of the offsets,
# which is Nothing exactly at non-trainable children. Only x and its offsets pass through the
# recursion, not a tuple of x's children, which is not part of the parent's type, so Julia's
# recursion limiter would widen it to Any.
_children_expr(::Type{<:NamedTuple{K}}, args) where K = :(NamedTuple{$K}(($(args...),)))
_children_expr(::Type{<:Tuple}, args) = :(($(args...),))
_istrainable(::Type{O}, i) where O = fieldtype(O, i) !== Nothing

function _rebuild(x, off, flat::AbstractVector, len::Int = length(flat), ntied::Int = 0)
  len == length(flat) || throw(DimensionMismatch("Rebuild expected a vector of length $len, got $(length(flat))"))
  ntied == 0 ? _build(x, off, flat, nothing) : _build(x, off, flat, Vector{Any}(nothing, ntied))
end

_getat(y::AbstractArray, o::Integer, flat::AbstractVector) =
  ProjectTo(y)(reshape(flat[o .+ (1:length(y))], axes(y)))  # ProjectTo is just correcting eltypes

_build(x, o::Integer, flat, cache) = _getat(x, o, flat)
function _build(x, t::Tied, flat, cache)
  y = cache[t.k]
  y === nothing || return y::Base.promote_op(_build, typeof(x), typeof(t.offsets), typeof(flat), typeof(cache))
  return cache[t.k] = _build(x, t.offsets, flat, cache)
end
@generated function _build(x, off::Union{Tuple, NamedTuple}, flat, cache)
  args = [_istrainable(off, i) ? :(_build(ch[$i], off[$i], flat, cache)) : :(ch[$i]) for i in 1:fieldcount(off)]
  :((ch, re) = functor(x); re($(_children_expr(off, args))))
end
function _build(x, off, flat, cache)  # Dict, Array of arrays
  ch, re = functor(x)
  re(mapvalue((c, o) -> o === nothing ? c : _build(c, o, flat, cache), ch, off))
end

function ChainRulesCore.rrule(::typeof(_rebuild), x, off, flat, len, ntied)
  _rebuild_back(dx) = (NoT, NoT, NoT, _accum!(x, off, dx, _zero(flat)), NoT, NoT)
  _rebuild(x, off, flat, len, ntied), _rebuild_back
end

_zero(x) = map!(zero, similar(x, float(eltype(x))), x)  # mutable zero array for _accum!
ChainRulesCore.@non_differentiable _zero(x)

###
### gradient of rebuild, accumulating at all tied positions
###

_accum!(x, off, dx, flat) = (dx = unthunk(dx); dx isa Zero ? flat : _accum_node!(x, off, dx, flat))

_accum_node!(x, t::Tied, dx, flat) = _accum!(x, t.offsets, dx, flat)
function _accum_node!(x, off::Integer, dx, flat::AbstractVector{T}) where T
  T2 = promote_type(T, eltype(dx))
  if T != T2  # then we must widen the type
    flat = copyto!(similar(flat, T2), flat)
  end
  @views flat[off .+ (1:length(x))] .+= vec(dx)
  flat
end
# Cotangents are structured like x's fields, hence functor(typeof(x), dx) for their children:
@generated function _accum_node!(x, off::Union{Tuple, NamedTuple}, dx, flat)
  steps = [:(flat = _accum!(ch[$i], off[$i], dch[$i], flat)) for i in 1:fieldcount(off) if _istrainable(off, i)]
  quote
    ch, _ = functor(x)
    dch, _ = functor(typeof(x), base(dx))
    $(steps...)
    flat
  end
end
function _accum_node!(x, off, dx, flat)  # Array of arrays
  ch, _ = functor(x)
  dch, _ = functor(typeof(x), base(dx))
  for (c, o, d) in zip(ch, off, dch)
    o === nothing || (flat = _accum!(c, o, d, flat))
  end
  flat
end
function _accum_node!(x::AbstractDict, off::AbstractDict, dx, flat)
  dch = base(dx)
  for (k, c) in pairs(x)
    off[k] === nothing || (flat = _accum!(c, off[k], get(dch, k, nothing), flat))
  end
  flat
end

###
### gradient of flatten, a tangent for the model
###

_tangent(x, off, dflat::AbstractVector, ntied::Int) = _tang(x, off, dflat, falses(ntied))
_tangent(x, off, ::Zero, ntied::Int) = NoT

# The gradient of flatten goes to the first tied position only. With seen = nothing, to all.
_firstvisit!(seen::AbstractVector{Bool}, k::Int) = seen[k] ? false : (seen[k] = true)
_firstvisit!(::Nothing, k::Int) = true
ChainRulesCore.@non_differentiable _firstvisit!(seen, k)

_tang(x, o::Integer, dflat, seen) = _getat(x, o, dflat)
_tang(x, t::Tied, dflat, seen) = _firstvisit!(seen, t.k) ? _tang(x, t.offsets, dflat, seen) : NoT
@generated function _tang(x, off::Union{Tuple, NamedTuple}, dflat, seen)
  args = [_istrainable(off, i) ? :(_tang(ch[$i], off[$i], dflat, seen)) : :NoT for i in 1:fieldcount(off)]
  :((ch, re) = functor(x); _structural(x, $(_children_expr(off, args)), re))
end
function _tang(x, off, dflat, seen)  # Dict, Array of arrays
  ch, re = functor(x)
  _structural(x, mapvalue((c, o) -> o === nothing ? NoT : _tang(c, o, dflat, seen), ch, off), re)
end

# From the tangents y of x's children: a natural tangent for wrappers like Transpose,
# else a Tangent with x's fields.
_structural(x, y, re) = all(d -> d isa AbstractZero, _values(y)) ? NoT : _structural_tangent(x, y, re)
_structural_tangent(x::AbstractArray, y::AbstractArray, re) = ProjectTo(x)(y)
_structural_tangent(x::AbstractArray, y, re) = re(y)
_structural_tangent(x::Union{Tuple, NamedTuple, AbstractDict}, y, re) = Tangent{typeof(x), typeof(y)}(y)
function _structural_tangent(x, y, re)
  y isa NamedTuple && keys(y) ⊆ fieldnames(typeof(x)) && return Tangent{typeof(x)}(; y...)
  throw(ArgumentError("cannot express the gradient of `destructure` for $(typeof(x)), as its `functor` children are not its fields"))
end

###
### second derivatives
###

function ChainRulesCore.rrule(::typeof(_accum!), x, off, dx, flat)
  @warn "second derivatives of Restructure may not work yet, sorry!" maxlog=3
  _accum_back(dflat) = (NoT, NoT, NoT, _tang(x, off, unthunk(dflat), nothing), NoT)
  _accum!(x, off, dx, flat), _accum_back
end
base(dx::Tangent{<:Tangent}) = backing(dx).backing  # might be needed for gradient(gradient(destructure))
base(dx::Tangent{Any, <:NamedTuple{(:backing,)}}) = base(backing(dx).backing)  # Zygote version
_maybewarn() = nothing
function ChainRulesCore.rrule(::typeof(_maybewarn))
  @warn "second derivatives of destructure may not work yet, sorry!" maxlog=3
  nothing, _ -> (NoT,)
end
