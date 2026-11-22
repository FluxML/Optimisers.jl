
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, unthunk
const NoT = NoTangent()

"""
    destructure(model; alias = false) -> vector, reconstructor

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

With `alias = true`, `re(p)` makes no copies: each parameter array of the new model is
a reshaped `view` of `p`, so writing into `p` changes this model.
Its element types follow `p`, as there is no conversion. Tied arrays are views of the same part of `p`.
Every parameter array must be a `StridedArray`, which excludes for example `SArray` and `Diagonal`.

```jldoctest
julia> v, re = destructure((x = [1.0, 2.0], y = (sin, [3.0 4.0])); alias = true)
([1.0, 2.0, 3.0, 4.0], Restructure(NamedTuple, ..., 4; alias = true))

julia> p = Float32[5, 6, 7, 8]; m = re(p)
(x = Float32[5.0, 6.0], y = (sin, Float32[7.0 8.0]))

julia> p[1] = 0; m.x
2-element view(::Vector{Float32}, 1:2) with eltype Float32:
 0.0
 6.0
```

See also [`destructure!`](@ref), which writes into the model instead of copying it.
"""
function destructure(x; alias::Bool = false)
  flat, off, len, ntied = _flatten(x)
  alias || return flat, Restructure(x, off, len, ntied)
  _check_aliasable(x)
  flat, Restructure(x, off, len, ntied, Val(:alias))
end

_check_aliasable(x) = foreach(_check_aliasable_leaf, trainables(x))
_check_aliasable_leaf(y) = y isa StridedArray ||
  throw(ArgumentError("destructure(model; alias = true) needs strided parameter arrays, which a view can replace, but got $(typeof(y))"))
ChainRulesCore.@non_differentiable _check_aliasable(x)

"""
    destructure!(model) -> vector, re!
    destructure!(vector, model) -> vector, re!

Like [`destructure`](@ref), but `re!(p)` copies `p` into the parameter arrays of `model` itself,
and returns `model`, instead of building a new one. Every parameter array must allow this,
see [`maywrite`](@ref Optimisers.maywrite).

The second method writes the parameters of `model` into the existing `vector`.

As `re!(p)` converts `p` to the element types of these arrays, it cannot accept for example
`ForwardDiff.Dual` numbers. Use `re(p)` from `destructure` for those.

!!! warning "Gradients"
    Zygote can differentiate `re!(p)`, but only through the model it returns.
    Use `m2 = re!(p)` and then `m2(x)`, even though `m2 === model`: a loss computed
    with the original `model` gets a silently wrong gradient.
    Call `re!` once per gradient, as a second call overwrites arrays which
    the first call's pullbacks may still read.
    With Enzyme, `re!` holds the arrays it writes into, so it must be passed as
    `Duplicated(re!, Enzyme.make_zero(re!))`: with `Const(re!)` the gradient is silently wrong.

# Example
```jldoctest
julia> m = (x = [1.0, 2.0], y = (sin, Float32[3.0 4.0]));

julia> v, re! = destructure!(m)
([1.0, 2.0, 3.0, 4.0], Restructure!(NamedTuple, ..., 4))

julia> re!([3, 5, 7, 9]) === m
true

julia> m
(x = [3.0, 5.0], y = (sin, Float32[7.0 9.0]))

julia> destructure!(v, (x = [0.1, 0.2], y = (cos, [0.3 0.4])));

julia> v
4-element Vector{Float64}:
 0.1
 0.2
 0.3
 0.4
```
"""
function destructure!(x)
  flat, off, len, ntied = _flatten(x)
  _check_writable(x)
  flat, Restructure(x, off, len, ntied, Val(:inplace))
end

function destructure!(flat::AbstractVector, x)
  off, leaves, len, ntied = _plan(x)
  len == length(flat) || throw(DimensionMismatch("destructure! expected a vector of length $len, got $(length(flat))"))
  foreach(_check_writable_leaf, leaves)
  _copyleaves!(flat, leaves)
  flat, Restructure(x, off, len, ntied, Val(:inplace))
end

_check_writable(x) = foreach(_check_writable_leaf, trainables(x))
_check_writable_leaf(y) = maywrite(y) ||
  throw(ArgumentError("destructure! needs parameter arrays it can write into, but got $(typeof(y)), for which `Optimisers.maywrite` is false"))
ChainRulesCore.@non_differentiable _check_writable(x)

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

The one returned by [`destructure!`](@ref) instead writes `p` into the model's arrays,
and is printed as `Restructure!`. The one from `destructure(model; alias = true)` builds the model
out of views of `p`.
"""
struct Restructure{T,S,M}
  model::T
  offsets::S
  length::Int
  ntied::Int
  mode::Val{M}  # :copy, :inplace or :alias. A field, not only a type parameter, so that Functors' reconstruction keeps it
end
Restructure(model, offsets, length::Int, ntied::Int = 0) = Restructure(model, offsets, length, ntied, Val(:copy))

(re::Restructure)(flat::AbstractVector) = _rebuild(re.model, re.offsets, flat, re.length, re.ntied)
(re::Restructure{<:Any, <:Any, :inplace})(flat::AbstractVector) = _rebuild!(re.model, re.offsets, flat, re.length, re.ntied)
(re::Restructure{<:Any, <:Any, :alias})(flat::AbstractVector) = _rebuild_alias(re.model, re.offsets, flat, re.length, re.ntied)
(re::Restructure)(x, flat::AbstractVector) = re(flat)(x)
Base.show(io::IO, re::Restructure{T,S,M}) where {T,S,M} = print(io, M === :inplace ? "Restructure!(" : "Restructure(",
  T.name.name, ", ..., ", re.length, M === :alias ? "; alias = true)" : ")")
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
  off, leaves, len, ntied = _plan(x)
  isempty(leaves) && return Bool[], off, 0, ntied
  T = mapreduce(eltype, promote_type, leaves)
  flat = _copyleaves!(similar(vec(first(leaves)), T, len), leaves)
  flat, off, len, ntied
end

function _plan(x)
  counts = IdDict{Any,Int}()
  _count!(counts, x)
  plan = _Plan(counts, IdDict{Any,Any}(), AbstractArray[], Ref(0), Ref(0))
  off = _plan!(plan, x)
  off, plan.leaves, plan.len[], plan.ntied[]
end

function _copyleaves!(flat::AbstractVector, leaves)
  o = 0
  for y in leaves
    copyto!(flat, o + 1, vec(y), 1, length(y))
    o += length(y)
  end
  flat
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
  _checklength(len, flat)
  _build(x, off, flat, _cache(ntied))
end

# Wrapping flat makes _build use views instead of copies:
struct _Alias{V<:AbstractVector}
  flat::V
end
function _rebuild_alias(x, off, flat::AbstractVector, len::Int, ntied::Int)
  _checklength(len, flat)
  _build(x, off, _Alias(flat), _cache(ntied))
end

_checklength(len, flat) = len == length(flat) ||
  throw(DimensionMismatch("Rebuild expected a vector of length $len, got $(length(flat))"))
_cache(ntied) = ntied == 0 ? nothing : Vector{Any}(nothing, ntied)

_getat(y::AbstractArray, o::Integer, flat::AbstractVector) =
  ProjectTo(y)(reshape(flat[o .+ (1:length(y))], axes(y)))  # ProjectTo is just correcting eltypes
_getat(y::AbstractArray, o::Integer, a::_Alias) = reshape(view(a.flat, o .+ (1:length(y))), axes(y))

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

_zero(x) = map!(zero, similar(x, float(eltype(x))), x)  # mutable zero array for _accum!
ChainRulesCore.@non_differentiable _zero(x)

###
### rebuild in place, writing each tied array once
###

function _rebuild!(x, off, flat::AbstractVector, len::Int, ntied::Int)
  _checklength(len, flat)
  _write!(x, off, flat, ntied == 0 ? nothing : falses(ntied))
  x
end

_write!(x, o::Integer, flat, seen) = (copyto!(x, 1, flat, o + 1, length(x)); nothing)
_write!(x, t::Tied, flat, seen) = _firstvisit!(seen, t.k) ? _write!(x, t.offsets, flat, seen) : nothing
@generated function _write!(x, off::Union{Tuple, NamedTuple}, flat, seen)
  steps = [:(_write!(ch[$i], off[$i], flat, seen)) for i in 1:fieldcount(off) if _istrainable(off, i)]
  quote
    ch, _ = functor(x)
    $(steps...)
    nothing
  end
end
function _write!(x, off, flat, seen)  # Array of arrays
  ch, _ = functor(x)
  for (c, o) in zip(ch, off)
    o === nothing || _write!(c, o, flat, seen)
  end
end
function _write!(x::AbstractDict, off::AbstractDict, flat, seen)
  for (k, c) in pairs(x)
    off[k] === nothing || _write!(c, off[k], flat, seen)
  end
end

# All three rebuilds are the same linear map, so they have the same pullback:
function ChainRulesCore.rrule(f::Union{typeof(_rebuild), typeof(_rebuild_alias), typeof(_rebuild!)}, x, off, flat, len, ntied)
  _rebuild_back(dx) = (NoT, NoT, NoT, _accum!(x, off, dx, _zero(flat)), NoT, NoT)
  f(x, off, flat, len, ntied), _rebuild_back
end

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
