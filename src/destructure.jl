
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, unthunk
const NoT = NoTangent()

"""
    destructure(model; alias = false) -> vector, reconstructor

Copies all [`trainable`](@ref Optimisers.trainable), [`isnumeric`](@ref Optimisers.isnumeric) parameters in the model
to a vector, and returns also a function which reverses this transformation.
Differentiable, also to higher order, such as a Hessian by Zygote over Zygote.

# Example
```jldoctest
julia> v, re = destructure((x=[1.0, 2.0], y=(sin, [3.0 + 4.0im])))
(ComplexF64[1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 4.0im], Restructure(NamedTuple, ..., 3))

julia> re([3, 5, 7+11im])
(x = [3.0, 5.0], y = (sin, ComplexF64[7.0 + 11.0im]))
```

If `model` contains various number types, they are promoted to make `vector`.

# Element types

Each parameter array of the new model `re(p)` has the element type of `model`'s array:
* A `Vector{Float64}` gives Float32 parameters where `model`'s were Float32.
  As this loses precision, `re(p)` warns once when `p` is more precise than `model`.
  To compute in Float64, convert `model` before `destructure`, e.g. with Flux's `f64`.
* Real parameters stay real, and complex ones complex. A complex `p` is an error
  when `model` has no complex parameters, otherwise the real ones take its real part.
* Structured arrays such as `Diagonal` keep their structure.
* Numbers which carry derivatives pass through, so that `re(p)` works with ForwardDiff and ReverseDiff.
  But `ForwardDiff.Dual` numbers take the precision of `model`, so that
  ForwardDiff and Zygote differentiate the same model.

Non-trainable arrays, such as the running statistics of a `BatchNorm` layer,
are not in `vector`. `re(p)` shares them with `model`, so they keep its element type.

# Devices

The parameter arrays of `re(p)` are slices of `p`, so they live where `p` does:
if `p` is on a GPU, then so are they.
All the parameter arrays of `model` must be on one device, and then so is `vector`.
When MLDataDevices.jl is loaded (as it is by Flux and Lux), `destructure` checks this.

# Aliasing

With `alias = true`, `re(p)` makes no copies: each parameter array of the new model is
a reshaped `view` of `p`, so writing into `p` changes this model.
As a view cannot convert, `p` must have the element type of `model`'s parameters,
or else be `ForwardDiff.Dual` numbers of their precision, and `re(p)` throws an error otherwise.
So all parameter arrays must have one element type, and must be `StridedArray`s,
which excludes for example `SArray` and `Diagonal`.
Tied arrays are views of the same part of `p`.

```jldoctest
julia> v, re = destructure((x = [1.0, 2.0], y = (sin, [3.0 4.0])); alias = true)
([1.0, 2.0, 3.0, 4.0], Restructure(NamedTuple, ..., 4; alias = true))

julia> p = [5.0, 6.0, 7.0, 8.0]; m = re(p)
(x = [5.0, 6.0], y = (sin, [7.0 8.0]))

julia> p[1] = 0; m.x
2-element view(::Vector{Float64}, 1:2) with eltype Float64:
 0.0
 6.0
```

See also [`destructure!`](@ref), which writes into the model instead of copying it.
"""
function destructure(x; alias::Bool = false)
  flat, off, len, ntied = _flatten(x)
  alias || return flat, Restructure(x, off, len, ntied, Val(:copy), Val(eltype(flat)))
  _check_aliasable(x)
  flat, Restructure(x, off, len, ntied, Val(:alias), Val(eltype(flat)))
end

function _check_aliasable(x)
  leaves = trainables(x)
  foreach(_check_aliasable_leaf, leaves)
  Ts = unique(map(eltype, leaves))
  length(Ts) <= 1 ||
    throw(ArgumentError("destructure(model; alias = true) needs parameter arrays of one element type, as views of one vector have its element type, but got $(join(Ts, ", "))"))
end
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
Like `re(p)`, it warns once when `p` is more precise than the model's parameters,
and a complex `p` is an error when the model has no complex parameters.

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
  flat, Restructure(x, off, len, ntied, Val(:inplace), Val(eltype(flat)))
end

function destructure!(flat::AbstractVector, x)
  off, leaves, len, ntied = _plan(x)
  len == length(flat) || throw(DimensionMismatch("destructure! expected a vector of length $len, got $(length(flat))"))
  _check_devices(push!(copy(leaves), flat))
  foreach(_check_writable_leaf, leaves)
  _copyleaves!(flat, leaves)
  flat, Restructure(x, off, len, ntied, Val(:inplace), Val(_promote_eltype(leaves)))
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
struct Restructure{T,S,M,E}
  model::T
  offsets::S
  length::Int
  ntied::Int
  mode::Val{M}  # :copy, :inplace or :alias
  eltype::Val{E}  # of the model's parameters, promoted, as for the vector from destructure
end
Restructure(model, offsets, length::Int, ntied::Int = 0, mode::Val = Val(:copy)) =
  Restructure(model, offsets, length, ntied, mode, Val(_promote_eltype(trainables(model))))

# Only the model is a child, and reconstruction finds its new eltype, e.g. after Flux's f64:
Functors.functor(::Type{<:Restructure}, re) = (; re.model), y -> Restructure(y.model, re.offsets, re.length, re.ntied, re.mode)

(re::Restructure)(flat::AbstractVector) = (_check_eltype(re, flat); _rebuild(re.model, re.offsets, flat, re.length, re.ntied))
(re::Restructure{<:Any, <:Any, :inplace})(flat::AbstractVector) = (_check_eltype(re, flat); _rebuild!(re.model, re.offsets, flat, re.length, re.ntied))
(re::Restructure{<:Any, <:Any, :alias})(flat::AbstractVector) = (_check_eltype(re, flat); _rebuild_alias(re.model, re.offsets, flat, re.length, re.ntied))
(re::Restructure)(x, flat::AbstractVector) = re(flat)(x)
Base.show(io::IO, re::Restructure{T,S,M}) where {T,S,M} = print(io, M === :inplace ? "Restructure!(" : "Restructure(",
  T.name.name, ", ..., ", re.length, M === :alias ? "; alias = true)" : ")")
Base.length(re::Restructure) = re.length

function _check_eltype(re::Restructure{<:Any, <:Any, M, E}, ::AbstractVector{P}) where {M, E, P}
  re.length == 0 && return nothing
  P <: Complex && !(E <: Complex) &&
    throw(ArgumentError("Restructure got a vector of $P, but the model has no complex parameters"))
  if M === :alias
    _eltype_for(E, P) === P || throw(ArgumentError("Restructure from `destructure(model; alias = true)` needs a vector of $(_eltype_for(E, P)), the element type of the model's parameters, as views cannot convert it, but got $P"))
  else
    p, e = _precision(P), _precision(E)
    p === nothing || e === nothing || promote_type(p, e) === e || _warn_precision(Val(M), e, p)
  end
  nothing
end

# Of plain floats, else nothing. ForwardDiff's extension finds the precision of Dual numbers:
_precision(::Type{P}) where P = P <: Union{AbstractFloat, Complex{<:AbstractFloat}} ? real(P) : nothing

@noinline _warn_precision(::Val, e, p) = @warn """Restructure got numbers of precision $p, for a model whose parameters are $e.
  The new model has the element types of the old one, so this loses precision.
  To compute in $p, convert the model before `destructure` (e.g. with Flux's `f64`), or else convert the vector.""" maxlog=1
@noinline _warn_precision(::Val{:inplace}, e, p) = @warn """Restructure! got numbers of precision $p, for a model whose parameters are $e.
  They are converted to the element types of the model's arrays, so this loses precision.""" maxlog=1

ChainRulesCore.@non_differentiable _check_eltype(re, flat)

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
  _check_devices(leaves)
  flat = _copyleaves!(similar(vec(first(leaves)), _promote_eltype(leaves), len), leaves)
  flat, off, len, ntied
end

_promote_eltype(leaves) = isempty(leaves) ? Bool : mapreduce(eltype, promote_type, leaves)

# Checks that all arrays are on one device, when MLDataDevices is loaded:
_check_devices(arrays) = nothing
ChainRulesCore.@non_differentiable _check_devices(arrays)

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

# ProjectTo restores structure such as Diagonal, and the element type, except for SArray and Dual numbers:
_getat(y::AbstractArray, o::Integer, flat::AbstractVector) =
  ProjectTo(y)(_numbers(_eltype_for(eltype(y), eltype(flat)), reshape(flat[o .+ (1:length(y))], axes(y))))
_getat(y::AbstractArray, o::Integer, a::_Alias) = reshape(view(a.flat, o .+ (1:length(y))), axes(y))

# The element type for a rebuilt array: that of the model's array Y, for plain numbers p::P.
# Other numbers pass through, except that ForwardDiff's extension gives Dual numbers the precision of Y.
_eltype_for(::Type{Y}, ::Type{P}) where {Y, P} =
  P <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}} ? Y : _retype(P, Y)
_retype(::Type{P}, ::Type) where P = P

_numbers(::Type{T}, z::AbstractArray{T}) where T = z
_numbers(::Type{T}, z::AbstractArray) where T = broadcast(x -> convert(T, T <: Real ? real(x) : x), z)

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
  _rebuild_back(dx) = (NoT, NoT, NoT, _gradflat(x, off, dx, flat, nothing), NoT, NoT)
  f(x, off, flat, len, ntied), _rebuild_back
end

###
### the two pullbacks, each the adjoint of the other
###

# _gradflat accumulates a tangent dx of x into a new vector like flat, the gradient of rebuild.
# _tangent slices a vector dflat into a tangent of x, the gradient of flatten.
# With ties = nothing, they use every position of a tied array; with ties = ntied, the first only.
# Each one's rrule calls the other, so that nested AD never differentiates their internals.
_gradflat(x, off, dx, flat::AbstractVector, ties) = _accum!(x, off, dx, _zero(flat), _seen(ties))
_gradflat(x, off, dx, ::Zero, ties) = NoT

_tangent(x, off, dflat::AbstractVector, ties) = _tang(x, off, dflat, _seen(ties))
_tangent(x, off, ::Zero, ties) = NoT

# As _tangent, but in the form of dx, as the cotangent of _gradflat's argument dx:
_tangent_like(dx, x, off, dflat::AbstractVector, ties) = _cot(dx, x, off, dflat, _seen(ties))
_tangent_like(dx, x, off, ::Zero, ties) = NoT

function ChainRulesCore.rrule(::typeof(_gradflat), x, off, dx, flat, ties)
  _gradflat_back(dflat) = (NoT, NoT, NoT, _tangent_like(dx, x, off, unthunk(dflat), ties), NoT, NoT)
  _gradflat(x, off, dx, flat, ties), _gradflat_back
end
function ChainRulesCore.rrule(::typeof(_tangent), x, off, dflat, ties)
  _tangent_back(dt) = (NoT, NoT, NoT, _gradflat(x, off, dt, dflat, ties), NoT)
  _tangent(x, off, dflat, ties), _tangent_back
end
function ChainRulesCore.rrule(::typeof(_tangent_like), dx, x, off, dflat, ties)
  _tangent_like_back(dt) = (NoT, NoT, NoT, NoT, _gradflat(x, off, dt, dflat, ties), NoT)
  _tangent_like(dx, x, off, dflat, ties), _tangent_like_back
end

_seen(::Nothing) = nothing
_seen(ntied::Int) = falses(ntied)

_firstvisit!(seen::AbstractVector{Bool}, k::Int) = seen[k] ? false : (seen[k] = true)
_firstvisit!(::Nothing, k::Int) = true

# Where dx is zero, the walk still marks the tied arrays it passes, so that both maps agree on which position is first:
_skip!(off, ::Nothing) = nothing
_skip!(::Union{Integer, Nothing}, seen::AbstractVector{Bool}) = nothing
_skip!(t::Tied, seen::AbstractVector{Bool}) = _firstvisit!(seen, t.k) ? _skip!(t.offsets, seen) : nothing
_skip!(off, seen::AbstractVector{Bool}) = foreach(o -> _skip!(o, seen), _values(off))

# The children of dx, matching those of functor(x). As well as ChainRules' Tangent and Zygote's NamedTuple,
# this accepts the form of nested AD, which treats a Tangent as a struct with one field, backing:
_children(x, dx) = functor(typeof(x), _strip(x, dx))[1]
_children(x::AbstractDict, dx) = _strip(x, dx)
_strip(x, dx::Tangent) = _strip(x, base(dx))
_strip(x, dx::NamedTuple{(:backing,)}) = hasfield(typeof(x), :backing) ? dx : _strip(x, dx.backing)
_strip(x, dx) = dx

###
### accumulate a tangent into a flat vector
###

function _accum!(x, off, dx, flat, seen)
  dx = unthunk(dx)
  dx isa Zero || return _accum_node!(x, off, dx, flat, seen)
  _skip!(off, seen)
  flat
end

_accum_node!(x, t::Tied, dx, flat, seen) = _firstvisit!(seen, t.k) ? _accum_node!(x, t.offsets, dx, flat, seen) : flat
function _accum_node!(x, off::Integer, dx, flat::AbstractVector{T}, seen) where T
  T2 = promote_type(T, eltype(dx))
  if T != T2  # then we must widen the type
    flat = copyto!(similar(flat, T2), flat)
  end
  @views flat[off .+ (1:length(x))] .+= vec(dx)
  flat
end
@generated function _accum_node!(x, off::Union{Tuple, NamedTuple}, dx, flat, seen)
  steps = [:(flat = _accum!(ch[$i], off[$i], dch[$i], flat, seen)) for i in 1:fieldcount(off) if _istrainable(off, i)]
  quote
    ch, _ = functor(x)
    dch = _children(x, dx)
    $(steps...)
    flat
  end
end
function _accum_node!(x, off, dx, flat, seen)  # Array of arrays
  ch, _ = functor(x)
  for (c, o, d) in zip(ch, off, _children(x, dx))
    o === nothing || (flat = _accum!(c, o, d, flat, seen))
  end
  flat
end
function _accum_node!(x::AbstractDict, off::AbstractDict, dx, flat, seen)
  dch = _children(x, dx)
  for (k, c) in pairs(x)
    off[k] === nothing || (flat = _accum!(c, off[k], get(dch, k, nothing), flat, seen))
  end
  flat
end

###
### slice a flat vector into a tangent of the model
###

_tang(x, o::Integer, dflat, seen) = ProjectTo(x)(reshape(dflat[o .+ (1:length(x))], axes(x)))
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
_structural(x, y, re) = _allzero(y) ? NoT : _structural_tangent(x, y, re)
_structural_tangent(x::AbstractArray, y::AbstractArray, re) = ProjectTo(x)(y)
_structural_tangent(x::AbstractArray, y, re) = re(y)
_structural_tangent(x::Union{Tuple, NamedTuple, AbstractDict}, y, re) = Tangent{typeof(x), typeof(y)}(y)
function _structural_tangent(x, y, re)
  y isa NamedTuple && keys(y) ⊆ fieldnames(typeof(x)) && return Tangent{typeof(x)}(; y...)
  throw(ArgumentError("cannot express the gradient of `destructure` for $(typeof(x)), as its `functor` children are not its fields"))
end
_allzero(y) = all(d -> d isa AbstractZero, _values(y))

###
### the same, in the form of a given tangent dx
###

# The walk is that of _accum!, so that the two are adjoint:
function _cot(dx, x, off, dflat, seen)
  dx = unthunk(dx)
  dx isa Zero || return _cot_node(dx, x, off, dflat, seen)
  _skip!(off, seen)
  NoT
end

_cot_node(dx, x, t::Tied, dflat, seen) = _firstvisit!(seen, t.k) ? _cot_node(dx, x, t.offsets, dflat, seen) : NoT
_cot_node(dx, x, o::Integer, dflat, seen) = ProjectTo(dx)(reshape(dflat[o .+ (1:length(x))], axes(x)))
@generated function _cot_node(dx, x, off::Union{Tuple, NamedTuple}, dflat, seen)
  args = [_istrainable(off, i) ? :(_cot(dch[$i], ch[$i], off[$i], dflat, seen)) : :NoT for i in 1:fieldcount(off)]
  quote
    ch, re = functor(x)
    dch = _children(x, dx)
    _like(dx, x, $(_children_expr(off, args)), re)
  end
end
function _cot_node(dx, x, off, dflat, seen)  # Dict, Array of arrays
  ch, re = functor(x)
  _like(dx, x, mapvalue((c, o, d) -> o === nothing ? NoT : _cot(d, c, o, dflat, seen), ch, off, _children(x, dx)), re)
end

# From the tangents y of x's children, a tangent in the form of dx. Nested AD sees a Tangent
# as a struct, so its tangent is a Tangent of a Tangent:
_like(dx, x, y, re) = _allzero(y) ? NoT : __like(dx, x, y, re)
__like(dx::Tangent, x, y, re) = Tangent{typeof(dx)}(backing = __like(backing(dx), x, y, re))
function __like(dx::NamedTuple{K}, x, y, re) where K
  K === (:backing,) && !hasfield(typeof(x), :backing) &&
    return Tangent{typeof(dx)}(backing = __like(dx.backing, x, y, re))
  y isa NamedTuple && all(k -> k in K || y[k] isa AbstractZero, keys(y)) ||
    throw(ArgumentError("cannot express the second derivative of `Restructure` for $(typeof(x)), as its `functor` children are not its fields"))
  z = NamedTuple{K}(map(k -> get(y, k, NoT), K))
  Tangent{typeof(dx), typeof(z)}(z)
end
__like(dx::Tuple, x, y::Tuple, re) = Tangent{typeof(dx), typeof(y)}(y)
__like(dx::AbstractArray, x::AbstractArray, y::AbstractArray, re) = y  # Array of arrays
__like(dx::AbstractArray, x::AbstractArray, y, re) = re(y)  # natural tangent of a wrapper like Transpose
__like(dx::AbstractDict, x, y::AbstractDict, re) = y
