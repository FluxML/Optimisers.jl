
using ChainRulesCore: canonicalize, backing, Tangent, AbstractZero, ZeroTangent
base(dx::Tangent) = backing(canonicalize(dx))
base(dx) = dx
const Zero = Union{Nothing, AbstractZero}  # Union{Zygote, Diffractor}

abstract type AbstractRule end

###
### setup
###

mutable struct Leaf{R,S}  # mutable so that its identity encodes parameter sharing
  rule::R
  state::S
end

@functor Leaf

Base.:(==)(a::Leaf, b::Leaf) = children(a) == children(b)

function setup(rule::AbstractRule, model)
  cnt = Ref(0)
  # Rely on Functors to identify shared arrays, they will share a Leaf in this tree:
  tree = fmapstructure(model, exclude = isnumeric) do x
    cnt[] += 1
    Leaf(rule, init(rule, x))
  end
  cnt[] == 0 && @warn "setup found no parameters in the given model"
  tree
end

function Base.show(io::IO, ℓ::Leaf)  # show method is mostly to hide its long type!
  ioc = IOContext(io, :compact => true)
  print(ioc, "Leaf(", ℓ.rule, ", ")
  show(ioc, ℓ.state)
  print(ioc, ")")
end

###
### update
###

function update!(tree, model, grad)
  # First walk is to accumulate the gradient. This recursion visits every copy of
  # shared leaves, but stops when branches are absent from the gradient:
  dict = IdDict{Leaf, Any}()
  grads!(dict, tree, model, grad)
  # Second walk is to update the model. The walk taken follows Leaf identity
  newmodel = fmap(tree, model; exclude = ℓ -> ℓ isa Leaf, walk = _second_walk, cache = LeafCache()) do ℓ, x
    haskey(dict, ℓ) || return x  # no gradient seen, nothing to do
    s′, x̄′ = apply!(ℓ.rule, ℓ.state, x, dict[ℓ])
    ℓ.state = s′  # to get state out of here, rely on mutability of Leaf
    subtract!(x, x̄′)
  end
  tree, newmodel  # note that tree is guaranteed to be updated
end

subtract!(x, x̄) = maywrite(x) ? (x .= x .- x̄) : eltype(x).(x .- x̄)

grads!(dict::IdDict, ℓ::Leaf, x, ::Zero) = nothing
function grads!(dict::IdDict, ℓ::Leaf, x, x̄)
  x̄₀ = get(dict, ℓ, false)
  dict[ℓ] = Broadcast.broadcasted(+, x̄, x̄₀)
  nothing
end
grads!(dict::IdDict, t, x, ::Zero) = nothing
function grads!(dict::IdDict, tree, x, x̄s...)
  # The only reason grads! takes model is that functor(typeof(x), base(x̄)) may differ from 
  # functor(typeof(tree), base(x̄)), for things like Transpose
  x̄s′ = map(x̄ -> functor(typeof(x), base(x̄))[1], x̄s)
  x′, _ = functor(typeof(x), x)
  foreach((tᵢ, xᵢ, x̄sᵢ...) -> grads!(dict, tᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
end

function update(tree, x, x̄s...)
  t′ = fmap(copy, tree; exclude = maywrite)  # goes inside Leaf
  x′ = fmap(copy, x; exclude = maywrite)
  update!(t′, x′, x̄s...)
end

# This differs from _default_walk(f,x,y) in taking re from 2nd argument, but cache will still operate on the first
function _second_walk(f, x, y)
  x′, _ = functor(typeof(y), x)
  y′, re = functor(y)
  re(map(f, x′, y′))
end

# When fmap reconstructs for update!, it should not cache results with trivial nodes like () in the state.
# This cache type has just enough methods to work in Functors, which possibly should be upgraded to just work.
struct LeafCache <: AbstractDict{Leaf,Any}
  dict::IdDict{Leaf,Any}
end
LeafCache() = LeafCache(IdDict{Leaf,Any}())

Base.setindex!(c::LeafCache, x, ℓ::Leaf) = setindex!(c.dict, x, ℓ)
Base.setindex!(c::LeafCache, x, _) = nothing
Base.in(k, c::LeafCache) = k in c.dict
Base.haskey(c::LeafCache, k) = haskey(c.dict, k)
Base.getindex(c::LeafCache, ℓ::Leaf) = getindex(c.dict, ℓ)
Base.iterate(c::LeafCache, i = 0) = iterate(c.dict, i)
Base.length(c::LeafCache) = length(c.dict)

# default all rules to first order calls
apply!(o, state, x, dx, dx2, dxs...) = apply!(o, state, x, dx)

###
### sources of truth
###

"""
    isnumeric(x) -> Bool

Returns `true` on any parameter to be adjusted by Optimisers.jl,
namely arrays of non-integer numbers. Returns `false` on all other types.

Requires also that `Functors.isleaf(x) == true`, to focus on e.g. the
parent of a transposed matrix, not the wrapper.
"""
isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Integer}) = false
isnumeric(x) = false

"""
    maywrite(x) -> Bool

Should return `true` if we are completely sure that `update!` can write new
values into `x`. Otherwise `false`, indicating a non-mutating path.
For now, simply `x isa DenseArray` allowing `Array`, `CuArray`, etc. 
"""
maywrite(::DenseArray) = true  # see https://github.com/FluxML/Optimisers.jl/issues/99 for discussion
maywrite(_) = false

@deprecate iswriteable maywrite false  # remove when releasing Optimisers@0.3

"""
    trainable(x::Layer) -> NamedTuple

This should be overloaded to make optimisers ignore some fields of
every `Layer`, which would otherwise contain trainable parameters.
(Elements such as functions and sizes are always ignored.)

The default is `Functors.children(x)`, usually a NamedTuple of all fields,
and `trainable(x)` must contain a subset of these.
"""
trainable(x) = functor(x)[1]

_trainable(x) = _trainable(functor(x)[1], trainable(x))
_trainable(ch::NamedTuple, tr::NamedTuple) = merge(map(_ -> nothing, ch), tr)
_trainable(ch::Tuple{Vararg{Any,N}}, tr::Tuple{Vararg{Any,N}}) where N = tr
_trainable(ch::AbstractArray, tr::AbstractArray) = tr
function _trainable(ch::NamedTuple, tr::Tuple)  # for old Flux-style no-names tuple
  @warn "trainable(x) should now return a NamedTuple with the field names, not a Tuple" maxlog=3
  map(c -> c in tr ? c : nothing, ch)
end

###
### rule definition helpers
###

"""
    @.. x = x + y

Sometimes in-place broadcasting macro, for use in `apply!` rules.
If `maywrite(x)` then it is just `@. x = rhs`, but if not, it becomes `x = @. rhs`.
"""
macro var".."(ex)
  Meta.isexpr(ex, :(=)) || throw("the macro @.. only accepts assignment, like @.. x = y + z")
  dst = esc(ex.args[1])
  src = esc(Broadcast.__dot__(ex.args[2]))
  :($dst = if $maywrite($dst)
      $dst .= $src
    else
      $src
    end)
end

"""
    x = @lazy y + z

Lazy broadcasting macro, for use in `apply!` rules. It broadcasts like `@.`
but does not materialise, returning a `Broadcasted` object for later use.
Beware that mutation of arguments will affect the result,
and that if it is used in two places, work will be done twice.
"""
macro lazy(ex)
  bc = esc(Broadcast.__dot__(ex))
  :($lazy.($bc))
end

function lazy end
Broadcast.broadcasted(::typeof(lazy), x) = Lazy(x)
struct Lazy{T}; bc::T; end
Broadcast.materialize(x::Lazy) = Broadcast.instantiate(x.bc)

onevalue(λ::T, x::AbstractArray{T}) where T = map(_ -> λ, x)
onevalue(λ, x::AbstractArray{T}) where T = onevalue(convert(float(T), λ), x)
