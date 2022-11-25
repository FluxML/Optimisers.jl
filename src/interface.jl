
using ChainRulesCore: canonicalize, backing, Tangent, AbstractZero, ZeroTangent
base(dx::Tangent) = backing(canonicalize(dx))
base(dx) = dx
const Zero = Union{Nothing, AbstractZero}  # Union{Zygote, Diffractor}

abstract type AbstractRule end

###
### setup
###

mutable struct Leaf{R,S}  # mutable so that its identity encodes parameter sharing...
  rule::R
  state::S
  frozen::Bool  # ... and to allow freeze! to act on this.
end
Leaf(rule, state; frozen::Bool = false) = Leaf(rule, state, frozen)

@functor Leaf

Base.:(==)(a::Leaf, b::Leaf) = children(a) == children(b)

function setup(rule::AbstractRule, model)
  cache = IdDict()
  tree = _setup(rule, model; cache)
  isempty(cache) && @warn "setup found no trainable parameters in this model"
  tree
end

# _setup is almost fmapstructure, but needs a _trainable_walk, and a cache which ignores numbers etc.
function _setup(rule, x; cache)
  haskey(cache, x) && return cache[x]
  if isnumeric(x)
    ℓ = Leaf(rule, init(rule, x))
    if isbits(x)
      cache[nothing] = nothing  # just to disable the warning
      ℓ
    else
      cache[x] = ℓ
    end
  else
    map(xᵢ -> _setup(rule, xᵢ; cache), _trainable(x))
  end
end

function Base.show(io::IO, ℓ::Leaf; colour = ℓ.frozen ? :cyan : :green)
  ioc = IOContext(io, :compact => true)
  str = sprint(show, ℓ.rule; context = ioc)  # produces Adam{Float32}(0.001, ... not 0.001f0
  printstyled(io, "Leaf(", str, ", "; color = colour)
  show(ioc, ℓ.state)
  printstyled(io, ℓ.frozen ? ", frozen = true)" : ")"; color = colour)
end

###
### update
###

function update(tree, model, grad, higher...)
  t′ = fmap(copy, tree; exclude = maywrite)  # walks inside Leaf
  x′ = fmap(copy, model; exclude = maywrite)
  update!(t′, x′, grad, higher...)
end

update!!(tree, model, grad, higher...) = old_update!(tree, model, grad, higher...)

function old_update!(tree, model, grad, higher...)
  # First walk is to accumulate the gradient. This recursion visits every copy of
  # shared leaves, but stops when branches are absent from the gradient:
  grads = IdDict{Leaf, Any}()
  _grads!(grads, tree, model, grad, higher...)
  # Second walk is to update the model. The params cache indexed by (tree,x),
  # so that identified Leafs can tie isbits parameters, but setup won't do that for you:
  newmodel = _update!(tree, model; grads, params = IdDict())
  tree, newmodel  # note that tree is guaranteed to be updated. Also that it's not necc a tree.
end

function _update!(tree, x; grads, params)
  haskey(params, (tree,x)) && return params[(tree,x)]
  isbits(tree) && return x  # means () is not cached, and also (((),),)
  x′, re = functor(x)
  x′′ = re(map((tᵢ, xᵢ) -> _update!(tᵢ, xᵢ; grads, params), tree, x′))
  if ismutable(x′′)
    params[(tree,x)] = x′′
  else  # no ties to preserve between immutable structs, right?
    x′′
  end
end
function _update!(ℓ::Leaf, x; grads, params)
  haskey(params, (ℓ,x)) && return params[(ℓ,x)]
  ℓ.frozen && return x
  params[(ℓ,x)] = if haskey(grads, ℓ)
    ℓ.state, x̄′ = apply!(ℓ.rule, ℓ.state, x, grads[ℓ]...)
    subtract!(x, x̄′)
  else
    x # no gradient seen
  end
end

subtract!(x, x̄) = maywrite(x) ? (x .= x .- x̄) : eltype(x).(x .- x̄)

_grads!(dict::IdDict, ℓ::Leaf, x, ::Zero...) = nothing
function _grads!(dict::IdDict, ℓ::Leaf, x, x̄s...)
  x̄s₀ = get(dict, ℓ, map(_ -> ZeroTangent(), x̄s))
  dict[ℓ] = map(+, x̄s, x̄s₀)  # adding Zero should be free. Lazy accumulation broadcasted(+, x̄, x̄₀) also possible.
  nothing
end
_grads!(dict::IdDict, t, x, ::Zero...) = nothing
function _grads!(dict::IdDict, tree, x, x̄s...)
  # The only reason _grads! takes model is that functor(typeof(x), base(x̄)) may differ from 
  # functor(typeof(tree), base(x̄)), for things like Transpose
  x̄s′ = map(x̄ -> functor(typeof(x), base(x̄))[1], x̄s)
  x′, _ = functor(typeof(x), x)
  foreach((tᵢ, xᵢ, x̄sᵢ...) -> _grads!(dict, tᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
end

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
    @.. x = y + z

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
