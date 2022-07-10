
using ChainRulesCore: canonicalize, backing, Tangent, AbstractZero
base(dx::Tangent) = backing(canonicalize(dx))
base(dx) = dx
const Zero = Union{Nothing, AbstractZero}  # Union{Zygote, Diffractor}

abstract type AbstractRule end

mutable struct Leaf{R,S}
  rule::R
  state::S
end

function setup(rule, x; cache = IdDict{Any,Leaf}())
  rule isa AbstractRule || Base.depwarn("In future, all optimisation rules should be <: AbstractRule", :setup)
  if isnumeric(x)
    leaf = get(cache, x, missing)
    ismissing(leaf) || return leaf
    leaf = Leaf(rule, init(rule, x))
    isbits(x) || (cache[x] = leaf)
    return leaf
  elseif isleaf(x)
    return nothing
  else
    return map(xᵢ -> setup(rule, xᵢ; cache), _trainable(x))
  end
end

_add!(x, x̄) = iswriteable(x) ? (x .= x .+ x̄) : eltype(x).(x .+ x̄)
subtract!(x, x̄) = iswriteable(x) ? (x .= x .- x̄) : eltype(x).(x .- x̄)

update!(::Nothing, x, ::Zero, ::Zero...) = nothing, x
update!(::Nothing, x, x̄s...) = nothing, x

update!(ℓ::Leaf, x, ::Zero, ::Zero...) = ℓ, x
function update!(ℓ::Leaf, x, x̄s...)
  ℓ.state, x̄′ = apply!(ℓ.rule, ℓ.state, x, map(base, x̄s)...)
  return ℓ, subtract!(x, x̄′)
end

update!(tree, x, ::Zero, ::Zero...) = tree, x
function update!(tree, x, x̄s...)
  cache = IdDict{Leaf,Any}()
  _accumulate!(cache, tree, x, x̄s...)
  return UpdateCallback(cache, IdDict{Leaf,Any}())(tree, x, x̄s...)
end

_accumulate!(::AbstractDict{Leaf,Any}, ::Nothing, _, _...) = nothing
_accumulate!(::AbstractDict{Leaf,Any}, ::Nothing, _, ::Zero, ::Zero...) = nothing
_accumulate!(::AbstractDict{Leaf,Any}, ℓ::Leaf, _, ::Zero, ::Zero...) = nothing
_accumulate!(::AbstractDict{Leaf,Any}, _, _, ::Zero, ::Zero...) = nothing

function _accumulate!(cache::AbstractDict{Leaf,Any}, ℓ::Leaf, _, x̄s...)
  acc_x̄s = get(cache, ℓ, missing)
  cache[ℓ] = ismissing(acc_x̄s) ? x̄s : map(_add!, acc_x̄s, x̄s)
  return
end
function _accumulate!(cache::AbstractDict{Leaf,Any}, tree, x, x̄s...)
  x̄s′ = map(x̄ -> functor(typeof(x), base(x̄))[1], x̄s)
  x′, _ = functor(typeof(x), x)
  foreach((stᵢ, xᵢ, x̄sᵢ...) -> _accumulate!(cache, stᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
end

# slightly cleaner way of closing over update! internal state
struct UpdateCallback
  acc_grads::IdDict{Leaf,Any}
  param_cache::IdDict{Leaf,Any}
end

(::UpdateCallback)(::Nothing, x, x̄s...) = nothing, x
(::UpdateCallback)(::Nothing, x, ::Zero, ::Zero...) = nothing, x
(::UpdateCallback)(ℓ::Leaf, x, ::Zero, ::Zero...) = ℓ, x
(::UpdateCallback)(tree, x, ::Zero, ::Zero...) = tree, x

(cb::UpdateCallback)(ℓ::Leaf, x, x̄s...) = get!(cb.param_cache, ℓ) do
  update!(ℓ, x, pop!(cb.acc_grads, ℓ)...)
end
function (cb::UpdateCallback)(tree, x, x̄s...)
  x̄s′ = map(x̄ -> functor(typeof(x), base(x̄))[1], x̄s)
  x′, re = functor(typeof(x), x)
  xtree = map(cb, tree, x′, x̄s′...)
  return map(first, xtree), re(map(last, xtree))
end

function update(tree, x, x̄s...)
  # because we rely on Leaf identity for tied parameters, they require special treatment
  cache = IdDict()
  tree′ = fmap(tree; cache, exclude = Base.Fix2(isa, Leaf)) do ℓ
    Leaf(ℓ.rule, fmap(copy, ℓ.state; cache, exclude = iswriteable))
  end
  x′ = fmap(copy, x; cache = empty!(cache), exclude = iswriteable)
  x̄s′ = fmap(copy, x̄s; cache = empty!(cache), exclude = iswriteable)
  return update!(tree′, x′, x̄s′...)
end

# default all rules to first order calls
apply!(o, state, x, dx, dxs...) = apply!(o, state, x, dx)

isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Integer}) = false
isnumeric(x) = false

iswriteable(::DenseArray) = true  # more elaborate versions are possible, wait until needed?
iswriteable(_) = false

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

"""
    @.. x = x + y

Sometimes in-place broadcasting macro, for use in `apply!` rules.
If `iswriteable(x)` then it is just `@. x = rhs`, but if not, it becomes `x = @. rhs`.
"""
macro var".."(ex)
  Meta.isexpr(ex, :(=)) || throw("the macro @.. only accepts assignment, like @.. x = y + z")
  dst = esc(ex.args[1])
  src = esc(Broadcast.__dot__(ex.args[2]))
  :($dst = if $iswriteable($dst)
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

function Base.show(io::IO, ℓ::Leaf)  # show method is mostly to hide its long type!
  ioc = IOContext(io, :compact => true)
  print(ioc, "Leaf(", ℓ.rule, ", ")
  show(ioc, ℓ.state)
  print(io, ")")
end

