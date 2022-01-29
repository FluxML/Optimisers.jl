
struct Leaf{R,S}
  rule::R
  state::S
end

function setup(rule, model)
  tiecheck(model)
  fmap(model; walk=_trainable_struct_walk, prune=nothing) do x
    isnumeric(x) ? Leaf(rule, init(rule, x)) : nothing
  end
end

subtract!(x, x̄) = iswriteable(x) ? (x .= x .- x̄) : (x .- x̄)

update!(::Nothing, x, x̄s...) = nothing, x

function update!(ℓ::Leaf, x, x̄s...)
  if all(isnothing, x̄s)
    return ℓ, x
  else
    s′, x̄′ = apply!(ℓ.rule, ℓ.state, x, x̄s...)
    return Leaf(ℓ.rule, s′), subtract!(x, x̄′)
  end
end

function update!(tree, x, x̄s...)
  if all(isnothing, x̄s)
    return tree, x
  else
    x̄s′ = map(x̄ -> functor(typeof(x), x̄)[1], x̄s)
    x′, re = functor(typeof(x), x)
    xtree = map((stᵢ, xᵢ, x̄sᵢ...) -> update!(stᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
    return map(first, xtree), re(map(last, xtree))
  end
end

function update(tree, x, x̄s...)
  t′ = fmap(copy, tree; exclude = iswriteable)
  x′ = fmap(copy, x; exclude = iswriteable)
  update!(t′, x′, x̄s...)
end

# default all rules to first order calls
apply!(o, state, x, dx, dxs...) = apply!(o, state, x, dx)

iswriteable(::DenseArray{<:AbstractFloat}) = true  # more elaborate versions are possible, wait until needed?
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
_trainable(ch::Tuple, tr::Tuple) = tr
function _trainable(ch::NamedTuple, tr::Tuple)  # for old Flux-style no-names tuple
  @warn "trainable(x) should now return a NamedTuple with the field names, not a Tuple"
  map(c -> c in tr ? c : nothing, ch)
end

function _trainable_walk(f, x)
  ch, re = functor(x)
  y = map(ch, _trainable(x)) do c, t
    isnothing(c) ? f(nothing) : isnothing(t) ? c : f(t)
  end
  re(y)
end

_trainable_struct_walk(f, x) = map(f, _trainable(x))

function tiecheck(x; set = Base.IdSet())
  if isnumeric(x) && !isbits(x)
    x in set && throw(ArgumentError(
      "Optimisers.jl does at present handle tied weights, sorry. Got $(summary(x)) twice."))
    push!(set, x)
  else
    map(y -> tiecheck(y; set), trainable(x))  # no need for _trainable
  end
end

"""
    destructure(model) -> vector, function

Concatenates all [`trainable`](@ref) parameters of a model into one vector,
and returns this along with a function which reconstructs the model.

# Example
```jldoctest
julia> m = (x=[1,2], y=[[3,4], nothing, (5,6)]);

julia> v, re = destructure(m)
([1, 2, 3, 4], Fix1(restructure, NamedTuple(...)))

julia> re([10, 20, 300, 400])
(x = [10, 20], y = Any[[300, 400], nothing, (5, 6)])
```
"""
function destructure(x)
  v = Functors.fvec(x; walk=_trainable_struct_walk)
  v, Base.Fix1(_restructure, (x, length(v)))  # bit of a hack to use Fix1 & save the length!
end

restructure(x, v) = _restructure((x, Functors.flatlength(x; walk=_trainable_struct_walk)), v)
_restructure((x, len), v) = Functors.fcopy(x, v; walk=_trainable_walk, len=len)

function Base.show(io::IO, re::Base.Fix1{typeof(_restructure)})
    print(io, "Fix1(restructure, ", typeof(re.x).name.name, "(...))")
end

"""
    @.. x = x + y
    @.. x + y / z

Magic broadcasting macro, for use in `apply!` rules:
* Applied to assignment `x = ...` it is like `@.` unless `!iswriteable(x)`,
  in which case it ignores `x`, and applies `@.` on the right.
* Applied to other expressions, it broadcasts like `@.` but does not materialise,
  returning a `Broadcasted` object for later use.
"""
macro var".."(ex)
  if Meta.isexpr(ex, :(=))
    dst = esc(ex.args[1])
    src = esc(Broadcast.__dot__(ex.args[2]))
    :(if $iswriteable($dst)
        $dst .= $src
      else
        $src
      end)
  else
    bc = esc(Broadcast.__dot__(ex))
    :($lazy.($bc))
  end
end

function lazy end
Broadcast.broadcasted(::typeof(lazy), x) = Lazy(x)
struct Lazy{T}; bc::T; end
Broadcast.materialize(x::Lazy) = Broadcast.instantiate(x.bc)

function Base.show(io::IO, ℓ::Leaf)  # show method is mostly to hide its long type!
  ioc = IOContext(io, :compact => true)
  print(ioc, "Leaf(", ℓ.rule, ", ")
  show(ioc, ℓ.state)
  print(io, ")")
end

