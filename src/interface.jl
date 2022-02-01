
using ChainRulesCore: canonicalize, backing, Tangent, AbstractZero
base(dx::Tangent) = backing(canonicalize(dx))
base(dx) = dx
const Zero = Union{Nothing, AbstractZero}  # Union{Zygote, Diffractor}

struct Leaf{R,S}
  rule::R
  state::S
  frozen::Bool
end

function setup(rule, x; seen = Base.IdSet())
  if isnumeric(x)
    x in seen && throw(ArgumentError("Optimisers.jl does not at present handle tied weights, sorry."))
    isbits(x) || push!(seen, x)
    return Leaf(rule, init(rule, x), false)
  elseif isleaf(x)
    return nothing
  else
    return map(xᵢ -> setup(rule, xᵢ; seen), _trainable(x))
  end
end

subtract!(x, x̄) = iswriteable(x) ? (x .= x .- x̄) : (x .- x̄)

update!(::Nothing, x, ::Zero...) = nothing, x
update!(::Nothing, x, x̄s...) = nothing, x

update!(ℓ::Leaf, x, ::Zero...) = ℓ, x
function update!(ℓ::Leaf, x, x̄s...)
  ℓ.frozen && return ℓ, x
  s′, x̄′ = apply!(ℓ.rule, ℓ.state, x, base.(x̄s)...)
  Leaf(ℓ.rule, s′, ℓ.frozen), subtract!(x, x̄′)
end

update!(tree, x, ::Zero...) = tree, x
function update!(tree, x, x̄s...)
  x̄s′ = map(x̄ -> functor(typeof(x), base(x̄))[1], x̄s)
  x′, re = functor(typeof(x), x)
  xtree = map((stᵢ, xᵢ, x̄sᵢ...) -> update!(stᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
  map(first, xtree), re(map(last, xtree))
end

function update(tree, x, x̄s...)
  t′ = fmap(copy, tree; exclude = iswriteable)
  x′ = fmap(copy, x; exclude = iswriteable)
  update!(t′, x′, x̄s...)
end

# default all rules to first order calls
apply!(o, state, x, dx, dxs...) = apply!(o, state, x, dx)

isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Bool}) = false  # convention of ChainRules is that Bool is non-differentiable
isnumeric(x) = false

iswriteable(::DenseArray{<:AbstractFloat}) = true  # more elaborate versions are possible, wait until needed?
iswriteable(_) = false

ids(x::NamedTuple{names}) where names = NamedTuple{names}(names)  # a map-friendly version of pairs
ids(x::Tuple) = propertynames(x)

"""
    freeze(state, branches = ()) -> state
    thaw(state, branches = ())

Disable training of a specified part of the model, by modifying the state returned by `setup`.
Specifying `:encoder` will shield all nodes within `model.encoder` from `update`,
specifying a tuple `(1, :first, 3)` will disable `model[1].first[3]`,
and a vector `[:enc, (:dec, 2), (:dec, 3)]` will disable all the given parts.
"""
freeze(ℓ::Leaf, addr::Tuple{}, b=true) = Leaf(ℓ.rule, ℓ.state, b)
freeze(::Nothing, addr::Tuple{}, b=true) = nothing

function freeze(tree, addr::Tuple, b=true)
  isempty(addr) && return map(t -> freeze(t, addr, b), tree)
  addr[1] in ids(tree) || throw("invalid index")  # sanity check since address is user-provided?
  map((t,i) -> i==addr[1] ? freeze(t, tail(addr), b) : t, tree, ids(tree))
end

# freeze(t::Tied, addr::Tuple, b=true) = Tied(t.ties, freeze(t.tree, addr, b))  # for PR42
# freeze(t::Tied, addr::Tuple{}=(), b=true) = Tied(t.ties, freeze(t.tree, addr, b))  # ambiguity?

freeze(tree, addr::Union{Symbol, Integer}, b=true) = freeze(tree, (addr,), b)
function freeze(tree, addr::Vector, b=true)  # allows freeze(state, [:x, (2, 3)])
  for a in addr
    tree = freeze(tree, a, b)
  end
  tree
end

thaw(tree, addr=()) = freeze(tree, addr, false)

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
  print(io, ", ", ℓ.frozen, ")")
end
