
using ChainRulesCore: canonicalize, backing, Tangent, AbstractZero
base(dx::Tangent) = backing(canonicalize(dx))
base(dx) = dx
base(::Nothing) = false
const Zero = Union{Nothing, AbstractZero}  # Union{Zygote, Diffractor}

abstract type AbstractRule end

struct Leaf{R,S}
  rule::R
  state::S
end

struct Tied; ties; tree; end

function setup(rule, x; ties = Pair[], cache = IdDict())
  rule isa AbstractRule || Base.depwarn("In future, all optimisation rules should be <: AbstractRule", :setup)
  tree = _setup(rule, x, (); ties, cache)
  isempty(ties) ? tree : Tied(ties, tree)
end

function _setup(rule, x, addr; ties, cache)
  usecache = !isbits(x) && cache !== false
  if isnumeric(x)
    if usecache && haskey(cache, x)
      push!(ties, addr => cache[x])
      return nothing
    else
      if usecache
        cache[x] = addr
      end
      return Leaf(rule, init(rule, x))
    end
  elseif isleaf(x)
    return nothing
  else
    x′ = _trainable(x)
    map((xᵢ, i) -> _setup(rule, xᵢ, (addr..., i); ties, cache), x′, ids(x′))
  end
end

subtract!(x, x̄) = iswriteable(x) ? (x .= x .- x̄) : eltype(x).(x .- x̄)

update!(::Nothing, x, ::Zero, ::Zero...) = nothing, x
update!(::Nothing, x, x̄s...) = nothing, x

update!(ℓ::Leaf, x, ::Zero, ::Zero...) = ℓ, x
function update!(ℓ::Leaf, x, x̄s...)
  s′, x̄′ = apply!(ℓ.rule, ℓ.state, x, base.(x̄s)...)
  Leaf(ℓ.rule, s′), subtract!(x, x̄′)
end

update!(tree, x, ::Zero, ::Zero...) = tree, x
function update!(tree, x, x̄s...)
  x̄s′ = map(x̄ -> functor(typeof(x), base(x̄))[1], x̄s)
  x′, re = functor(typeof(x), x)
  xtree = map((stᵢ, xᵢ, x̄sᵢ...) -> update!(stᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
  map(first, xtree), re(map(last, xtree))
end

update!(t::Tied, x, ::Zero) = t, x
function update!(t::Tied, x, x̄)
  # accumulate tied gradients:
  for (dup, orig) in t.ties
    x̄ = place(x, x̄, orig) do x̄ᵢ
      x̄ᵢᵢ = pick(x̄, dup)
      Broadcast.broadcasted(+, base(x̄ᵢ), base(x̄ᵢᵢ))
    end
  end
  # run the optimisers:
  t′, x′ = update!(t.tree, x, x̄)
  # restore tied weights:
  for (dup, orig) in t.ties
    x′ = place(x′, dup) do
      pick(x′, orig)
    end
  end
  Tied(t.ties, t′), x′
end

function update(tree, x, x̄s...)
  t′ = fmap(copy, tree; exclude = iswriteable)
  x′ = fmap(copy, x; exclude = iswriteable)
  update!(t′, x′, x̄s...)
end

# default all rules to first order calls
apply!(o, state, x, dx, dxs...) = apply!(o, state, x, dx)

isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Integer}) = false
isnumeric(x) = false

iswriteable(::DenseArray) = true  # more elaborate versions are possible, wait until needed?
iswriteable(_) = false

ids(x::NamedTuple{names}) where names = NamedTuple{names}(names)  # a map-friendly version of pairs
ids(x::Tuple) = propertynames(x)

function pick(x, addr::Tuple)  # used for both x and x̄
  (isempty(addr) || x isa Zero) && return x
  x′, _ = functor(x)
  pick(get(x′, addr[1], nothing), tail(addr))
end

place(f, x, addr::Tuple{}) = f()
function place(f, x, addr::Tuple)
  x′, re = functor(x)
  re(map((xᵢ, i) -> i == addr[1] ? place(f, xᵢ, tail(addr)) : xᵢ, x′, ids(x′)))
end

place(f, x, x̄, addr::Tuple{}) = f(x̄)  # placed into x̄
# This function needs x to know how to restore missing branches of x̄
function place(f, x, x̄, addr::Tuple)
  x̄′, _ = functor(typeof(x), x̄ isa Zero ? map(_->nothing, x) : base(x̄))
  x′, _ = functor(typeof(x), x)
  map((xᵢ, x̄ᵢ, i) -> i == addr[1] ? place(f, xᵢ, x̄ᵢ, tail(addr)) : x̄ᵢ, x′, x̄′, ids(x′))
end

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

