
using ChainRulesCore: canonicalize, backing, Tangent, AbstractZero
base(dx::Tangent) = backing(canonicalize(dx))
base(dx) = dx
const Zero = Union{Nothing, AbstractZero}  # Union{Zygote, Diffractor}

abstract type AbstractRule end

struct Leaf{R,S}
  rule::R
  state::S
end

function setup(rule, x; seen = Base.IdSet())
  rule isa AbstractRule || Base.depwarn("In future, all optimisation rules should be <: AbstractRule", :setup)
  if isnumeric(x)
    x in seen && throw(ArgumentError("Optimisers.jl does not at present handle tied weights, sorry."))
    isbits(x) || push!(seen, x)
    return Leaf(rule, init(rule, x))
  elseif isleaf(x)
    return nothing
  else
    return map(xᵢ -> setup(rule, xᵢ; seen), _trainable(x))
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

function Base.show(io::IO, ℓ::Leaf)
  ioc = IOContext(io, :compact => true)
  str = sprint(show, ℓ.rule; context = ioc)
  printstyled(io, "Leaf(", str, ", "; color = :green)
  str = sprint(show, ℓ.state; context = ioc)
  if length(str) < 80
    print(io, str)
    printstyled(io, ")"; color = :green)
  else
    print(io, first(str, 66))
    printstyled(io, " ...)"; color = :green)
  end
end

