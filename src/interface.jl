using ChainRulesCore: canonicalize, backing, Tangent, AbstractZero, ZeroTangent

base(dx::Tangent) = backing(canonicalize(dx))
base(dx) = dx

const Zero = Union{Nothing, AbstractZero}  # Union{Zygote, Diffractor}

abstract type AbstractRule end

function Base.show(io::IO, rule::AbstractRule)  # makes Adam(0.01f0) prettier
  invoke(show, Tuple{IO,Any}, IOContext(io, :compact => true), rule)
end

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
    mapvalue(xᵢ -> _setup(rule, xᵢ; cache), _trainable(x))
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

function update!(tree, model, grad, higher...)
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
  x′′ = re(mapvalue((tᵢ, xᵢ) -> _update!(tᵢ, xᵢ; grads, params), tree, x′))
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
subtract!(x, x̄::Zero) = x

# If we get Zero from AD on a leaf we skip the optimizer step. See
# https://github.com/FluxML/Optimisers.jl/issues/140
_grads!(dict::IdDict, ℓ::Leaf, x, ::Zero...) = nothing

function _grads!(dict::IdDict, ℓ::Leaf, x, x̄s...)
  x̄s₀ = get(dict, ℓ, map(_ -> ZeroTangent(), x̄s))
  dict[ℓ] = map(+, x̄s, x̄s₀)  # adding Zero should be free. Lazy accumulation broadcasted(+, x̄, x̄₀) also possible.
  nothing
end

# If we get Zero from AD in correspondence of a non-leaf node
# we end the recursion. The optimizer step won't be taken.
# https://github.com/FluxML/Optimisers.jl/issues/140
_grads!(dict::IdDict, t, x, ::Zero...) = nothing

function _grads!(dict::IdDict, tree, x, x̄s...)
  # The only reason _grads! takes model is that functor(typeof(x), base(x̄)) may differ from 
  # functor(typeof(tree), base(x̄)), for things like Transpose
  x̄s′ = map(x̄ -> functor(typeof(x), base(x̄))[1], x̄s)
  x′, _ = functor(typeof(x), x)
  foreachvalue((tᵢ, xᵢ, x̄sᵢ...) -> _grads!(dict, tᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
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

This may be overloaded to make optimisers ignore some fields of
every `Layer`, which would otherwise contain trainable parameters.

!!! warning
    This is very rarely required. Fields of `struct Layer` which contain
    functions, or integers like sizes, are always ignored anyway.
    Overloading `trainable` is only necessary when some arrays of numbers
    are to be optimised, and some arrays of numbers are not.

The default is `Functors.children(x)`, usually a NamedTuple of all fields,
and `trainable(x)` must contain a subset of these.
"""
trainable(x) = functor(x)[1]

# like trainable(x), but also tries to output non-trainable children giving value nothing
_trainable(x) = _trainable(functor(x)[1], trainable(x))
_trainable(ch::NamedTuple, tr::NamedTuple) = merge(map(_ -> nothing, ch), tr)
_trainable(ch::Tuple{Vararg{Any,N}}, tr::Tuple{Vararg{Any,N}}) where N = tr
_trainable(ch::AbstractArray, tr::AbstractArray) = tr
_trainable(ch::Dict, tr::Dict) = merge(mapvalue(_ -> nothing, ch), tr)

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

nonneg(η::Real) = η < 0 ? throw(DomainError(η, "the learning rate cannot be negative")) : η

"""
    @def struct Rule; eta = 0.1; beta = (0.7, 0.8); end

Helper macro for defining rules with default values.
The types of the literal values are used in the `struct`,
like this:
```julia
struct Rule{T1, T2}
  eta::T1
  beta::T2

  Rule(eta, beta = (0.7, 0.8)) = eta < 0 ? error() : new(eta, beta)
  Rule(; eta = 0.1, beta = (0.7, 0.8)) = Rule(eta, beta)
end

```
Any field called `eta` is assumed to be a learning rate, and cannot be negative.
"""
macro def(expr)
  Meta.isexpr(expr, :struct) || throw("@def must act on a struct definition")
  lines = expr.args[3].args
  names, default_vals = [], []
  default_types, type_params = [], []
  for i in eachindex(lines)
    lines[i] isa Symbol && throw("@def requires a default for every field")
    Meta.isexpr(lines[i], :(=)) || continue
    name, val = lines[i].args
    push!(names, name)
    push!(default_vals, val)
    push!(default_types, _def_typeof(val))
    # @show name, val, typeof(val)
    type = Symbol("T$name")
    push!(type_params, type)
    lines[i] = :($name::$type)
  end
  rule = Meta.isexpr(expr.args[2], :<:) ? expr.args[2].args[1] : expr.args[2]
  expr.args[2] = Expr(:(<:), Expr(:curly, rule, type_params...), :AbstractRule)
  params = [Expr(:kw, nv...) for nv in zip(names, default_vals)]
  check_sign_eta = :eta in names ? :($(_assert_positive_eta)(eta)) : nothing
  vars_with_checks = [
    :($(_def_check_type)($name, $type)) for (name, type) in zip(names, default_types)
  ]
  # Positional-argument method, has defaults for all but the first arg:
  positional = :(function $rule($(names[1]), $(params[2:end]...))
    vars = ($(vars_with_checks...),)
    $check_sign_eta
    return new{typeof.(vars)...}(vars...)
  end)
  # Keyword-argument method. (Made an inner constructor only to allow
  # resulting structs to be @doc... cannot if macro returns a block.)
  kwmethod = :($rule(; $(params...)) = $rule($(names...)))
  push!(lines, positional, kwmethod)
  # return esc(expr)
# end

  return quote 
          Base.@__doc__ $expr

          function Base.show(io::IO, r::$rule)
            pairs = ["$n=$(getfield(r, n))" for n in [($names...)]]
            print(io, $rule,"(", join(pairs, ", "), ")")
          end
        end |> esc
end

@inline _assert_positive_eta(eta) = _assert_positive_eta(eta, eta < 0)
@inline function _assert_positive_eta(eta, cond::Bool)
  cond && throw(DomainError(eta, "the learning rate cannot be negative"))
end

_def_typeof(val) = typeof(val)
_def_typeof(val::Expr) = typeof(eval(val))

_def_check_type(x, T) = x
_def_check_type(x, T::Type{<:Number}) = throw(ArgumentError("$x is not a number"))
_def_check_type(x::Number, T::Type{<:Number}) = x
_def_check_type(x::Number, T::Type{<:AbstractFloat}) = float(x)
_def_check_type(x::Complex, T::Type{<:AbstractFloat}) = throw(ArgumentError("cannot convert complex to real"))
_def_check_type(x, T::Type{<:AbstractFloat}) = throw(ArgumentError("$x is not a number"))
_def_check_type(x::Tuple, T::Type{<:Tuple}) = _def_check_type.(x, (T.parameters...,))
_def_check_type(x, T::Type{<:Tuple}) = throw(ArgumentError("$x is not a tuple"))
