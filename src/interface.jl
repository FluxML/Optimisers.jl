patch(x, x̄) = x .- x̄

function state(o, x)
  if isnumeric(x)
    return init(o, x)
  elseif isleaf(x)
    return nothing
  else
    x′, _ = functor(x)
    return map(xᵢ -> state(o, xᵢ), x′)
  end
end

function _update(o, st, x, x̄s...; copy::Bool = true)
  st′, x̄′ = apply(o, st, x, x̄s...; copy)
  x′ = copy ? patch(x, x̄′) : patch!!(x, x̄′)
  return st′, x′
end

function update(o, state, x::T, x̄s...; copy::Bool = true) where T
  if all(isnothing, x̄s)
    return state, x
  elseif isnumeric(x)
    return _update(o, state, x, x̄s...; copy)
  else
    x̄s′ = map(x̄ -> functor(typeof(x), x̄)[1], x̄s)
    x′, re = functor(typeof(x), x)
    xstate = map((stᵢ, xᵢ, x̄sᵢ...) -> update(o, stᵢ, xᵢ, x̄sᵢ...; copy), state, x′, x̄s′...)
    return map(first, xstate), re(map(last, xstate))
  end
end

update!(o, state, x, x̄s...) = update(o, state, x::T, x̄s...; copy = false)

# default all rules to first order calls
apply(o, state, x, dx, dxs...; kw...) = apply(o, state, x, dx; kw...)

isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Bool}) = false  # convention of ChainRules is that Bool is non-differentiable
isnumeric(x) = false

iswriteable(::DenseArray{<:AbstractFloat}) = true
iswriteable(_) = false

"""
    x′ = @.. x = x + y
    bc = @.. x + y / z

Magic broadcasting macro:
* Applied to assignment `x = ...` it overwrites `x` when `iswriteable(x)`,
  *unless* you set `copy === true` in local scope. Always applies `@.` on the right.
* Applied to other expressions, it broadcasts like `@.` but does not materialise,
  returning a `Broadcasted` object for later use.
"""
macro var".."(ex)
  if Meta.isexpr(ex, :(=))
    dst = esc(ex.args[1])
    src = esc(Broadcast.__dot__(ex.args[2]))
    cpy = esc(:copy)
    :(if $cpy !== true && $iswriteable($dst)
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

patch!!(x, x̄) = @.. x = x - x̄
