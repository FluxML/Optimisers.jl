
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

patch!(x, x̄) = iswriteable(x) ? (x .= x .- x̄) : (x .- x̄)

function _update!(o, st, x, x̄s...)
  st′, x̄′ = apply!(o, st, x, x̄s...)
  return st′, patch!(x, x̄′)
end

function update!(o, state, x, x̄s...)
  if all(isnothing, x̄s)
    return state, x
  elseif isnumeric(x)
    return _update!(o, state, x, x̄s...)
  else
    x̄s′ = map(x̄ -> functor(typeof(x), x̄)[1], x̄s)
    x′, re = functor(typeof(x), x)
    xstate = map((stᵢ, xᵢ, x̄sᵢ...) -> update!(o, stᵢ, xᵢ, x̄sᵢ...), state, x′, x̄s′...)
    return map(first, xstate), re(map(last, xstate))
  end
end

function update(o, state, x, x̄s...)
  state′ = fmap(copy, state; exclude = iswriteable)
  x′ = fmap(copy, x; exclude = iswriteable)
  update!(o, state′, x′, x̄s...)
end

# default all rules to first order calls
apply!(o, state, x, dx, dxs...) = apply!(o, state, x, dx)

isnumeric(x::AbstractFloat) = true
isnumeric(x::Complex{<:AbstractFloat}) = true
isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Bool}) = false  # convention of ChainRules is that Bool is non-differentiable
isnumeric(x) = false

iswriteable(::DenseArray{<:AbstractFloat}) = true  # more elaborate versions are possible, wait until needed?
iswriteable(_) = false

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
