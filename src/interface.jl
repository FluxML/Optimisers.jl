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

function _update(o, st, x, x̄s...)
  st′, x̄′ = apply(o, st, x, x̄s...)
  return st′, patch(x, x̄′)
end

function update(o, state, x::T, x̄s...) where T
  if all(isnothing, x̄s)
    return state, x
  elseif isnumeric(x)
    return _update(o, state, x, x̄s...)
  else
    x̄s′ = map(x̄ -> functor(typeof(x), x̄)[1], x̄s)
    x′, re = functor(typeof(x), x)
    xstate = map((stᵢ, xᵢ, x̄sᵢ...) -> update(o, stᵢ, xᵢ, x̄sᵢ...), state, x′, x̄s′...)
    return map(first, xstate), re(map(last, xstate))
  end
end

# default all rules to first order calls
apply(o, state, x, dx, dxs...) = apply(o, state, x, dx)

isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Bool}) = false  # convention of ChainRules is that Bool is non-differentiable
isnumeric(x) = false
