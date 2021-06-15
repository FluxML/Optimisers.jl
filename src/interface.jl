patch(x, x̄) = x .- x̄

function state(o, x)
  if isleaf(x)
    return init(o, x)
  else
    x, _ = functor(x)
    return map(x -> state(o, x), x)
  end
end

function _update(o, st, x, x̄s...)
  st, x̄ = apply(o, st, x, x̄s...)
  return st, patch(x, x̄)
end

function update(o, state, x::T, x̄s...) where T
  if all(isnothing, x̄s)
    return state, x
  elseif isleaf(x)
    return _update(o, state, x, x̄s...)
  else
    x̄s = map(x̄ -> functor(typeof(x), x̄)[1], x̄s)
    x, restructure = functor(typeof(x), x)
    xstate = map((state, x, x̄s...) -> update(o, state, x, x̄s...), state, x, x̄s...)
    return map(first, xstate), restructure(map(last, xstate))
  end
end

# default all rules to first order calls
apply(o, state, x, dx, dxs...) = apply(o, state, x, dx)
