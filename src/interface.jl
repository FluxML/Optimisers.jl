patch(x, x̄) = x .- x̄

function state(o, x)
  if isleaf(x)
    return init(o, x)
  else
    x, _ = functor(x)
    return map(x -> state(o, x), x)
  end
end

function _update(o, st, x, x̄)
  x̄, st = apply(o, st, x, x̄)
  return patch(x, x̄), st
end

function update(o, state, x::T, x̄) where T
  if x̄ === nothing
    return x, state
  elseif isleaf(x)
    return _update(o, state, x, x̄)
  else
    x̄, _  = functor(typeof(x), x̄)
    x, restructure = functor(typeof(x), x)
    xstate = map((x, x̄, state) -> update(o, state, x, x̄), x, x̄, state)
    return restructure(map(first, xstate)), map(x -> x[2], xstate)
  end
end
