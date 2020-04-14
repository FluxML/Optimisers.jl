function patch(x, x̄)
  return x .- x̄
end

function state(o, x)
  if isleaf(x)
    return init(o, x)
  else
    x, _ = functor(x)
    map(x -> state(o, x), x)
  end
end

function _update(o, x, x̄, state)
  x̄, state = apply(o, x, x̄, state)
  return patch(x, x̄), state
end

function update(o, x, x̄, state)
  if x̄ === nothing
    return x, state
  elseif isleaf(x)
    _update(o, x, x̄, state)
  else
    x̄, _  = functor(typeof(x), x̄)
    x, re = functor(typeof(x), x)
    xstate = map((x, x̄, state) -> update(o, x, x̄, state), x, x̄, state)
    re(map(first, xstate)), map(x -> x[2], xstate)
  end
end
