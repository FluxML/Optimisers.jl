patch(x, x̄) = x .- x̄

function state(o, x)
  if isleaf(x)
    return init(o, x)
  else
    x, _ = functor(x)
    return map(x -> state(o, x), x)
  end
end

function _update(o, x, x̄, st)
  x̄, st = ismutable(x) ? apply!(o, x, x̄, st) : apply(o, x, x̄, st)
  return patch(x, x̄), st
end

function update(o, x::T, x̄, state) where T
  if x̄ === nothing
    return x, state
  elseif isleaf(x)
    return _update(o, x, x̄, state)
  else
    x̄, _  = functor(typeof(x), x̄)
    x, restructure = functor(typeof(x), x)
    xstate = map((x, x̄, state) -> update(o, x, x̄, state), x, x̄, state)
    return restructure(map(first, xstate)), map(x -> x[2], xstate)
  end
end
