patch(x, x̄) = x .- x̄

function state(o, x)
  if isleaf(x)
    return init(o, x)
  else
    x, _ = _functor(x)
    return map(x -> state(o, x), x)
  end
end

function _update(o, x, x̄, st)
  st, x = apply(o, st, x, x̄)
  return patch(x, x̄), st
end

function update(o, x::T, x̄, state) where T
  if x̄ === nothing
    return x, state
  elseif isleaf(x)
    return _update(o, x, x̄, state)
  else
    x̄, _  = _functor(typeof(x), x̄)
    x, restructure = _functor(typeof(x), x)
    xstate = map((x, x̄, state) -> update(o, x, x̄, state), x, x̄, state)
    return restructure(map(first, xstate)), map(x -> x[2], xstate)
  end
end

_functor(x) = Functors.functor(x)
_functor(ref::Base.RefValue) = Functors.functor(ref[])
_functor(T, x) = Functors.functor(T, x)

# may be risky since Optimisers may silently call
# this if some structures don't have appropriate overrides
init(o, x) = nothing

# default all rules to first order calls
apply(o, x, dx, dxs, state) = apply(o, x, dx, state)
