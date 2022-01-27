
struct State{T} backing::T end
State(st::State) = st
backing(st::State) = getfield(st, :backing)
function Base.show(io::IO, st::State)
  print(io, "State(")
  show(IOContext(io, :compact => true), backing(st))
  print(io, ")")
end

patch(x, x̄) = x .- x̄

function state(o, x)
  if isnumeric(x)
    return init(o, x)
  elseif isleaf(x)
    return nothing
  else
    x, _ = functor(x)
    return State(map(x -> state(o, x), x))
  end
end

function update(o, st, x, x̄s...)
  st, x̄ = apply(o, st, x, x̄s...)
  return st, patch(x, x̄)
end

function update(o, state::State, x::T, x̄s...) where T
  if all(isnothing, x̄s)
    return state, x
  elseif isnumeric(x)
    return update(o, backing(state), x, x̄s...)
  else
    x̄s = map(x̄ -> functor(typeof(x), x̄)[1], x̄s)
    x, restructure = functor(typeof(x), x)
    xstate = map((state, x, x̄s...) -> update(o, State(state), x, x̄s...), backing(state), x, x̄s...)
    return State(map(first, xstate)), restructure(map(last, xstate))
  end
end

update(::State, args...) = throw(ArgumentError("the optimiser state must be the 2nd argument"))

# default all rules to first order calls
apply(o, state, x, dx, dxs...) = apply(o, state, x, dx)

isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Bool}) = false  # convention of ChainRules is that Bool is non-differentiable
isnumeric(x) = false
