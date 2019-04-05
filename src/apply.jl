const Param{T<:Number} = Union{AbstractArray{T},T}

_apply(opt, x, x̄, state) = apply(opt, x, x̄, state)
_apply(opt, x, x̄, ::Nothing) = apply(opt, x, x̄)

# Immutable updates

function update(opt, x::Param, x̄::Param, state = nothing)
  Δ, state = _apply(opt, x, x̄, state)
  return x .- Δ, state
end

# Mutable updates

# Figure out if we can do in-place
inplace(x, y) = false
inplace(x, y::Nothing) = true
inplace(x::AbstractArray, x̄::AbstractArray) = true
inplace(x, x̄::NamedTuple) = all(inplace(getfield(x, f), getfield(x̄, f)) for f in fieldnames(typeof(x̄)))

function update!(opt, x::AbstractArray{<:Number}, x̄::AbstractArray, state = nothing)
  Δ, state = _apply(opt, x, x̄, state)
  x .-= Δ
  return state
end

function update!(opt, x, x̄::NamedTuple)
  for f in fieldnames(typeof(x̄))
    f̄ = getfield(x̄, f)
    f̄ === nothing || update!(opt, getfield(x, f), f̄)
  end
end

# Package Integration

using Requires

@init @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" begin
  function update(opt, x::Colors.RGB{T}, x̄::NamedTuple) where T
    Colors.RGB{T}(clamp(update(opt, x.r, x̄.r)[1], 0, 1),
                  clamp(update(opt, x.g, x̄.g)[1], 0, 1),
                  clamp(update(opt, x.b, x̄.b)[1], 0, 1)), nothing
  end
end
