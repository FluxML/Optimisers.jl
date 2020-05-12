"""
    Descent(η)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `p̄`, this runs `p -= η*p̄`.
"""
mutable struct Descent
  eta::Float64
end

init(o::Descent, x) = nothing

function apply(o::Descent, x, x̄, state)
  η = convert(eltype(x̄), o.eta)
  x̄ .* η, state
end

function (o::Descent)(m, m̄)
  update(o, m, m̄, state(o, m))[1]
end

function (o::Descent)(m, m̄, st)
  update(o, m, m̄, st)
end
