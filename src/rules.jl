"""
    Descent(η)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`.
"""
mutable struct Descent
  eta::Float64
end

function apply(o::Descent, x, x̄, state = nothing)
  x̄ .* o.eta, state
end
