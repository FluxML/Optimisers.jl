"""
    Descent(η)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `p̄`, this runs `p -= η*p̄`.
"""
mutable struct Descent
  eta::Float64
end

init(o::Descent, x) = nothing

function apply(o::Descent, x, x̄, st)
  η = convert(eltype(x̄), o.eta)
  x̄ .* η, st
end

function (o::Descent)(m, m̄)
  update(o, m, m̄, state(o, m))[1]
end

function (o::Descent)(m, m̄, state)
  update(o, m, m̄, state)
end

mutable struct ADAM{T,K}
  eta::T
  beta::Tuple{K,K}
end

const ϵ = 1e-8
init(o::ADAM, x) = IdDict()

function apply(o::ADAM, x, Δ, st)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(st, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  st[x] = (mt, vt, βp .* β)
  return Δ, st
end
