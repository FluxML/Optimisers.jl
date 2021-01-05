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

struct Momentum{T,S}
  eta::T
  rho::S
end

Momentum(η = 0.01, ρ = 0.9) = Momentum{typeof(η), typeof(ρ)}(η, ρ)

function apply(o::Momentum, x, Δ, st)
  η, ρ = o.eta, o.rho
  v = st
  v = @. ρ * v - η * Δ
  Δ = @. -v
  Δ, v
end

function (o::Momentum)(m, m̄, state)
  update(o, m, m̄, state)
end

init(o::Momentum, x::AbstractArray) = zero(x)

struct Nesterov{T,S}
  eta::T
  rho::S
  # velocity::IdDict
end

Nesterov(η = 0.001, ρ = 0.9) = Nesterov{typeof(η), typeof(ρ)}(η, ρ)

init(o::Nesterov, x::AbstractArray) = zero(x)

function (o::Nesterov)(m, m̄, state)
  update(o, m, m̄, state)
end

function apply(o::Nesterov, x, Δ, st)
  η, ρ = o.eta, o.rho
  v = st
  d = @. ρ^2 * v - (1+ρ) * η * Δ
  v = @. ρ*v - η*Δ
  Δ = -d
  Δ, v
end

struct ADAM{T,K}
  eta::T
  beta::Tuple{K,K}
end

const ϵ = 1f-8

function (o::ADAM)(m, m̄, state)
  update(o, m, m̄, state)
end

init(o::ADAM, x::AbstractArray) = (zero(x), zero(x), o.beta)
init(o::ADAM, x) = nothing

function apply(o::ADAM, x, Δ, st)
  η, β = o.eta, o.beta
  mt, vt, βp = st
  mt = β[1] .* mt .+ (1f0 .- β[1]) .* Δ
  vt = β[2] .* vt .+ (1f0 .- β[2]) .* Δ .^ 2
  Δ =  mt ./ (1 .- βp[1]) ./ (.√(vt ./ (1f0 .- βp[2])) .+ ϵ) .* η
  return Δ, (mt, vt, βp .* β)
end


