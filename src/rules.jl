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
