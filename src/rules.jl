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
end

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

struct RMSProp{T,S}
  eta::T
  rho::S
end

init(o::RMSProp, x::AbstractArray) = zero(x)

function apply(o::RMSProp, x, Δ, st)
  η, ρ = o.eta, o.rho
  acc = st
  acc = ρ .* acc .+ (1 .- ρ) .* Δ.^2
  Δ = Δ .* (η ./ (.√acc .+ ϵ))
  Δ, acc
end

function (o::RMSProp)(m, m̄, state)
  update(o, m, m̄, state)
end

struct ADAM{T,K}
  eta::T
  beta::K
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

struct RADAM{T,S}
  eta::T
  beta::S
end

init(o::RADAM, x::AbstractArray) = (zero(x), zero(x), [β[1], β[2]], 1)

function apply(o::RADAM, x, Δ, st)
  η, β = o.eta, o.beta
  ρ∞ = 2/(1-β[2])-1

  mt, vt, βp, t = st

  mt = β[1] .* mt .+ (1 .- β[1]) .* Δ
  vt = β[2] .* vt .+ (1 .- β[2]) .* Δ .^ 2
  ρ = ρ∞ - 2t[] * βp[2] / (1 - βp[2])
  if ρ > 4
    r = sqrt((ρ-4)*(ρ-2)*ρ∞/((ρ∞-4)*(ρ∞-2)*ρ))
    Δ =  mt ./ (1 .- βp[1]) ./ (.√(vt ./ (1 .- βp[2])) .+ ϵ) .* η .* r
  else
    Δ =  mt ./ (1 .- βp[1]) .* η
  end
  βp .= βp .* β
  t_ = t + 1

  return Δ, (mt, vt, βp, t_)
end

struct AdaMax{T,S}
  eta::T
  beta::S
end

init(o::AdaMax, x::AbstractArray) = (zero(x), zero(x), [β[1], β[2]])

function apply(o::AdaMax, x, Δ, st)
  η, β = o.eta, o.beta

  mt, ut, βp = st 

  mt = β[1] .* mt .+ (1 .- β[1]) .* Δ
  ut = max.(β[2] .* ut, abs.(Δ))
  Δ = (η ./ (1 .- βp[1])) .* mt ./ (ut .+ ϵ)
  βp = βp .* β

  return Δ, (mt, ut, βp)
end

struct OADAM{T,S}
  eta::T
  beta::S
end

init(o::OADAM, x::AbstractArray) = (zero(x), zero(x), zero(x), [β[1], β[2]])

function apply(o::OADAM, x, Δ, st)
  η, β = o.eta, o.beta

  mt, vt, Δ_, βp = st

  mt = β[1] .* mt .+ (1 .- β[1]) .* Δ
  vt = β[2] .* vt .+ (1 .- β[2]) .* Δ .^ 2
  Δ = -Δ_
  Δ_ = η .* mt ./ (1 .- βp[1]) ./ (.√(vt ./ (1 .- βp[2])) .+ ϵ)
  Δ = Δ .+ 2Δ_
  βp = βp .* β

  return Δ, (mt, vt, Δ_, βp)
end

struct ADAGrad{T}
  eta::T
end

init(o::ADAGrad, x::AbstractArray) = (fill!(similar(x), ϵ),)

function apply(o::ADAGrad, x, Δ, st)
  η = o.eta
  acc, = st
  acc = acc .+ Δ .^ 2
  Δ = Δ .* η ./ (.√acc .+ ϵ)
  Δ, (acc,)
end

struct ADADelta
  rho::Float64
end

init(o::ADADelta, x::AbstractArray) = (zero(x), zero(x))

function apply(o::ADADelta, x, Δ, st)
  ρ = o.rho
  acc, Δacc = st
  acc = ρ .* acc .+ (1 .- ρ) .* Δ .^ 2
  # DON'T remove epsilon from numerator
  # or even out of the square roots
  Δ = Δ .* .√(Δacc .+ ϵ) ./ .√(acc .+ ϵ)
  Δacc = ρ .* Δacc .+ (1 .- ρ) .* Δ .^ 2
  return Δ, (acc, Δacc)
end

struct AMSGrad{T,S}
  eta::T
  beta::S
end

init(o::AMSGrad, x::AbstractArray) = (fill!(similar(x), ϵ), fill!(similar(x), ϵ), fill!(similar(x), ϵ))

function apply(o::AMSGrad, x, Δ, st)
  η, β = o.eta, o.beta

  mt, vt, v̂t = st 

  mt = β[1] .* mt .+ (1 .- β[1]) .* Δ
  vt = β[2] .* vt .+ (1 .- β[2]) .* Δ .^ 2
  v̂t = max.(v̂t, vt)
  Δ = η .* mt ./ (.√v̂t .+ ϵ)
  Δ, (mt, vt, v̂t)
end

struct NADAM
  eta::Float64
  beta::Tuple{Float64, Float64}
end

init(o::NADAM, x::AbstractArray) = (zero(x), zero(x), [o.beta[1], o.beta[2]])

function apply(o::NADAM, x, Δ, st)
  η, β = o.eta, o.beta

  mt, vt, βp = st 
  β1p, β2p = βp

  mt = β[1] .* mt .+ (1 .- β[1]) .* Δ
  vt = β[2] .* vt .+ (1 .- β[2]) .* Δ .^ 2
  Δ = (β[1] .* mt ./ (1 .- β[1] .* β1p) .+ (1 .- β[1]) .* Δ ./ (1 .- β1p)) ./ (.√(vt .* β[2] ./ (1 .- β2p)) .+ ϵ) .* η
  βp = βp .* β

  return Δ, (mt, vt, βp)
end

struct AdaBelief{T,S}
  eta::T
  beta::S
end

init(o::AdaBelief, x::AbstractArray) = (zero(x), zero(x))

function apply(o::AdaBelief, x, Δ, st)
  η, β = o.eta, o.beta
  mt, st = st 
  mt = β[1] .* mt .+ (1 .- β[1]) .* Δ
  st = β[2] .* st .+ (1 .- β[2]) * (Δ .- mt) .^ 2
  Δ =  η .* mt ./ (.√(st) .+ ϵ)
  Δ, (mt, st)
end

struct InvDecay{T}
  gamma::T
end

init(o::InvDecay, x::AbstractArray) = 1

function apply(o::InvDecay, x, Δ, st)
  γ = o.gamma
  n, = st
  Δ = Δ .* 1 ./ (1 .+ γ .* n)
  return Δ, (n + 1,)
end

mutable struct ExpDecay
  eta::Float64
  decay::Float64
  step::Int64
  clip::Float64
end

init(o::ExpDecay, x::AbstractArray) = (0,)

function apply(o::ExpDecay, x, Δ, st)
  η, s, decay = o.eta, o.step, o.decay
  n, = st .+ 1
  if n%s == 0 && count(x -> x%s == 0, st) == 1
    η = max(η * decay, o.clip)
    o.eta = η
  end
  Δ = Δ * η
  Δ, (n,)
end

struct WeightDecay{T}
  wd::T
end

init(o::WeightDecay, x) = nothing

function apply!(o::WeightDecay, x, Δ, st)
  wd = o.wd
  Δ = Δ + wd * x
  Δ, st
end
