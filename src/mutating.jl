# This file contains duplicated rules as in rules.jl
# where all the operations are done in-place for a softer deprecation

function apply!(o::Descent, x, dx, state)
  η = convert(eltype(dx), o.eta)
  dx .*= η

  return dx, state
end

function apply!(o::Momentum, x, dx, state)
  η, ρ, v = o.eta, o.rho, state
  @. v = ρ * v - η * dx

  return -v, v
end

function apply(o::Nesterov, x, dx, state)
  η, ρ, v = o.eta, o.rho, state
  @. d = ρ^2 * v - (1+ρ) * η * dx
  @. v = ρ * v - η * dx

  return -d, v
end

function apply(o::RMSProp, x, dx, state)
  η, ρ, ϵ, acc = o.eta, o.rho, o.epsilon, state
  @. acc = ρ * acc + (1 - ρ) * dx^2
  @. dx = dx * (η / (sqrt(acc) + ϵ))

  return dx, acc
end

function apply(o::ADAM{T}, x, dx, state) where T
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, vt, βt = state

  @. mt = β[1] * mt + (one(T) - β[1]) * dx
  @. vt = β[2] * vt + (one(T) - β[2]) * dx ^ 2
  @. dx = mt / (one(T) - βt[1]) / (sqrt(vt / (one(T) - βt[2])) + ϵ) * η

  return dx, (mt, vt, βt .* β)
end

function apply(o::RADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  ρ∞ = 2/(1-β[2])-1

  mt, vt, βt, t = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  ρ = ρ∞ - 2*t * βt[2] / (1 - βt[2])
  if ρ > 4
    r = sqrt((ρ - 4) * (ρ - 2) * ρ∞/((ρ∞ - 4) * (ρ∞ - 2) * ρ))
    @. dx = mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η * r
  else
    @. dx = mt / (1 - βt[1]) * η
  end

  return dx, (mt, vt, βt .* β, t + 1)
end

function apply!(o::AdaMax, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  (mt, ut), βt = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. ut = max(β[2] * ut, abs(dx))
  @. dx = (η/(1 - βt[1])) * mt/(ut + ϵ)

  return dx, ((mt, ut), βt .* β)
end

function apply!(o::AdaMax, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  (mt, ut), βt = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. ut = max(β[2] * ut, abs(dx))
  @. dx = (η/(1 - βt[1])) * mt/(ut + ϵ)

  return dx, ((mt, ut), βt .* β)
end

function apply!(o::OADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  (mt, vt), βt, dx_ = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  @. dx = -dx_
  @. dx_ = η * mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)
  @. dx += 2*dx_

  return dx, ((mt, vt), βt .* β, dx_)
end

function apply!(o::ADAGrad, x, dx, state)
  η, ϵ = o.eta, o.epsilon
  acc, = state

  @. acc += dx^2
  @. dx *= η / (sqrt(acc) + ϵ)

  return dx, (acc,)
end

function apply!(o::ADADelta, x, dx, state)
  ρ, ϵ = o.rho, o.epsilon
  acc, Δacc = state

  @. acc = ρ * acc + (1 - ρ) * dx^2
  # DON'T remove epsilon from numerator
  # or even out of the square roots
  @. dx *= sqrt(Δacc + ϵ) / sqrt(acc + ϵ)
  @. Δacc = ρ * Δacc + (1 - ρ) * dx^2

  return dx, (acc, Δacc)
end

function apply!(o::AMSGrad, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt, v̂t = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx ^ 2
  @. v̂t = max(v̂t, vt)
  @. dx = η * mt / (sqrt(v̂t) + ϵ)

  return dx, (mt, vt, v̂t)
end

function apply!(o::NADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  (mt, vt), βt = state.decays

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  @. dx = (β[1] * mt / (1 - β[1] * βt[1]) + (1 - β[1]) * dx / (1 - βt[1])) / 
          (sqrt(vt * β[2] / (1 - βt[2])) + ϵ) * η

  return dx, ((mt, vt), βt .* β)
end

function apply!(o::AdaBelief, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, st = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. st = β[2] * st + (1 - β[2]) * (dx - mt)^2
  @. dx =  η * mt / (sqrt(st) + ϵ)

  return dx, (mt, st)
end
