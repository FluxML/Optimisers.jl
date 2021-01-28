"""
    Descent(;η = 0.1)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `dp`, this runs `p -= η*dp`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
"""
mutable struct Descent{T}
  eta::T
end
Descent(;η = 0.1) = Descent(η)

init(o::Descent, x::AbstractArray) = nothing

function apply!(o::Descent, dx, state)
  η = convert(eltype(dx), o.eta)
  dx .*= η
  
  return dx, state
end

(o::Descent)(m, dm, st) = update!(o, m, dm, st)

"""
    Momentum(;η = 0.01, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
"""
mutable struct Momentum{T,S}
  eta::T
  rho::S
end
Momentum(;η = 0.01, ρ = 0.9) = Momentum(η, ρ)

init(o::Momentum, x::AbstractArray) = (velocity = zero(x),)

function apply!(o::Momentum, dx, state)
  η, ρ, v = o.eta, o.rho, state.velocity
  @. v = ρ * v - η * dx
  @. dx = -v
  
  return dx, (velocity = v,)
end

(o::Momentum)(m, dm, state) = update!(o, m, dm, state)

"""
    Nesterov(;η = 0.001, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect dampening oscillations.
"""
mutable struct Nesterov{T,S}
  eta::T
  rho::S
end
Nesterov(;η = 0.001, ρ = 0.9) = Nesterov(η, ρ)

init(o::Nesterov, x::AbstractArray) = (velocity = zero(x),)

(o::Nesterov)(m, dm, state) = update!(o, m, dm, state)

function apply!(o::Nesterov, dx, state)
  η, ρ, v = o.eta, o.rho, state.velocity
  d = @. ρ^2 * v - (1+ρ) * η * dx
  @. v = ρ * v - η * dx
  @. dx = -d
  
  return dx, (velocity = v,)
end

"""
    RMSProp(;η = 0.001, ρ = 0.9, ϵ = 1f-8)

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                  (no need to change default)
"""
mutable struct RMSProp{T,S}
  eta::T
  rho::S
  epsilon::Float32
end
RMSProp(;η = 0.001, ρ = 0.9, ϵ = 1f-8) = RMSProp(η, ρ, ϵ)

init(o::RMSProp, x::AbstractArray) = (acceleration = zero(x),)

function apply!(o::RMSProp, dx, state)
  η, ρ, ϵ, acc = o.eta, o.rho, o.epsilon, state.acceleration
  @. acc = ρ * acc + (1 - ρ) * dx^2
  @. dx = dx * (η / (sqrt(acc) + ϵ))
  
  return dx, (acceleration = acc,)
end

(o::RMSProp)(m, dm, state) = update!(o, m, dm, state)

"""
    ADAM(;η = 0.001, β = (0.9, 0.999), ϵ = 1f-8)

[ADAM](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
mutable struct ADAM{T,K}
  eta::T
  beta::Tuple{K,K}
  epsilon::Float32
end
ADAM(;η = 0.001, β = (0.9, 0.999), ϵ = 1f-8) = ADAM(η, β, ϵ)

init(o::ADAM, x::AbstractArray) = (moments = (zero(x), zero(x)), decays = o.beta)

(o::ADAM)(m, dm, state) = update!(o, m, dm, state)

function apply!(o::ADAM{T}, dx, state) where T
  η, β, βt, ϵ = o.eta, o.beta, state.decays, o.epsilon
  mt, vt = state.moments
  @. mt = β[1] * mt + (one(T) - β[1]) * dx
  @. vt = β[2] * vt + (one(T) - β[2]) * dx ^ 2
  @. dx =  mt / (one(T) - βt[1]) / (sqrt(vt / (one(T) - βt[2])) + ϵ) * η
  return dx, (moments = (mt, vt), decays = βt .* β)
end
