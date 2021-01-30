const mach_eps = 1f-8

"""
    Descent(; η = 0.1)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `dp`, this runs `p -= η*dp`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
"""
struct Descent{T}
  eta::T
end
Descent(; η = 0.1) = Descent(η)

init(o::Descent, x::AbstractArray) = nothing

function apply(o::Descent, x, dx, state)
  η = convert(eltype(dx), o.eta)
  dx .*= η
  
  return dx .* η, state
end

(o::Descent)(m, dm, st) = update(o, m, dm, st)

"""
    Momentum(; η = 0.01, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
"""
struct Momentum{T,S}
  eta::T
  rho::S
end
Momentum(; η = 0.01, ρ = 0.9) = Momentum(η, ρ)

init(o::Momentum, x::AbstractArray) = (velocity = zero(x),)

function apply(o::Momentum, x, dx, state)
  η, ρ, v = o.eta, o.rho, state.velocity
  @. v = ρ * v - η * dx
  
  return -v, (velocity = v,)
end

(o::Momentum)(m, dm, state) = update(o, m, dm, state)

"""
    Nesterov(; η = 0.001, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect dampening oscillations.
"""
struct Nesterov{T,S}
  eta::T
  rho::S
end
Nesterov(; η = 0.001, ρ = 0.9) = Nesterov(η, ρ)

init(o::Nesterov, x::AbstractArray) = (velocity = zero(x),)

(o::Nesterov)(m, dm, state) = update(o, m, dm, state)

function apply(o::Nesterov, x, dx, state)
  η, ρ, v = o.eta, o.rho, state.velocity
  d = @. ρ^2 * v - (1+ρ) * η * dx
  @. v = ρ * v - η * dx
  
  return -d, (velocity = v,)
end

"""
    RMSProp(; η = 0.001, ρ = 0.9, ϵ = 1f-8)

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
struct RMSProp{T,S}
  eta::T
  rho::S
  epsilon::Float32
end
RMSProp(; η = 0.001, ρ = 0.9, ϵ = mach_eps) = RMSProp(η, ρ, ϵ)

init(o::RMSProp, x::AbstractArray) = (acceleration = zero(x),)

function apply(o::RMSProp, x, dx, state)
  η, ρ, ϵ, acc = o.eta, o.rho, o.epsilon, state.acceleration
  @. acc = ρ * acc + (1 - ρ) * dx^2
  dx = @. dx * (η / (sqrt(acc) + ϵ))
  
  return dx, (acceleration = acc,)
end

(o::RMSProp)(m, dm, state) = update(o, m, dm, state)

"""
    ADAM(; η = 0.001, β = (0.9, 0.999), ϵ = 1f-8)

[ADAM](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct ADAM{T,K}
  eta::T
  beta::Tuple{K,K}
  epsilon::Float32
end
ADAM(; η = 0.001, β = (0.9, 0.999), ϵ = mach_eps) = ADAM(η, β, ϵ)

init(o::ADAM, x::AbstractArray) = (moments = (zero(x), zero(x)), decays = o.beta)

(o::ADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::ADAM{T}, x, dx, state) where T
  η, β, βt, ϵ = o.eta, o.beta, state.decays, o.epsilon
  mt, vt = state.moments

  @. mt = β[1] * mt + (one(T) - β[1]) * dx
  @. vt = β[2] * vt + (one(T) - β[2]) * dx ^ 2
  dx = @. mt / (one(T) - βt[1]) / (sqrt(vt / (one(T) - βt[2])) + ϵ) * η

  return dx, (moments = (mt, vt), decays = βt .* β)
end

"""
    RADAM(; η = 0.001, β = (0.9, 0.999), ϵ = mach_eps)

[Rectified ADAM](https://arxiv.org/abs/1908.03265) optimizer.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct RADAM{T,S}
  eta::T
  beta::Tuple{S,S}
  epsilon::Float32
end
RADAM(; η = 0.001, β = (0.9, 0.999), ϵ = mach_eps) = RADAM(η, β, ϵ)

init(o::RADAM, x::AbstractArray) = (moments = (zero(x), zero(x)), decays = o.beta, t = 1)

(o::RADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::RADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  ρ∞ = 2/(1-β[2])-1

  mt, vt = state.moments
  βt, t = state.decays, state.t

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  ρ = ρ∞ - 2*t * βt[2] / (1 - βt[2])
  if ρ > 4
    r = sqrt((ρ - 4) * (ρ - 2) * ρ∞/((ρ∞ - 4) * (ρ∞ - 2) * ρ))
    dx = @. mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η * r
  else
    dx = @. mt / (1 - βt[1]) * η
  end

  return dx, (moments = (mt, vt), decays = βt .* β, t = t + 1)
end

"""
    AdaMax(; η = 0.001, β = (0.9, 0.999), ϵ = 1f-8)

[AdaMax](https://arxiv.org/abs/1412.6980) is a variant of ADAM based on the ∞-norm.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct AdaMax{T,S}
  eta::T
  beta::Tuple{S,S}
  epsilon::Float32
end
AdaMax(; η = 0.001, β = (0.9, 0.999), ϵ = 1f-8) = AdaMax(η, β, ϵ)

init(o::AdaMax, x::AbstractArray) = (moments = (zero(x), zero(x)), decays = o.beta)

(o::AdaMax)(m, dm, state) = update(o, m, dm, state)

function apply(o::AdaMax, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, ut = state.moments
  βt = state.decays

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. ut = max(β[2] * ut, abs(dx))
  dx = @. (η/(1 - βt[1])) * mt/(ut + ϵ)

  return dx, (moments = (mt, ut), decays = βt .* β)
end

"""
    OADAM(; η = 0.001, β = (0.5, 0.9), ϵ = 1f-8)

[OADAM](https://arxiv.org/abs/1711.00141) (Optimistic ADAM)
is a variant of ADAM adding an "optimistic" term suitable for adversarial training.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct OADAM{T,S}
  eta::T
  beta::Tuple{S,S}
  epsilon::Float32
end
OADAM(; η = 0.001, β = (0.5, 0.9), ϵ = mach_eps) = OADAM(η, β, ϵ)

init(o::OADAM, x::AbstractArray) = (moments = (zero(x), zero(x)), decays = o.beta, dx = zero(x))

(o::OADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::OADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt = state.moments
  βt, dx_ = state.decays, state.dx

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  @. dx = -dx_
  @. dx_ = η * mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)
  dx = @. dx + 2*dx_

  return dx, (moments = (mt, vt), decays = βt .* β, dx = dx_)
end

"""
    ADAGrad(; η = 0.1, ϵ = 1f-8)

[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct ADAGrad{T}
  eta::T
  epsilon::Float32
end
ADAGrad(; η = 0.1, ϵ = mach_eps) = ADAGrad(η, ϵ)

init(o::ADAGrad, x::AbstractArray) = (acceleration = fill!(similar(x), o.epsilon),)

(o::ADAGrad)(m, dm, state) = update(o, m, dm, state)

function apply(o::ADAGrad, x, dx, state)
  η, ϵ = o.eta, o.epsilon
  acc = state.acceleration

  @. acc += dx^2
  dx = @. dx * η / (sqrt(acc) + ϵ)

  return dx, (acceleration = acc,)
end

"""
    ADADelta(; ρ = 0.9, ϵ = 1f-8)

[ADADelta](https://arxiv.org/abs/1212.5701) is a version of ADAGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ`): Factor by which the gradient is decayed at each time step.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct ADADelta{T}
  rho::T
  epsilon::Float32
end
ADADelta(; ρ = 0.9, ϵ = mach_eps) = ADADelta(ρ, ϵ)

init(o::ADADelta, x::AbstractArray) = (acceleration = zero(x), Δacceleration = zero(x))

(o::ADADelta)(m, dm, state) = update(o, m, dm, state)

function apply(o::ADADelta, x, dx, state)
  ρ, ϵ = o.rho, o.epsilon
  acc, Δacc = state.acceleration, state.Δacceleration

  @. acc = ρ * acc + (1 - ρ) * dx^2
  # DON'T remove epsilon from numerator
  # or even out of the square roots
  dx = @. dx * sqrt(Δacc + ϵ) / sqrt(acc + ϵ)
  @. Δacc = ρ * Δacc + (1 - ρ) * dx^2
  
  return dx, (acceleration = acc, Δacceleration = Δacc)
end

"""
    AMSGrad(; η = 0.001, β = (0.9, 0.999), ϵ = 1f-8)

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the ADAM
optimiser. Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct AMSGrad{T,S}
  eta::T
  beta::Tuple{S,S}
  epsilon::Float32
end
AMSGrad(; η = 0.001, β = (0.9, 0.999), ϵ = mach_eps) = AMSGrad(η, β, ϵ)

init(o::AMSGrad, x::AbstractArray) =
  (moments = (fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon)),)

(o::AMSGrad)(m, dm, state) = update(o, m, dm, state)

function apply(o::AMSGrad, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt, v̂t = state.moments

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx ^ 2
  @. v̂t = max(v̂t, vt)
  dx = @. η * mt / (sqrt(v̂t) + ϵ)

  return dx, (moments = (mt, vt, v̂t),)
end

"""
    NADAM(; η = 0.001, β = (0.9, 0.999), ϵ = 1f-8)

[NADAM](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) is a Nesterov variant of ADAM.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct NADAM{T,S}
  eta::T
  beta::Tuple{S,S}
  epsilon::Float32
end
NADAM(; η = 0.001, β = (0.9, 0.999), ϵ = mach_eps) = NADAM(η, β, ϵ)

init(o::NADAM, x::AbstractArray) = (moments = (zero(x), zero(x)), decays = o.beta)

(o::NADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::NADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt = state.moments
  βt = state.decays

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  dx = @. (β[1] * mt / (1 - β[1] * βt[1]) + (1 - β[1]) * dx / (1 - βt[1])) / 
          (sqrt(vt * β[2] / (1 - βt[2])) + ϵ) * η

  return dx, (moments = (mt, vt), decays = βt .* β)
end

"""
    ADAMW(; η = 0.001, β = (0.9, 0.999), γ = 0, ϵ = 1f-8)

[ADAMW](https://arxiv.org/abs/1711.05101) is a variant of ADAM fixing (as in repairing) its
weight decay regularization.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- `γ`: Decay applied to weights during optimisation.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
ADAMW(; η = 0.001, β = (0.9, 0.999), γ = 0, ϵ = mach_eps) =
  SequenceOptimiser(ADAM(η, β, ϵ), WeightDecay(γ))

"""
    AdaBelief(;η = 0.001, β = (0.9, 0.999), ϵ = 1f-8)

The [AdaBelief](https://arxiv.org/abs/2010.07468) optimiser is a variant of the well-known
ADAM optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct AdaBelief{T,S}
  eta::T
  beta::Tuple{S,S}
  epsilon::Float32
end
AdaBelief(; η = 0.001, β = (0.9, 0.999), ϵ = mach_eps) = AdaBelief(η, β, ϵ)

init(o::AdaBelief, x::AbstractArray) = (moments = (zero(x), zero(x)),)

(o::AdaBelief)(m, dm, state) = update(o, m, dm, state)

function apply(o::AdaBelief, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, st = state.moments

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. st = β[2] * st + (1 - β[2]) * (dx - mt)^2
  dx = @. η * mt / (sqrt(st) + ϵ)
  
  return dx, (moments = (mt, st),)
end

"""
    WeightDecay(; γ = 0)

Decay weights by `wd`.

# Parameters
- Weight decay (`γ`)
"""
struct WeightDecay{T}
  wd::T
end
WeightDecay(; γ = 1e-4) = WeightDecay(γ)

init(o::WeightDecay, x::AbstractArray) = nothing

(o::WeightDecay)(m, dm, state) = update(o, m, dm, state)

function apply(o::WeightDecay, x, dx, state)
  dx = @. dx + o.wd * x

  return dx, state
end

"""
    SequenceOptimiser(opts...)

Compose a sequence of optimisers so that each `opt` in `opts`
updates the gradient in the order specified.
"""
struct SequenceOptimiser{O}
  opts::O
end
SequenceOptimiser(opts...) = SequenceOptimiser(opts)

init(o::SequenceOptimiser, x::AbstractArray) = [init(opt, x) for opt in o.opts]

(o::SequenceOptimiser)(m, dm, state) = update(o, m, dm, state)

function apply(o::SequenceOptimiser, x, dx, states)
  new_states = similar(states)
  for (i, (opt, state)) in enumerate(zip(o.opts, states))
    dx, new_states[i] = apply(opt, x, dx, state)
  end

  return dx, new_states
end