"""
    Descent(η = 1f-1)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `dp`, this runs `p -= η*dp`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
"""
struct Descent{T}
  eta::T
end
Descent() = Descent(1f-1)

init(o::Descent, x::AbstractArray) = nothing

function apply(o::Descent, x, dx, state)
  η = convert(eltype(dx), o.eta)
  
  return dx .* η, state
end

(o::Descent)(m, dm, state) = update(o, m, dm, state)

"""
    Momentum(η = 1f-2, ρ = 9f-1)

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
"""
struct Momentum{T}
  eta::T
  rho::T
end
Momentum(η = 1f-2, ρ = 9f-1) = Momentum{typeof(η)}(η, ρ)

init(o::Momentum, x::AbstractArray) = zero(x)

function apply(o::Momentum, x, dx, state)
  η, ρ, v = o.eta, o.rho, state
  @. v = ρ * v - η * dx
  
  return v, -v
end

(o::Momentum)(m, dm, state) = update(o, m, dm, state)

"""
    Nesterov(η = 1f-3, ρ = 9f-1)

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect dampening oscillations.
"""
struct Nesterov{T}
  eta::T
  rho::T
end
Nesterov(η = 1f-3, ρ = 9f-1) = Nesterov{typeof(η)}(η, ρ)

init(o::Nesterov, x::AbstractArray) = zero(x)

(o::Nesterov)(m, dm, state) = update(o, m, dm, state)

function apply(o::Nesterov, x, dx, state)
  η, ρ, v = o.eta, o.rho, state
  d = @. ρ^2 * v - (1+ρ) * η * dx
  @. v = ρ * v - η * dx
  
  return -d, v
end

"""
    RMSProp(η = 1f-3, ρ = 9f-1, ϵ = eps(typeof(η)))

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct RMSProp{T}
  eta::T
  rho::T
  epsilon::T
end
RMSProp(η = 1f-3, ρ = 9f-1, ϵ = eps(typeof(η))) = RMSProp{typeof(η)}(η, ρ, ϵ)

init(o::RMSProp, x::AbstractArray) = zero(x)

function apply(o::RMSProp, x, dx, state)
  η, ρ, ϵ, acc = o.eta, o.rho, o.epsilon, state
  @. acc = ρ * acc + (1 - ρ) * dx^2
  dx = @. dx * (η / (sqrt(acc) + ϵ))
  
  return dx, acc
end

(o::RMSProp)(m, dm, state) = update(o, m, dm, state)

"""
    ADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[ADAM](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct ADAM{T}
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
ADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = ADAM{typeof(η)}(η, β, ϵ)

init(o::ADAM, x::AbstractArray) = (zero(x), zero(x), o.beta)

(o::ADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::ADAM{T}, x, dx, state) where T
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, vt, βt = state

  @. mt = β[1] * mt + (one(T) - β[1]) * dx
  @. vt = β[2] * vt + (one(T) - β[2]) * dx ^ 2
  dx = @. mt / (one(T) - βt[1]) / (sqrt(vt / (one(T) - βt[2])) + ϵ) * η

  return dx, (mt, vt, βt .* β)
end

"""
    RADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[Rectified ADAM](https://arxiv.org/abs/1908.03265) optimizer.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct RADAM{T}
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
RADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = RADAM{typeof(η)}(η, β, ϵ)

init(o::RADAM, x::AbstractArray) = (zero(x), zero(x), o.beta, 1)

(o::RADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::RADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  ρ∞ = 2/(1-β[2])-1

  mt, vt, βt, t = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  ρ = ρ∞ - 2*t * βt[2] / (1 - βt[2])
  if ρ > 4
    r = sqrt((ρ - 4) * (ρ - 2) * ρ∞/((ρ∞ - 4) * (ρ∞ - 2) * ρ))
    dx = @. mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η * r
  else
    dx = @. mt / (1 - βt[1]) * η
  end

  return dx, (mt, vt, βt .* β, t + 1)
end

"""
    AdaMax(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[AdaMax](https://arxiv.org/abs/1412.6980) is a variant of ADAM based on the ∞-norm.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct AdaMax{T}
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
AdaMax(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = AdaMax{typeof(η)}(η, β, ϵ)

init(o::AdaMax, x::AbstractArray) = (zero(x), zero(x), o.beta)

(o::AdaMax)(m, dm, state) = update(o, m, dm, state)

function apply(o::AdaMax, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, ut, βt = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. ut = max(β[2] * ut, abs(dx))
  dx = @. (η/(1 - βt[1])) * mt/(ut + ϵ)

  return dx, (mt, ut, βt .* β)
end

"""
    OADAM(η = 1f-3, β = (5f-1, 9f-1), ϵ = eps(typeof(η)))

[OADAM](https://arxiv.org/abs/1711.00141) (Optimistic ADAM)
is a variant of ADAM adding an "optimistic" term suitable for adversarial training.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct OADAM{T}
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
OADAM(η = 1f-3, β = (5f-1, 9f-1), ϵ = eps(typeof(η))) = OADAM{typeof(η)}(η, β, ϵ)

init(o::OADAM, x::AbstractArray) = (zero(x), zero(x), o.beta, zero(x))

(o::OADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::OADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt, βt, dx_ = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  @. dx = -dx_
  @. dx_ = η * mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)
  dx = @. dx + 2*dx_

  return dx, (mt, vt, βt .* β, dx_)
end

"""
    ADAGrad(η = 1f-1, ϵ = eps(typeof(η)))

[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct ADAGrad{T}
  eta::T
  epsilon::T
end
ADAGrad(η = 1f-1, ϵ = eps(typeof(η))) = ADAGrad{typeof(η)}(η, ϵ)

init(o::ADAGrad, x::AbstractArray) = fill!(similar(x), o.epsilon)

(o::ADAGrad)(m, dm, state) = update(o, m, dm, state)

function apply(o::ADAGrad, x, dx, state)
  η, ϵ = o.eta, o.epsilon
  acc = state

  @. acc += dx^2
  dx = @. dx * η / (sqrt(acc) + ϵ)

  return dx, acc
end

"""
    ADADelta(ρ = 9f-1, ϵ = eps(typeof(ρ)))

[ADADelta](https://arxiv.org/abs/1212.5701) is a version of ADAGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ`): Factor by which the gradient is decayed at each time step.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct ADADelta{T}
  rho::T
  epsilon::T
end
ADADelta(ρ = 9f-1, ϵ = eps(typeof(ρ))) = ADADelta{typeof(ρ)}(ρ, ϵ)

init(o::ADADelta, x::AbstractArray) = (zero(x), zero(x))

(o::ADADelta)(m, dm, state) = update(o, m, dm, state)

function apply(o::ADADelta, x, dx, state)
  ρ, ϵ = o.rho, o.epsilon
  acc, Δacc = state

  @. acc = ρ * acc + (1 - ρ) * dx^2
  # DON'T remove epsilon from numerator
  # or even out of the square roots
  dx = @. dx * sqrt(Δacc + ϵ) / sqrt(acc + ϵ)
  @. Δacc = ρ * Δacc + (1 - ρ) * dx^2
  
  return dx, (acc, Δacc)
end

"""
    AMSGrad(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the ADAM
optimiser. Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct AMSGrad{T}
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
AMSGrad(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = AMSGrad{typeof(η)}(η, β, ϵ)

init(o::AMSGrad, x::AbstractArray) =
  (fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon))

(o::AMSGrad)(m, dm, state) = update(o, m, dm, state)

function apply(o::AMSGrad, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt, v̂t = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx ^ 2
  @. v̂t = max(v̂t, vt)
  dx = @. η * mt / (sqrt(v̂t) + ϵ)

  return dx, (mt, vt, v̂t)
end

"""
    NADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[NADAM](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) is a Nesterov variant of ADAM.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct NADAM{T}
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
NADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = NADAM{typeof(η)}(η, β, ϵ)

init(o::NADAM, x::AbstractArray) = (zero(x), zero(x), o.beta)

(o::NADAM)(m, dm, state) = update(o, m, dm, state)

function apply(o::NADAM, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt, βt = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. vt = β[2] * vt + (1 - β[2]) * dx^2
  dx = @. (β[1] * mt / (1 - β[1] * βt[1]) + (1 - β[1]) * dx / (1 - βt[1])) / 
          (sqrt(vt * β[2] / (1 - βt[2])) + ϵ) * η

  return dx, (mt, vt, βt .* β)
end

"""
    ADAMW(η = 1f-3, β = (9f-1, 9.99f-1), γ = 0, ϵ = eps(typeof(η)))

[ADAMW](https://arxiv.org/abs/1711.05101) is a variant of ADAM fixing (as in repairing) its
weight decay regularization.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Weight decay (`γ`): Decay applied to weights during optimisation.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
ADAMW(η = 1f-3, β = (9f-1, 9.99f-1), γ = 0, ϵ = eps(typeof(η))) =
  OptimiserChain(ADAM{typeof(η)}(η, β, ϵ), WeightDecay(γ))

"""
    AdaBelief(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

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
struct AdaBelief{T}
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
AdaBelief(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = AdaBelief{typeof(η)}(η, β, ϵ)

init(o::AdaBelief, x::AbstractArray) = (zero(x), zero(x))

(o::AdaBelief)(m, dm, state) = update(o, m, dm, state)

function apply(o::AdaBelief, x, dx, state)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, st = state

  @. mt = β[1] * mt + (1 - β[1]) * dx
  @. st = β[2] * st + (1 - β[2]) * (dx - mt)^2
  dx = @. η * mt / (sqrt(st) + ϵ)
  
  return dx, (mt, st)
end

"""
    WeightDecay(γ = 5f-4)

Decay weights by `γ`.

# Parameters
- Weight decay (`γ`): Decay applied to weights during optimisation.
"""
struct WeightDecay{T}
  wd::T
end
WeightDecay() = WeightDecay(5f-4)

init(o::WeightDecay, x::AbstractArray) = nothing

(o::WeightDecay)(m, dm, state) = update(o, m, dm, state)

function apply(o::WeightDecay, x, dx, state)
  dx = @. dx + o.wd * x

  return dx, state
end

"""
    OptimiserChain(opts...)

Compose a chain (sequence) of optimisers so that each `opt` in `opts`
updates the gradient in the order specified.
"""
struct OptimiserChain{O}
  opts::O
end
OptimiserChain(opts...) = OptimiserChain(opts)

init(o::OptimiserChain, x::AbstractArray) = [init(opt, x) for opt in o.opts]

(o::OptimiserChain)(m, dm, states) = update(o, m, dm, states)

function apply(o::OptimiserChain, x, dx, states)
  new_states = similar(states)
  for (i, (opt, state)) in enumerate(zip(o.opts, states))
    dx, new_states[i] = apply(opt, x, dx, state)
  end

  return dx, new_states
end

for Opt in (:Descent, :ADAM, :Momentum, :Nesterov, :RMSProp,
            :ADAGrad, :AdaMax, :ADADelta, :AMSGrad, :NADAM,
            :RADAM, :OADAM, :AdaBelief)
  @eval function $Opt(m::$Opt; kwargs...)
    fs = fieldnames($Opt)
    args = NamedTuple{fs}(f in keys(kwargs) ? getindex(kwargs, f) : getfield(m, f) for f in fs)
    $Opt(args...)
  end
end
