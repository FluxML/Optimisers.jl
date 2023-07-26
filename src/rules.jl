@deprecate ADAM Adam
@deprecate NADAM NAdam
@deprecate ADAMW AdamW
@deprecate RADAM RAdam
@deprecate OADAM OAdam
@deprecate ADAGrad AdaGrad
@deprecate ADADelta AdaDelta

"""
    Descent(η = 1f-1)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `dp`, this runs `p -= η*dp`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
"""
struct Descent{T} <: AbstractRule
  eta::T
end
Descent() = Descent(1f-1)

init(o::Descent, x::AbstractArray) = nothing

function apply!(o::Descent, state, x, dx)
  η = convert(float(eltype(x)), o.eta)

  return state, @lazy dx * η  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

"""
    Momentum(η = 1f-2, ρ = 9f-1)

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
"""
struct Momentum{T} <: AbstractRule
  eta::T
  rho::T
end
Momentum(η = 1f-2, ρ = 9f-1) = Momentum{typeof(η)}(η, ρ)

init(o::Momentum, x::AbstractArray) = zero(x)

function apply!(o::Momentum, state, x, dx)
  η, ρ, mvel = o.eta, o.rho, state
  @.. mvel = ρ * mvel + η * dx  # Macro @.. broadcasts into mvel if it can, else @. of rhs.

  return mvel, mvel
end

"""
    Nesterov(η = 1f-3, ρ = 9f-1)

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect dampening oscillations.
"""
struct Nesterov{T} <: AbstractRule
  eta::T
  rho::T
end
Nesterov(η = 1f-3, ρ = 9f-1) = Nesterov{typeof(η)}(η, ρ)

init(o::Nesterov, x::AbstractArray) = zero(x)

function apply!(o::Nesterov, state, x, dx)
  η, ρ, vel = o.eta, o.rho, state

  newdx = @. - ρ^2 * vel + (1+ρ) * η * dx  # Cannot be lazy as this needs the old velocity
  @.. vel = ρ * vel - η * dx

  return vel, newdx
end

"""
    RMSProp(η = 1f-3, ρ = 9f-1, ϵ = eps(typeof(η)); centred = false)

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

[Centred RMSProp](http://arxiv.org/abs/1308.08500) is a variant which normalises
gradients by an estimate their variance, instead of their second moment.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
- Keyword `centred` (or `centered`): Indicates whether to use centred variant
                                     of the algorithm.
"""
struct RMSProp{T} <: AbstractRule
  eta::T
  rho::T
  epsilon::T
  centred::Bool
end

RMSProp(η = 1f-3, ρ = 9f-1, ϵ = eps(typeof(η)); centred::Bool = false, centered::Bool = false) =
  RMSProp{typeof(η)}(η, ρ, ϵ, centred | centered)

init(o::RMSProp, x::AbstractArray) = (zero(x), o.centred ? zero(x) : false)

function apply!(o::RMSProp, state, x, dx)
  η, ρ, ϵ = o.eta, o.rho, o.epsilon
  quad, lin = state

  @.. quad = ρ * quad + (1 - ρ) * abs2(dx)
  if o.centred
    @.. lin = ρ * lin + (1 - ρ) * dx
  end
  dx′ = @lazy dx * η / (sqrt(quad - abs2(lin)) + ϵ)

  return (quad, lin), dx′
end

function adjust(r::RMSProp; kw...)
  :centred in keys(kw) && throw(ArgumentError("adjust(::RMSProp; centred) is not allowed, as the variants store different states"))
  _adjust(r, NamedTuple(kw))  # that's why _adjust exists!
end

function Base.show(io::IO, o::RMSProp)
  show(io, typeof(o))
  print(io, "(")
  join(io, [o.eta, o.rho, o.epsilon], ", ")
  print(io, "; centred = ", o.centred, ")")
end


"""
    Rprop(η = 1f-3, ℓ = (5f-1, 1.2f0), Γ = (1f-6, 50f0))

Optimizer using the
[Rprop](https://ieeexplore.ieee.org/document/298623) algorithm. A full-batch
learning algorithm that depends only on the sign of the gradient.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.

- Scaling factors (`ℓ::Tuple`): Multiplicative increase and decrease factors.

- Step sizes (`Γ::Tuple`): Mminimal and maximal allowed step sizes.
"""
struct Rprop{T} <: AbstractRule
    eta::T
    ell::Tuple{T,T}
    gamma::Tuple{T,T}
end

Rprop(η = 1f-3, ℓ = (5f-1, 1.2f0), Γ = (1f-6, 50f0)) = Rprop{typeof(η)}(η, ℓ, Γ)

init(o::Rprop, x::AbstractArray) = (zero(x), onevalue(o.eta, x))

function apply!(o::Rprop, state, x, dx)
    T = eltype(x)
    ℓ = map(T, o.ell)
    Γ = map(T, o.gamma)
    g, η = state

    η = broadcast(g, η, dx) do g, η, dx
        g * dx > 0 ? min(η * ℓ[2], Γ[2]) : g * dx < 0 ? max(η * ℓ[1], Γ[1]) : η
    end
    g = broadcast(g, dx) do g, dx
        g * dx < 0 ? zero(T) : T(dx)
    end
    dx′ = @lazy η * sign(g)

    return (g, η), dx′
end

"""
    Adam(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[Adam](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct Adam{T} <: AbstractRule
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
Adam(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = Adam{typeof(η)}(η, β, ϵ)

init(o::Adam, x::AbstractArray) = (zero(x), zero(x), o.beta)

function apply!(o::Adam, state, x, dx)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, vt, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  dx′ = @lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η

  return (mt, vt, βt .* β), dx′
end

"""
    Lion(η = 0.001, β::Tuple = (0.9, 0.999))

[Lion](https://arxiv.org/abs/2302.06675) optimiser.

# Parameters
- Learning rate (`η`): Magnitude by which gradients are updating the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
"""
struct Lion{T} <: AbstractRule
  eta::T
  beta::Tuple{T,T}
end
Lion(η = 1f-3, β = (9f-1, 9.99f-1)) = Lion{typeof(η)}(η, β)

init(o::Lion, x::AbstractArray) = zero(x)

function apply!(o::Lion, state, x, dx)
  η, β = o.eta, o.beta

  @.. state = β[2] * dx + (1-β[2]) * state

  # The paper writes the update in terms of the old momentum,
  # but easy to solve in terms of the current momentum instead:
  dx′ = @lazy η * sign((β[2]-β[1]) * dx + β[1] * state)

  return state, dx′
end

"""
    RAdam(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[Rectified Adam](https://arxiv.org/abs/1908.03265) optimizer.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct RAdam{T} <: AbstractRule
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
RAdam(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = RAdam{typeof(η)}(η, β, ϵ)

init(o::RAdam, x::AbstractArray) = (zero(x), zero(x), o.beta, 1)

function apply!(o::RAdam, state, x, dx)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  ρ∞ = 2/(1-β[2])-1

  mt, vt, βt, t = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  ρ = ρ∞ - 2*t * βt[2] / (1 - βt[2])
  if ρ > 4
    r = sqrt((ρ - 4) * (ρ - 2) * ρ∞/((ρ∞ - 4) * (ρ∞ - 2) * ρ))
    dx′ = @lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η * r
  else
    dx′ = @lazy mt / (1 - βt[1]) * η
  end

  return (mt, vt, βt .* β, t + 1), dx′
end

"""
    AdaMax(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[AdaMax](https://arxiv.org/abs/1412.6980) is a variant of Adam based on the ∞-norm.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct AdaMax{T} <: AbstractRule
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
AdaMax(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = AdaMax{typeof(η)}(η, β, ϵ)

init(o::AdaMax, x::AbstractArray) = (zero(x), zero(x), o.beta)

function apply!(o::AdaMax, state, x, dx)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, ut, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. ut = max(β[2] * ut, abs(dx))
  dx′ = @lazy (η/(1 - βt[1])) * mt/(ut + ϵ)

  return (mt, ut, βt .* β), dx′
end

"""
    OAdam(η = 1f-3, β = (5f-1, 9f-1), ϵ = eps(typeof(η)))

[OAdam](https://arxiv.org/abs/1711.00141) (Optimistic Adam)
is a variant of Adam adding an "optimistic" term suitable for adversarial training.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct OAdam{T} <: AbstractRule
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
OAdam(η = 1f-3, β = (5f-1, 9f-1), ϵ = eps(typeof(η))) = OAdam{typeof(η)}(η, β, ϵ)

init(o::OAdam, x::AbstractArray) = (zero(x), zero(x), o.beta, zero(x))

function apply!(o::OAdam, state, x, dx)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, vt, βt, term = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  prev = copy(term)
  @.. term = η * mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)
  dx′ = @lazy 2 * term - prev

  return (mt, vt, βt .* β, term), dx′
end

"""
    AdaGrad(η = 1f-1, ϵ = eps(typeof(η)))

[AdaGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct AdaGrad{T} <: AbstractRule
  eta::T
  epsilon::T
end
AdaGrad(η = 1f-1, ϵ = eps(typeof(η))) = AdaGrad{typeof(η)}(η, ϵ)

init(o::AdaGrad, x::AbstractArray) = onevalue(o.epsilon, x)

function apply!(o::AdaGrad, state, x, dx)
  η, ϵ = o.eta, o.epsilon
  acc = state

  @.. acc = acc + abs2(dx)
  dx′ = @lazy dx * η / (sqrt(acc) + ϵ)

  return acc, dx′
end

"""
    AdaDelta(ρ = 9f-1, ϵ = eps(typeof(ρ)))

[AdaDelta](https://arxiv.org/abs/1212.5701) is a version of AdaGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ`): Factor by which the gradient is decayed at each time step.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct AdaDelta{T} <: AbstractRule
  rho::T
  epsilon::T
end
AdaDelta(ρ = 9f-1, ϵ = eps(typeof(ρ))) = AdaDelta{typeof(ρ)}(ρ, ϵ)

init(o::AdaDelta, x::AbstractArray) = (zero(x), zero(x))

function apply!(o::AdaDelta, state, x, dx)
  ρ, ϵ = o.rho, o.epsilon
  acc, Δacc = state

  @.. acc = ρ * acc + (1 - ρ) * abs2(dx)
  # DON'T remove epsilon from numerator or even out of the square roots!
  dx′ = @. dx * sqrt(Δacc + ϵ) / sqrt(acc + ϵ)  # Cannot be lazy as this needs the old Δacc
  @.. Δacc = ρ * Δacc + (1 - ρ) * abs2(dx′)

  return (acc, Δacc), dx′
end

"""
    AMSGrad(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the Adam
optimiser. Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct AMSGrad{T} <: AbstractRule
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
AMSGrad(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = AMSGrad{typeof(η)}(η, β, ϵ)

init(o::AMSGrad, x::AbstractArray) =
  (onevalue(o.epsilon, x), onevalue(o.epsilon, x), onevalue(o.epsilon, x))

function apply!(o::AMSGrad, state, x, dx)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, vt, v̂t = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  @.. v̂t = max(v̂t, vt)
  dx′ = @lazy η * mt / (sqrt(v̂t) + ϵ)

  return (mt, vt, v̂t), dx′
end

"""
    NAdam(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))

[NAdam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) is a Nesterov variant of Adam.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
struct NAdam{T} <: AbstractRule
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
NAdam(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η))) = NAdam{typeof(η)}(η, β, ϵ)

init(o::NAdam, x::AbstractArray) = (zero(x), zero(x), o.beta)

function apply!(o::NAdam, state, x, dx)
  η, β, ϵ = o.eta, o.beta, o.epsilon

  mt, vt, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  dx′ = @lazy (β[1] * mt / (1 - β[1] * βt[1]) + (1 - β[1]) * dx / (1 - βt[1])) /
          (sqrt(vt * β[2] / (1 - βt[2])) + ϵ) * η

  return (mt, vt, βt .* β), dx′
end

"""
    AdamW(η = 1f-3, β = (9f-1, 9.99f-1), γ = 0, ϵ = eps(typeof(η)))

[AdamW](https://arxiv.org/abs/1711.05101) is a variant of Adam fixing (as in repairing) its
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
AdamW(η = 1f-3, β = (9f-1, 9.99f-1), γ = 0, ϵ = eps(typeof(η))) =
  OptimiserChain(Adam{typeof(η)}(η, β, ϵ), WeightDecay{typeof(η)}(γ))

"""
    AdaBelief(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = 1e-16)

The [AdaBelief](https://arxiv.org/abs/2010.07468) optimiser is a variant of the well-known
Adam optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ::Float32`): Constant to prevent division by zero
                                  (no need to change default)
"""
struct AdaBelief{T} <: AbstractRule
  eta::T
  beta::Tuple{T, T}
  epsilon::T
end
AdaBelief(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = oftype(η, 1e-16)) = AdaBelief{typeof(η)}(η, β, ϵ)

init(o::AdaBelief, x::AbstractArray) = (zero(x), zero(x), o.beta)

function apply!(o::AdaBelief, state, x, dx)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, st, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. st = β[2] * st + (1 - β[2]) * abs2(dx - mt) + ϵ
  dx′ = @lazy η * mt / (1 - βt[1]) / (sqrt(st / (1 - βt[2])) + ϵ)

  return (mt, st, βt .* β), dx′
end

"""
    WeightDecay(γ = 5f-4)

Decay weights by ``γ``, that is, add `γ .* x` to the gradient `x̄` which will be
subtracted from `x`.

Typically composed  with other optimisers as the first transformation in an [`OptimiserChain`](@ref).
This is equivalent to adding ``L_2`` regularization with coefficient ``γ`` to the loss.

# Parameters
- Weight decay (`γ`): Decay applied to weights during optimisation.
"""
struct WeightDecay{T} <: AbstractRule
  gamma::T
end
WeightDecay() = WeightDecay(5f-4)

init(o::WeightDecay, x::AbstractArray) = nothing

function apply!(o::WeightDecay, state, x, dx)
  dx′ = @lazy dx + o.gamma * x

  return state, dx′
end

"""
    ClipGrad(δ = 10f0)

Restricts every gradient component to obey `-δ ≤ dx[i] ≤ δ`.

See also [`ClipNorm`](@ref).
"""
struct ClipGrad{T<:Real} <: AbstractRule
  delta::T
end
ClipGrad(δ::Integer = 10) = ClipGrad(Float32(δ))  # float is to ensure adjust can change this

init(o::ClipGrad, x::AbstractArray) = nothing

function apply!(o::ClipGrad, state, x, dx)
  δ = convert(float(eltype(x)), o.delta)
  dx′ = @lazy clamp(dx, -δ, δ)

  return state, dx′
end

"""
    ClipNorm(ω = 10f0, p = 2; throw = true)

Scales any gradient array for which `norm(dx, p) > ω`
to stay at this threshold (unless `p==0`).

Throws an error if the norm is infinite or `NaN`,
which you can turn off with `throw = false`.

See also [`ClipGrad`](@ref).
"""
struct ClipNorm{T<:Real} <: AbstractRule
  omega::T
  p::T
  throw::Bool
end
ClipNorm(ω = 10f0, p = 2; throw::Bool = true) = ClipNorm{float(typeof(ω))}(ω, p, throw)

init(o::ClipNorm, x::AbstractArray) = nothing

function apply!(o::ClipNorm, state, x, dx)
  nrm = _norm(dx, o.p)
  if o.throw && !isfinite(nrm)
    throw(DomainError("gradient has $(o.p)-norm $nrm, for array $(summary(x))"))
  end
  λ = min(o.omega / nrm, 1)

  return state, @lazy dx * λ
end

_norm(dx::AbstractArray, p::Real) = norm(dx, p)  # LinearAlgebra, CUDA
function _norm(dx::Broadcast.Broadcasted, p::Real)
  if p == 2
    # This lacks the undeflow/overflow tests of LinearAlgebra's version
    sqrt(sum(abs2, dx))
  elseif p == 1
    float(sum(abs, dx))
  elseif p == Inf
    float(maximum(abs, dx))
  elseif p == 0
    cnt = count(!iszero, dx)
    T = Base.@default_eltype dx
    T <: Number ? convert(float(T), cnt) : cnt
  elseif p == -Inf
    float(minimum(abs, dx))
  else
    # This isn't optimally fast but does ensure p::Float64 doesn't promote
    tmp = abs.(dx)
    q = convert(float(eltype(tmp)), p)
    sum(tmp .^ q) ^ (1/q)
  end
end

"""
    OptimiserChain(opts...)

Compose a sequence of optimisers so that each `opt` in `opts`
updates the gradient, in the order specified.

With an empty sequence, `OptimiserChain()` is the identity,
so `update!` will subtract the full gradient from the parameters.
This is equivalent to `Descent(1)`.

# Example

```jldoctest
julia> o = OptimiserChain(ClipGrad(1.0), Descent(0.1));

julia> m = (zeros(3),);

julia> s = Optimisers.setup(o, m)
(Leaf(OptimiserChain(ClipGrad{Float64}(1.0), Descent{Float64}(0.1)), (nothing, nothing)),)

julia> Optimisers.update(s, m, ([0.3, 1, 7],))[2]  # clips before discounting
([-0.03, -0.1, -0.1],)
```
"""
struct OptimiserChain{O<:Tuple} <: AbstractRule
  opts::O
end
OptimiserChain(opts...) = OptimiserChain(opts)

@functor OptimiserChain

init(o::OptimiserChain, x::AbstractArray) = map(opt -> init(opt, x), o.opts)

function apply!(o::OptimiserChain, states, x, dx, dxs...)
  foldl(tuple.(o.opts, states); init = ((), dx)) do (states′, dx′), (opt, state)
    if dx′ isa Zero
      return (states′..., state), dx′
    else 
      state′, dx′ = apply!(opt, state, x, dx′, dxs...)
      return (states′..., state′), dx′
    end
  end
end

function Base.show(io::IO, c::OptimiserChain)
  print(io, "OptimiserChain(")
  join(io, c.opts, ", ")
  print(io, ")")
end

adjust(ℓ::OptimiserChain, eta::Real) = OptimiserChain(map(opt -> adjust(opt, eta), ℓ.opts)...)
adjust(ℓ::OptimiserChain; kw...) = OptimiserChain(map(opt -> adjust(opt; kw...), ℓ.opts)...)


"""
    AccumGrad(n::Int)

A rule constructed `OptimiserChain(AccumGrad(n), Rule())` will accumulate for `n` steps,
before applying `Rule` to the mean of these `n` gradients.

This is useful for training with effective batch sizes too large for the available memory.
Instead of computing the gradient for batch size `b` at once, compute it for size `b/n` and
accumulate `n` such gradients.

# Example
```jldoctest
julia> m = (x=[1f0], y=[2f0]);

julia> r = OptimiserChain(AccumGrad(2), WeightDecay(0.01), Descent(0.1));

julia> s = Optimisers.setup(r, m);

julia> Optimisers.update!(s, m, (x=[33], y=[0]));

julia> m  # model not yet changed
(x = Float32[1.0], y = Float32[2.0])

julia> Optimisers.update!(s, m, (x=[0], y=[444]));

julia> m  # n=2 gradients applied at once
(x = Float32[-0.651], y = Float32[-20.202])
```
"""
struct AccumGrad <: AbstractRule
  n::Int
  
  function AccumGrad(n::Int)
    n > 0 || throw(ArgumentError("AccumGrad must accumulate at least one gradient"))
    return new(n)  
  end
end

function init(o::AccumGrad, x)
  return (zero(x), 1)
end

function apply!(o::AccumGrad, state, x, dx)
  accum_dx, counter = state
  if counter == 1
    @.. accum_dx = dx / o.n
  else
    @.. accum_dx = accum_dx + dx / o.n
  end
  if counter == o.n
    return (accum_dx, 1), accum_dx
  else
    return (accum_dx, counter + 1), nothing
  end
end

"""
    MixedPrecision(opt)
    MixedPrecision{T}(opt)

An optimiser that wraps another optimiser `opt` in order to perform mixed precision
training [1]. 

The state of `MixedPrecision` will contain a copy in precision `T` of the trainable parameter `x`, 
call it `xT`. 
The internal state of `opt` also operates at precision `T`.
If `T` is not specified, it defaults to `Float32`.

Call `g` the gradient of `x`. Both `g` and `x` are typically in a precision lower than `T`
(e.g. `Float16`).

In the `update!(opt_state, x, g)` call, `opt` is used to update `xT` instead of `x`, 
then `x` is updated with the value of `xT`. 

[1] Micikevicius et al. '17, "Mixed Precision Training", https://arxiv.org/abs/1710.03740 .

# Examples

```julia
x = rand(Float16, 2) # a trainable parameter in low precision

opt = MixedPrecision(Adam(1e-3)) 
opt_state = Optimisers.setup(opt, x) # the state contains a copy of x in Float32 precision

g = rand(Float16, 2) # a gradient in low precision

# accumulation is performed in high precision,
# then also the low precision x is synced
Optimisers.update!(opt_state, x, g)  
```
"""
struct MixedPrecision{T<:Number, O<:AbstractRule} <: AbstractRule
  opt::O
end

MixedPrecision(opt::AbstractRule) = MixedPrecision{Float32, typeof(opt)}(opt)
MixedPrecision{T}(opt::AbstractRule) where T = MixedPrecision{T, typeof(opt)}(opt)

to_precision(::Type{T}, x::AbstractArray) where T = convert(AbstractArray{T}, x)

function init(o::MixedPrecision{T}, x) where T
  xT = to_precision(T, x)
  return (xT, init(o.opt, xT))
end

function apply!(o::MixedPrecision{T}, state, x, dx) where T
  xT, st = state
  st′, dx′ = apply!(o.opt, st, xT, dx)
  subtract!(xT, dx′)
  @. x = xT
  return (xT, st′), nothing
end
