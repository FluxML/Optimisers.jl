@deprecate ADAM Adam
@deprecate NADAM NAdam
@deprecate ADAMW AdamW
@deprecate RADAM RAdam
@deprecate OADAM OAdam
@deprecate ADAGrad AdaGrad
@deprecate ADADelta AdaDelta

"""
    Descent(η = 1f-1)
    Descent(; [eta])

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `dp`, this runs `p -= η*dp`.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
"""
struct Descent{T} <: AbstractRule
  eta::T
end

Descent(; eta = 1f-1) = Descent(eta)

init(o::Descent, x::AbstractArray) = nothing

function apply!(o::Descent, state, x, dx)
  η = ofeltype(x, o.eta)

  return state, @lazy dx * η  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

function Base.show(io::IO, o::Descent)
  print(io, "Descent(")
  show(io, o.eta)
  print(io, ")")
end

"""
    Momentum(η = 0.01, ρ = 0.9)
    Momentum(; [eta, rho])

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ == rho`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
"""
@def struct Momentum <: AbstractRule
  eta = 0.01  # Macro @def uses 0.01 as default value, and Float64 as the type
  rho = 0.9
end

init(o::Momentum, x::AbstractArray) = zero(x)

function apply!(o::Momentum, mvel, x::AbstractArray{T}, dx) where T
  η, ρ = T(o.eta), T(o.rho)
  @.. mvel = ρ * mvel + η * dx  # Macro @.. broadcasts into mvel if it can, else @. of rhs.

  return mvel, mvel
end

"""
    Nesterov(η = 0.001, ρ = 0.9)
    Nesterov(; [eta, rho])


Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect dampening oscillations.
"""
@def struct Nesterov <: AbstractRule
  eta = 0.001
  rho = 0.9
end

init(o::Nesterov, x::AbstractArray) = zero(x)

function apply!(o::Nesterov, vel, x::AbstractArray{T}, dx) where T
  η, ρ = T(o.eta), T(o.rho)

  newdx = @. - ρ^2 * vel + (1+ρ) * η * dx  # Cannot be lazy as this needs the old velocity
  @.. vel = ρ * vel - η * dx

  return vel, newdx
end

"""
    RMSProp(η = 0.001, ρ = 0.9, ϵ = 1e-8; centred = false)
    RMSProp(; [eta, rho, epsilon, centred])

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

[Centred RMSProp](http://arxiv.org/abs/1308.08500) is a variant which normalises
gradients by an estimate their variance, instead of their second moment.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ == rho`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
- Keyword `centred` (or `centered`): Indicates whether to use centred variant
                                     of the algorithm.
"""
struct RMSProp{Teta,Trho,Teps} <: AbstractRule
  eta::Teta
  rho::Trho
  epsilon::Teps
  centred::Bool
end

function RMSProp(η, ρ = 0.9, ϵ = 1e-8; centred::Bool = false, centered::Bool = false)
  η < 0 && throw(DomainError(η, "the learning rate cannot be negative"))
  return RMSProp(float(η), float(ρ), float(ϵ), centred | centered)
end
RMSProp(; eta = 0.001, rho = 0.9, epsilon = 1e-8, kw...) = RMSProp(eta, rho, epsilon; kw...)

init(o::RMSProp, x::AbstractArray) = (zero(x), o.centred ? zero(x) : false)

function apply!(o::RMSProp, state, x::AbstractArray{T}, dx) where T
  η, ρ, ϵ = T(o.eta), T(o.rho), _eps(T, o.epsilon)
  quad, lin = state

  @.. quad = ρ * quad + (1 - ρ) * abs2(dx)
  if o.centred
    @.. lin = ρ * lin + (1 - ρ) * dx
    dx′ = @lazy dx * η / (sqrt(max(real(quad) - abs2(lin), 0)) + ϵ)  # rounding can make the difference slightly negative
  else
    # `lin` is a `false` placeholder, and Reactant cannot trace `abs2` of a `Bool`.
    dx′ = @lazy dx * η / (sqrt(quad) + ϵ)
  end

  return (quad, lin), dx′
end

function adjust(r::RMSProp; kw...)
  :centred in keys(kw) && throw(ArgumentError("adjust(::RMSProp; centred) is not allowed, as the variants store different states"))
  _adjust(r, NamedTuple(kw))  # that's why _adjust exists!
end

function Base.show(io::IO, o::RMSProp)
  print(io, "RMSProp(eta=$(o.eta), rho=$(o.rho), epsilon=$(o.epsilon), centred=$(o.centred)")
end


"""
    Rprop(η = 1e-3, ℓ = (0.5, 1.2), Γ = (1e-6, 50.0))
    Rprop(; [eta, ell, gamma])

Optimizer using the
[Rprop](https://ieeexplore.ieee.org/document/298623) algorithm. A full-batch
learning algorithm that depends only on the sign of the gradient.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.

- Scaling factors (`ℓ::Tuple == ell`): Multiplicative increase and decrease factors.

- Step sizes (`Γ::Tuple == gamma`): Mminimal and maximal allowed step sizes.
"""
@def struct Rprop <: AbstractRule
    eta =  1e-3
    ell = (0.5, 1.2)
    gamma = (1e-6, 50.0)
end

init(o::Rprop, x::AbstractArray) = (zero(x), onevalue(o.eta, x))

function apply!(o::Rprop, state, x::AbstractArray{T}, dx) where T
    ℓ, Γ = T.(o.ell), T.(o.gamma)
    g, η = state
  
    # Branch-free, so that array types which trace the broadcast (Reactant) work too.
    η = broadcast(g, η, dx) do g, η, dx
        ifelse(g * dx > 0, min(η * ℓ[2], Γ[2]), ifelse(g * dx < 0, max(η * ℓ[1], Γ[1]), η))
    end
    g = broadcast(g, dx) do g, dx
        ifelse(g * dx < 0, zero(g), oftype(g, dx))
    end
    dx′ = @lazy η * sign(g)

    return (g, η), dx′
end

"""
    Adam(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8)
    Adam(; [eta, beta, epsilon])

[Adam](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct Adam <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
  epsilon = 1e-8
end

init(o::Adam, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta))

function apply!(o::Adam, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), _eps(T, o.epsilon)
  mt, vt, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  dx′ = @lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η

  return (mt, vt, βt .* β), dx′
end

"""
    Lion(η = 0.001, β = (0.9, 0.999))
    Lion(; [eta, beta])

[Lion](https://arxiv.org/abs/2302.06675) optimiser.

# Parameters
- Learning rate (`η == eta`): Magnitude by which gradients are updating the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
"""
@def struct Lion <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
end

init(o::Lion, x::AbstractArray) = zero(x)

function apply!(o::Lion, state, x::AbstractArray{T}, dx) where T
  η, β = T(o.eta), T.(o.beta)

  @.. state = β[2] * state + (1-β[2]) * dx

  # The paper writes the update in terms of the old momentum,
  # but easy to solve in terms of the current momentum instead:
  dx′ = @lazy η * sign((β[2]-β[1]) * dx + β[1] * state)

  return state, dx′
end

"""
    RAdam(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8)
    RAdam(; [eta, beta, epsilon])

[Rectified Adam](https://arxiv.org/abs/1908.03265) optimizer.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct RAdam <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
  epsilon = 1e-8
end

init(o::RAdam, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta), 1)

function apply!(o::RAdam, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), _eps(T, o.epsilon)
  ρ∞ = 2/(1-β[2]) - 1 |> real

  mt, vt, βt, t = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  ρ = ρ∞ - 2*t * βt[2] / (1 - βt[2]) |> real
  if ρ > 4
    r = sqrt((ρ - 4) * (ρ - 2) * ρ∞/((ρ∞ - 4) * (ρ∞ - 2) * ρ))
    dx′ = @lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η * r
  else
    dx′ = @lazy mt / (1 - βt[1]) * η
  end

  return (mt, vt, βt .* β, t + 1), dx′
end

"""
    AdaMax(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8)
    AdaMax(; [eta, beta, epsilon])

[AdaMax](https://arxiv.org/abs/1412.6980) is a variant of Adam based on the ∞-norm.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct AdaMax <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
  epsilon = 1e-8
end

init(o::AdaMax, x::AbstractArray{T}) where T = (zero(x), zero(real.(x)), T.(o.beta))

function apply!(o::AdaMax, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), _eps(T, o.epsilon)
  mt, ut, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. ut = max(real(β[2]) * ut, abs(dx))
  dx′ = @lazy (η/(1 - βt[1])) * mt/(ut + ϵ)

  return (mt, ut, βt .* β), dx′
end

"""
    OAdam(η = 0.001, β = (0.5, 0.9), ϵ = 1e-8)
    OAdam(; [eta, beta, epsilon])

[OAdam](https://arxiv.org/abs/1711.00141) (Optimistic Adam)
is a variant of Adam adding an "optimistic" term suitable for adversarial training.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct OAdam <: AbstractRule
  eta = 0.001
  beta = (0.5, 0.9)
  epsilon = 1e-8
end

init(o::OAdam, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta), zero(x))

function apply!(o::OAdam, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), _eps(T, o.epsilon)
  mt, vt, βt, term = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  prev = copy(term)
  @.. term = η * mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)
  dx′ = @lazy 2 * term - prev

  return (mt, vt, βt .* β, term), dx′
end

"""
    AdaGrad(η = 0.1, ϵ = 1e-8)
    AdaGrad(; [eta, epsilon])

[AdaGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct AdaGrad <: AbstractRule
  eta = 0.1
  epsilon = 1e-8
end

init(o::AdaGrad, x::AbstractArray) = onevalue(o.epsilon, x)

function apply!(o::AdaGrad, state, x::AbstractArray{T}, dx) where T
  η, ϵ = T(o.eta), _eps(T, o.epsilon)
  acc = state

  @.. acc = acc + abs2(dx)
  dx′ = @lazy dx * η / (sqrt(acc) + ϵ)

  return acc, dx′
end

"""
    AdaDelta(ρ = 0.9, ϵ = 1e-8)
    AdaDelta(; [rho, epsilon])

[AdaDelta](https://arxiv.org/abs/1212.5701) is a version of AdaGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ == rho`): Factor by which the gradient is decayed at each time step.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct AdaDelta <: AbstractRule
  rho = 0.9
  epsilon = 1e-8
end

init(o::AdaDelta, x::AbstractArray) = (zero(x), zero(x))

function apply!(o::AdaDelta, state, x::AbstractArray{T}, dx) where T
  ρ, ϵ = T(o.rho), _eps(T, o.epsilon)
  acc, Δacc = state

  @.. acc = ρ * acc + (1 - ρ) * abs2(dx)
  # DON'T remove epsilon from numerator or even out of the square roots!
  dx′ = @. dx * sqrt(Δacc + ϵ) / sqrt(acc + ϵ)  # Cannot be lazy as this needs the old Δacc
  @.. Δacc = ρ * Δacc + (1 - ρ) * abs2(dx′)

  return (acc, Δacc), dx′
end

"""
    AMSGrad(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8)
    AMSGrad(; [eta, beta, epsilon])

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the Adam
optimiser. Parameters don't need tuning.

As in Algorithm 2 of the paper, the moment estimates start from zero and
are not bias-corrected, unlike in [`Adam`](@ref).

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct AMSGrad <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
  epsilon = 1e-8
end

init(o::AMSGrad, x::AbstractArray) = (zero(x), zero(real.(x)), zero(real.(x)))

function apply!(o::AMSGrad, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), _eps(T, o.epsilon)
  mt, vt, v̂t = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = real(β[2]) * vt + (1 - real(β[2])) * abs2(dx)
  @.. v̂t = max(v̂t, vt)
  dx′ = @lazy η * mt / (sqrt(v̂t) + ϵ)

  return (mt, vt, v̂t), dx′
end

"""
    NAdam(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8)
    NAdam(; [eta, beta, epsilon])

[NAdam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) is a Nesterov variant of Adam.
Parameters don't need tuning.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct NAdam <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
  epsilon = 1e-8
end

init(o::NAdam, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta))

function apply!(o::NAdam, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), _eps(T, o.epsilon)
  mt, vt, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  dx′ = @lazy (β[1] * mt / (1 - β[1] * βt[1]) + (1 - β[1]) * dx / (1 - βt[1])) /
          (sqrt(vt * β[2] / (1 - βt[2])) + ϵ) * η

  return (mt, vt, βt .* β), dx′
end

"""
    AdamW(η = 0.001, β = (0.9, 0.999), λ = 0, ϵ = 1e-8; couple = true)
    AdamW(; [eta, beta, lambda, epsilon, couple])

[AdamW](https://arxiv.org/abs/1711.05101) is a variant of Adam fixing (as in repairing) its
weight decay regularization.
Implemented as an [`OptimiserChain`](@ref) of [`Adam`](@ref) and [`WeightDecay`](@ref)`.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Weight decay (`λ == lambda`): Controls the strength of ``L_2`` regularisation.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
- Keyword `couple`: If `true`, the weight decay is coupled with the learning rate, as in pytorch's AdamW.
                    This corresponds to an update of the form `x = x - η * (dx + λ * x)`, where `dx` is the
                    update from Adam with learning rate 1.
                    If `false`, the weight decay is decoupled from the learning rate, in the spirit of the original paper.
                    This corresponds to an update of the form `x = x - η * dx - λ * x`.
                    Default is `true`.

!!! warning "Breaking change in v0.4"
    With version 0.4 the default update rule for AdamW has changed to match the pytorch implementation.
    The previous rule, which is closer to the original paper, can be obtained by setting `AdamW(..., couple=false)`.
    See [this issue](https://github.com/FluxML/Flux.jl/issues/2433) for more details.
"""
struct AdamW{Teta,Tbeta<:Tuple,Tlambda,Teps} <: AbstractRule
  eta::Teta
  beta::Tbeta
  lambda::Tlambda
  epsilon::Teps
  couple::Bool
end

function AdamW(η, β = (0.9, 0.999), λ = 0.0, ϵ = 1e-8; couple::Bool = true)
  η < 0 && throw(DomainError(η, "the learning rate cannot be negative"))
  return AdamW(float(η), β, float(λ), float(ϵ), couple)
end

AdamW(; eta = 0.001, beta = (0.9, 0.999), lambda= 0.0,  epsilon = 1e-8, kw...) =
  AdamW(eta, beta, lambda, epsilon; kw...)

init(o::AdamW, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta))

function apply!(o::AdamW, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ, λ = T(o.eta), T.(o.beta), _eps(T, o.epsilon), T(o.lambda)
  mt, vt, βt = state

  # standard Adam update with learning rate eta=1
  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  dx′ = @lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)

  # apply learning rate and weight decay
  if o.couple
    dx′′ = @lazy η * (dx′ + λ * x)
  else
    dx′′ = @lazy η * dx′ + λ * x
  end

  return (mt, vt, βt .* β), dx′′
end

function Base.show(io::IO, o::AdamW)
  print(io, "AdamW(eta=$(o.eta), beta=$(o.beta), lambda=$(o.lambda), epsilon=$(o.epsilon), couple=$(o.couple))")
end


"""
    AdaBelief(η = 0.001, β = (0.9, 0.999), ϵ = 1e-16)
    AdaBelief(; [eta, beta, epsilon])

The [AdaBelief](https://arxiv.org/abs/2010.07468) optimiser is a variant of the well-known
Adam optimiser.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple == beta`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                                  (no need to change default)
"""
@def struct AdaBelief <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
  epsilon = 1e-16
end

init(o::AdaBelief, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta))

function apply!(o::AdaBelief, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), _eps(T, o.epsilon)
  mt, st, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. st = β[2] * st + (1 - β[2]) * abs2(dx - mt) + ϵ
  dx′ = @lazy η * mt / (1 - βt[1]) / (sqrt(st / (1 - βt[2])) + ϵ)

  return (mt, st, βt .* β), dx′
end

"""
    Muon(η = 0.02, μ = 0.95, λ = 0; nesterov = true)
    Muon(; [eta, mu, lambda, nesterov])

[Muon](https://kellerjordan.github.io/posts/muon/) optimiser for matrix parameters,
following Keller Jordan's [reference implementation](https://github.com/KellerJordan/Muon).

Each step updates the momentum `m = μ * m + (1 - μ) * dx`, then orthogonalises it: `O` has
the same singular vectors as `m`, but singular values close to one. (With `nesterov = true`,
the default, this acts on `μ * m + (1 - μ) * dx` instead.) A parameter `x` of size `(r, c)`
then changes by `x .-= η .* (sqrt(max(1, r / c)) .* O .+ λ .* x)`.
For Flux's `Dense`, whose weight is `(out, in)` like that of PyTorch's `Linear`,
this is the same scaling as in the reference.

`O` comes from five steps of a quintic Newton-Schulz iteration, starting from `m / norm(m)`.
Its coefficients are tuned for speed rather than convergence: singular values of
`m / norm(m)` above about `0.003` end up between `0.68` and `1.2`, and smaller ones stay smaller.
This runs in the element type of `x`, while the reference uses `BFloat16`.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`μ == mu`): Decay of the exponential moving average `m` of the gradient,
                  with `0 ≤ μ < 1`.
- Weight decay (`λ == lambda`): Decoupled weight decay, multiplied by the learning rate,
                  as in the reference and in [`AdamW`](@ref).
- Keyword `nesterov`: If `true`, orthogonalise `μ * m + (1 - μ) * dx` (with the new `m`)
                  instead of `m`.

# Which parameters

Muon is meant for the hidden weight matrices of a network. Everything else should use a
standard rule such as [`AdamW`](@ref): biases and normalisation gains, and also the input
embedding and the output layer, although these are matrices too. As the shape of a parameter
does not tell which is which, `setup` throws an error on any array which is not a matrix,
rather than silently using another rule. This includes convolution kernels: the reference
flattens PyTorch's `(c_out, c_in, k...)` kernels to `(c_out, :)` matrices, but Flux stores
them as `(k..., c_in, c_out)`.

To use Muon for some matrices only, set up the whole model with another rule,
then replace the states of those matrices:

```jldoctest
julia> using Functors: fmap, fmap_with_path

julia> model = (embed = ones(2, 5), hidden = (weight = ones(2, 2), bias = zeros(2)), head = ones(5, 2));

julia> state = fmap_with_path(Optimisers.setup(AdamW(), model);
                              exclude = (kp, s) -> s isa Optimisers.Leaf) do kp, s
         x = getkeypath(model, kp)
         kp[1] == :hidden && ndims(x) == 2 ? Optimisers.setup(Muon(), x) : s
       end;

julia> fmap(s -> nameof(typeof(s.rule)), state; exclude = s -> s isa Optimisers.Leaf)
(embed = :AdamW, hidden = (weight = :Muon, bias = :AdamW), head = :AdamW)

julia> state.hidden.weight
Leaf(Muon(eta=0.02, mu=0.95, lambda=0.0, nesterov=true), [0.0 0.0; 0.0 0.0])
```
"""
struct Muon{Teta,Tmu,Tlambda} <: AbstractRule
  eta::Teta
  mu::Tmu
  lambda::Tlambda
  nesterov::Bool
end

function Muon(η, μ = 0.95, λ = 0.0; nesterov::Bool = true)
  η < 0 && throw(DomainError(η, "the learning rate cannot be negative"))
  0 <= μ < 1 || throw(DomainError(μ, "the momentum must satisfy 0 ≤ μ < 1"))
  return Muon(float(η), float(μ), float(λ), nesterov)
end

Muon(; eta = 0.02, mu = 0.95, lambda = 0.0, kw...) = Muon(eta, mu, lambda; kw...)

init(o::Muon, x::AbstractMatrix) = zero(x)
init(o::Muon, x::AbstractArray) = _muon_matrix_only(x)

function apply!(o::Muon, mom, x::AbstractMatrix{T}, dx) where T
  η, μ, λ = T(o.eta), T(o.mu), T(o.lambda)

  @.. mom = μ * mom + (1 - μ) * dx
  O = _newton_schulz5(o.nesterov ? (@. μ * mom + (1 - μ) * dx) : mom)
  γ = real(T)(sqrt(max(1, size(x, 1) / size(x, 2))))

  return mom, @lazy η * (γ * O + λ * x)
end
apply!(o::Muon, state, x::AbstractArray, dx) = _muon_matrix_only(x)

_muon_matrix_only(x) = throw(ArgumentError("Muon only accepts matrices, not $(summary(x)). " *
  "Use another rule such as AdamW for this parameter, as explained in the docstring of Muon."))

function Base.show(io::IO, o::Muon)
  print(io, "Muon(eta=$(o.eta), mu=$(o.mu), lambda=$(o.lambda), nesterov=$(o.nesterov))")
end

# Keller Jordan's quintic iteration, X ← a X + (b A + c A²) X with A = X X', from X = G / ‖G‖.
# The Gram matrix is taken on the shorter side, and the buffers are reused across steps.
function _newton_schulz5(G::AbstractMatrix{T}) where T
  a, b, c = real(T).((3.4445, -4.7750, 2.0315))
  X = similar(G)
  X .= G ./ (sqrt.(_sumabs2(G)) .+ _eps(T, 1e-7))  # reduces with dims to avoid a GPU readback
  tall = size(X, 1) > size(X, 2)
  k = min(size(X)...)
  A, B, Y = similar(X, k, k), similar(X, k, k), similar(X)
  for _ in 1:5
    tall ? mul!(A, X', X) : mul!(A, X, X')
    copyto!(B, A)
    mul!(B, A, A, c, b)
    copyto!(Y, X)
    tall ? mul!(Y, X, B, one(a), a) : mul!(Y, B, X, one(a), a)
    X, Y = Y, X
  end
  return X
end

_sumabs2(G) = sum(abs2, vec(G); dims = 1)
_sumabs2(G::AbstractArray{<:Union{Float16, Complex{Float16}}}) =
  sum(x -> abs2(convert(promote_type(eltype(G), Float32), x)), vec(G); dims = 1)  # avoids overflow and underflow

"""
    WeightDecay(λ = 5e-4)
    WeightDecay(; [lambda])

Implements ``L_2`` regularisation, also known as ridge regression, 
when composed  with other rules as the first transformation in an [`OptimiserChain`](@ref).

It does this by adding `λ .* x` to the gradient. This is equivalent to adding 
`λ/2 * sum(abs2, x) == λ/2 * norm(x)^2` to the loss.

See also [`SignDecay`] for ``L_1`` normalisation.

# Parameters
- Penalty (`λ ≥ 0`): Controls the strength of the regularisation.
"""
@def struct WeightDecay <: AbstractRule
  lambda = 5e-4
end

init(o::WeightDecay, x::AbstractArray) = nothing

function apply!(o::WeightDecay, state, x::AbstractArray{T}, dx) where T
  λ = T(o.lambda)
  dx′ = @lazy dx + λ * x

  return state, dx′
end

function adjust(r::WeightDecay; gamma = nothing, kw...)
   if isnothing(gamma)
     return _adjust(r, NamedTuple(kw))
   else
     Base.depwarn("The strength of WeightDecay is now field :lambda, not :gamma", :adjust, force=true)
     nt = (; lambda = gamma, NamedTuple(kw)...)
     return _adjust(r, nt)
   end
 end

"""
    SignDecay(λ = 1e-3)
    SignDecay(; [lambda])

Implements ``L_1`` regularisation, also known as LASSO regression,
when composed  with other rules as the first transformation in an [`OptimiserChain`](@ref).

It does this by adding `λ .* sign(x)` to the gradient. This is equivalent to adding 
`λ * sum(abs, x) == λ * norm(x, 1)` to the loss.

See also [`WeightDecay`] for ``L_2`` normalisation.
They can be used together: `OptimiserChain(SignDecay(0.012), WeightDecay(0.034), Adam())`
is equivalent to adding `0.012 * norm(x, 1) + 0.017 * norm(x, 2)^2` to the loss function.

# Parameters
- Penalty (`λ ≥ 0`): Controls the strength of the regularisation.
"""
@def struct SignDecay <: AbstractRule
  lambda = 1e-3
end

init(o::SignDecay, x::AbstractArray) = nothing

function apply!(o::SignDecay, state, x::AbstractArray{T}, dx) where T
  λ = T(o.lambda)
  dx′ = @lazy dx + λ * sign(x)

  return state, dx′
end


"""
    ClipGrad(δ = 10)
    ClipGrad(; [delta])

Restricts every gradient component to obey `-δ ≤ dx[i] ≤ δ`.

Typically composed with other rules using [`OptimiserChain`](@ref).

See also [`ClipNorm`](@ref).
"""
@def struct ClipGrad <: AbstractRule
  delta = 10.0
end

init(o::ClipGrad, x::AbstractArray) = nothing

function apply!(o::ClipGrad, state, x::AbstractArray{T}, dx) where T
  δ = T(o.delta)
  dx′ = @lazy clamp(dx, -δ, δ)

  return state, dx′
end

"""
    ClipNorm(ω = 10, p = 2; throw = true)

Scales any gradient array for which `norm(dx, p) > ω`
to stay at this threshold (unless `p==0`).

Throws an error if the norm is infinite or `NaN`,
which you can turn off with `throw = false`.

Typically composed with other rules using [`OptimiserChain`](@ref).

See also [`ClipGrad`](@ref).
"""
struct ClipNorm{To,Tp} <: AbstractRule
  omega::To
  p::Tp
  throw::Bool
end
ClipNorm(ω = 10, p = 2; throw::Bool = true) = ClipNorm(float(ω), float(p), throw)

init(o::ClipNorm, x::AbstractArray) = nothing

function apply!(o::ClipNorm, state, x::AbstractArray{T}, dx) where T
  nrm = _norm(dx, o.p)
  if o.throw && !isfinite(nrm)
    throw(DomainError("gradient has $(o.p)-norm $nrm, for array $(summary(x))"))
  end
  λ = T(min(o.omega / nrm, 1))

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
(Leaf(OptimiserChain(ClipGrad(1.0), Descent(0.1)), (nothing, nothing)),)

julia> Optimisers.update(s, m, ([0.3, 1, 7],))[2]  # clips before discounting
([-0.03, -0.1, -0.1],)
```
"""
struct OptimiserChain{O<:Tuple} <: AbstractRule
  opts::O
end

function OptimiserChain(opts...)
  any(opt -> opt isa MixedPrecision, opts) && throw(ArgumentError("MixedPrecision optimisers should wrap the entire OptimiserChain, not be inside it."))
  return OptimiserChain(opts)
end

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

_adjust_rule(f, kp, ℓ::OptimiserChain, args, nt) =
  OptimiserChain(map(opt -> _adjust_rule(f, kp, opt, args, nt), ℓ.opts)...)


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
(x = Float32[-0.651], y = Float32[-20.202002])
```
"""
struct AccumGrad{N} <: AbstractRule
  n::N
end

function AccumGrad(n::Integer)
  n > 0 || throw(ArgumentError("AccumGrad must accumulate at least one gradient"))
  return AccumGrad{typeof(n)}(n)
end

init(::AccumGrad{N}, x) where N = (zero(x), N(1))

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
    MixedPrecision([T = Float32,] opt)

An optimiser that wraps another optimiser `opt` in order to perform mixed precision
training [1]. 

The state of `MixedPrecision{T}` will contain a copy in precision `T` of any trainable parameter `x`, 
call it `xT`, as well as the internal state of `opt` also at precision `T`.
If `T` is not specified, it defaults to `Float32`.

Call `g` the gradient of `x`. Both `g` and `x` are typically in a precision lower than `T`
(e.g. `Float16`).

In the `update!(opt_state, x, g)` call, `opt` is used to update `xT` instead of `x`, 
then `x` is updated with the value of `xT`. 

# Reference

[1] Micikevicius et al. '17, "Mixed Precision Training", https://arxiv.org/abs/1710.03740 .

# Examples

```julia
x = rand(Float16, 2) # A trainable parameter in low precision

opt = MixedPrecision(Adam(1e-3)) # Equivalent to MixedPrecision(Float32, Adam(1e-3))
opt_state = Optimisers.setup(opt, x) # The state contains a copy of x in Float32 precision

g = rand(Float16, 2) # A gradient in low precision

# Accumulation is performed in high precision,
# then also the low precision x is synced
Optimisers.update!(opt_state, x, g)  
```
"""
struct MixedPrecision{T<:Number, O<:AbstractRule} <: AbstractRule
  rule::O
end

@functor MixedPrecision

MixedPrecision(rule::AbstractRule) = MixedPrecision{Float32, typeof(rule)}(rule)
MixedPrecision(T::Type, rule::AbstractRule) = MixedPrecision{T, typeof(rule)}(rule)

function init(o::MixedPrecision{T}, x::AbstractArray) where T
  xT = T.(x)
  return (xT, init(o.rule, xT))
end

function apply!(o::MixedPrecision{T}, state, x, dx) where T
  xT, st = state
  st′, dx′ = apply!(o.rule, st, xT, dx)
  xT = subtract!(xT, dx′)
  if maywrite(x)
    x .= xT
    dx′ = nothing
  else
    dx′ = eltype(x).(x .- xT)
  end
  return (xT, st′), dx′
end

adjust(o::MixedPrecision{T}, eta::Real) where T = MixedPrecision(T, adjust(o.rule, eta))
adjust(o::MixedPrecision{T}; kw...) where T = MixedPrecision(T, adjust(o.rule; kw...))
_adjust_rule(f, kp, o::MixedPrecision{T}, args, nt) where T =
  MixedPrecision(T, _adjust_rule(f, kp, o.rule, args, nt))
