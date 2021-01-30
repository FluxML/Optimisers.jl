"""
    Descent(η = 0.1)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.

# Examples
```julia
opt = Descent()

opt = Descent(0.3)

ps = params(model)

gs = gradient(ps) do
    loss(x, y)
end

Flux.Optimise.update!(opt, ps, gs)
```
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

"""
    Momentum(η = 0.01, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.

# Examples
```julia
opt = Momentum()

opt = Momentum(0.01, 0.99)
```
"""
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

"""
    Nesterov(η = 0.001, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect dampening oscillations.

# Examples
```julia
opt = Nesterov()

opt = Nesterov(0.003, 0.95)
```
"""
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

"""
    RMSProp(η = 0.001, ρ = 0.9)

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.

# Examples
```julia
opt = RMSProp()

opt = RMSProp(0.002, 0.95)
```
"""
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

"""
    ADAM(η = 0.001, β::Tuple = (0.9, 0.999))

[ADAM](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = ADAM()

opt = ADAM(0.001, (0.9, 0.8))
```
"""
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

"""
    RADAM(η = 0.001, β::Tuple = (0.9, 0.999))

[Rectified ADAM](https://arxiv.org/abs/1908.03265) optimizer.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = RADAM()

opt = RADAM(0.001, (0.9, 0.8))
```
"""
struct RADAM{T,S}
  eta::T
  beta::S
end

init(o::RADAM, x::AbstractArray) = (zero(x), zero(x), [o.beta[1], o.beta[2]], 1)

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

"""
    AdaMax(η = 0.001, β::Tuple = (0.9, 0.999))

[AdaMax](https://arxiv.org/abs/1412.6980) is a variant of ADAM based on the ∞-norm.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = AdaMax()

opt = AdaMax(0.001, (0.9, 0.995))
```
"""
struct AdaMax{T,S}
  eta::T
  beta::S
end

init(o::AdaMax, x::AbstractArray) = (zero(x), zero(x), [o.beta[1], o.beta[2]])

function apply(o::AdaMax, x, Δ, st)
  η, β = o.eta, o.beta

  mt, ut, βp = st 

  mt = β[1] .* mt .+ (1 .- β[1]) .* Δ
  ut = max.(β[2] .* ut, abs.(Δ))
  Δ = (η ./ (1 .- βp[1])) .* mt ./ (ut .+ ϵ)
  βp = βp .* β

  return Δ, (mt, ut, βp)
end

"""
    OADAM(η = 0.0001, β::Tuple = (0.5, 0.9))

[OADAM](https://arxiv.org/abs/1711.00141) (Optimistic ADAM)
is a variant of ADAM adding an "optimistic" term suitable for adversarial training.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = OADAM()

opt = OADAM(0.001, (0.9, 0.995))
```
"""
struct OADAM{T,S}
  eta::T
  beta::S
end

init(o::OADAM, x::AbstractArray) = (zero(x), zero(x), zero(x), [o.beta[1], o.beta[2]])

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

"""
    ADAGrad(η = 0.1)

[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.

# Examples
```julia
opt = ADAGrad()

opt = ADAGrad(0.001)
```
"""
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

"""
    ADADelta(ρ = 0.9)

[ADADelta](https://arxiv.org/abs/1212.5701) is a version of ADAGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ`): Factor by which the gradient is decayed at each time step.

# Examples
```julia
opt = ADADelta()

opt = ADADelta(0.89)
```
"""
struct ADADelta{T}
  rho::T
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

"""
    AMSGrad(η = 0.001, β::Tuple = (0.9, 0.999))

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the ADAM
optimiser. Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = AMSGrad()

opt = AMSGrad(0.001, (0.89, 0.995))
```
"""
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

"""
    NADAM(η = 0.001, β::Tuple = (0.9, 0.999))

[NADAM](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) is a Nesterov variant of ADAM.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = NADAM()

opt = NADAM(0.002, (0.89, 0.995))
```
"""
struct NADAM{T,S}
  eta::T
  beta::S
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

"""
    AdaBelief(η = 0.001, β::Tuple = (0.9, 0.999))

The [AdaBelief](https://arxiv.org/abs/2010.07468) optimiser is a variant of the well-known
ADAM optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = AdaBelief()

opt = AdaBelief(0.001, (0.9, 0.8))
```
"""
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

"""
    InvDecay(γ = 0.001)

Apply inverse time decay to an optimiser, so that the effective step size at
iteration `n` is `eta / (1 + γ * n)` where `eta` is the initial step size.
The wrapped optimiser's step size is not modified.

# Examples
```julia
Optimiser(InvDecay(..), Opt(..))
```
"""
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

"""
    ExpDecay(η = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4)

Discount the learning rate `η` by the factor `decay` every `decay_step` steps till
a minimum of `clip`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- `decay`: Factor by which the learning rate is discounted.
- `decay_step`: Schedule decay operations by setting the number of steps between
                two decay operations.
- `clip`: Minimum value of learning rate.

# Examples
To apply exponential decay to an optimiser:
```julia
Optimiser(ExpDecay(..), Opt(..))

opt = Optimiser(ExpDecay(), ADAM())
```
"""
mutable struct ExpDecay{T,S}
  eta::T
  decay::T
  step::S
  clip::T
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

"""
    WeightDecay(wd = 0)

Decay weights by `wd`.

# Parameters
- Weight decay (`wd`)
"""
struct WeightDecay{T}
  wd::T
end

init(o::WeightDecay, x) = nothing

function apply(o::WeightDecay, x, Δ, st)
  wd = o.wd
  Δ = Δ + wd * x
  Δ, st
end
