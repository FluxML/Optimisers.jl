###
### freezing
###

"""
    Optimisers.freeze!(tree)

Temporarily alters the state `tree = setup(rule, model)` so that parameters
will not be updated. Un-done by [`thaw!`](@ref Optimisers.thaw!).

Can be applied to the state corresponding to only part of a model,
for instance with `model::Chain`, to freeze `model.layers[1]` you
should call `freeze!(tree.layers[1])`.

# Example
```jldoctest
julia> m = (x = ([1.0], 2.0), y = [3.0]);

julia> s = Optimisers.setup(Momentum(), m);

julia> Optimisers.freeze!(s.x)

julia> Optimisers.update!(s, m, (x = ([pi], 10pi), y = [100pi]));  # with fake gradient

julia> m
(x = ([1.0], 2.0), y = [-0.14159265358979312])

julia> s
(x = (Leaf(Momentum(0.01, 0.9), [0.0], frozen = true), ()), y = Leaf(Momentum(0.01, 0.9), [3.14159]))

julia> Optimisers.thaw!(s)

julia> s.x
(Leaf(Momentum(0.01, 0.9), [0.0]), ())
```
"""
freeze!(tree) = foreach(freeze!, tree)
freeze!(ℓ::Leaf) = (ℓ.frozen = true; nothing)

"""
    Optimisers.thaw!(tree)

The reverse of [`freeze!`](@ref Optimisers.freeze!). Applies to all parameters,
mutating every `Leaf(rule, state, frozen = true)` to `Leaf(rule, state, frozen = false)`.
"""
thaw!(tree) = foreach(thaw!, tree)
thaw!(ℓ::Leaf) = (ℓ.frozen = false; nothing)

freeze!(::Union{Number, AbstractArray{<:Number}}) = throw(ArgumentError(
  "`freeze!` must not be applied to a model, only to the state tree from `setup`"))
thaw!(::Union{Number, AbstractArray{<:Number}}) = throw(ArgumentError(
  "`thaw!` must not be applied to a model, only to the state tree from `setup`"))

###
### adjust
###

"""
    Optimisers.adjust!(tree, η)

Alters the state `tree = setup(rule, model)` to change the parameters of the
optimisation rule, without destroying its stored state. Typically used mid-way
through training.

Can be applied to part of a model, by acting only on the corresponding part
of the state `tree`.

To change just the learning rate, provide a number `η::Real`.

# Example
```jldoctest adjust
julia> m = (vec = rand(Float32, 2), fun = sin);

julia> st = Optimisers.setup(Nesterov(), m)  # stored momentum is initialised to zero
(vec = Leaf(Nesterov(0.001, 0.9), Float32[0.0, 0.0]), fun = ())

julia> st, m = Optimisers.update(st, m, (vec = [16, 88], fun = nothing));  # with fake gradient

julia> st
(vec = Leaf(Nesterov(0.001, 0.9), Float32[-0.016, -0.088]), fun = ())

julia> Optimisers.adjust!(st, 0.123)  # change learning rate, stored momentum untouched

julia> st
(vec = Leaf(Nesterov(0.123, 0.9), Float32[-0.016, -0.088]), fun = ())
```

To change other parameters, `adjust!` also accepts keyword arguments matching the field
names of the optimisation rule's type.

```jldoctest adjust
julia> fieldnames(Adam)
(:eta, :beta, :epsilon)

julia> st2 = Optimisers.setup(OptimiserChain(ClipGrad(), Adam()), m)
(vec = Leaf(OptimiserChain(ClipGrad(10.0), Adam(0.001, (0.9, 0.999), 1.0e-8)), (nothing, (Float32[0.0, 0.0], Float32[0.0, 0.0], (0.9, 0.999)))), fun = ())

julia> Optimisers.adjust(st2; beta = (0.777, 0.909), delta = 11.1)  # delta acts on ClipGrad
(vec = Leaf(OptimiserChain(ClipGrad(11.1), Adam(0.001, (0.777, 0.909), 1.0e-8)), (nothing, (Float32[0.0, 0.0], Float32[0.0, 0.0], (0.9, 0.999)))), fun = ())

julia> Optimisers.adjust(st; beta = "no such field")  # silently ignored!
(vec = Leaf(Nesterov(0.123, 0.9), Float32[-0.016, -0.088]), fun = ())
```
"""
adjust!(tree, eta::Real) = foreach(st -> adjust!(st, eta), tree)
adjust!(tree; kw...) = foreach(st -> adjust!(st; kw...), tree)

adjust!(ℓ::Leaf, eta::Real) = (ℓ.rule = adjust(ℓ.rule, eta); nothing)
adjust!(ℓ::Leaf; kw...) = (ℓ.rule = adjust(ℓ.rule; kw...); nothing)

adjust(ℓ::Leaf, eta::Real) = Leaf(adjust(ℓ.rule, eta), ℓ.state, ℓ.frozen)
adjust(ℓ::Leaf; kw...) = Leaf(adjust(ℓ.rule; kw...), ℓ.state, ℓ.frozen)

"""
    adjust(tree, η) -> tree

Like [`adjust!`](@ref Optimisers.adjust), but returns a new tree instead of mutating the old one.
"""
function adjust(tree, eta::Real)
  t′ = fmap(copy, tree; exclude = maywrite)  # same as used for update / update!
  adjust!(t′, eta)
  t′
end
function adjust(tree; kw...)
  t′ = fmap(copy, tree; exclude = maywrite)
  adjust!(t′; kw...)
  t′
end

"""
    Optimisers.adjust(rule::RuleType, η::Real) -> rule

If a new optimisation rule has a learning rate which is *not* stored in field `rule.eta`,
then you may should add a method to `adjust`. (But simpler to just use the standard name.)
"""
adjust(r::AbstractRule, eta::Real) = _adjust(r, (; eta))
adjust(r::AbstractRule; kw...) = _adjust(r, NamedTuple(kw))

function _adjust(r::T, nt::NamedTuple) where T <: AbstractRule
  isempty(nt) && throw(ArgumentError("adjust must be given something to act on!"))
  fs = fieldnames(T)
  vals = map(fs) do field
    get(nt, field, getfield(r, field))
  end
  return ConstructionBase.constructorof(T)(vals...)
end
