
"""
    Optimisers.adjust(tree, η) -> tree

Alters the state `tree = setup(rule, model)` to change the parameters of the
optimisation rule, without destroying its stored state. Typically used mid-way
through training.

To change just the learning rate, provide a number `η::Real`.

# Example
```jldoctest
julia> m = (vec = rand(Float32, 2), fun = sin);

julia> st = Optimisers.setup(Nesterov(), m)  # stored momentum is initialised to zero
(vec = Leaf(Nesterov{Float32}(0.001, 0.9), Float32[0.0, 0.0]), fun = nothing)

julia> st, m = Optimisers.update(st, m, (vec = [16, 88], fun = nothing));  # with fake gradient

julia> st
(vec = Leaf(Nesterov{Float32}(0.001, 0.9), Float32[-0.016, -0.088]), fun = nothing)

julia> st = Optimisers.adjust(st, 0.123)  # change learning rate, stored momentum untouched
(vec = Leaf(Nesterov{Float32}(0.123, 0.9), Float32[-0.016, -0.088]), fun = nothing)
```

To change other parameters, `adjust` also accepts keyword arguments matching the field
names of the optimisation rule's type.

```
julia> fieldnames(Adam)
(:eta, :beta, :epsilon)

julia> st2 = Optimisers.setup(OptimiserChain(ClipGrad(), Adam()), m)
(vec = Leaf(OptimiserChain(ClipGrad{Float32}(10.0), Adam{Float32}(0.001, (0.9, 0.999), 1.19209f-7)), [nothing, (Float32[0.0, 0.0], Float32[0.0, 0.0], (0.9, 0.999))]), fun = nothing)

julia> Optimisers.adjust(st2; beta = (0.777, 0.909), delta = 11.1)  # delta acts on ClipGrad
(vec = Leaf(OptimiserChain(ClipGrad{Float32}(11.1), Adam{Float32}(0.001, (0.777, 0.909), 1.19209f-7)), [nothing, (Float32[0.0, 0.0], Float32[0.0, 0.0], (0.9, 0.999))]), fun = nothing)

julia> Optimisers.adjust(st; beta = "no such field")  # silently ignored!
(vec = Leaf(Nesterov{Float32}(0.001, 0.9), Float32[-0.016, -0.088]), fun = nothing)
```
"""
adjust(tree, eta::Real) = map(st -> adjust(st, eta), tree)
adjust(tree; kw...) = map(st -> adjust(st; kw...), tree)

adjust(::Nothing, ::Real) = nothing
adjust(::Nothing; kw...) = nothing

adjust(ℓ::Leaf, eta::Real) = ℓ.frozen ? ℓ : Leaf(adjust(ℓ.rule, eta), ℓ.state, ℓ.frozen)
adjust(ℓ::Leaf; kw...) = ℓ.frozen ? ℓ : Leaf(adjust(ℓ.rule; kw...), ℓ.state, ℓ.frozen)


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
  T(vals...)  # relies on having the default constructor
end

