
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

julia> st2 = Optimisers.setup(Adam(), m)
(vec = Leaf(Adam{Float32}(0.001, (0.9, 0.999), 1.19209f-7), (Float32[0.0, 0.0], Float32[0.0, 0.0], (0.9, 0.999))), fun = nothing)

julia> Optimisers.adjust(st2; beta = (0.777, 0.909))
(vec = Leaf(Adam{Float32}(0.001, (0.777, 0.909), 1.19209f-7), (Float32[0.0, 0.0], Float32[0.0, 0.0], (0.9, 0.999))), fun = nothing)

julia> Optimisers.adjust(st; beta = "no such field")  # silently ignored!
(vec = Leaf(Nesterov{Float32}(0.001, 0.9), Float32[-0.016, -0.088]), fun = nothing)
```
"""
adjust(tree, eta::Real) = map(st -> adjust(st, eta), tree)
adjust(tree; kw...) = map(st -> adjust(st; kw...), tree)

adjust(::Nothing, ::Real) = nothing
adjust(::Nothing; kw...) = nothing

adjust(ℓ::Leaf, eta::Real) = Leaf(adjust(ℓ.rule, eta), ℓ.state)
adjust(ℓ::Leaf; kw...) = Leaf(adjust(ℓ.rule; kw...), ℓ.state)


"""
    Optimisers.adjust(rule::RuleType, η::Real) -> rule

Replaces the learning rate of the optimisation rule with the given number.

This method is only called by `adjust(tree, η)`, and if `RuleType` has a field
called `:eta` and the default constructor, then the standard definition will work.

It should only be necessary to provide a method if your custom `struct` stores its
learning rate a field with a different name. Or if an inner constructor blocks the
default behaviour.

# Example
```jldoctest
julia> Optimisers.adjust(Adam(), 0.12345)
(Adam{Float32}(0.12345f0, (0.9f0, 0.999f0), 1.1920929f-7), true)

julia> struct DecayDescent{T} <: Optimisers.AbstractRule  # as in the documentation
         eta::T
       end

julia> Optimisers.adjust(DecayDescent(0.1f0), 0.2345)  # works automatically
DecayDescent{Float32}(0.2345f0)

julia> Optimisers.adjust(ClipNorm(), 0.345)  # does nothing, as this has no learning rate
ClipNorm{Float32}(10.0f0, 2.0f0, true)
```
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

