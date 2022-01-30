module Optimisers

using Functors: functor, fmap, isleaf
using LinearAlgebra
using Base: tail

include("interface.jl")
export AbstractRule

include("destructure.jl")
export destructure

include("rules.jl")
export Descent, Adam, Momentum, Nesterov, RMSProp,
       AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, AdamW, RAdam, OAdam, AdaBelief,
       WeightDecay, ClipGrad, ClipNorm, OptimiserChain

import Functors: functor
# This is https://github.com/FluxML/Functors.jl/pull/33, put here just to simplify trying it out.
begin
  using LinearAlgebra
  # The reason for these is to let W and W' be seen as tied weights in Flux models.
  # Can't treat ReshapedArray very well, as its type doesn't include enough details for reconstruction.

  functor(::Type{<:Adjoint}, x) = (parent = _adjoint(x),), y -> adjoint(only(y))

  _adjoint(x) = adjoint(x)  # _adjoint is the inverse, and also understands more types:
  _adjoint(x::NamedTuple{(:parent,)}) = x.parent  # "structural" gradient, and lazy broadcast used by Optimisers:
  _adjoint(bc::Broadcast.Broadcasted{S}) where S = Broadcast.Broadcasted{S}(_conjugate(bc.f, adjoint), _adjoint.(bc.args))

  functor(::Type{<:Transpose}, x) = (parent = _transpose(x),), y -> transpose(only(y))

  _transpose(x) = transpose(x)
  _transpose(x::NamedTuple{(:parent,)}) = x.parent
  _transpose(bc::Broadcast.Broadcasted{S}) where S = Broadcast.Broadcasted{S}(_conjugate(bc.f, transpose), _transpose.(bc.args))

  _conjugate(f::F, ::typeof(identity)) where F = f
  _conjugate(f::F, op::Union{typeof(transpose), typeof(adjoint)}) where F = (xs...,) -> op(f(op.(xs)...))
end

"""
    Optimisers.apply!(rule::RuleType, state, parameters, gradient) -> (state, gradient)

This defines the action of any optimisation rule. It should return the modified gradient
which will be subtracted from the parameters, and the updated state (if any) for use at
the next iteration, as a tuple `(state, gradient)`.

For efficiency it is free to mutate the old state, but only what is returned will be used.
Ideally this should check `iswriteable(x)`, which the built-in rules do via [`@..`](@ref).

The initial state is `init(rule::RuleType, parameters)`.

# Example
```jldoctest
julia> Optimisers.init(Descent(0.1), Float32[1,2,3]) === nothing
true

julia> Optimisers.apply!(Descent(0.1), nothing, Float32[1,2,3], [4,5,6])
(nothing, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}}(*, ([4, 5, 6], 0.1f0)))
```
"""
apply!

"""
    Optimisers.init(rule::RuleType, parameters) -> state

Sets up the initial state for a given optimisation rule, and an array of parameters.
This and [`apply!`](@ref) are the two functions which any new optimisation rule must define.

# Examples
```jldoctest
julia> Optimisers.init(Descent(), Float32[1,2,3])  # is `nothing`

julia> Optimisers.init(Momentum(), [1.0, 2.0])
2-element Vector{Float64}:
 0.0
 0.0
```
"""
init

"""
    Optimisers.setup(rule, model) -> tree

Initialises the given optimiser for every trainable parameter within the model.
Returns a tree of the relevant states, which must be passed to [`update`](@ref)
or [`update!`](@ref).

# Example
```jldoctest
julia> m = (x = rand(3), y = (true, false), z = tanh);

julia> Optimisers.setup(Momentum(), m)  # same field names as m
(x = Leaf(Momentum{Float32}(0.01, 0.9), [0.0, 0.0, 0.0]), y = (nothing, nothing), z = nothing)
```

The recursion into structures uses Functors.jl, and any new `struct`s containing parameters
need to be marked with `Functors.@functor` before use.
See [the Flux docs](https://fluxml.ai/Flux.jl/stable/models/advanced/) for more about this.

```
julia> struct Layer; mat; fun; end

julia> model = (lay = Layer([1 2; 3 4f0], sin), vec = [5, 6f0]);

julia> Optimisers.setup(Momentum(), model)  # new struct is by default ignored
(lay = nothing, vec = Leaf(Momentum{Float32}(0.01, 0.9), [0.0, 0.0]))

julia> destructure(model)
(Float32[5.0, 6.0], Restructure(NamedTuple, ..., 2))

julia> using Functors; @functor Layer  # annotate this type as containing parameters

julia> Optimisers.setup(Momentum(), model)
(lay = (mat = Leaf(Momentum{Float32}(0.01, 0.9), [0.0 0.0; 0.0 0.0]), fun = nothing), vec = Leaf(Momentum{Float32}(0.01, 0.9), [0.0, 0.0]))

julia> destructure(model)
(Float32[1.0, 3.0, 2.0, 4.0, 5.0, 6.0], Restructure(NamedTuple, ..., 6))
```
"""
setup

"""
    Optimisers.update(tree, model, gradient) -> (tree, model)

Uses the optimiser and the gradient to change the trainable parameters in the model.
Returns the improved model, and the optimiser states needed for the next update.
The initial tree of states comes from [`setup`](@ref).

See also [`update!`](@ref), which will be faster for models of ordinary `Array`s or `CuArray`s.

# Example
```jldoctest
julia> m = (x = Float32[1,2,3], y = tanh);

julia> t = Optimisers.setup(Descent(0.1f0), m)
(x = Leaf(Descent{Float32}(0.1), nothing), y = nothing)

julia> g = (x = [1,1,1], y = nothing);  # fake gradient

julia> Optimisers.update(t, m, g)
((x = Leaf(Descent{Float32}(0.1), nothing), y = nothing), (x = Float32[0.9, 1.9, 2.9], y = tanh))
```
"""
update

"""
    Optimisers.update!(tree, model, gradient) -> (tree, model)

Uses the optimiser and the gradient to change the trainable parameters in the model.
Returns the improved model, and the optimiser states needed for the next update.
The initial tree of states comes from [`setup`](@ref).

This is used in exactly the same manner as [`update`](@ref), but because it may mutate
arrays within the old model (and the old state), it will be faster for models of ordinary
`Array`s or `CuArray`s. However, you should not rely on the old model being fully updated.
"""
update!

end # module
