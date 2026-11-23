module Optimisers

using Functors: Functors, functor, fmap, fmap_with_path,
                KeyPath, haskeypath, getkeypath,
                isleaf, @functor, fmapstructure, children, AbstractWalk
using LinearAlgebra
using ConstructionBase: ConstructionBase
using Compat: @compat

include("interface.jl")
export AbstractRule

include("utils.jl")

include("adjust.jl")

include("destructure.jl")
export destructure, destructure!

include("trainables.jl")
export trainables
export KeyPath, haskeypath, getkeypath # from Functors.jl

include("nospecialize.jl")

include("rules.jl")
export Descent, Adam, Momentum, Nesterov, Rprop, RMSProp,
       AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, AdamW, RAdam, OAdam, AdaBelief,
       WeightDecay, SignDecay, ClipGrad, ClipNorm, OptimiserChain, Lion, Muon,
       AccumGrad, MixedPrecision

"""
    Optimisers.make_reactant_compatible(rule::AbstractRule, [dev]) -> rule

Wrap `rule` so its scalar hyper-parameters (learning rate, momenta, step counters)
are stored as tracked numbers on the Reactant device `dev`, instead of being baked
into the compiled program as constants — letting e.g. the learning rate be changed
with [`adjust!`](@ref) without recompilation. `dev` must be a
`MLDataDevices.ReactantDevice`; it defaults to `MLDataDevices.reactant_device()`.

The wrapped rule is safe to `setup` either eagerly or under `@jit`, e.g.
`@jit Optimisers.setup(rule, ps)`. It is also idempotent: wrapping an
already-wrapped rule returns it unchanged.

[`setup`](@ref Optimisers.setup) applies this automatically when the model's
parameters live on a Reactant device, so calling it explicitly is usually unnecessary.

Methods are provided by the extension loaded when both Reactant.jl and
MLDataDevices.jl are present.
"""
function make_reactant_compatible end

@compat public apply!, init, setup, update, update!, make_reactant_compatible
@compat public adjust!, adjust, freeze!, thaw!
@compat public trainable, isnumeric, maywrite, Restructure
@compat public nospecialize_trainable, nospecialize_trainables
@compat public @.., @lazy, @def, opmath

###
### one-array functions
###

"""
    Optimisers.apply!(rule::RuleType, state, parameters, gradient) -> (state, gradient)

This defines the action of any optimisation rule. It should return the modified gradient
which will be subtracted from the parameters, and the updated state (if any) for use at
the next iteration, as a tuple `(state, gradient)`.

For efficiency it is free to mutate the old state, but only what is returned will be used.
Ideally this should check `maywrite(x)`, which the built-in rules do via [`@..`](@ref).

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
This and [`apply!`](@ref Optimisers.apply!) are the two functions which any new optimisation rule must define.

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

###
### whole-model functions
###

"""
    Optimisers.setup(rule, model) -> state_tree
    Optimisers.setup(f, model) -> state_tree

Initialises the given optimiser for every trainable parameter within the model.
Returns a tree of the relevant states, which must be passed to [`update`](@ref Optimisers.update)
or [`update!`](@ref Optimisers.update!).

With a function `f` in place of the `rule`, each trainable array `x` gets the rule `f(kp, x)`,
where `kp` is the [`KeyPath`](@ref) of `x` in the model, as in `trainables(model; path = true)`.
An array which appears twice in the model gets the rule of its first position.
This can also give different rules to different arrays, see [`Muon`](@ref Optimisers.Muon).

```jldoctest
julia> model = (layer = (weight = ones(3, 2), bias = zeros(3)), head = ones(1, 3));

julia> st = Optimisers.setup(model) do kp, x  # no weight decay on vectors, smaller η for the head
         AdamW(eta = kp[1] == :head ? 1e-4 : 1e-3, lambda = ndims(x) == 1 ? 0 : 0.1)
       end;

julia> fmap(ℓ -> (ℓ.rule.eta, ℓ.rule.lambda), st; exclude = ℓ -> ℓ isa Optimisers.Leaf)
(layer = (weight = (0.001, 0.1), bias = (0.001, 0.0)), head = (0.0001, 0.1))
```

When the model's parameters live on a Reactant device (with Reactant.jl and
MLDataDevices.jl loaded), `setup` automatically applies
[`make_reactant_compatible`](@ref Optimisers.make_reactant_compatible) to `rule`, so its
hyper-parameters are tracked on-device rather than baked into the compiled program as
constants. This is a no-op under `@jit`/tracing and for already-compatible rules.

# Example
```jldoctest
julia> m = (x = rand(3), y = (true, false), z = tanh);

julia> Optimisers.setup(Momentum(), m)  # same field names as m
(x = Leaf(Momentum(eta=0.01, rho=0.9), [0.0, 0.0, 0.0]), y = ((), ()), z = ())
```

The recursion into structures uses Functors.jl, which by default looks into the fields of any `struct`.
A type which should not be traversed can be marked with `Functors.@leaf`.
See [the Flux docs](https://fluxml.ai/Flux.jl/stable/models/advanced/) for more about this.

```jldoctest
julia> struct Layer; mat; fun; end

julia> model = (lay = Layer([1 2; 3 4f0], sin), vec = [5, 6f0]);

julia> Optimisers.setup(Momentum(), model)  # fields of a new struct are visited by default
(lay = (mat = Leaf(Momentum(eta=0.01, rho=0.9), Float32[0.0, 0.0, 0.0, …]), fun = ()), vec = Leaf(Momentum(eta=0.01, rho=0.9), Float32[0.0, 0.0]))

julia> destructure(model)
(Float32[1.0, 3.0, 2.0, 4.0, 5.0, 6.0], Restructure(NamedTuple, ..., 6))

julia> using Functors; @leaf Layer  # annotate this type as containing no parameters

julia> Optimisers.setup(Momentum(), model)
(lay = (), vec = Leaf(Momentum(eta=0.01, rho=0.9), Float32[0.0, 0.0]))

julia> destructure(model)
(Float32[5.0, 6.0], Restructure(NamedTuple, ..., 2))
```
"""
setup

"""
    Optimisers.update(tree, model, gradient) -> (tree, model)

Uses the optimiser and the gradient to change the trainable parameters in the model.
Returns the improved model, and the optimiser states needed for the next update.
The initial tree of states comes from [`setup`](@ref Optimisers.setup).

See also [`update!`](@ref Optimisers.update!), which will be faster for models of ordinary `Array`s or `CuArray`s.

# Example
```jldoctest
julia> m = (x = Float32[1,2,3], y = tanh);

julia> t = Optimisers.setup(Descent(0.1), m)
(x = Leaf(Descent(0.1), nothing), y = ())

julia> g = (x = [1,1,1], y = nothing);  # fake gradient

julia> Optimisers.update(t, m, g)
((x = Leaf(Descent(0.1), nothing), y = ()), (x = Float32[0.9, 1.9, 2.9], y = tanh))
```
"""
update

"""
    Optimisers.update!(tree, model, gradient) -> (tree, model)

Uses the optimiser and the gradient to change the trainable parameters in the model.
Returns the improved model, and the optimiser states needed for the next update.
The initial tree of states comes from [`setup`](@ref Optimisers.setup).

This is used in exactly the same manner as [`update`](@ref Optimisers.update), but because it may mutate
arrays within the old model (and the old state), it will be faster for models of ordinary
`Array`s or `CuArray`s. However, you should not rely on the old model being fully updated
but rather use the returned model.
(The original state tree is always mutated, as each `Leaf` is mutable.)

# Example

```jldoctest
julia> using StaticArrays, Zygote, Optimisers

julia> m = (x = [1f0, 2f0], y = SA[4f0, 5f0]);  # partly mutable model

julia> t = Optimisers.setup(Momentum(1/30, 0.9), m)  # tree of states
(x = Leaf(Momentum(eta=0.03333333333333333, rho=0.9), Float32[0.0, 0.0]), y = Leaf(Momentum(eta=0.03333333333333333, rho=0.9), Float32[0.0, 0.0]))

julia> g = gradient(m -> sum(abs2.(m.x .+ m.y)), m)[1]  # structural gradient
(x = Float32[10.0, 14.0], y = Float32[10.0, 14.0])

julia> t2, m2 = Optimisers.update!(t, m, g);

julia> m2  # after update or update!, this is the new model
(x = Float32[0.6666666, 1.5333333], y = Float32[3.6666667, 4.5333333])

julia> m2.x === m.x  # update! has re-used this array, for efficiency
true

julia> m  # original should be discarded, may be mutated but no guarantee
(x = Float32[0.6666666, 1.5333333], y = Float32[4.0, 5.0])

julia> t == t2  # original state tree is guaranteed to be mutated
true
```
"""
update!

end # module
