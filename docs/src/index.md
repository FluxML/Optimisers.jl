# Optimisers.jl

Optimisers.jl defines many standard gradient-based optimisation rules, and tools for applying them to deeply nested models.

This was written as the new training system for [Flux.jl](https://github.com/FluxML/Flux.jl) neural networks,
and also used by [Lux.jl](https://github.com/LuxDL/Lux.jl).
But it can be used separately on any array, or anything else understood by [Functors.jl](https://github.com/FluxML/Functors.jl).

## Installation

In the Julia REPL, type
```julia
]add Optimisers
```

or
```julia-repl
julia> import Pkg; Pkg.add("Optimisers")
```

## An optimisation rule

A new optimiser must overload two functions, [`apply!`](@ref Optimisers.apply!) and [`init`](@ref Optimisers.init).
These act on one array of parameters:

```julia
# Define a container to hold any optimiser specific parameters (if any):
struct DecayDescent <: Optimisers.AbstractRule
  eta::Float64
end

# Define an `apply!` rule which encodes how the gradients will be used to
# update the parameters:
function Optimisers.apply!(o::DecayDescent, state, x, x̄)
  T = eltype(x)
  newx̄ = T(o.eta / √state) .* x̄
  nextstate = state + 1
  return nextstate, newx̄
end

# Define the function which sets up the initial state (if any):
Optimisers.init(o::DecayDescent, x::AbstractArray) = 1
```

The parameters will be immediately updated to `x .- newx̄`, while `nextstate` is
caried to the next iteration.

Notice that the state is handled separately from the optimiser itself. This
is a key design principle and allows users to manage their own state explicitly.
It of course also makes it easier to store the state.

## Usage with [Flux.jl](https://github.com/FluxML/Flux.jl)

To apply such an optimiser to a whole model, [`setup`](@ref Optimisers.setup) builds a tree containing any initial
state for every trainable array. Then at each step, [`update`](@ref Optimisers.update) uses this and the gradient
to adjust the model:

```julia
using Flux, Metalhead, Zygote, Optimisers

model = Metalhead.ResNet(18) |> gpu  # define a model to train
image = rand(Float32, 224, 224, 3, 1) |> gpu;  # dummy data
@show sum(model(image));  # dummy loss function

rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
state_tree = Optimisers.setup(rule, model);  # initialise this optimiser's momentum etc.

∇model, _ = gradient(model, image) do m, x  # calculate the gradients
  sum(m(x))
end;

state_tree, model = Optimisers.update(state_tree, model, ∇model);
@show sum(model(image));  # reduced
```

Notice that a completely new instance of the model is returned. Internally, this
is handled by [Functors.jl](https://fluxml.ai/Functors.jl), where we do a walk over the
tree formed by the model and update the parameters using the gradients.

There is also [`Optimisers.update!`](@ref) which similarly returns a new model,
but is free to mutate arrays within the old one for efficiency.
(The method of `apply!` above is likewise free to mutate arrays within its state;
they are defensively copied when this rule is used with `update`.)
For `Adam()`, there are two momenta per parameter, thus `state` is about twice the size of `model`:

```julia
Base.summarysize(model) / 1024^2  # about 45MB
Base.summarysize(state) / 1024^2  # about 90MB
```

Optimisers.jl does not depend on any one automatic differentiation package,
but for now the most likely source of gradients is [Zygote.jl](https://fluxml.ai/Zygote.jl).
Note that `update` always wants the gradient from Zygote's "explicit" mode, as shown above.
This `∇model` is another tree structure, rather than the dictionary-like object from 
Zygote's "implicit" mode `gradient(() -> loss(...), Flux.params(model))` -- see 
[Zygote's documentation](https://fluxml.ai/Zygote.jl/dev/#Explicit-and-Implicit-Parameters-1) for more about this difference.


## Usage with [Lux.jl](https://github.com/avik-pal/Lux.jl)

The main design difference of Lux from Flux is that the tree of parameters is separate from
the layer structure. It is these parameters which `setup` and `update` need to know about.

Lux describes this separation of parameter storage from model description as "explicit" parameters.
Beware that it has nothing to do with Zygote's notion of "explicit" gradients.
(If the same model is written in Flux and Lux, `∇model` above and `∇params` below will be nearly
identical trees of nested `NamedTuple`s.)

```julia
using Lux, Boltz, Zygote, Optimisers

lux_model, params, lux_state = Boltz.resnet(:resnet18) |> gpu;  # define and initialise model
images = rand(Float32, 224, 224, 3, 4) |> gpu;  # batch of dummy data
y, lux_state = Lux.apply(lux_model, images, params, lux_state);  # run the model
@show sum(y);  # initial dummy loss

rule = Optimisers.Adam()
opt_state = Optimisers.setup(rule, params);  # optimiser state based on model parameters

(loss, lux_state), back = Zygote.pullback(params, images) do p, x
  y, st = Lux.apply(lux_model, x, p, lux_state)
  sum(y), st  # return both the loss, and the updated lux_state
end;
∇params, _ = back((one.(loss), nothing));  # gradient of only the loss, with respect to parameter tree
loss == sum(y)  # not yet changed

opt_state, params = Optimisers.update!(opt_state, params, ∇params);

y, lux_state = Lux.apply(lux_model, images, params, lux_state);
@show sum(y);  # now reduced
```

Besides the parameters stored in `params` and gradually optimised, any other model state
is stored in `lux_state`, and updated by `Lux.apply`. (In this example, BatchNorm has state.)
This is completely unrelated to Optimisers.jl's state, although designed in a similar spirit.

```julia
Base.summarysize(lux_model) / 1024   # just 2KB
Base.summarysize(params) / 1024^2    # about 45MB, same as Flux model
Base.summarysize(lux_state) / 1024   # 40KB
Base.summarysize(opt_state) / 1024^2 # about 90MB, with Adam
```

If you are certain there is no model state, then the gradient calculation can
be simplified to use `Zygote.gradient` instead of `Zygote.pullback`:

```julia
∇params, _ = gradient(params, images) do p, x
  y, _ = Lux.apply(lux_model, x, p, lux_state)  # discards new lux_state
  sum(y)
end;
```


## Non-`trainable` Parameters

Optimisers.jl uses [Functors.jl](https://fluxml.ai/Functors.jl) to walk the `struct`s
making up the model, for which they must be annotated `@functor Type`. 
By default optimisation will alter all [`isnumeric`](@ref Optimisers.isnumeric) arrays. 

If some arrays of a particular layer should not be treated this way,
you can define a method for [`trainable`](@ref Optimisers.trainable)

```julia
struct Layer{T}
  alpha::T
  beta::T
  length::Int
end
Layer(n::Int) = Layer(randn(n), zeros(n), n)

Functors.@functor Layer

# Both array fields will be, for example, moved to the GPU:
Functors.children(Layer(3))  # (alpha = [...], beta = [...], length)

Optimisers.trainable(x::Layer) = (; alpha = x.alpha)  # must be a subset of children

# Only the first field will be optimised:
st = Optimisers.setup(DecayDescent(0.1), Layer(3))
```

## Frozen Parameters

To temporarily prevent training from affecting some parameters,
use [freeze!](@ref Optimisers.freeze!) and `thaw!`.
They work by mutating all `Leaf`s of the state tree, or part of it.

```julia
using Flux, Optimisers

x = randn(Float32, 28, 28, 1, 1);
net = @autosize (size(x)...,) Chain(
  Conv((3, 3), 1 => 3, stride=2, bias=false), Flux.flatten, Dense(_ => 2, relu),
)
opt = Optimisers.setup(Optimisers.Momentum(), net);

net.layers[3] isa Dense  # now freeze this layer's parameters:
Optimisers.freeze!(opt.layers[3])
opt.layers[3].bias  # confirm: Leaf(Momentum(...), [0.0, 0.0], frozen = true)

Optimisers.update!(opt, net, gradient(m -> sum(m(x)), net)...);

net.layers[3].bias  # stil zero, and its momentum is too:

Optimisers.thaw!(opt)
opt.layers[3].bias  # Leaf(Momentum(...), [0.0, 0.0])
```

## Adjusting Hyperparameters

To change the learning rate during training, use [`adjust!`](@ref Optimisers.adjust!).
This works much like `freeze!` by mutating the state tree, or part of it,
without discarding the momenta. For the Flux model from just above:

```julia
Optimisers.adjust!(opt, 0.03)  # change η for the whole model...

Optimisers.adjust!(opt.layers[3], 0.04)  # ... or just for one layer.
```

To change other fields of the optimisation rule, it accepts keyword arguments:

```julia
Momentum |> fieldnames  # (:eta, :rho)

Optimisers.adjust!(opt, rho = 0.95)  # change ρ for the whole model.
```

## Tied Parameters

If the same array appears twice (or more) in the model, [Functors.jl](https://fluxml.ai/Functors.jl) should recognise this.
Within Optimisers.jl, `setup` will initialise once, and use the same `Leaf` for both parameters. 
Then `update` will accumulate the gradient from both, and the updated model returned will have the tie maintained.

```julia
using Flux, Optimisers

enc = Chain(Dense(40 => 20, tanh), Dense(20 => 10));
dec = Chain(Dense(enc[1].weight', true, tanh), Dense(enc[2].weight', true, tanh));
model = Chain(; enc, dec)

st = Optimisers.setup(Optimisers.Adam(), model);

st.layers.enc.layers[1].weight === st.layers.dec.layers[1].weight.parent  # true
```

This identification relies on `===`, and will work for ordinary `Array`s and `CuArray`s.
It will not at present work for `reshape`d arrays, nor for immutable arrays such as those
from StaticArrays.jl.


## Obtaining a flat parameter vector

Instead of a nested tree-like structure, sometimes is is convenient to have all the
parameters as one simple vector. Optimisers.jl contains a function [`destructure`](@ref Optimisers.destructure)
which creates this vector, and also creates way to re-build the original structure
with new parameters. Both flattening and re-building may be used within `gradient` calls.

An example with Flux's `model`:

```julia
using ForwardDiff  # an example of a package which only likes one array

model = Chain(  # much smaller model example, as ForwardDiff is a slow algorithm here
          Conv((3, 3), 3 => 5, pad=1, bias=false), 
          BatchNorm(5, relu), 
          Conv((3, 3), 5 => 3, stride=16),
        )
image = rand(Float32, 224, 224, 3, 1);
@show sum(model(image));

flat, re = destructure(model)
st = Optimisers.setup(rule, flat)  # state is just one Leaf now

∇flat = ForwardDiff.gradient(flat) do v
  m = re(v)      # rebuild a new object like model
  sum(m(image))  # call that as before
end

st, flat = Optimisers.update(st, flat, ∇flat)
@show sum(re(flat)(image));
```

Here `flat` contains only the 283 trainable parameters, while the non-trainable
ones are preserved inside `re`, an object of type `Restructure`.
When defining new layers, these can be specified if necessary by overloading [`trainable`](@ref Optimisers.trainable).
By default, all numeric arrays visible to [Functors.jl](https://github.com/FluxML/Functors.jl)
are assumed to contain trainable parameters.
Tied parameters (arrays appearing in different layers) are included only once in `flat`.

Lux stores only the trainable parameters in `params`.
This can also be flattened to a plain `Vector` in the same way:

```julia
params, lux_state = Lux.setup(Random.default_rng(), lux_model);

flat, re = destructure(params)

∇flat = ForwardDiff.gradient(flat) do v
  p = re(v)  # rebuild an object like params
  y, _ = Lux.apply(lux_model, images, p, lux_state)
  sum(y)
end
```

## Collecting all trainable parameters

Sometimes it is useful to collect all trainable parameters in a model,
similarly to what [`destructure`](@ref Optimisers.destructure) does but without
concatenating the arrays into a flat vector.
This is done by [`trainables`](@ref Optimisers.trainables), which returns a list of arrays:

```julia-repl
julia> using Flux, Optimisers

julia> model = Chain(Dense(2 => 3, tanh), BatchNorm(3), Dense(3 => 2));

julia> trainables(model)
6-element Vector{AbstractArray}:
 Float32[0.5756773 -0.1975264; 0.4723181 -0.7546912; -0.91631395 0.07392061]
 Float32[0.0, 0.0, 0.0]
 Float32[0.0, 0.0, 0.0]
 Float32[1.0, 1.0, 1.0]
 Float32[-0.8764882 0.40812716 0.1919528; -0.9123545 -0.4462516 0.6751252]
 Float32[0.0, 0.0]

julia> l2reg(model) = sum([sum(abs2, p) for p in trainables(model)]);

julia> g = gradient(l2reg, model)[1];
```
Notice that the `BatchNorm` layer has two trainable parameters, `γ` and `β`, which are included in the list, while the `μ ` and `σ²` buffers are not.

Sometimes one wants to iterate over all trainable parameters in a model and the corresponding parameters of a matched structure such a gradient or the moving average of the model. 
This can be done using `trainables(model, path=true)`. For instance, here is how to update the parameters
of a moving average model with the parameters of the model:

```julia
for (kp, p_avg) in trainables(model_avg, path=true)
    p = getkeypath(model, kp)  
    p_avg .= 0.99 .* p_avg .+ 0.01 .* p
end
```

## Incomplete or nothing gradients

If the gradient is not available for some parameters, or branches of the model, 
`update` will not take an optimisation step for those parameters.
This is the case when the gradient is `nothing` or a subtype of `ChainRules.AbstractZero`.

For stateful optimisers, skipping an update it is generaly not the same as updating with a zero gradient.
For example, in the case of Adam, the momentum and variance are updated even if the gradient is zero:

```julia-repl
julia> x = (a = ones(2), b = ones(2));
(a = [1.0, 1.0], b = [1.0, 1.0])

julia> opt_state = Optimisers.setup(Adam(0.1), x)
(a = Leaf(Adam(0.1, (0.9, 0.999), 1.0e-8), ([0.0, 0.0], [0.0, 0.0], (0.9, 0.999))), b = Leaf(Adam(0.1, (0.9, 0.999), 1.0e-8), ([0.0, 0.0], [0.0, 0.0], (0.9, 0.999))))

julia> g = (; a = ones(2), b = ones(2)); # First an update with a non-zero gradient to increase the momentum and variance

julia> Optimisers.update!(opt_state, x, g);

julia> opt_state # the state in `a` and `b` are the same
(a = Leaf(Adam(0.1, (0.9, 0.999), 1.0e-8), ([0.1, 0.1], [0.001, 0.001], (0.81, 0.998001))), b = Leaf(Adam(0.1, (0.9, 0.999), 1.0e-8), ([0.1, 0.1], [0.001, 0.001], (0.81, 0.998001))))

julia> g = (; a = zeros(2), b = nothing); # Now an update with a zero gradient for a and no gradient for b

julia> Optimisers.update!(opt_state, x, g);

julia> opt_state # the state in `a` and `b` differ
(a = Leaf(Adam(0.1, (0.9, 0.999), 1.0e-8), ([0.09, 0.09], [0.000999, 0.000999], (0.729, 0.997003))), b = Leaf(Adam(0.1, (0.9, 0.999), 1.0e-8), ([0.1, 0.1], [0.001, 0.001], (0.81, 0.998001))))
```

## Usage with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)

Enzyme.jl is a new automatic differentiation package, an alternative to Zygote.jl.
It likes to store the model and the gradient together, as an object `Duplicated(x, dx)`.

Optimisers.jl now has some methods to handle this:
* `update!(opt_state, Duplicated(model, grad))` uses the gradient to update both the model and the optimiser state, and
* `setup(::AbstractRule, ::Duplicated)` ignores the gradient and returns `setup(rule, model)`.
