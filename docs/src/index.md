# Optimisers.jl

## Defining an optimisation rule

A new optimiser must overload two functions, [`apply!`](@ref) and [`init`](@ref).
These act on one array of parameters:

```julia
# Define a container to hold any optimiser specific parameters (if any):
struct DecayDescent{T} <: Optimisers.AbstractRule
  eta::T
end

# Define an `apply!` rule which encodes how the gradients will be used to
# update the parameters:
function Optimisers.apply!(o::DecayDescent, state, x, x̄)
  newx̄ = (o.eta / √state) .* x̄
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

To apply such an optimiser to a whole model, [`setup`](@ref) builds a tree containing any initial
state for every trainable array. Then at each step, [`update`](@ref) uses this and the gradient
to adjust the model:

```julia

using Flux, Metalhead, Optimisers

model = Metalhead.ResNet(18) |> gpu  # define a model to train
image = rand(Float32, 224, 224, 3, 1) |> gpu;  # dummy data
@show sum(model(image));  # dummy loss function

rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
state = Optimisers.setup(rule, model);  # initialise this optimiser's momentum etc.

∇model, _ = gradient(model, image) do m, x  # calculate the gradients
  sum(m(x))
end;

state, model = Optimisers.update(state, model, ∇model);
@show sum(model(image));

```

Notice that a completely new instance of the model is returned. Internally, this
is handled by [Functors.jl](https://fluxml.ai/Functors.jl), where we do a walk over the
tree formed by the model and update the parameters using the gradients.

Optimisers.jl does not depend on any one automatic differentiation package,
but for now the most likely source of gradients is [Zygote.jl](https://fluxml.ai/Zygote.jl).
Note that `update` always wants the gradient from Zygote's "explicit" mode, as shown above.
This `∇model` is another tree structure, rather than the dictionary-like object from 
Zygote's "implicit" mode `gradient(() -> loss(...), Flux.params(model))` -- see 
[Zygote's documentation](https://fluxml.ai/Zygote.jl/dev/#Explicit-and-Implicit-Parameters-1) for more about this difference.

There is also [`Optimisers.update!`](@ref) which similarly returns a new model and new state,
but is free to mutate arrays within the old one for efficiency.
The method of `apply!` you write is likewise free to mutate arrays within its state;
they are defensively copied when this rule is used with `update`.

## Usage with [Lux.jl](https://github.com/avik-pal/Lux.jl)

The main design difference of Lux is that the tree of parameters is separate from
the layer structure. It is these parameters which `setup` and `update` need to know about.

Lux describes this separation of parameter storage from model description as "explicit" parameters.
Beware that it has nothing to do with Zygote's notion of "explicit" gradients.
(If the same model is written in Flux and Lux, `∇model` above and `∇params` below will often be
identical trees of nested `NamedTuple`s.)

```julia

using Lux, Boltz, Zygote, Optimisers

lux_model, params, lux_state = Boltz.resnet(:resnet18) |> gpu;  # define and initialise model
images = rand(Float32, 224, 224, 3, 4) |> gpu;  # batch of dummy data
y, _ = Lux.apply(lux_model, images, params, lux_state);  # run the model
@show sum(y)  # initial dummy loss

rule = Optimisers.Adam()
opt_state = Optimisers.setup(rule, params);  # optimiser state based on model parameters

∇params, _ = gradient(params, images) do p, x  # gradient with respect to parameter tree
  y, _ = Lux.apply(lux_model, x, p, lux_state)
  sum(y)
end;

opt_state, params = Optimisers.update!(opt_state, params, ∇params);

y, _ = Lux.apply(lux_model, images, params, lux_state);
@show sum(y)

```

Besides the parameters stored in `params` and gradually optimised, any other model state
is stored in `lux_state`. For simplicity this example does not show how to propagate the 
updated `lux_state` to the next iteration, see Lux's documentation.

## Obtaining a flat parameter vector

Instead of a nested tree-like structure, sometimes is is convenient to have all the
parameters as one simple vector. Optimisers.jl contains a function [`destructure`](@ref)
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
ones are preserved inside `re`.
When defining new layers, these can be specified if necessary by overloading [`trainable`](@ref).
By default, all numeric arrays visible to [Functors.jl](https://github.com/FluxML/Functors.jl)
are assumed to contain trainable parameters.

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

