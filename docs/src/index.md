# Optimisers.jl

## Defining an Optimiser

A new optimiser must overload two functions, `apply!` and `init`:

```julia
# Define a container to hold any optimiser specific parameters (if any):
struct DecayDescent{T}
  η::T
end

# Define an `apply!` rule which encodes how the gradients will be used to
# update the parameters:
function Optimisers.apply!(o::DecayDescent, state, x, x̄)
  newx̄ = (o.η / √state) .* x̄
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

To apply such an optimiser to a whole model, `setup` builds a tree containing any initial
state for every trainable array. Then at each step, `update` uses this and the gradient
to adjust the model:

```julia

using Flux, Metalhead, Optimisers

model = Metalhead.ResNet18() |> gpu  # define a model to train
image = rand(Float32, 224, 224, 3, 1) |> gpu;  # dummy data
@show sum(model(image));  # dummy loss function

rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
state = Optimisers.setup(rule, model);  # initialize this optimiser's momentum etc.

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

There is also `Optimisers.update!` which similarly returns a new model and new state,
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

lux_model, params, lux_state = Boltz.resnet(:resnet18) .|> gpu;  # define and initialise model
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

