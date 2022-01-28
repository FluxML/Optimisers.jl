# Optimisers.jl

## Define an Optimiser

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

## Usage

To apply such an optimiser to a whole model, `setup` builds a tree containing any initial
state for every trainable array. Then at each step, `update` uses this and the gradient
to adjust the model:

```julia

using Flux, Metalhead, Optimisers

o = Optimisers.ADAM() # define an ADAM optimiser with default settings
st = Optimisers.setup(o, m)  # initialize the optimiser before using it

model = ResNet18() # define a model to train on
ip = rand(Float32, 224, 224, 3, 1) # dummy data

m̄, _ = gradient(model, ip) do m, x # calculate the gradients
  sum(m(x)) # dummy loss function
end

st, mnew = Optimisers.update(st, m, m̄)

```

Notice that a completely new instance of the model is returned. Internally, this
is handled by [Functors.jl](https://fluxml.ai/Functors.jl), where we do a walk over the
tree formed by the model and update the parameters using the gradients. Optimisers can
work with different forms of gradients, but most likely use case are the gradients as
returned by [Zygote.jl](https://fluxml.ai/Zygote.jl).

There is also `Optimisers.update!` which similarly returns a new model and new state,
but is free to mutate arrays within the old one for efficiency.
The method of `apply!` you write is likewise free to mutate arrays within its state;
they are defensively copied when this rule is used with `update`.
