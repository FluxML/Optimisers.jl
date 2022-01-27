# Optimisers.jl

## Define an Optimiser

```julia
# Define a container to hold any optimiser specific parameters (if any)
struct Descent{T}
  η::T
end

# Define an `apply` rule with which to update the current params
# using the gradients
function Optimisers.apply(o::Descent, state, m, m̄)
    o.η .* m̄, state
end

Optimisers.init(o, x::AbstractArray) = nothing
```

Notice that the state is handled separately from the optimiser itself. This
is a key design principle and allows users to manage their own state explicitly.

It of course also makes it easier to store the state.

## Usage

```julia

using Flux, Metalhead, Optimisers

o = Optimisers.ADAM() # define an ADAM optimiser with default settings
st = Optimisers.state(o, m)  # initialize the optimiser before using it

model = ResNet() # define a model to train on
ip = rand(Float32, 224, 224, 3, 1) # dummy data

m̄, _ = gradient(model, ip) do m, x # calculate the gradients
  sum(m(x))
end


st, mnew = Optimisers.update(o, st, m, m̄)

# or

st, mnew = o(m, m̄, st)
```

Notice that a completely new instance of the model is returned. Internally, this
is handled by [Functors.jl](https://fluxml.ai/Functors.jl), where we do a walk over the
tree formed by the model and update the parameters using the gradients. Optimisers can
work with different forms of gradients, but most likely use case are the gradients as
returned by [Zygote.jl](https://fluxml.ai/Zygote.jl).

