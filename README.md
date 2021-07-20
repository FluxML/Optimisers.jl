# Optimisers.jl

[![][docs-stable-img]][docs-stable-url]
[![][docs-dev-img]][docs-dev-url]
[![][action-img]][action-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://fluxml.ai/Optimisers.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://fluxml.ai/Optimisers.jl/dev/

[action-img]: https://github.com/FluxML/Optimisers.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Optimisers.jl/actions

Optimisers.jl defines many standard optimisers and utilities for learning loops.

The API for defining an optimiser, and using it is simple.

## Define an Optimiser

```julia
# Define a container to hold any optimiser specific parameters
# it's not a strict requirement - just a convention
struct Descent{T}
  \eta::T
end

# Define an `apply` rule with which to update the current params
# using the gradients
function Optimisers.apply(o::Descent, st, m, m\bar)
  m .- o.\eta .* m\bar, st
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

model = ResNet().layers # define a model to train on
ip = rand(Float32, 224, 224, 3, 1) # dummy data

m\bar, _ = gradient(model, ip) do m, x # calculate the gradients
  sum(m(x))
end


st, mnew = Optimisers.update(o, st, m, m\bar)

# or

st, mnew = o(m, m\bar, st)
```

Notice that a completely new instance of the model is returned. Internally, this
is handled by [Functors.jl](https://fluxml.ai/Functors.jl), where we do a walk over the
tree formed by the model and update the parameters using the gradients. Optimisers can
work with different forms of gradients, but most likely use case are the gradients as
returned by [Zygote.jl](https://fluxml.ai/Zygote.jl).
