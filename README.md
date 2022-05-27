# Optimisers.jl

<!-- [![][docs-stable-img]][docs-stable-url] -->
[![][docs-dev-img]][docs-dev-url]
[![][action-img]][action-url]
[![][coverage-img]][coverage-url] 

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://fluxml.ai/Optimisers.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://fluxml.ai/Optimisers.jl/dev/

[action-img]: https://github.com/FluxML/Optimisers.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Optimisers.jl/actions

[coverage-img]: https://codecov.io/gh/FluxML/Optimisers.jl/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/gh/FluxML/Optimisers.jl

Optimisers.jl defines many standard gradient-based optimisation rules, and tools for applying them to deeply nested models.

This is the future of training for [Flux.jl](https://github.com/FluxML/Flux.jl) neural networks,
and the present for [Lux.jl](https://github.com/avik-pal/Lux.jl).
But it can be used separately on anything understood by [Functors.jl](https://github.com/FluxML/Functors.jl).

## Installation

```julia
] add Optimisers
```

## Usage

The core idea is that optimiser state (such as momentum) is explicitly handled.
It is initialised by `setup`, and then at each step, `update` returns both the new
state, and the model with its trainable parameters adjusted:

```julia
state = Optimisers.setup(Optimisers.Adam(), model)  # just once

grad = Zygote.gradient(m -> loss(m(x), y), model)[1]

state, model = Optimisers.update(state, model, grad)  # at every step
```

For models with deeply nested layers containing the parameters (like [Flux.jl](https://github.com/FluxML/Flux.jl) models),
this state is a similarly nested tree. As is the gradient: if using Zygote, this must be done in "explicit" mode,
not the "implicit" mode with `Params`.

The function `destructure` collects all the trainable parameters into one vector,
and returns this along with a function to re-build a similar model:

```julia
vector, re = Optimisers.destructure(model)

model2 = re(2 .* vector)
```

[The documentation](https://fluxml.ai/Optimisers.jl/dev/) explains usage in more detail,
describes all the optimization rules, and shows how to define new ones.
