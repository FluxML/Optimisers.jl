<img align="right" width="200px" src="https://github.com/FluxML/Optimisers.jl/raw/master/docs/src/assets/logo.png">

# Optimisers.jl

[![][docs-stable-img]][docs-stable-url]
[![][docs-dev-img]][docs-dev-url]
[![][action-img]][action-url]
[![][coverage-img]][coverage-url] 

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://fluxml.ai/Optimisers.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-dev-url]: https://fluxml.ai/Optimisers.jl/dev/

[action-img]: https://github.com/FluxML/Optimisers.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Optimisers.jl/actions

[coverage-img]: https://codecov.io/gh/FluxML/Optimisers.jl/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/gh/FluxML/Optimisers.jl

Optimisers.jl defines many standard gradient-based optimisation rules, and tools for applying them to deeply nested models.

This was written as the new training system for [Flux.jl](https://github.com/FluxML/Flux.jl) neural networks,
and also used by [Lux.jl](https://github.com/avik-pal/Lux.jl).
But it can be used separately on any array, or anything else understood by [Functors.jl](https://github.com/FluxML/Functors.jl).


> [!WARNING]
> With version 0.4 the default update rule for AdamW has changed to match the pytorch implementation.
> The previous rule, which is closer to the original paper, can be obtained by setting `AdamW(..., couple=false)`.
> See [this issue](https://github.com/FluxML/Flux.jl/issues/2433) for more details.

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
this state is a similarly nested tree. As is the gradient: if using Zygote, you must use the "explicit" style as shown,
not the "implicit" one with `Params`.

The function `destructure` collects all the trainable parameters into one vector,
and returns this along with a function to re-build a similar model:

```julia
vector, re = Optimisers.destructure(model)

model2 = re(2 .* vector)
```

[The documentation](https://fluxml.ai/Optimisers.jl/dev/) explains usage in more detail,
describes all the optimization rules, and shows how to define new ones.
