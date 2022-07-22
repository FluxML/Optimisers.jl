
## Optimisation Rules

```@docs
Optimisers.Descent
Optimisers.Momentum
Optimisers.Nesterov
Optimisers.Rprop
Optimisers.RMSProp
Optimisers.Adam
Optimisers.RAdam
Optimisers.AdaMax
Optimisers.OAdam
Optimisers.AdaGrad
Optimisers.AdaDelta
Optimisers.AMSGrad
Optimisers.NAdam
Optimisers.AdamW
Optimisers.AdaBelief
```

In addition to the main course, you may wish to order some of these condiments:

```@docs
Optimisers.ClipGrad
Optimisers.ClipNorm
Optimisers.WeightDecay
Optimisers.OptimiserChain
```

## Model Interface

```@docs
Optimisers.setup
Optimisers.update
Optimisers.update!
Optimisers.adjust(::Any, ::Real)
```

Calling `Functors.@functor` on your model's layer types by default causes the
optimiser to act on all suitable fields. To restrict this, define `trainable`:

```@docs
Optimisers.trainable
```

Such restrictions are also obeyed by this function for flattening a model:

```@docs
Optimisers.destructure
Optimisers.Restructure
```

## Rule Definition

```@docs
Optimisers.apply!
Optimisers.init
Optimisers.@..
Optimisers.@lazy
Optimisers.adjust(::AbstractRule, ::Real)
```
