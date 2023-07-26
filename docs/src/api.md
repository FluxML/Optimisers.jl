
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
Optimisers.AccumGrad
Optimisers.ClipGrad
Optimisers.ClipNorm
Optimisers.MixedPrecision
Optimisers.OptimiserChain
Optimisers.WeightDecay
```

## Model Interface

```@docs
Optimisers.setup
Optimisers.update
Optimisers.update!
Optimisers.adjust!
Optimisers.adjust(::Any, ::Real)
Optimisers.freeze!
Optimisers.thaw!
```

Calling `Functors.@functor` on your model's layer types by default causes
these functions to recurse into all children, and ultimately optimise
all `isnumeric` leaf nodes.
To further restrict this by ignoring some fields of a layer type, define `trainable`:

```@docs
Optimisers.trainable
Optimisers.isnumeric
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
