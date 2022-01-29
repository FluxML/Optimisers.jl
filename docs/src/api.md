
## Optimisation Rules

```@docs
Optimisers.Descent
Optimisers.Momentum
Optimisers.Nesterov
Optimisers.RMSProp
Optimisers.ADAM
Optimisers.RADAM
Optimisers.AdaMax
Optimisers.OADAM
Optimisers.ADAGrad
Optimisers.ADADelta
Optimisers.AMSGrad
Optimisers.NADAM
Optimisers.ADAMW
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
Optimisers.destructure
```

Calling `Functors.@functor` on your model's layer types by default causes the
optimiser to act on all suitable fields. To restrict this, define `trainable`:

```@docs
Optimisers.trainable
```

## Rule Definition

```@docs
Optimisers.apply!
Optimisers.init
Optimisers.@..
```
