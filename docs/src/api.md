```@meta
CollapsedDocStrings = true
```

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
Optimisers.Lion
Optimisers.Muon
```

In addition to the main course, you may wish to order some of these condiments:

```@docs
Optimisers.AccumGrad
Optimisers.ClipGrad
Optimisers.ClipNorm
Optimisers.MixedPrecision
Optimisers.OptimiserChain
Optimisers.SignDecay
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
Optimisers.make_reactant_compatible
```

Calling `Functors.@functor` on your model's layer types by default causes
these functions to recurse into all children, and ultimately optimise
all `isnumeric` leaf nodes.
To further restrict this by ignoring some fields of a layer type, define `trainable`:

```@docs
Optimisers.trainable
Optimisers.isnumeric
Optimisers.maywrite
```

Such restrictions are also obeyed by this function for flattening a model:

```@docs
Optimisers.destructure
Optimisers.Restructure
Optimisers.trainables
```

Code that only reads a model, such as printing it, can use these instead of `trainable` and `trainables`.
They find the same children and arrays, but return them in containers of `Any` or `AbstractArray`,
and compile nothing for a new layer type that uses the default `functor`, `trainable` and `isleaf`:

```@docs
Optimisers.nospecialize_trainable
Optimisers.nospecialize_trainables
```

## Rule Definition

```@docs
Optimisers.apply!
Optimisers.init
Optimisers.@..
Optimisers.@lazy
Optimisers.adjust(::AbstractRule, ::Real)
Optimisers.@def
```

## KeyPath

A `KeyPath` is a sequence of keys that can be used to access a value within a nested structure.
It is defined in Functors.jl and re-exported by Optimisers.jl here for convenience.

```@docs
Functors.KeyPath
Functors.haskeypath
Functors.getkeypath
Functors.setkeypath!
```
