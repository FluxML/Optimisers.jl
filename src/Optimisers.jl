module Optimisers

using Functors: functor, fmap, isleaf

include("interface.jl")
include("rules.jl")

export init, update!
export Descent, ADAM, Momentum, Nesterov, RMSProp,
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM, OADAM, AdaBelief,
       WeightDecay, Sequence

end # module
