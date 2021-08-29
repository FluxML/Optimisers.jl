module Optimisers

using Functors
using Functors: functor, fmap, isleaf

include("interface.jl")
include("rules.jl")

export Descent, ADAM, Momentum, Nesterov, RMSProp,
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM, OADAM, AdaBelief,
       WeightDecay, OptimiserChain

end # module
