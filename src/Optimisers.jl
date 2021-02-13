module Optimisers

using Functors: functor, fmap, isleaf
using ArrayInterface

include("interface.jl")
include("rules.jl")
include("mutating.jl")

export Descent, ADAM, Momentum, Nesterov, RMSProp,
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM, OADAM, AdaBelief,
       WeightDecay, OptimiserChain

end # module
