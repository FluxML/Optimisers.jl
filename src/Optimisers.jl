module Optimisers

using Functors: functor, fmap, isleaf
using LinearAlgebra

include("interface.jl")
include("rules.jl")

export Descent, ADAM, Momentum, Nesterov, RMSProp,
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM, OADAM, AdaBelief,
       WeightDecay, ClipGrad, ClipNorm, OptimiserChain

end # module
