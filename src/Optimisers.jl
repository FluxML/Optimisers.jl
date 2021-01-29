module Optimisers

using Functors: functor, fmap, isleaf

include("interface.jl")
include("rules.jl")

export Descent, Momentum, Nesterov, RMSProp,
       ADAM, ADAGrad, AdaMax, ADADelta, AMSGrad,
       NADAM, RADAM, OADAM, AdaBelief,
       InvDecay, ExpDecay, WeightDecay

end # module
