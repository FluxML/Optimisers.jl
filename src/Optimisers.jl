module Optimisers

using Functors: functor, fmap, isleaf

include("interface.jl")
include("rules.jl")

export init, update!
export Descent, Momentum, Nesterov, RMSProp,
       ADAM

end # module
