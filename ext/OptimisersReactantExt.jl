module OptimisersReactantExt

using Optimisers: Optimisers
using Reactant: TracedRNumber

# Once https://github.com/EnzymeAD/Reactant.jl/pull/835 we can support throwing errors
# from compiled MLIR
@inline function Optimisers._assert_positive_eta(eta, ::TracedRNumber{Bool})
    return
end

end
