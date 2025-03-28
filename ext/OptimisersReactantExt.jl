module OptimisersReactantExt

using Optimisers: Optimisers
using Reactant: TracedRNumber, @trace

# Once https://github.com/EnzymeAD/Reactant.jl/pull/835 we can support throwing errors
# from compiled MLIR
@inline function Optimisers._assert_positive_eta(eta, ::TracedRNumber{Bool})
    return
end

function Optimisers.apply!(
    o::Optimisers.AccumGrad{<:TracedRNumber{<:Integer}}, state, x, dx
)
    accum_dx, counter = state
    @trace if counter == 1
        @. accum_dx = dx / o.n
    else
        @. accum_dx = accum_dx .+ dx / o.n
    end
    @trace if counter == o.n
        dx_final = accum_dx
        counter = 1
    else
        dx_final = zero.(dx)
        counter += 1
    end
    return (accum_dx, counter), dx_final
end

end
