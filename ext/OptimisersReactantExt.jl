module OptimisersReactantExt

import Optimisers
import Reactant

Optimisers._eps(T::Type{<:Reactant.TracedRNumber{<:AbstractFloat}}, e) = T(e)

end
