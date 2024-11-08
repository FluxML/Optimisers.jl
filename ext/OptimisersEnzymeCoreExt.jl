module OptimisersEnzymeCoreExt

import Optimisers: trainable, setup, update!, isnumeric, AbstractRule
import EnzymeCore: Duplicated, Const

using Functors: fmapstructure

println("loaded!")

trainable(x::Duplicated) = (; val = x.val)

"""
    setup(rule::AbstractRule, model_grad::Duplicated)

For use with Enzyme's Duplicated, this just calls `setup(rule, model_grad.val)`.
"""
setup(rule::AbstractRule, model_grad::Duplicated) = setup(rule, model_grad.val)

"""
    update!(opt_state, model_grad::Duplicated)

For use with Enzyme's `Duplicated`, which holds both a model/parameters
and the corresponding gradient.

# Example

```jldoctest
julia> using Optimisers, EnzymeCore

julia> x_dx = Duplicated(Float16[1,2,3], Float16[1,0,-4])
Duplicated{Vector{Float16}}(Float16[1.0, 2.0, 3.0], Float16[1.0, 0.0, -4.0])

julia> st = Optimisers.setup(Momentum(1/9), x_dx)  # acts only on x not on dx
Leaf(Momentum(0.111111, 0.9), Float16[0.0, 0.0, 0.0])

julia> Optimisers.update!(st, x_dx)  # mutates both arguments
model_grad.val = Float16[0.8887, 2.0, 3.445]

julia> x_dx
Duplicated{Vector{Float16}}(Float16[0.8887, 2.0, 3.445], Float16[1.0, 0.0, -4.0])

julia> st
Leaf(Momentum(0.111111, 0.9), Float16[0.1111, 0.0, -0.4443])
```
"""
function update!(opt_state, model_grad::Duplicated)
  _, _ = update!(opt_state, model_grad.val, _grad_or_nothing(model_grad))
  nothing
end

# This function strips the returned gradient to be Zygote-like:
_grad_or_nothing(dup::Duplicated) = fmapstructure(_grad_or_nothing, dup.dval; prune=nothing)
_grad_or_nothing(::Const) = nothing
_grad_or_nothing(x) = isnumeric(x) ? x : nothing

end
