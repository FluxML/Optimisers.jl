
"""
    Optimisers.adjust(tree, adj) -> tree

Alters the state `tree = setup(rule, model)` to change the parameters of the
optimisation rule, without destroying its stored state. Typically used mid-way
through training.
* To change just the learning rate, provide a number `adj::Real`.
* To change all parameters, provide a new rule `adj::AbstractRule`.
  (This will affect only leaves of the same type.)

# Example
```jldoctest
julia> m = (vec = rand(Float32, 2), fun = sin);

julia> st = Optimisers.setup(Nesterov(), m)  # stored momentum is initialised to zero
(vec = Leaf(Nesterov{Float32}(0.001, 0.9), Float32[0.0, 0.0]), fun = nothing)

julia> st, m = Optimisers.update(st, m, (vec = [16, 88], fun = nothing));  # with fake gradient

julia> st
(vec = Leaf(Nesterov{Float32}(0.001, 0.9), Float32[-0.016, -0.088]), fun = nothing)

julia> st = Optimisers.adjust(st, 0.123)  # change learning rate, stored momentum untouched
(vec = Leaf(Nesterov{Float32}(0.123, 0.9), Float32[-0.016, -0.088]), fun = nothing)

julia> Optimisers.adjust(st, Nesterov(0.10101, 0.90909))  # change both η and ρ
(vec = Leaf(Nesterov{Float64}(0.10101, 0.90909), Float32[-0.016, -0.088]), fun = nothing)

julia> Optimisers.adjust(st, Adam())  # this does nothing -- Adam stores two vectors
┌ Warning: adjust did not find any rules to act on!
└ @ Optimisers ~/.julia/dev/Optimisers/src/adjust.jl:33
(vec = Leaf(Nesterov{Float32}(0.123, 0.9), Float32[-0.016, -0.088]), fun = nothing)
```
"""
function adjust(tree, a)
  ok = Ref(false)
  newtree = _adjust(tree, a, ok)
  ok[] || @warn "adjust did not find any rules to act on!"
  newtree
end

# """
#     Optimisers.adjust(tree; kw...) -> tree
#
# I'm not sure we want this keyword story, or not yet?
# """
# adjust(tree; kw...) = adjust(tree, NamedTuple(kw))

_adjust(tree, a, ok::Ref) = map(st -> _adjust(st, a, ok), tree)
_adjust(::Nothing, a, ok::Ref) = nothing
function _adjust(ℓ::Leaf, a, ok::Ref)
  newrule, flag = adjust(ℓ.rule, a)
  ok[] |= flag
  Leaf(newrule, ℓ.state)
end

"""
    Optimisers.adjust(rule::RuleType, η::Real) -> rule, flag

Replaces the learning rate of the optimisation rule with the given number.
This method is only called by `adjust(tree, η)`, and if `RuleType` has a field
called `:eta` and the default constructor, then the standard definition will work.
It is only necessary to provide a method if your `struct` stores its learning rate
in a different field.

# Example
```jldoctest
julia> struct DecayDescent{T} <: Optimisers.AbstractRule  # as in the documentation
         eta::T
       end

julia> Optimisers.adjust(DecayDescent(0.1f0), 0.23)  # works automatically
(DecayDescent{Float32}(0.23f0), true)
```
"""
function adjust(r::T, η::Real) where T <: AbstractRule
  fs = fieldnames(T)
  if :eta in fs
    vals = map(fs) do field
      field == :eta ? η : getfield(r, field)
    end
    T(vals...), true  # relies on having the default constructor
  else
    r, false
  end
end

# adjust(r::AbstractRule, η::Real) = adjust(r, (eta = η,))
# function adjust(r::T, nt::NamedTuple) where T <: AbstractRule
#   fs = fieldnames(T)
#   if all(k -> k in fs, keys(nt))
#     vals = map(fs) do field
#       get(nt, field, getfield(r, field))
#     end
#     T(vals...), true  # relies on having the default constructor
#   else
#     r, false
#   end
# end

function adjust(oldr::AbstractRule, newr::AbstractRule)
  if typeof(newr).name.wrapper == typeof(oldr).name.wrapper
    newr, true
  else
    oldr, false
  end
end
