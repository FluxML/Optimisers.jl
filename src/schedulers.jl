using Base.Iterators: Cycle, Zip, Stateful,
                      Repeated, Take, TakeWhile

# f(t, p) = (2 / π) * abs(asin(sin(π * (t - 1) / p)))
# sine(t, p = 1.) = (2 / π) * abs(asin(sin(π * (t - 1) / p)))

# Simple scheduling can happen as a basic closure.
triangle(t) = (1 - 2 * abs(round(Int, t/2) - t/2))

const BaseIterators = Union{Cycle, Zip,
                            Stateful, Repeated,
                            Take, TakeWhile}

mutable struct Schedule{O,F}
  f::F
  opt::O
  cursor::Float32
end

"""
    Schedule(f, opt)

Create a scheduled optimiser which can update the optimiser fields defined by `f`.

`f` can be any callable, while `opt` can be any optimiser.

See also [next](@ref), [init](@ref)
"""
function Schedule(f, opt)
  Schedule(f, opt, 1.f0 + eps(Float32))
end

# Schedule(v::AbstractVector, opt) = Schedule(Iterators.cycle(v), opt)

# Timestep: t - always depends on run
# Phase: p - mostly likely constant

# What we want to "schedule" - everything (so will have to let people mess with things)
# Most likely learning rate
# If `f` relies on only time - it can be generated on the fly

# st = (t, timestep)

check_empty(x) = x
check_empty(::Nothing) = (nothing, nothing)


"""
   next(itr, st)

Returns the next value of the scheduling function and its updated state

For general iterators, this amounts to a call to [Base.iterate](@ref).
"""
function next(itr::T, st) where T <: Union{AbstractVector{<:Real}, BaseIterators}
  ft, itrst = st
  @show st
  res = iterate(itr, itrst)
  check_empty(res) 
end

next(f, (t,ts)) = f(t .+ ts), (t .+ ts, ts)

getUnionAll(t) = Base.typename(t).wrapper
next_opt(o::T; kwargs...) where T = getUnionAll(T)(o, kwargs...)

"""
    next(s::Schedule, state)

Returns a new optimiser as described by the scheduler function
and the optimiser. This allocates a new optimiser on the stack, keeping the original one intact.

The state is a tuple of the state as defined by the scheduling policy.
"""

# just need to handle what happens if an iteration runs out (returns nothing)
# In that case just switch to the next thing in the vector `s.f`
# it could also be a Sequence(::Vector)
function next(s::Schedule{O}, st) where O
  tnew, schedst = next(s.f, st)
  ADAM(s.opt, eta = tnew * s.opt.eta), (tnew, schedst) #replace with O(..)
end

# function next(v::AbstractVector, st)
#   o = s.opt
#   for f in v
#     s_ = Schedule(f, o)
#     st_ = init(s_)
#     next(s_, st_) 
#   end
# end

init(f) = (1, 1)
init(itr::T) where T <: BaseIterators = iterate(itr)
init(s::Schedule, x) = (init(s.f, x), init(s.opt, x))
init(s::Schedule{O, <: AbstractVector}, x) where O = init(Schedule(s.f[1], s.opt), x)
 
function apply(s::Schedule, x, dx, st)
  schedst, optst = st
  o, schedst2 = next(s, schedst)
  @show o
  Δ, optst2 = apply(o, x, dx, optst)
  Δ, (schedst2, optst2)
end

struct InvDecay{T}
  decay::T
end

# (start, step) -> consider step to be a function/ vector of step sizes?
(inv::InvDecay)(t) = 1 / (1 + inv.decay * t)

InvDecay(s = 0.1f0) = InvDecay{typeof(s)}(s)

sine(p) = t -> sin(Float32(π) * t / p)

