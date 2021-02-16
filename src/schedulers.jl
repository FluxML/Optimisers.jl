f(t, p) = (2 / π) * abs(asin(sin(π * (t - 1) / p)))
sine(t, p = 1.) = (2 / π) * abs(asin(sin(π * (t - 1) / p)))
sine(p) = t -> (2 / Float32(π)) * abs(asin(sin(Float32(π) * (t - 1) / p)))

mutable struct Schedule{O,F}
  f::F
  opt::O
  cursor::Float32
  # cursor_step::Float64
end

function Schedule(f, opt)
  Schedule(f, opt, 1.f0 + eps(Float32)) 
end


# Timestep: t - always depends on run
# Phase: p - mostly likely constant

# What we want to "schedule" - everything (so will have to let people mess with things)
# Most likely learning rate
# If `f` relies on only time - it can be generated on the fly

function next(s::Schedule{O}, (cursor, cursor_step)) where O
  cursor += cursor_step
  ADAM(s.opt, eta = s.f(cursor) * s.opt.eta) #replace with O(..)
end

init(f, x) = (1.f0, 0.1f0)
init(s::Schedule, x) = (init(s.f, x), init(s.opt, x))

function apply(s::Schedule, x, dx, st)
  schedst, optst = st
  o = next(s, schedst)
  # @show o
  Δ, optst2 = apply(o, x, dx, optst)
  Δ, ((cursor .+ cursor_step, cursor_step), optst2)
end

struct InvDecay{T}
  decay::T
end

init(inv::InvDecay, x) = (1, 1)
(inv::InvDecay)(t) = 1 / (1 + inv.decay * t)

InvDecay(s = 0.1f0) = InvDecay{typeof(s)}(s)
