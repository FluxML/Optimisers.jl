module OptimisersReactantMLDataDevicesExt

using Optimisers: Optimisers
using Reactant: Reactant
using MLDataDevices: ReactantDevice, get_device, with_track_numbers, reactant_device

# --- device-preserving eltype conversion ---------------------------------
# On-device numbers must be rebuilt through their concrete constructor so they
# stay tracked; `convert(T, x)` / `T(x)` would pull them back to the host as
# compile-time constants. (Reproduces Lux's `Utils.convert_eltype`.)
# The eager path hits the `Concrete*` methods (keeping the scalar on-device). Under
# `@jit setup` the scalars are `TracedRNumber`s AND `T === eltype(x) === TracedRNumber{…}`,
# so the `x::Number` method runs `convert(TracedRNumber{…}, x::TracedRNumber)`, which
# routes through the `TracedRNumber{…}` constructor and stays traced. (Do NOT add a
# `TracedRNumber` method here: with `T` already a `TracedRNumber` type it would try to
# build the invalid `TracedRNumber{TracedRNumber{…}}`.)
_convert_eltype(::Type{T}, x::Number) where {T<:Number} = convert(T, x)
_convert_eltype(::Type{T}, x::Reactant.ConcretePJRTNumber) where {T<:Number} =
    Reactant.ConcretePJRTNumber{T}(x)
_convert_eltype(::Type{T}, x::Reactant.ConcreteIFRTNumber) where {T<:Number} =
    Reactant.ConcreteIFRTNumber{T}(x)

# --- the wrapper ----------------------------------------------------------
# Wrapping a rule in `ReactantOptimiser` lets us override `init`/`_adjust` so the
# learning rate, momenta and step counters are stored/updated as tracked numbers
# rather than being frozen into the compiled program as constants.
struct ReactantOptimiser{T} <: Optimisers.AbstractRule
    opt::T
end

function Base.show(io::IO, opt::ReactantOptimiser)
    print(io, "ReactantOptimiser(", opt.opt, ")")
    return nothing
end

Optimisers.apply!(opt::ReactantOptimiser, state, x, y) =
    Optimisers.apply!(opt.opt, state, x, y)

Optimisers.init(opt::ReactantOptimiser, ps) = Optimisers.init(opt.opt, ps)

# Beta-carrying rules: the stock inits use `T.(o.beta)`, which materialises the
# tracked betas to host constants; swap in the device-preserving conversion.
for common_opt in (:Adam, :AdaMax, :NAdam, :AdamW, :AdaBelief)
    @eval function Optimisers.init(
        opt::ReactantOptimiser{<:Optimisers.$(common_opt)}, x::AbstractArray{T}
    ) where {T}
        return zero(x), zero(x), _convert_eltype.((T,), opt.opt.beta)
    end
end

function Optimisers.init(
    opt::ReactantOptimiser{<:Optimisers.RAdam}, x::AbstractArray{T}
) where {T}
    betas = _convert_eltype.((T,), opt.opt.beta)
    # The step counter must stay tracked so it flows through a compiled `update!` as a
    # runtime variable (rather than being frozen as a constant, forcing a recompile per
    # step). Derive a tracked `1` from an already-tracked scalar (the promoted `beta`)
    # instead of `with_track_numbers(get_device(opt), Integer)(1)`: `get_device` errors
    # inside a trace, and it was the only thing stopping `@jit Optimisers.setup(opt, ps)`
    # working for RAdam. `+ 1` (not `+ one(T)`: under a trace `T === TracedRNumber{…}`)
    # promotes correctly for both concrete and traced scalars. A tracked `Float` counter
    # is numerically identical to the stock integer one — RAdam's `apply!` only uses `t`
    # in float arithmetic (`2t·βt`) and as `t + 1`.
    t = betas[1] - betas[1] + 1
    return zero(x), zero(x), betas, t
end

function Optimisers.init(
    opt::ReactantOptimiser{<:Optimisers.OAdam}, x::AbstractArray{T}
) where {T}
    return zero(x), zero(x), _convert_eltype.((T,), opt.opt.beta), zero(x)
end
# (AccumGrad needs no init override: the stock `init(::AccumGrad{N}, x) = (zero(x), N(1))`
#  already tracks the counter when `N` is the tracked-integer type.)

# --- adjust ---------------------------------------------------------------
# Move the adjustment NamedTuple onto the device (as tracked numbers) BEFORE
# delegating to the inner rule, so the new value stays traced, then re-wrap.
# Mirrors `Lux.DistributedOptimizer`'s `_adjust` idiom.
function Optimisers._adjust(opt::ReactantOptimiser, nt::NamedTuple)
    dev = with_track_numbers(get_device(opt), AbstractFloat)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, dev(nt)))
end

function Optimisers._adjust(
    opt::ReactantOptimiser{<:Optimisers.AccumGrad}, nt::NamedTuple
)
    dev = with_track_numbers(get_device(opt), Integer)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, dev(nt)))
end

# --- construction helpers -------------------------------------------------
# Device-less form: default to the current Reactant device. Dispatches to the
# device-aware methods below, so `OptimiserChain` / `AccumGrad` / `ClipNorm` (all
# `<: AbstractRule`) and the idempotency guard are all reached correctly.
Optimisers.make_reactant_compatible(opt::Optimisers.AbstractRule) =
    Optimisers.make_reactant_compatible(opt, reactant_device())

Optimisers.make_reactant_compatible(opt::Optimisers.AbstractRule, dev::ReactantDevice) =
    ReactantOptimiser(with_track_numbers(dev, AbstractFloat)(opt))

# Idempotent: re-wrapping an already-wrapped rule is a no-op. Guards against double
# wrapping (a rule passed through `make_reactant_compatible` twice, or an
# `OptimiserChain` that already holds wrapped rules).
Optimisers.make_reactant_compatible(opt::ReactantOptimiser, ::ReactantDevice) = opt

Optimisers.make_reactant_compatible(opt::Optimisers.OptimiserChain, dev::ReactantDevice) =
    ReactantOptimiser(Optimisers.OptimiserChain(
        Optimisers.make_reactant_compatible.(opt.opts, (dev,))...))

Optimisers.make_reactant_compatible(opt::Optimisers.AccumGrad, dev::ReactantDevice) =
    ReactantOptimiser(with_track_numbers(dev, Integer)(opt))

# `throw` forced to false: raising a host-side error is illegal under tracing.
Optimisers.make_reactant_compatible(opt::Optimisers.ClipNorm, dev::ReactantDevice) =
    ReactantOptimiser(Optimisers.ClipNorm(
        with_track_numbers(dev, Integer)(opt.omega), opt.p, false))

end # module
