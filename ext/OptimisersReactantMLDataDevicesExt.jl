module OptimisersReactantMLDataDevicesExt

using Optimisers: Optimisers
using Reactant: Reactant
using MLDataDevices: ReactantDevice, get_device, get_device_type, with_track_numbers,
                     reactant_device

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

# `adjust` keeps the eltype of a tracked number, as it keeps the type of a host one.
Optimisers._keeptype(old::Reactant.ConcretePJRTNumber{T}, new::Number) where {T} =
    _convert_eltype(T, new)
Optimisers._keeptype(old::Reactant.ConcreteIFRTNumber{T}, new::Number) where {T} =
    _convert_eltype(T, new)

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
    # runtime variable (rather than being frozen as a constant). Derive a tracked `1`
    # from an already-tracked scalar (the promoted `beta`) instead of
    # `with_track_numbers(get_device(opt), Integer)(1)`: `get_device` errors inside a
    # trace, which would break `@jit Optimisers.setup(opt, ps)`. A tracked `Float`
    # counter is numerically identical to the stock integer one — RAdam's `apply!` only
    # uses `t` in float arithmetic (`2t·βt`) and as `t + 1`.
    return zero(x), zero(x), betas, _one_like(betas[1])
end

# A tracked `1` on the same device as `x`. Arithmetic on a concrete number returns a host
# number, and `one(x)` would land on the default device.
_one_like(x::Number) = one(x)
_one_like(x::Reactant.ConcretePJRTNumber{T}) where {T} = Reactant.ConcretePJRTNumber{T}(
    one(T); client = Reactant.XLA.client(x), device = Reactant.XLA.device(x), x.sharding)
_one_like(x::Reactant.ConcreteIFRTNumber{T}) where {T} = Reactant.ConcreteIFRTNumber{T}(
    one(T); client = Reactant.XLA.client(x), device = Reactant.XLA.device(x), x.sharding)

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
# Mirrors `Lux.DistributedOptimizer`'s `_adjust` idiom. Delegating to the inner rule's
# own `adjust` (not `_adjust`) keeps its checks and aliases, e.g. `RMSProp`'s `centred`.
function Optimisers._adjust(opt::ReactantOptimiser, nt::NamedTuple)
    dev = get_device(opt)
    # A rule holding no tracked number, e.g. an integer `Descent(1)`, keeps the new values
    # on the host too.
    dev isa ReactantDevice || return ReactantOptimiser(Optimisers.adjust(opt.opt; nt...))
    nt′ = with_track_numbers(dev, AbstractFloat)(nt)
    return ReactantOptimiser(Optimisers.adjust(opt.opt; nt′...))
end

# Containers hold individually wrapped rules, and each of those moves the values itself.
const WrappedContainer = Union{Optimisers.OptimiserChain, Optimisers.MixedPrecision}

Optimisers._adjust(opt::ReactantOptimiser{<:WrappedContainer}, nt::NamedTuple) =
    ReactantOptimiser(Optimisers.adjust(opt.opt; nt...))

function Optimisers._adjust(
    opt::ReactantOptimiser{<:Optimisers.AccumGrad}, nt::NamedTuple
)
    dev = with_track_numbers(get_device(opt), Integer)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, dev(nt)))
end

# `adjust!(tree, filter; ...)`: the filter sees the rule inside the wrapper, while the new
# values still go through the wrapper to stay tracked.
Optimisers._adjust_rule(f, kp, opt::ReactantOptimiser{<:WrappedContainer}, args, nt) =
    ReactantOptimiser(Optimisers._adjust_rule(f, kp, opt.opt, args, nt))

Optimisers._adjust_rule(f, kp, opt::ReactantOptimiser, args, nt) =
    f(kp, opt.opt) ? Optimisers.adjust(opt, args...; nt...) : opt

# A rule's own `adjust(rule, η)` method, for a learning rate not stored in field `eta`,
# must still be called, and with a host `η::Real`. Then the fields it replaced are put
# back on the device: only those, as some fields stay on the host, e.g. `ClipNorm`'s `p`.
function Optimisers.adjust(opt::ReactantOptimiser, eta::Real)
    new = Optimisers.adjust(opt.opt, eta)
    dev = get_device(opt)
    dev isa ReactantDevice || return ReactantOptimiser(new)
    track = with_track_numbers(dev, AbstractFloat)
    tracked(x) = x isa Reactant.ConcreteRNumber
    isleaf(x) = tracked(x) || Optimisers.Functors.isleaf(x)
    new′ = Optimisers.fmap(opt.opt, new; exclude = isleaf, cache = nothing) do old, x
        tracked(old) && !tracked(x) ? track(Optimisers._keeptype(old, x)) : x
    end
    return ReactantOptimiser(new′)
end

Optimisers.adjust(opt::ReactantOptimiser{<:WrappedContainer}, eta::Real) =
    ReactantOptimiser(Optimisers.adjust(opt.opt, eta))

# --- one copy of the tracked numbers per leaf -----------------------------
# A step compiled while leaves share tracked numbers reads one of them for all leaves,
# and writes it back to all, undoing a later `adjust!` of only some leaves.
Optimisers._unshare(opt::ReactantOptimiser) = _copy_tracked(opt)

# A foreign rule already holding on-device numbers, e.g. Lux's. Inside a trace there is
# nothing to copy, and `get_device` would throw.
function Optimisers._unshare(opt::Optimisers.AbstractRule)
    Reactant.within_compile() && return opt
    get_device(opt) isa ReactantDevice || return opt
    return _copy_tracked(opt)
end

_copy_tracked(opt::ReactantOptimiser) = ReactantOptimiser(_copy_tracked(opt.opt))
_copy_tracked(opt::Optimisers.OptimiserChain) =
    Optimisers.OptimiserChain(map(_copy_tracked, opt.opts)...)
_copy_tracked(opt::Optimisers.MixedPrecision{T}) where {T} =
    Optimisers.MixedPrecision(T, _copy_tracked(opt.rule))
_copy_tracked(opt) = Optimisers.fmap(copy, opt; exclude = x -> x isa Reactant.ConcreteRNumber)

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

# Wrap the inner rule too, so that its `init` override (if any) keeps its scalars tracked.
function Optimisers.make_reactant_compatible(
    opt::Optimisers.MixedPrecision{T}, dev::ReactantDevice
) where {T}
    return ReactantOptimiser(
        Optimisers.MixedPrecision(T, Optimisers.make_reactant_compatible(opt.rule, dev)))
end

Optimisers.make_reactant_compatible(opt::Optimisers.AccumGrad, dev::ReactantDevice) =
    ReactantOptimiser(with_track_numbers(dev, Integer)(opt))

# `throw` forced to false: raising a host-side error is illegal under tracing. `p` stays
# on the host, as `_norm` branches on it.
Optimisers.make_reactant_compatible(opt::Optimisers.ClipNorm, dev::ReactantDevice) =
    ReactantOptimiser(Optimisers.ClipNorm(
        with_track_numbers(dev, AbstractFloat)(opt.omega), opt.p, false))

# --- automatic wrapping at setup ------------------------------------------
# `Optimisers.setup(rule, model)` calls this hook. When `model` sits on a Reactant
# device (eager, concrete arrays) we transparently `make_reactant_compatible(rule)`, so
# the user gets on-device-tracked hyper-parameters without an explicit call. No-op when:
#   * inside a trace (`@jit setup`): `get_device` would throw, and the caller (e.g. Lux)
#     has already wrapped eagerly — leave the rule untouched. This MUST be checked first,
#     before the `get_device(rule)` probe below, which would throw on traced scalars.
#   * the model is not on a Reactant device (CPU/CUDA/…);
#   * the rule is already Reactant-compatible: our own `ReactantOptimiser`, or a foreign
#     wrapper such as Lux's whose scalars are already on-device (caught by the
#     `get_device(rule)` probe) — this avoids double-wrapping.
function Optimisers._prepare_rule(rule::Optimisers.AbstractRule, model)
    Reactant.within_compile() && return rule
    get_device_type(model) <: ReactantDevice || return rule
    rule isa ReactantOptimiser && return rule
    get_device(rule) isa ReactantDevice && return rule
    return Optimisers.make_reactant_compatible(rule, get_device(model))
end

end # module
