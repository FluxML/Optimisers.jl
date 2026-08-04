module OptimisersReactantMLDataDevicesExt

using Optimisers: Optimisers
using Reactant: Reactant
using MLDataDevices: ReactantDevice, get_device, with_track_numbers

# --- device-preserving eltype conversion ---------------------------------
# On-device numbers must be rebuilt through their concrete constructor so they
# stay tracked; `convert(T, x)` / `T(x)` would pull them back to the host as
# compile-time constants. (Reproduces Lux's `Utils.convert_eltype`.)
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
    # NOTE: `get_device` errors inside a Reactant trace, so a `ReactantOptimiser{RAdam}`
    # must have `setup` run eagerly (host-side), not under `@jit`.
    dev = with_track_numbers(get_device(opt), Integer)
    return zero(x), zero(x), _convert_eltype.((T,), opt.opt.beta), dev(1)
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
Optimisers.make_reactant_compatible(opt::Optimisers.AbstractRule, dev::ReactantDevice) =
    ReactantOptimiser(with_track_numbers(dev, AbstractFloat)(opt))

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
