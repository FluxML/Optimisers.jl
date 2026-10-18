module OptimisersReactantExt

using Optimisers: Optimisers, @..
using Reactant: Reactant, TracedRNumber, TracedRArray, ConcretePJRTArray, ConcreteIFRTArray

# Once https://github.com/EnzymeAD/Reactant.jl/pull/835 we can support throwing errors
# from compiled MLIR
@inline function Optimisers._assert_positive_eta(eta, ::TracedRNumber{Bool})
    return
end

# The generic `map` scalar-indexes, which is disallowed on GPU. Build on the host and
# place it like `zero(x)` does.
for CA in (:ConcretePJRTArray, :ConcreteIFRTArray)
    @eval Optimisers.onevalue(λ::T, x::$CA{T}) where {T} = $CA(
        fill(λ, size(x)); client = Reactant.XLA.client(x), device = Reactant.XLA.device(x),
        x.sharding)
end

# `T(::TracedRNumber)` is not defined, so the stock `T(o.eta)` / `T.(o.beta)` lines blow
# up when the rule's scalars are traced (promoted via `to_rarray(opt; track_numbers=...)`
# or `make_reactant_compatible`). Convert traced scalars within the trace instead, so the
# arithmetic runs in the parameters' eltype as in the eager rule.
@inline _to(::Type{T}, x::TracedRNumber) where {T} = convert(TracedRNumber{T}, x)
@inline _to(::Type{T}, x) where {T} = T(x)
@inline _eps(::Type{T}, x::TracedRNumber) where {T} = _to(T, x)
@inline _eps(::Type{T}, x) where {T} = Optimisers._eps(T, x)

# A traced `AccumGrad` cannot return `nothing` on the steps where it is still accumulating.
# It returns its (zero until complete) update wrapped together with the traced flag
# `ready`, which the traced `OptimiserChain` below uses to hold the downstream rules'
# state. Anything else consuming the update just broadcasts over `dx`.
struct AccumulatedGrad{D,R}
    dx::D
    ready::R
end

Base.broadcastable(g::AccumulatedGrad) = g.dx

function Optimisers.apply!(
    o::Optimisers.AccumGrad{<:TracedRNumber{<:Integer}}, state, x, dx
)
    accum_dx, counter = state
    restart = counter == 1
    @.. accum_dx = ifelse(restart, zero(accum_dx), accum_dx) + dx / o.n
    ready = counter == o.n
    counter = ifelse(ready, one(counter), counter + 1)
    dx′ = @. ifelse(ready, accum_dx, zero(accum_dx))
    return (accum_dx, counter), AccumulatedGrad(dx′, ready)
end

# Under tracing the rules after an `AccumGrad` cannot be skipped, so they run on every
# step and their previous state is selected back until the accumulation is complete.
function Optimisers.apply!(
    o::Optimisers.OptimiserChain, states, x::TracedRArray, dx, dxs...
)
    init = ((), dx, nothing)
    states′, dx′, ready = foldl(tuple.(o.opts, states); init) do (states′, dx′, ready), (opt, state)
        dx′ isa Optimisers.Zero && return (states′..., state), dx′, ready
        old = ready === nothing ? state : _snapshot(state)
        state′, dx′ = Optimisers.apply!(opt, state, x, dx′, dxs...)
        ready === nothing || (state′ = _select(ready, state′, old))
        if dx′ isa AccumulatedGrad
            ready = ready === nothing ? dx′.ready : ready & dx′.ready
            dx′ = dx′.dx
        end
        return (states′..., state′), dx′, ready
    end
    (ready === nothing || dx′ isa Optimisers.Zero) && return states′, dx′
    dx′ = @. ifelse(ready, dx′, zero(dx′))
    return states′, AccumulatedGrad(dx′, ready)
end

# Rules update their state arrays in place, so keep a (free, under tracing) copy of the
# old values to select from.
_snapshot(x::TracedRArray) = copy(x)
_snapshot(x::Union{Tuple,NamedTuple}) = map(_snapshot, x)
_snapshot(x) = x

_select(c, new::TracedRArray, old::TracedRArray) = ifelse.(c, new, old)
_select(c, new::TracedRNumber{T}, old::TracedRNumber) where {T} = ifelse(c, new, _to(T, old))
_select(c, new::Tuple, old::Tuple) = map((n, o) -> _select(c, n, o), new, old)
_select(c, new::NamedTuple{K}, old::NamedTuple{K}) where {K} =
    map((n, o) -> _select(c, n, o), new, old)
_select(c, new, old) = new

# RAdam under tracing. Stock `apply!` (rules.jl) takes a host-side
# `if ρ > 4` branch, which fires `TypeError: non-boolean
# (TracedRNumber{Bool}) used in boolean context` because `ρ` is a
# TracedRNumber whenever the state's `βt` tuple or its `mt`/`vt`
# arrays trace, regardless of whether the rule's own `beta` field was
# promoted. This variant uses `ifelse` / `ifelse.` so both arms lower
# to XLA selects with no host branching. The `one(ρ)` fallback inside
# `sqrt` ensures the argument is valid when ρ ≤ 4 (the value is
# discarded by the outer broadcasted `ifelse.`). Dispatch fires
# whenever `x` is a `TracedRArray`, which is the only situation in
# which `apply!` runs inside a Reactant trace, so eager / CPU users
# continue to hit the stock method.
function Optimisers.apply!(
    o::Optimisers.RAdam, state, x::TracedRArray{T}, dx,
) where {T}
    η = _to(T, o.eta)
    β = map(b -> _to(T, b), o.beta)
    ϵ = _eps(T, o.epsilon)
    ρ∞ = real(2 / (1 - β[2]) - 1)

    mt, vt, βt, t = state

    @.. mt = β[1] * mt + (1 - β[1]) * dx
    @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
    ρ = real(ρ∞ - 2 * t * βt[2] / (1 - βt[2]))

    r = sqrt(ifelse(ρ > 4,
        (ρ - 4) * (ρ - 2) * ρ∞ / ((ρ∞ - 4) * (ρ∞ - 2) * ρ),
        one(ρ)))

    mt_bc = mt ./ (1 .- βt[1])
    dx′ = ifelse.(ρ > 4,
        mt_bc ./ (sqrt.(vt ./ (1 .- βt[2])) .+ ϵ) .* η .* r,
        mt_bc .* η)

    return (mt, vt, βt .* β, t + 1), dx′
end

end
