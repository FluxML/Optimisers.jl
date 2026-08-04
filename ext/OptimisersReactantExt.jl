module OptimisersReactantExt

using Optimisers: Optimisers, @..
using Reactant: Reactant, TracedRNumber, TracedRArray, @trace

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
# `T(::TracedRNumber{T})` is not defined, so the stock `T(o.eta)` /
# `T.(o.beta)` lines blow up when the rule's scalars were already
# promoted via `to_rarray(opt; track_numbers=...)`. Treat already-traced
# scalars as no-ops and only convert host numbers.
@inline _radam_to(::Type{T}, x::TracedRNumber) where {T} = x
@inline _radam_to(::Type{T}, x) where {T} = T(x)
@inline _radam_eps(::Type{T}, x::TracedRNumber) where {T} = x
@inline _radam_eps(::Type{T}, x) where {T} = Optimisers._eps(T, x)

function Optimisers.apply!(
    o::Optimisers.RAdam, state, x::TracedRArray{T}, dx,
) where {T}
    η = _radam_to(T, o.eta)
    β = map(b -> _radam_to(T, b), o.beta)
    ϵ = _radam_eps(T, o.epsilon)
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
