using Reactant, MLDataDevices, Optimisers

# A concrete Reactant device to move host data onto (CPU backend in CI).
cdev = reactant_device()

# Move params onto the device, recover the (TN=Union{}) device from them, wrap.
function _wrap(opt, ps)
    dev = get_device(ps)
    opt_ra = Optimisers.make_reactant_compatible(opt, dev)
    # NOTE: `setup` must run EAGERLY (not under @jit): the RAdam init override
    # calls `get_device`, which errors inside a Reactant trace.
    st = Optimisers.setup(opt_ra, ps)
    return opt_ra, st
end

@testset "make_reactant_compatible: $(typeof(opt).name.name)" for opt in (
        Adam(0.01f0),
        RAdam(0.01f0),
        OptimiserChain(AccumGrad(2), Descent(0.1f0)),
    )
    ps = cdev((randn(Float32, 3), randn(Float32, 2)))
    opt_ra, st = _wrap(opt, ps)

    # The rule's scalar hyper-parameters now live on the Reactant device.
    @test get_device(opt_ra) isa ReactantDevice

    gs = cdev((randn(Float32, 3), randn(Float32, 2)))
    st2, ps2 = @jit Optimisers.update!(st, ps, gs)
    @test all(isfinite, Array(ps2[1]))
    @test all(isfinite, Array(ps2[2]))

    # A second step exercises the AccumGrad counter branch under tracing.
    gs2 = cdev((randn(Float32, 3), randn(Float32, 2)))
    st3, ps3 = @jit Optimisers.update!(st2, ps2, gs2)
    @test all(isfinite, Array(ps3[1]))
end

@testset "learning rate is tracked, not a compiled constant" begin
    # Float64 / 0.011 reproduces the exact HLO constant string used by
    # `test/reactant.jl`, so the presence/absence check is stable.
    ps = cdev((randn(3), randn(2)))
    gs = cdev((randn(3), randn(2)))

    # Eager `setup` on a Reactant device auto-wraps the rule, so even a plain rule
    # keeps its learning rate tracked (not a baked constant).
    st = Optimisers.setup(Descent(0.011), ps)
    hlo = repr(@code_hlo Optimisers.update!(st, ps, gs))
    @test !contains(hlo, "1.100000e-02")

    # Contrast: under `@jit`, `setup` does NOT auto-wrap (`get_device` would throw in a
    # trace), so the plain rule bakes the learning rate in as a constant.
    st_plain = @jit Optimisers.setup(Descent(0.011), ps)
    hlo_plain = repr(@code_hlo Optimisers.update!(st_plain, ps, gs))
    @test contains(hlo_plain, "1.100000e-02")
end

@testset "adjust! runs host-side and updates the tracked eta" begin
    ps = cdev((randn(Float32, 3),))
    opt_ra, st = _wrap(Adam(0.01f0), ps)

    Optimisers.adjust!(st, 0.001f0)
    @test st[1].rule.opt.eta ≈ 0.001f0

    st2 = Optimisers.adjust(st; eta = 0.005f0)
    @test st2[1].rule.opt.eta ≈ 0.005f0
end

# Lux's actual flow: wrap EAGERLY, then run `setup` UNDER `@jit`. This worked for
# every rule except RAdam, whose init called `get_device` (errors in a trace); the
# tracked counter is now derived from an already-tracked scalar instead.
@testset "setup runs under @jit: $(typeof(opt).name.name)" for opt in (
        Adam(0.01f0),
        RAdam(0.01f0),
    )
    ps = cdev((randn(Float32, 4), randn(Float32, 2)))
    opt_ra = Optimisers.make_reactant_compatible(opt, get_device(ps))
    st = @jit Optimisers.setup(opt_ra, ps)

    gs = cdev((randn(Float32, 4), randn(Float32, 2)))
    st2, ps2 = @jit Optimisers.update!(st, ps, gs)
    @test all(isfinite, Array(ps2[1]))

    # A second step advances the tracked RAdam counter as a runtime variable.
    gs2 = cdev((randn(Float32, 4), randn(Float32, 2)))
    st3, ps3 = @jit Optimisers.update!(st2, ps2, gs2)
    @test all(isfinite, Array(ps3[1]))
end

@testset "make_reactant_compatible is idempotent" begin
    ps = cdev((randn(Float32, 3),))
    dev = get_device(ps)
    opt1 = Optimisers.make_reactant_compatible(Adam(0.01f0), dev)
    @test Optimisers.make_reactant_compatible(opt1, dev) === opt1   # two-arg form
    @test Optimisers.make_reactant_compatible(opt1) === opt1        # device-less form
end

@testset "device-less make_reactant_compatible uses the default device" begin
    opt = Optimisers.make_reactant_compatible(Adam(0.01f0))
    @test get_device(opt) isa ReactantDevice

    ps = cdev((randn(Float32, 3),))
    st = Optimisers.setup(opt, ps)
    gs = cdev((randn(Float32, 3),))
    st2, ps2 = @jit Optimisers.update!(st, ps, gs)
    @test all(isfinite, Array(ps2[1]))
end

# `setup` auto-detects a model on a Reactant device and wraps the rule, so a plain
# eager `Optimisers.setup(opt, ps)` (the Flux flow) needs no `make_reactant_compatible`.
@testset "setup auto-wraps a rule on a Reactant device: $(typeof(opt).name.name)" for opt in (
        Adam(0.01f0),
        RAdam(0.01f0),
        OptimiserChain(AccumGrad(2), Descent(0.1f0)),
    )
    ps = cdev((randn(Float32, 3), randn(Float32, 2)))
    st = Optimisers.setup(opt, ps)   # eager, no explicit wrap, no @jit

    # Each Leaf now holds the tracked wrapper, so its scalars live on the device.
    @test get_device(st[1].rule) isa ReactantDevice
    @test get_device(st[2].rule) isa ReactantDevice

    gs = cdev((randn(Float32, 3), randn(Float32, 2)))
    st2, ps2 = @jit Optimisers.update!(st, ps, gs)
    @test all(isfinite, Array(ps2[1]))

    # A second step exercises the AccumGrad counter branch under tracing.
    gs2 = cdev((randn(Float32, 3), randn(Float32, 2)))
    st3, ps3 = @jit Optimisers.update!(st2, ps2, gs2)
    @test all(isfinite, Array(ps3[1]))
end

_step!(st, ps, gs) = (Optimisers.update!(st, ps, gs); nothing)

# Eager `setup` on a Reactant device followed by a compiled `update!` must follow the
# eager rule step by step. 8 steps reach RAdam's rectified branch (t ≥ 5).
@testset "compiled update! matches eager: $(rule)" for rule in (
        Adam(0.01f0),
        RAdam(0.01f0),
        Rprop(),
        RMSProp(),
        AdaGrad(),
        AMSGrad(),
        MixedPrecision(Adam(0.01f0)),
        OptimiserChain(Adam(0.1f0), ClipNorm(1)),
        OptimiserChain(AccumGrad(3), Adam(0.1f0)),
        OptimiserChain(ClipGrad(1), AccumGrad(2), Adam(0.1f0)),
        OptimiserChain(AccumGrad(2), AccumGrad(2), Momentum()),
        OptimiserChain(OptimiserChain(AccumGrad(2), Adam(0.1f0)), WeightDecay()),
    )
    rng = Random.MersenneTwister(0)
    p = randn(rng, Float32, 5)
    grads = [randn(rng, Float32, 5) for _ in 1:8]

    st, p_cpu = Optimisers.setup(rule, p), copy(p)
    p_ra = cdev(copy(p))
    st_ra = Optimisers.setup(rule, p_ra)
    step_ra! = Reactant.@compile _step!(st_ra, p_ra, cdev(grads[1]))
    for g in grads
        Optimisers.update!(st, p_cpu, g)
        step_ra!(st_ra, p_ra, cdev(g))
    end
    @test Array(p_ra) ≈ p_cpu rtol=1e-5
end

# On GPU an eager broadcast over device arrays is compiled, so the rules' broadcast
# closures are traced and must not branch on element values, and scalar indexing is
# disallowed. 6 steps reach RAdam's rectified branch (t ≥ 5).
@testset "eager update! on device matches CPU: $(rule)" for rule in (
        Rprop(),
        RMSProp(),
        ClipNorm(1),
        # On Julia 1.10 Reactant cannot materialise the longer lazy updates of `RAdam` and
        # `Adam` eagerly: `Broadcast.flatten` of them infers no concrete eltype.
        (VERSION >= v"1.11" ? (RAdam(0.01f0), OptimiserChain(Adam(0.1f0), ClipNorm(1))) : ())...,
    )
    rng = Random.MersenneTwister(0)
    p = randn(rng, Float32, 5)
    grads = [randn(rng, Float32, 5) for _ in 1:6]

    st, p_cpu = Optimisers.setup(rule, p), copy(p)
    p_ra = cdev(copy(p))
    st_ra = Optimisers.setup(rule, p_ra)
    for g in grads
        Optimisers.update!(st, p_cpu, g)
        Optimisers.update!(st_ra, p_ra, cdev(g))
    end
    @test Array(p_ra) ≈ p_cpu rtol=1e-5
end

# Muon needs matrices. Eagerly, its Newton-Schulz iteration is compiled on its own.
@testset "Muon on device matches CPU" begin
    rule = Muon(0.01f0, 0.9f0, 0.1f0)
    rng = Random.MersenneTwister(0)
    p = randn(rng, Float32, 6, 4)
    grads = [randn(rng, Float32, 6, 4) for _ in 1:3]

    st, p_cpu = Optimisers.setup(rule, p), copy(p)
    p_ra, p_eager = cdev(copy(p)), cdev(copy(p))
    st_ra, st_eager = Optimisers.setup(rule, p_ra), Optimisers.setup(rule, p_eager)
    step_ra! = Reactant.@compile _step!(st_ra, p_ra, cdev(grads[1]))
    for g in grads
        Optimisers.update!(st, p_cpu, g)
        step_ra!(st_ra, p_ra, cdev(g))
        Optimisers.update!(st_eager, p_eager, cdev(g))
    end
    @test Array(p_ra) ≈ p_cpu rtol=1e-5
    @test Array(p_eager) ≈ p_cpu rtol=1e-5
end

# The new value must reach the rules inside the container, and stay tracked so that an
# already-compiled `update!` uses it.
@testset "adjust! reaches the rules in a wrapped $(nameof(typeof(rule)))" for rule in (
        OptimiserChain(WeightDecay(0f0), Descent(0.1f0)),
        OptimiserChain(OptimiserChain(AccumGrad(1), Descent(0.1f0)), WeightDecay(0f0)),
        MixedPrecision(Descent(0.1f0)),
    )
    ps = cdev((; w = Float32[0.1, 0.2, 0.3, 0.4]))
    gs = cdev((; w = ones(Float32, 4)))
    st = Optimisers.setup(rule, ps)
    step_ra! = Reactant.@compile _step!(st, ps, gs)

    Optimisers.adjust!(st, 0f0)
    w = Array(ps.w)
    Optimisers.update!(st, ps, gs)
    @test Array(ps.w) == w
    step_ra!(st, ps, gs)
    @test Array(ps.w) == w

    st2 = Optimisers.adjust(st; eta = 0.5f0)
    @test typeof(st2) == typeof(st)
    step_ra!(st2, ps, gs)
    @test Array(ps.w) ≈ w .- 0.5f0
end

# A filter sees the rules inside the wrappers, and the new values stay tracked.
@testset "adjust! with a filter on wrapped rules" begin
    ps = cdev((; w = Float32[1, 2], b = Float32[3, 4]))
    gs = cdev((; w = zeros(Float32, 2), b = zeros(Float32, 2)))
    st = Optimisers.setup(OptimiserChain(SignDecay(0f0), WeightDecay(0f0)), ps)
    step_ra! = Reactant.@compile _step!(st, ps, gs)
    lambdas(l) = (l.rule.opt.opts[1].opt.lambda, l.rule.opt.opts[2].opt.lambda)

    Optimisers.adjust!(st, SignDecay; lambda = 0.5f0)  # not WeightDecay's lambda
    @test lambdas(st.w) == lambdas(st.b) == (0.5f0, 0f0)
    w = Array(ps.w)
    b = Array(ps.b)
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ w .- 0.5f0 .* sign.(w)
    @test Array(ps.b) ≈ b .- 0.5f0 .* sign.(b)

    st2 = Optimisers.adjust(st, WeightDecay; lambda = 0.5f0)  # out of place
    @test typeof(st2) == typeof(st)
    @test lambdas(st2.w) == (0.5f0, 0.5f0)
    w = Array(ps.w)
    step_ra!(st2, ps, gs)
    @test Array(ps.w) ≈ 0.5f0 .* w .- 0.5f0 .* sign.(w)

    only_b(kp, r) = r isa SignDecay && last(kp) == :b
    st3 = Optimisers.adjust(st, only_b; lambda = 0f0)
    @test typeof(st3) == typeof(st)
    @test lambdas(st3.w) == (0.5f0, 0f0) && lambdas(st3.b) == (0f0, 0f0)
    w, b = Array(ps.w), Array(ps.b)
    step_ra!(st3, ps, gs)
    @test Array(ps.w) ≈ w .- 0.5f0 .* sign.(w)
    @test Array(ps.b) == b

    # Wrapped atomic rules: one holding no tracked number, and one with a tracked integer.
    st = Optimisers.setup(Descent(1), ps)
    Optimisers.adjust!(st, SignDecay, 3)
    @test st.w.rule.opt.eta === 1
    Optimisers.adjust!(st, Descent, 2)
    @test st.w.rule.opt.eta === 2

    st = Optimisers.setup(AccumGrad(1), ps)
    Optimisers.adjust!(st, AccumGrad; n = 2)
    @test st.w.rule.opt.n isa Reactant.ConcretePJRTNumber{Int}
    @test st.w.rule.opt.n == 2
end

# Each leaf holds its own tracked numbers, so a step compiled while they are equal still
# sees a later adjustment of only some leaves, instead of writing the old value back.
@testset "adjust! of some leaves after compiling" begin
    ps = cdev((; w = Float32[1, 2], b = Float32[3, 4]))
    gs = cdev((; w = ones(Float32, 2), b = ones(Float32, 2)))
    st = Optimisers.setup(Descent(0.1f0), ps)
    step_ra! = Reactant.@compile _step!(st, ps, gs)
    Optimisers.adjust!(st.b, 0f0)
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ [0.9, 1.9]
    @test Array(ps.b) == [3, 4]
    @test st.b.rule.opt.eta == 0

    Optimisers.adjust!(st, Optimisers.make_reactant_compatible(Descent(0.2f0), cdev))
    step_ra! = Reactant.@compile _step!(st, ps, gs)
    Optimisers.adjust!(st.b, 0f0)
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ [0.7, 1.7]
    @test Array(ps.b) == [3, 4]
end

@testset "setup with a function wraps each rule" begin
    ps = cdev((; w = Float32[1 2; 3 4], b = Float32[3, 4]))
    gs = cdev((; w = ones(Float32, 2, 2), b = ones(Float32, 2)))
    st = Optimisers.setup((kp, x) -> ndims(x) == 2 ? Momentum(0.1f0) : Descent(0.1f0), ps)
    @test get_device(st.w.rule) isa ReactantDevice
    @test st.b.rule.opt isa Descent
    step_ra! = Reactant.@compile _step!(st, ps, gs)
    Optimisers.adjust!(st, Descent, 0f0)
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ [0.9 1.9; 2.9 3.9]
    @test Array(ps.b) == [3, 4]
end

# A value of another type keeps the tracked numbers' eltype, so a compiled step still applies.
@testset "adjust! with a value of another number type" begin
    ps = cdev((; w = Float32[1, 2]))
    gs = cdev((; w = ones(Float32, 2)))
    st = Optimisers.setup(OptimiserChain(WeightDecay(0.5f0), Descent(0.1f0)), ps)
    step_ra! = Reactant.@compile _step!(st, ps, gs)
    Optimisers.adjust!(st, 0.5)
    Optimisers.adjust!(st; lambda = 0)
    @test st.w.rule.opt.opts[2].opt.eta isa Reactant.ConcreteRNumber{Float32}
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ [0.5, 1.5]
    @test typeof(Optimisers.adjust(st, Descent, 1)) == typeof(st)
end

# The rule's own `adjust(rule, η)` is used, and the new value is tracked.
struct _LR{T} <: Optimisers.AbstractRule
    lr::T
end
Optimisers.init(::_LR, x::AbstractArray) = nothing
Optimisers.apply!(o::_LR, state, x, dx) = state, dx .* o.lr
Optimisers.adjust(::_LR, eta::Real) = _LR(eta)

@testset "adjust! calls a wrapped rule's own adjust(rule, η)" begin
    ps = cdev((; w = Float32[1, 2]))
    gs = cdev((; w = ones(Float32, 2)))
    st = Optimisers.setup(_LR(1f0), ps)
    step_ra! = Reactant.@compile _step!(st, ps, gs)
    Optimisers.adjust!(st, 0.5f0)
    @test st.w.rule.opt.lr isa Reactant.ConcreteRNumber
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ [0.5, 1.5]
    st2 = Optimisers.adjust(st, _LR, 0.25)
    @test typeof(st2) == typeof(st)
    @test st2.w.rule.opt.lr == 0.25f0
end

# `ClipNorm` has no `eta`, and its `omega` stays tracked so that an already-compiled
# `update!` uses the new value.
@testset "adjust! on a wrapped ClipNorm: $(rule)" for rule in (
        ClipNorm(1f0),
        OptimiserChain(ClipNorm(1f0), Descent(1f0)),
    )
    ps = cdev((; w = Float32[0.1, 0.2, 0.3, 0.4]))
    gs = cdev((; w = fill(10f0, 4)))
    st = Optimisers.setup(rule, ps)
    step_ra! = Reactant.@compile _step!(st, ps, gs)

    Optimisers.adjust!(st, 1f0)
    w = Array(ps.w)
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ w .- 0.5f0

    Optimisers.adjust!(st; omega = 2f0)
    w = Array(ps.w)
    step_ra!(st, ps, gs)
    @test Array(ps.w) ≈ w .- 1f0
end

@testset "adjust! on a wrapped rule holding no tracked number" begin
    ps = cdev((; w = Float32[0.1, 0.2, 0.3, 0.4]))
    st = Optimisers.setup(Descent(1), ps)
    Optimisers.adjust!(st, 2)
    @test st.w.rule.opt.eta === 2
end

@testset "onevalue keeps a Reactant array on its device" begin
    x = cdev(randn(Float32, 3, 2))
    y = Optimisers.onevalue(0.5f0, x)
    @test y isa typeof(x)
    @test Reactant.XLA.device(y) == Reactant.XLA.device(x)
    @test Array(y) == fill(0.5f0, 3, 2)
end

@testset "setup does not double-wrap already-compatible rules" begin
    ps = cdev((randn(Float32, 3),))
    dev = get_device(ps)

    # (a) Our own wrapper is not re-wrapped (each leaf gets a copy, of the same type).
    opt_ours = Optimisers.make_reactant_compatible(Adam(0.01f0), dev)
    st = Optimisers.setup(opt_ours, ps)
    @test typeof(st[1].rule) == typeof(opt_ours)
    @test st[1].rule.opt.eta == opt_ours.opt.eta
    @test st[1].rule.opt.eta !== opt_ours.opt.eta

    # (b) A bare rule already carrying on-device scalars (mimics Lux's foreign wrapper)
    #     is detected via `get_device` and left alone.
    opt_foreign = Reactant.to_rarray(Adam(0.01f0); track_numbers = AbstractFloat)
    st2 = Optimisers.setup(opt_foreign, ps)
    @test typeof(st2[1].rule) == typeof(opt_foreign)
    @test st2[1].rule.eta !== opt_foreign.eta
end

@testset "setup leaves non-Reactant models untouched" begin
    # Plain host arrays: the hook must be a no-op even though the extension is loaded.
    ps = (randn(Float32, 3), randn(Float32, 2))
    st = Optimisers.setup(Adam(0.01f0), ps)
    @test st[1].rule isa Adam
    @test st[1].rule === st[2].rule                  # shared, un-wrapped rule
    @test !(get_device(st[1].rule) isa ReactantDevice)
end

@testset "destructure checks devices, also while tracing" begin
    m = Reactant.to_rarray((a = [1.0, 2.0], b = [3.0]))
    @test destructure(m)[1] isa Reactant.ConcretePJRTArray
    flatsum(m) = sum(destructure(m)[1])
    @test (@jit flatsum(m)) ≈ 6

    m2 = Reactant.to_rarray((a = [1.0 2.0; 3.0 4.0], b = Float32[5.0]))
    v2, re2 = destructure(m2)
    @test Array(v2) == [1, 3, 2, 4, 5]
    m2′ = re2(v2)
    @test Array(m2′.a) == [1 2; 3 4]
    @test m2′.b isa Reactant.ConcretePJRTArray{Float32}
    rebuildsum(v) = sum(re2(v).b)
    @test (@jit rebuildsum(v2)) ≈ 5
    v2, re2! = destructure!(m2)
    @test re2!([10.0, 20, 30, 40, 50]) === m2
    @test Array(m2.a) == [10 30; 20 40] && Array(m2.b) == Float32[50]
end
