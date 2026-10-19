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
        RAdam(0.01f0),
        RMSProp(),
        ClipNorm(1),
        OptimiserChain(Adam(0.1f0), ClipNorm(1)),
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

    # (a) Our own wrapper passes through unchanged (same object, not re-wrapped).
    opt_ours = Optimisers.make_reactant_compatible(Adam(0.01f0), dev)
    st = Optimisers.setup(opt_ours, ps)
    @test st[1].rule === opt_ours

    # (b) A bare rule already carrying on-device scalars (mimics Lux's foreign wrapper)
    #     is detected via `get_device` and left alone.
    opt_foreign = Reactant.to_rarray(Adam(0.01f0); track_numbers = AbstractFloat)
    st2 = Optimisers.setup(opt_foreign, ps)
    @test st2[1].rule === opt_foreign
end

@testset "setup leaves non-Reactant models untouched" begin
    # Plain host arrays: the hook must be a no-op even though the extension is loaded.
    ps = (randn(Float32, 3), randn(Float32, 2))
    st = Optimisers.setup(Adam(0.01f0), ps)
    @test st[1].rule isa Adam
    @test st[1].rule === st[2].rule                  # shared, un-wrapped rule
    @test !(get_device(st[1].rule) isa ReactantDevice)
end
