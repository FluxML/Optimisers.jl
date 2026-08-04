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

    opt_ra = Optimisers.make_reactant_compatible(Descent(0.011), get_device(ps))
    st = Optimisers.setup(opt_ra, ps)
    hlo = repr(@code_hlo Optimisers.update!(st, ps, gs))
    @test !contains(hlo, "1.100000e-02")

    # Contrast: a plain (untracked) rule bakes the learning rate in as a constant.
    st_plain = Optimisers.setup(Descent(0.011), ps)
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
