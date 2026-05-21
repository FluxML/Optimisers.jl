using Pkg
Pkg.add("Reactant")

using Reactant, Optimisers

@testset for opt in (Descent(0.011), Momentum(0.011), Adam(0.011), AdamW(0.011), RAdam(0.011))
    opt_ra = Reactant.to_rarray(opt; track_numbers=AbstractFloat)

    x_ra = Reactant.to_rarray((rand(3), rand(2)))
    gs_ra = Reactant.to_rarray((rand(3), rand(2)))

    st_opt = @jit Optimisers.setup(opt, x_ra)
    st_opt_ra = @jit Optimisers.setup(opt_ra, x_ra)

    @testset "out of place" begin
        hlo = @code_hlo Optimisers.update(st_opt, x_ra, gs_ra)
        @test length(findall("1.100000e-02>", repr(hlo))) == 2

        st_opt2, x_ra2 = @jit Optimisers.update(st_opt, x_ra, gs_ra)

        hlo = @code_hlo Optimisers.update(st_opt_ra, x_ra, gs_ra)
        @test !contains(repr(hlo), "1.100000e-02>")

        st_opt2_ra, x_ra2 = @jit Optimisers.update(st_opt_ra, x_ra, gs_ra)
    end

    @testset "in place" begin
        hlo = @code_hlo Optimisers.update!(st_opt, x_ra, gs_ra)
        @test length(findall("1.100000e-02>", repr(hlo))) == 2

        st_opt2, x_ra2 = @jit Optimisers.update!(st_opt, x_ra, gs_ra)

        hlo = @code_hlo Optimisers.update!(st_opt_ra, x_ra, gs_ra)
        @test !contains(repr(hlo), "1.100000e-02>")

        st_opt2_ra, x_ra2 = @jit Optimisers.update!(st_opt_ra, x_ra, gs_ra)
    end
end

@testset "RAdam parity with CPU" begin
    # Branch coverage: the rectification threshold `ρ > 4` is a
    # deterministic function of `β₂` and the step counter `t` (it does
    # not see the gradients at all — see Liu et al. 2019), and for any
    # reasonable `β₂` the warmup arm fires at t = 1..4 and the
    # rectified arm at t ≥ 5. Running 8 steps therefore exercises both
    # arms regardless of how `grads` are sampled. With the default
    # `β₂ = 0.999` used below: ρ ≈ {1.0, 2.0, 3.0, 4.0} at t = 1..4,
    # then ≈ {5.0, 6.0, 7.0, 8.0} at t = 5..8.
    rng = Random.MersenneTwister(0)
    p_cpu = randn(rng, Float32, 6)
    grads = [randn(rng, Float32, 6) for _ in 1:8]

    opt_cpu = RAdam(0.01f0, (0.9f0, 0.999f0))
    st_cpu = Optimisers.setup(opt_cpu, p_cpu)
    p_iter = copy(p_cpu)
    for g in grads
        st_cpu, p_iter = Optimisers.update(st_cpu, p_iter, g)
    end
    p_cpu_final = p_iter

    opt_ra = Reactant.to_rarray(opt_cpu; track_numbers=AbstractFloat)
    p_ra = Reactant.to_rarray(copy(p_cpu))
    st_ra = @jit Optimisers.setup(opt_ra, p_ra)

    p_outofplace = p_ra
    st_outofplace = st_ra
    for g in grads
        g_ra = Reactant.to_rarray(g)
        st_outofplace, p_outofplace = @jit Optimisers.update(st_outofplace, p_outofplace, g_ra)
    end
    @test Array(p_outofplace) ≈ p_cpu_final rtol=1e-5 atol=1e-6

    p_inplace = Reactant.to_rarray(copy(p_cpu))
    st_inplace = @jit Optimisers.setup(opt_ra, p_inplace)
    for g in grads
        g_ra = Reactant.to_rarray(g)
        st_inplace, p_inplace = @jit Optimisers.update!(st_inplace, p_inplace, g_ra)
    end
    @test Array(p_inplace) ≈ p_cpu_final rtol=1e-5 atol=1e-6
end

@testset "AccumGrad" begin
    opt = OptimiserChain(AccumGrad(2), Descent(1.0))
    opt_ra = Reactant.to_rarray(opt; track_numbers=Number)

    x_ra = Reactant.to_rarray(rand(3))
    gs_ra = Reactant.to_rarray(rand(3))
    gs_ra2 = Reactant.to_rarray(rand(3))

    st_opt = @jit Optimisers.setup(opt_ra, x_ra)
    @test Int64(st_opt.state[1][2]) == 1

    st_opt, x_ra2 = @jit Optimisers.update(st_opt, x_ra, gs_ra)
    @test Int64(st_opt.state[1][2]) == 2
    @test Array(x_ra2) == Array(x_ra)

    st_opt, x_ra3 = @jit Optimisers.update(st_opt, x_ra2, gs_ra2)
    @test Int64(st_opt.state[1][2]) == 1
    @test Array(x_ra3) ≈ Array(x_ra) .- 0.5 .* Array(gs_ra) .- 0.5 .* Array(gs_ra2) rtol = 1e-2
end
