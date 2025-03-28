using Pkg
Pkg.add("Reactant")

using Reactant, Optimisers

@testset for opt in (Descent(0.011), Momentum(0.011), Adam(0.011), AdamW(0.011))
    opt_ra = Reactant.to_rarray(opt; track_numbers=AbstractFloat)

    x_ra = Reactant.to_rarray((rand(3), rand(2)))
    gs_ra = Reactant.to_rarray((rand(3), rand(2)))

    st_opt = @jit Optimisers.setup(opt, x_ra)
    st_opt_ra = @jit Optimisers.setup(opt_ra, x_ra)

    @testset "out of place" begin
        hlo = @code_hlo Optimisers.update(st_opt, x_ra, gs_ra)
        @test length(findall("dense<1.100000e-02>", repr(hlo))) == 2

        st_opt2, x_ra2 = @jit Optimisers.update(st_opt, x_ra, gs_ra)

        hlo = @code_hlo Optimisers.update(st_opt_ra, x_ra, gs_ra)
        @test !contains(repr(hlo), "dense<1.100000e-02>")

        st_opt2_ra, x_ra2 = @jit Optimisers.update(st_opt_ra, x_ra, gs_ra)
    end

    @testset "in place" begin
        hlo = @code_hlo Optimisers.update!(st_opt, x_ra, gs_ra)
        @test length(findall("dense<1.100000e-02>", repr(hlo))) == 2

        st_opt2, x_ra2 = @jit Optimisers.update!(st_opt, x_ra, gs_ra)

        hlo = @code_hlo Optimisers.update!(st_opt_ra, x_ra, gs_ra)
        @test !contains(repr(hlo), "dense<1.100000e-02>")

        st_opt2_ra, x_ra2 = @jit Optimisers.update!(st_opt_ra, x_ra, gs_ra)
    end
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
