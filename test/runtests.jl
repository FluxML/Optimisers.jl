using Optimisers, Test
using Zygote
using Statistics, Random, LinearAlgebra
Random.seed!(1)
using Optimisers: @..

@testset verbose=true "Optimisers.jl" begin

  @testset "very basics" begin
    m = ([1.0, 2.0],)
    mid = objectid(m[1])
    g = ([25, 33],)
    o = Descent(0.1)
    s = Optimisers.setup(o, m)
    
    s2, m2 = Optimisers.update(s, m, g)
    @test m[1] == 1:2  # not mutated
    @test Optimisers.iswriteable(m[1])
    @test m2[1] ≈ [1,2] .- 0.1 .* [25, 33]

    s3, m3 = Optimisers.update!(s, m, g)
    @test objectid(m3[1]) == mid
    @test m3[1] ≈ [1,2] .- 0.1 .* [25, 33]
  end

  @testset for o in (Descent(), ADAM(), Momentum(), Nesterov(), RMSProp(),
                     ADAGrad(), AdaMax(), ADADelta(), AMSGrad(), NADAM(),
                     ADAMW(), RADAM(), OADAM(), AdaBelief())
    w′ = (α = rand(3, 3), β = rand(3, 3))

    # Original example
    w = (α = 5rand(3, 3), β = rand(3, 3))
    st = Optimisers.setup(o, w)
    loss(x, y) = mean((x.α .* x.β .- y.α .* y.β) .^ 2)
    @test loss(w, w′) > 1
    for i = 1:10^4
      gs = gradient(x -> loss(x, w′), w)
      st, w = Optimisers.update(st, w, gs...)
    end
    lw = loss(w, w′)
    if o isa ADADelta
      @test_broken lw < 0.001
    else
      @test lw < 0.001
    end

    # Slightly harder variant
    m = (α = randn(3), β = transpose(5rand(3,3)), γ = (rand(2), tanh))  # issue 28
    st = Optimisers.setup(o, m)
    @test loss(m, w′) > 1
    for i = 1:10^4
      gs = gradient(x -> loss(x, w′), m)
      st, m = Optimisers.update!(st, m, gs...)
    end
    lm = loss(m, w′)
    if lm < 0.1
      @test lm < 0.1
    else
      @test_broken lm < 0.1  # @test keyword broken doesn't exist on Julia 1.6
    end

  end

  @testset "OptimiserChain with $pre" for pre in (WeightDecay(), ClipGrad(), ClipNorm())
    Random.seed!(84)
    w = randn(10, 10)
    w′ = randn(10, 10)
    loss(x, w, w′) = mean((w*x .- w′*x) .^ 2)
    @test loss(rand(10, 10), w, w′) > 1
    opt = OptimiserChain(pre, ADAM(0.001))
    st = Optimisers.init(opt, w)
    for t = 1:10^5
      x = rand(10)
      gs = gradient(w -> loss(x, w, w′), w)
      st, w = Optimisers.update!(st, w, gs...)
    end
    @test_broken loss(rand(10, 10), w, w′) < 0.01
  end

  @testset "gradient clipping" begin
    m = (α = ([0], sin), γ = rand(3))
    s1 = Optimisers.setup(ClipGrad(13), m)
    _, m1 = Optimisers.update(s1, m, (α = nothing, γ = [1,10,100],))
    @test m.γ .- m1.γ ≈ [1, 10, 13]

    s2 = Optimisers.setup(ClipNorm(10), m)
    _, m2 = Optimisers.update(s2, m, (α = ([0.1], nothing), γ = [1,10,100],))
    @test only(m.α[1] .- m2.α[1]) ≈ 0.1
    @test norm(m.γ .- m2.γ) ≈ 10
    @test_throws DomainError Optimisers.update(s2, m, (α = [0.1], γ = [1,10,NaN],))

    s3 = Optimisers.setup(ClipNorm(5, 1; throw=false), m)
    _, m3 = Optimisers.update(s3, m, (α = ([0.1], nothing), γ = [1,10,100],))
    @test only(m.α[1] .- m3.α[1]) ≈ 0.1
    @test norm(m.γ .- m3.γ, 1) ≈ 5
    _, m3n = Optimisers.update!(s3, m, (α = nothing, γ = [1,10,Inf],))
    @test isnan(m3n.γ[3])
  end

  @testset "broadcasting macro" begin
    x = [1.0, 2.0]; y = [3,4]; z = [5,6]
    @test (@.. x + y * z) isa Broadcast.Broadcasted
    bc = @.. x + y * z
    @test (y .+ 2 .* bc) == [35,56]

    @test (@.. x = y * z) isa Array
    @test x == y .* z  # mutated
    r = 1.0:2.0  # immutable
    @test (@.. r = y * z) isa Array
    @test (@.. r = y * z) == y .* z
  end

end
