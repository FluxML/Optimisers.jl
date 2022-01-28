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
    s = Optimisers.state(o, m)

    s2, m2 = Optimisers.update(o, s, m, g)
    @test m[1] == 1:2  # not mutated
    @test Optimisers.iswriteable(m[1])
    @test m2[1] ≈ [1,2] .- 0.1 .* [25, 33]

    s3, m3 = Optimisers.update!(o, s, m, g)
    @test objectid(m3[1]) == mid
    @test m3[1] ≈ [1,2] .- 0.1 .* [25, 33]
  end

  ALL_OPTS = (Descent(), ADAM(), Momentum(), Nesterov(), RMSProp(),
              ADAGrad(), AdaMax(), ADADelta(), AMSGrad(), NADAM(),
              ADAMW(), RADAM(), OADAM(), AdaBelief())
  ALL_TRANSFORMS = (ALL_OPTS..., ClipGrad(), ClipNorm(), WeightDecay())

  @testset "Training with gradients" begin
    @testset for o in ALL_OPTS
      w′ = (α = rand(3, 3), β = rand(3, 3))

      # Original example
      w = (α = 5rand(3, 3), β = rand(3, 3))
      st = Optimisers.state(o, w)
      loss(x, y) = mean((x.α .* x.β .- y.α .* y.β) .^ 2)
      @test loss(w, w′) > 1
      for i = 1:10^4
        gs = gradient(x -> loss(x, w′), w)
        st, w = Optimisers.update(o, st, w, gs...)
      end
      lw = loss(w, w′)
      if o isa ADADelta
        @test_broken lw < 0.001
      else
        @test lw < 0.001
      end

      # Slightly harder variant
      m = (α = randn(3), β = transpose(5rand(3,3)), γ = (rand(2), tanh))  # issue 28
      st = Optimisers.state(o, m)
      @test loss(m, w′) > 1
      for i = 1:10^4
        gs = gradient(x -> loss(x, w′), m)
        st, m = o(st, m, gs...)
      end
      lm = loss(m, w′)
      if lm < 0.1
        @test lm < 0.1
      else
        @test_broken lm < 0.1  # @test keyword broken doesn't exist on Julia 1.6
      end
    end
  end

  @testset "Scalar Params" begin
    m = (α = 3, β = 5f0, γ = 2.0+2.0im)
    gs = (α = nothing, β = 1f0, γ = 1.0+1.0im)

    # General compatibility
    @testset for t in ALL_TRANSFORMS
      m2, gs2 = m, gs
      # These can't handle complex numbers
      if t isa Union{AdaMax,AMSGrad,ClipGrad}
        m2, gs2 = m[(:α, :β)], gs[(:α, :β)]
      end
      st = Optimisers.state(t, m2)
      Optimisers.update(t, st, m2, gs2)
    end

    # End-to-end
    o = Descent(0.1)
    st = Optimisers.state(o, m)
    _, m′ = Optimisers.update(o, st, m, gs)
    @test m.β - 0.1gs.β ≈ m′.β
    @test m.γ - 0.1gs.γ ≈ m′.γ
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
      st, w = Optimisers.update(opt, st, w, gs...)
    end
    @test loss(rand(10, 10), w, w′) < 0.01
  end

  @testset "gradient clipping" begin
    @test_skip m = (α = ([0], sin), γ = rand(3))  # https://github.com/FluxML/Optimisers.jl/issues/28
    m = (α = ([0], [0]), γ = rand(3))
    c1 = ClipGrad(13)
    s1 = Optimisers.state(c1, m)
    _, g1 = Optimisers.update(c1, s1, m, (α = nothing, γ = [1,10,100],))
    @test m.γ .- g1.γ ≈ [1, 10, 13]

    c2 = ClipNorm(10)
    s2 = Optimisers.state(c2, m)
    _, g2 = Optimisers.update(c2, s2, m, (α = ([0.1], nothing), γ = [1,10,100],))
    @test only(m.α[1] .- g2.α[1]) ≈ 0.1
    @test norm(m.γ .- g2.γ) ≈ 10
    @test_throws DomainError Optimisers.update(c2, s2, m, (α = [0.1], γ = [1,10,NaN],))

    c3 = ClipNorm(5, 1; throw=false)
    _, g3 = Optimisers.update(c3, s2, m, (α = ([0.1], nothing), γ = [1,10,100],))
    @test only(m.α[1] .- g3.α[1]) ≈ 0.1
    @test norm(m.γ .- g3.γ, 1) ≈ 5
    _, g3n = Optimisers.update(c3, s2, m, (α = nothing, γ = [1,10,Inf],))
    @test isnan(g3n.γ[3])
  end

  @testset "Optimiser Updates" begin
    opt = ADAM()
    new_opt = ADAM(opt, eta = 9.f0)
    @test new_opt.eta == 9.f0
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
