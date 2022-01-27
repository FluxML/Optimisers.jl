using Optimisers, Test
using Zygote, Random
using Statistics

@testset "Optimisers" begin
  Random.seed!(84)
  w′ = (α = rand(3, 3), β = rand(3, 3))
  @testset for o in (Descent(), ADAM(), Momentum(), Nesterov(), RMSProp(),
                     ADAGrad(), AdaMax(), ADADelta(), AMSGrad(), NADAM(),
                     ADAMW(), RADAM(), OADAM(), AdaBelief())

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
    @test lw < 0.001  broken = lw > 0.001

    # Slightly harder variant
    m = (α = 5randn(3), β = transpose(randn(3,3)), γ = (rand(2), tanh))  # issue 28
    st = Optimisers.state(o, m)
    @test loss(m, w′) > 1
    for i = 1:10^4
      gs = gradient(x -> loss(x, w′), m)
      st, m = o(st, m, gs...)
    end
    lm = loss(m, w′)
    @test lm < 0.1  broken = lm > 0.1

  end
end

@testset "OptimiserChain" begin
  Random.seed!(84)
  w = randn(10, 10)
  w′ = randn(10, 10)
  loss(x, w, w′) = mean((w*x .- w′*x) .^ 2)
  opt = OptimiserChain(WeightDecay(), ADAM(0.001))
  st = Optimisers.state(opt, w)
  for t = 1:10^5
    x = rand(10)
    gs = gradient(w -> loss(x, w, w′), w)
    st, w = Optimisers.update(opt, st, w, gs...)
  end
  @test loss(rand(10, 10), w, w′) < 0.01
end

@testset "Optimiser Updates" begin
  opt = ADAM()
  new_opt = ADAM(opt, eta = 9.f0)
  @test new_opt.eta == 9.f0
end
