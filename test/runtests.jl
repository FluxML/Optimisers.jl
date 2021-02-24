using Optimisers, Test
using Zygote, Random
using Statistics

@testset "Optimisers" begin
  Random.seed!(84)
  w′ = (α = rand(3, 3), β = rand(3, 3))
  @testset for o in (Descent(), ADAM(), Momentum(), Nesterov(), RMSProp(),
                     ADAGrad(), AdaMax(), ADADelta(), AMSGrad(), NADAM(),
                     ADAMW(), RADAM(), OADAM(), AdaBelief())
    w = (α = rand(3, 3), β = rand(3, 3))
    st = Optimisers.state(o, w)
    loss(x, y) = mean((x.α .* x.β .- y.α .* y.β) .^ 2)
    l = loss(w, w′)
    for i = 1:10^4
      gs = gradient(x -> loss(x, w′), w)
      w, st = o(st, w, gs...)
    end
    @test loss(w, w′) < 0.01
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
    w, st = Optimisers.update(opt, st, w, gs...)
  end
  @test loss(rand(10, 10), w, w′) < 0.01
end

@testset "Optimiser Updates" begin
  opt = ADAM()
  new_opt = ADAM(opt, eta = 9.f0)
  @test new_opt.eta == 9.f0
end
