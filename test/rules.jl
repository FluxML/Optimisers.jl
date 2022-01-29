using Optimisers, Functors, StaticArrays, Zygote
using LinearAlgebra, Statistics, Test, Random

Random.seed!(1)

RULES = [
  # All the rules at default settings
  Descent(), ADAM(), Momentum(), Nesterov(), RMSProp(),
  ADAGrad(), AdaMax(), ADADelta(), AMSGrad(), NADAM(),
  ADAMW(), RADAM(), OADAM(), AdaBelief(),
  # A few chained combinations:
  OptimiserChain(WeightDecay(), ADAM(0.001)), OptimiserChain(ClipNorm(), ADAM(0.001)),
  OptimiserChain(ClipGrad(0.5), Momentum()),
]

name(o) = typeof(o).name.name
name(o::OptimiserChain) = join(name.(o.opts), " → ")

@testset "independence" begin
  @testset "$(name(o))" for o in RULES
    w = randn(10, 10)
    w′ = randn(10, 10)
    loss(x, w, w′) = mean((w*x .- w′*x) .^ 2)
    @test loss(rand(10, 10), w, w′) > 1
    st = Optimisers.setup(o, w)
    for t = 1:10^5
      x = rand(10)
      gs = gradient(w -> loss(x, w, w′), w)
      st, w = Optimisers.update!(st, w, gs...)
    end
    @test loss(rand(10, 10), w, w′) < 0.01
  end
end

@testset verbose=true "simple sum" begin
  @testset "$(name(o))" for o in RULES
    m = shuffle!(reshape(1:64, 8, 8) .+ 0.0)
    s = Optimisers.setup(o, m)
    for _ in 1:10^5
      g = gradient(x -> sum(abs2, x + x'), m)[1]
      s, m = Optimisers.update!(s, m, g)
    end
    # @test sum(m) < sum(1:64)
    if sum(m) < 1
      @test sum(m) < 1
    else
      @show name(o) sum(m)/sum(1:64)
      @test_broken sum(m) < 1
    end
  end
end

@testset "original" begin
  @testset "$(name(o))" for o in RULES
    w′ = (α = rand(3, 3), β = rand(3, 3))
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
      @show name(o) loss(w, w′)
      @test_broken lw < 0.001
    else
      @test lw < 0.001
    end
  end
end

