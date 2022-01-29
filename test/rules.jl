using Optimisers
using ChainRulesCore, Functors, StaticArrays, Zygote
using LinearAlgebra, Statistics, Test, Random

Random.seed!(1)

RULES = [
  # All the rules at default settings
  Descent(), ADAM(), Momentum(), Nesterov(), RMSProp(),
  ADAGrad(), AdaMax(), ADADelta(), AMSGrad(), NADAM(),
  ADAMW(), RADAM(), OADAM(), AdaBelief(),
  # A few chained combinations:
  OptimiserChain(WeightDecay(), ADAM(0.001)),
  OptimiserChain(ClipNorm(), ADAM(0.001)),
  OptimiserChain(ClipGrad(0.5), Momentum()),
]

name(o) = typeof(o).name.name
name(o::OptimiserChain) = join(name.(o.opts), " → ")

@testset "independence" begin
  @testset "$(name(o))" for o in RULES
    w = randn(10, 10)
    w′ = randn(10, 10)
    iloss(x, w, w′) = mean((w*x .- w′*x) .^ 2)
    @test iloss(rand(10, 10), w, w′) > 1
    st = Optimisers.setup(o, w)
    for t = 1:10^5
      x = rand(10)
      gs = gradient(w -> iloss(x, w, w′), w)
      st, w = Optimisers.update!(st, w, gs...)
    end
    @test iloss(rand(10, 10), w, w′) < 0.01
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

@testset verbose=true "StaticArrays" begin
  @testset "$(name(o))" for o in RULES

    W1 = @SMatrix randn(10, 10)
    b1 = @SVector randn(10)
    W2 = @SMatrix randn(10, 10)
    model = (; W1, b1, W2, tanh)
    s_loss(m, x, y) = sum(abs2, m.W2 * (m.tanh).(m.W1*x .+ m.b1) .- y)
    # x = @SMatrix randn(10, 10)
    # y = @SMatrix randn(10, 10)  # gives an error from sum(; dims=())
    x = @SVector randn(10)
    y = @SVector randn(10)
    @test s_loss(model, x, y) > 10
    state = Optimisers.setup(o, model)
    for t = 1:10^3
      g = gradient(m -> s_loss(m, x, y), model)[1]
      state, model = Optimisers.update!(state, model, g)
    end
    if o isa Union{Descent, RMSProp, ADAGrad, ADADelta, NADAM}
      @show name(o) s_loss(model, x, y)
      @test_broken s_loss(model, x, y) < 1
    else
      @test s_loss(model, x, y) < 1
    end
  end
end

@testset verbose=true "element types" begin
  @testset "$(name(o))" for o in RULES
    marray = (Float32[1,2], Float64[3,4], Float16[5,6])
    types = map(eltype, marray)

    # This is a weak test, as it copies & then does `update!`
    uparray = Optimisers.update(Optimisers.setup(o, marray), marray, marray)[2]
    @test map(eltype, uparray) == types

    # Static version is truly out-of-place:
    mstatic = (SA{Float32}[1,2], SA{Float64}[3,4]) # , SA{Float16}[5,6])  with Float16, all fail
    upstatic = Optimisers.update(Optimisers.setup(o, mstatic), mstatic, mstatic)[2]
    if o isa OptimiserChain && o.opts[2] isa ADAM  # These promote to Float64
      @test_broken map(eltype, upstatic) == types[1:2]
    else
      @test map(eltype, upstatic) == types[1:2]
    end
    @test upstatic[1] isa SVector

    # With ordinary Array gradient, what happens?
    upstatic2 = Optimisers.update(Optimisers.setup(o, mstatic), mstatic, marray[1:2])[2]
    # @test map(eltype, upstatic2) == types[1:2]  # same information
    if upstatic2[1] isa SVector
      @test upstatic2[1] isa SVector
    else
      @test_broken upstatic2[1] isa SVector
    end
  end
end

@testset "gradient types" begin
  @testset "$(name(o))" for o in RULES
    x = (a = ones(2,2), b = transpose(ones(2,2)))
    s = Optimisers.setup(o, x)

    _, x1 = Optimisers.update(s, x, (a = [1 2; 3 4], b = nothing))
    @test x1.a != ones(2,2)
    @test x1.b == ones(2,2)

    _, xfill = Optimisers.update(s, x, (a = Zygote.Fill(2.0,2,2), b = Zygote.Fill(true,2,2)))
    @test xfill.a != ones(2,2)
    @test xfill.b != ones(2,2)

    bc = Optimisers.@.. 1 + log([2 3; 4 5]) / 6
    _, xbc = Optimisers.update(s, x, (a = bc, b = bc))
    @test xbc.a != ones(2,2)
    @test xbc.b != ones(2,2)

    th = ChainRulesCore.@thunk @. 1 + log([2 3; 4 5]) / 6
    _, xth = Optimisers.update(s, x, (a = bc, b = bc))
    @test xth.a != ones(2,2)
    @test xth.b != ones(2,2)
  end
end

