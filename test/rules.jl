using Optimisers
using ChainRulesCore, Functors, StaticArrays, Zygote
using LinearAlgebra, Statistics, Test, Random

Random.seed!(1)

RULES = [
  # All the rules at default settings:
  Descent(), Adam(), Momentum(), Nesterov(), Rprop(), RMSProp(),
  AdaGrad(), AdaMax(), AdaDelta(), AMSGrad(), NAdam(),
  AdamW(), RAdam(), OAdam(), AdaBelief(), Lion(),
  MixedPrecision(Float64, Adam()),
  # A few chained combinations:
  OptimiserChain(SignDecay(0.001), Adam(0.001)),
  OptimiserChain(ClipNorm(), Adam(0.001)),
  OptimiserChain(ClipGrad(0.5), Momentum()),
  OptimiserChain(WeightDecay(), OAdam(), ClipGrad(1)),
  # Not the default:
  RMSProp(centred = true), AdamW(couple=false),
]

MATRIX_RULES = [
  # All the rules at default settings:
  Muon(),
  # A few chained combinations:
  OptimiserChain(WeightDecay(), Muon()),
  # Not the default:
  Muon(nesterov=false),
]

name(o) = typeof(o).name.name  # just for printing testset headings
name(o::OptimiserChain) = join(name.(o.opts), " → ")
name(o::RMSProp) = o.centred ? "RMSProp(centred = true)" : :RMSProp
name(o::Muon) = o.nesterov ? :Muon : "Muon(nesterov = false)"

LOG = Dict()  # for debugging these testsets, this makes it easy to plot each optimiser's loss

loggradient(o) = (f, xs...) -> begin
  y, dxs = Zygote.withgradient(f, xs...)
  push!(get!(() -> Float32[], LOG, name(o)), y)
  dxs  # save the loss, return the gradient
end

@testset "independence" begin
  empty!(LOG)
  @testset "$(name(o))" for o in vcat(RULES, MATRIX_RULES)
    w = randn(10, 10)
    w′ = randn(10, 10)
    iloss(x, w, w′) = mean((w*x .- w′*x) .^ 2)
    @test iloss(rand(10, 10), w, w′) > 1
    st = Optimisers.setup(o, w)
    for t = 1:(10^5)
      x = rand(10, 20)
      gs = loggradient(o)(w -> iloss(x, w, w′), w)
      st, w = Optimisers.update!(st, w, gs...)
    end
    @test iloss(rand(10, 10), w, w′) < 0.01
  end
end

@testset "simple sum" begin
  empty!(LOG)
  @testset "$(name(o))" for o in vcat(RULES, MATRIX_RULES)
    m = shuffle!(reshape(1:64, 8, 8) .+ 0.0)
    s = Optimisers.setup(o, m)
    for _ in 1:(10^5)
      g = loggradient(o)(x -> sum(abs2, x + x'), m)[1]
      s, m = Optimisers.update!(s, m, g)
    end
    @test sum(m) < sum(1:64)
    if sum(m) < 1
      @test sum(m) < 1
    else
      @show name(o) sum(m)/sum(1:64)
      @test_broken sum(m) < 1
    end
  end
end

@testset "original" begin
  empty!(LOG)
  @testset "$(name(o))" for o in vcat(RULES, MATRIX_RULES)
    w′ = (α = rand(3, 3), β = rand(3, 3))
    w = (α = 5rand(3, 3), β = rand(3, 3))
    st = Optimisers.setup(o, w)
    loss(x, y) = mean((x.α .* x.β .- y.α .* y.β) .^ 2)
    @test loss(w, w′) > 1
    for i = 1:(10^4)
      gs = loggradient(o)(x -> loss(x, w′), w)
      st, w = Optimisers.update(st, w, gs...)
    end
    @test loss(w, w′) < 0.001
  end
end

@testset "StaticArrays" begin
  empty!(LOG)
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
    for t = 1:(10^3)
      g = loggradient(o)(m -> s_loss(m, x, y), model)[1]
      state, model = Optimisers.update!(state, model, g)
    end
    if o isa Descent
      @show name(o) s_loss(model, x, y)
      @test_broken s_loss(model, x, y) < 1
    else
      @test s_loss(model, x, y) < 1
    end
  end
end

@testset "element types" begin
  # Rules which accept any shape are checked on vectors and on matrices, while
  # those which require a matrix, such as Muon, are checked only on the latter.
  @testset "$(name(o)), size $sz" for (sz, rules) in ((2,) => RULES, (2, 2) => vcat(RULES, MATRIX_RULES)),
                                       o in rules
    marray = map(T -> reshape(T.(1:prod(sz)), sz), (Float32, Float64, Float16))
    types = map(eltype, marray)

    # This is a weak test, as it copies & then does `update!`
    uparray = Optimisers.update(Optimisers.setup(o, marray), marray, marray)[2]
    @test map(eltype, uparray) == types

    # Static version is truly out-of-place:
    mstatic = map(a -> SArray{Tuple{sz...}}(a), marray[1:2])  # with Float16, all fail
    upstatic = Optimisers.update(Optimisers.setup(o, mstatic), mstatic, mstatic)[2]
    @test map(eltype, upstatic) == types[1:2]
    @test upstatic[1] isa SArray

    # With ordinary Array gradient, what happens? Not so important!
    upstatic2 = Optimisers.update(Optimisers.setup(o, mstatic), mstatic, marray[1:2])[2]
    # @test map(eltype, upstatic2) == types[1:2]  # same information
    if upstatic2[1] isa SArray
      @test upstatic2[1] isa SArray
    else
      @test_broken upstatic2[1] isa SArray
    end
  end
end

@testset "gradient types" begin
  @testset "$(name(o))" for o in vcat(RULES, MATRIX_RULES)
    x = (a = ones(2,2), b = transpose(ones(2,2)))
    s = Optimisers.setup(o, x)

    _, x1 = Optimisers.update(s, x, (a = [1 2; 3 4], b = nothing))
    @test x1.a != ones(2,2)
    @test x1.b == ones(2,2)

    _, xfill = Optimisers.update(s, x, (a = Zygote.Fill(2.0,2,2), b = Zygote.Fill(true,2,2)))
    @test xfill.a != ones(2,2)
    @test xfill.b != ones(2,2)

    bc = Optimisers.@lazy 1 + log([2 3; 4 5]) / 6
    _, xbc = Optimisers.update(s, x, (a = bc, b = bc))
    @test xbc.a != ones(2,2)
    @test xbc.b != ones(2,2)

    th = ChainRulesCore.@thunk @. 1 + log([2 3; 4 5]) / 6
    _, xth = Optimisers.update(s, x, (a = bc, b = bc))
    @test xth.a != ones(2,2)
    @test xth.b != ones(2,2)
  end
end

@testset "mutation check" begin
  # If @lazy captures a matrix which is later mutated, the results won't agree here:
  @testset "$(name(o))" for o in RULES
    model = Float64.(rand(Int8, 8))
    s_model = SVector{8}(model)
    grads = [Float64.(rand(Int8, 8)) for t in 1:13]
    s_grads = [SVector{8}(x) for x in grads]
    state = Optimisers.setup(o, model)
    s_state = Optimisers.setup(o, s_model)
    for t in 1:13
      state, model = Optimisers.update!(state, model, grads[t])
      s_state, s_model = Optimisers.update!(s_state, s_model, s_grads[t])
    end
    @test model == s_model
  end
end

@testset "with complex numbers: Flux#1776" begin
  empty!(LOG)
  @testset "$(name(opt))" for opt in [
              # The Flux PR had 1e-2 for all. But AdaDelta(ρ) needs ρ≈0.9 not small. And it helps to make ε not too small too:
              Adam(1e-2), RMSProp(1e-2), RAdam(1e-2), OAdam(1e-2), AdaGrad(1e-2), AdaDelta(0.9, 1e-5), NAdam(1e-2), AdaBelief(1e-2),
              # These weren't in Flux PR:
              Descent(1e-2), Momentum(1e-2), Nesterov(1e-2), AdamW(1e-2), 
              ]
    # Our "model" is just a complex number
    model = (w = zeros(ComplexF64, 1),)

    # Our model attempts to learn `f(x) = conj(x)` where `f(x) = w*x`
    function loss(m)
      # Deterministic training data is the best training data
      x = ones(1, 1) + 1im*ones(1, 1)
      # Manually implement `mse()` to allow demonstration of brokenness
      # on older Flux builds that don't have a fixed `mse()`
      return sum(abs2.(m.w * x .- conj(x)))
    end
    @test loss(model) ≈ 2.0

    state = Optimisers.setup(opt, model)

    # Train for 10 iterations, enforcing that loss is monotonically decreasing
    last_loss = Inf
    for idx in 1:10
      grads = loggradient(opt)(loss, model)
      state, model = Optimisers.update!(state, model, grads...)
      opt isa Union{Momentum, Nesterov} && idx > 8 && continue  # these are very flat at the end
      @test loss(model) < last_loss
      last_loss = loss(model)
    end
    @test loss(model) < 1.9

    # Repeat with StaticArrays
    static_model = (w = SA[0.0 + 0im],)
    static_state = Optimisers.setup(opt, static_model)
    function static_loss(m)
      x = hcat(SA[1.0 + im])
      sum(abs2.(m.w * x .- conj(x)))
    end
    last_loss = Inf
    for idx in 1:10
      grads = gradient(static_loss, static_model)
      static_state, static_model = Optimisers.update!(static_state, static_model, grads...)
      opt isa Union{Momentum, Nesterov} && idx > 8 && continue
      @test static_loss(static_model) < last_loss
      last_loss = static_loss(static_model)
    end
    @test static_loss(static_model) < 1.9 
  end
end

@testset "AccumGrad" begin
  x0 = rand(5)
  x = copy(x0)
  lr = 0.01
  tree = Optimisers.setup(OptimiserChain(AccumGrad(3), Descent(lr)), x)

  g1 = rand(5)
  tree, x1 = Optimisers.update(tree, x, g1)
  @test x1 ≈ x
  @test x1 ≈ x0 
  g2 = rand(5)
  tree, x2 = Optimisers.update(tree, x1, g2)
  @test x2 ≈ x
  @test x2 ≈ x0 
  g3 = rand(5)
  tree, x3 = Optimisers.update(tree, x2, g3)
  @test x3 ≈ x0 .- lr .* (g1 .+ g2 .+ g3) ./ 3
  g4 = rand(5)
  
  tree, x4 = Optimisers.update(tree, x3, g4)
  @test x4 ≈ x3
end

@testset "Float16 epsilon" begin
  # issue https://github.com/FluxML/Optimisers.jl/issues/167
  x = Float16[0.579, -0.729, 0.5493]
  δx = Float16[-0.001497, 0.0001875, -0.013176]

  os = Optimisers.setup(Adam(1e-4), x);
  os, x = Optimisers.update(os, x, δx)
  @test x ≈ Float16[1.835, -0.886, 0.5493] rtol=1e-3
end

@testset "MixedPrecision" begin
  x = rand(Float16, 2)
  opt_state = Optimisers.setup(MixedPrecision(Adam(1e-3)), x)
  @test opt_state.state[1] isa Vector{Float32}
  @test opt_state.state[2][1] isa Vector{Float32}
  g = rand(Float16, 2)
  new_state, new_x = Optimisers.update(opt_state, x, rand(Float16, 2))
  @test new_x == Float16.(new_state.state[1])
  @test new_x ≈ x .- 1e-3 .* g

  x = rand(Float16, 2)
  opt_state = Optimisers.setup(MixedPrecision(Float64, Adam(1e-3)), x)
  @test opt_state.state[1] isa Vector{Float64}
  @test opt_state.state[2][1] isa Vector{Float64}

  opt = MixedPrecision(Float64, Adam(1e-3))
  opt2 = Optimisers.adjust(opt, 2e-3)
  @test opt2.rule.eta == 2e-3
  @test opt2 isa MixedPrecision{Float64, <:Adam}

  @test_throws ArgumentError OptimiserChain(MixedPrecision(Adam()))
end

@testset "Muon" begin
  @testset "only matrices" begin
    # The error must come from setup, not from the first update, so that a model
    # with unsuitable parameters fails before any training happens.
    @test_throws ArgumentError Optimisers.setup(Muon(), rand(2, 3, 4))
    @test_throws ArgumentError Optimisers.setup(Muon(), (α = rand(3, 3), β = rand(3)))
  end

  @testset "reference algorithm" begin
    # Momentum is an EMA, and the update is scaled by sqrt(max(1, rows/cols)), which
    # orthogonalisation alone would not give. Two steps are needed to tell `nesterov`
    # apart: from B₀ = 0 the lookahead is 1.95 * B₁, a positive multiple which
    # _newton_schulz5 normalises away, so both settings agree on the first step.
    η, μ = 0.02, 0.95
    @testset "nesterov = $nesterov, size ($m, $n)" for nesterov in (true, false),
                                                       (m, n) in ((6, 4), (4, 6))
      x = randn(m, n)
      γ = sqrt(max(1, m / n))
      st = Optimisers.setup(Muon(η, μ, nesterov), x)

      B = zero(x)
      for _ in 1:2
        g = randn(m, n)
        B = μ * B + (1 - μ) * g
        U = nesterov ? (1 - μ) * g + μ * B : B
        expected = x .- η * γ * Optimisers._newton_schulz5(U)

        st, x = Optimisers.update(st, x, g)
        @test x ≈ expected
      end
    end
  end

  @testset "_newton_schulz5" begin
    @testset "size ($m, $n)" for (m, n) in ((32, 32), (64, 16), (16, 64))
      # Give the input a known spectrum: the iteration acts only on the singular
      # values, so how close it gets depends on the conditioning of the input, not
      # on the random draw.
      F = svd(randn(m, n))
      G = F.U * Diagonal(range(1, 2, length=min(m, n))) * F.Vt

      O = Optimisers._newton_schulz5(G)
      @test size(O) == (m, n)
      # The tuned quintic deliberately stops short of exactly orthogonal.
      @test all(s -> 0.68 < s < 1.14, svdvals(Matrix(O)))
    end

    # Five fixed steps cannot fully orthogonalise a badly conditioned matrix, but
    # they still improve it by two orders of magnitude.
    F = svd(randn(32, 32))
    G = F.U * Diagonal(range(1, 1e-3, length=32)) * F.Vt
    @test cond(Matrix(Optimisers._newton_schulz5(G))) < cond(G) / 100

    # Orientation is handled internally, so the iteration commutes with transposing.
    G = randn(64, 16)
    @test Optimisers._newton_schulz5(collect(G')) ≈ Optimisers._newton_schulz5(G)'

    # Normalising by the Frobenius norm discards any positive scalar multiple, which
    # is what makes `nesterov` a no-op on the first step above.
    @test Optimisers._newton_schulz5(2G) ≈ Optimisers._newton_schulz5(G) rtol=1e-5

    @testset "$T" for T in (Float32, Float64, Float16)
      @test eltype(Optimisers._newton_schulz5(randn(T, 8, 6))) == T
    end
  end
end
