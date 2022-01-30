using Optimisers
using ChainRulesCore, Functors, StaticArrays, Zygote
using LinearAlgebra, Statistics, Test, Random

Random.seed!(1)

RULES = [
  # All the rules at default settings:
  Descent(), ADAM(), Momentum(), Nesterov(), RMSProp(),
  ADAGrad(), AdaMax(), ADADelta(), AMSGrad(), NADAM(),
  ADAMW(), RADAM(), OADAM(), AdaBelief(),
  # A few chained combinations:
  OptimiserChain(WeightDecay(), ADAM(0.001)),
  OptimiserChain(ClipNorm(), ADAM(0.001)),
  OptimiserChain(ClipGrad(0.5), Momentum()),
  OptimiserChain(WeightDecay(), OADAM(), ClipGrad(1)),
]

name(o) = typeof(o).name.name
name(o::OptimiserChain) = join(name.(o.opts), " → ")

LOG = Dict()

loggradient(o) = (f, xs...) -> begin
  y, dxs = Zygote.withgradient(f, xs...)
  push!(get!(() -> Float32[], LOG, name(o)), y)
  dxs  # save the loss, return the gradient
end

@testset "independence" begin
  empty!(LOG)
  @testset "$(name(o))" for o in RULES
    w = randn(10, 10)
    w′ = randn(10, 10)
    iloss(x, w, w′) = mean((w*x .- w′*x) .^ 2)
    @test iloss(rand(10, 10), w, w′) > 1
    st = Optimisers.setup(o, w)
    for t = 1:10^5
      x = rand(10)
      gs = loggradient(o)(w -> iloss(x, w, w′), w)
      st, w = Optimisers.update!(st, w, gs...)
    end
    @test iloss(rand(10, 10), w, w′) < 0.01
  end
end

@testset verbose=true "simple sum" begin
  empty!(LOG)
  @testset "$(name(o))" for o in RULES
    m = shuffle!(reshape(1:64, 8, 8) .+ 0.0)
    s = Optimisers.setup(o, m)
    for _ in 1:10^5
      g = loggradient(o)(x -> sum(abs2, x + x'), m)[1]
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

#=
using Plots
_plot(s; kw...) = (plot(); _plot!(s; kw...))
_plot!(s; kw...) = plot!(LOG[s]; label=string(s), yguide="loss", xguide="iter", kw...)

_plot(:ADAGrad)  # decline
LOG[:ADAGrad][end]  # 3869.4075f0

_plot!(:AMSGrad)  # decline
LOG[:AMSGrad][end]  # 2742.004f0

findfirst(isnan, LOG[:ADADelta]) # 182
plot(LOG[:ADADelta][1:100], yaxis=:log10)  # exp growth
=#

@testset "original" begin
  empty!(LOG)
  @testset "$(name(o))" for o in RULES
    w′ = (α = rand(3, 3), β = rand(3, 3))
    w = (α = 5rand(3, 3), β = rand(3, 3))
    st = Optimisers.setup(o, w)
    loss(x, y) = mean((x.α .* x.β .- y.α .* y.β) .^ 2)
    @test loss(w, w′) > 1
    for i = 1:10^4
      gs = loggradient(o)(x -> loss(x, w′), w)
      st, w = Optimisers.update(st, w, gs...)
    end
    @test loss(w, w′) < 0.001
  end
end

@testset verbose=true "StaticArrays" begin
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
    for t = 1:10^3
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

#=
plot(LOG[:Descent])  # decline
plot(LOG[:RMSProp])  # const 10^11
plot(LOG[:ADAGrad])  # const 10^8
plot(LOG[:ADADelta][1:30], yaxis=:log10)  # exp growth
plot(LOG[name(NADAM())])  # stuck at 10^11
=#

@testset "element types" begin
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

    # With ordinary Array gradient, what happens? Not so important!
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

@testset verbose=true "mutation check" begin
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

@testset verbose=true "with complex numebers: Flux#1776" begin
  empty!(LOG)
  @testset "$(name(f(1e-2)))" for f in [
              ADAM, RMSProp, RADAM, OADAM, ADAGrad, ADADelta, NADAM, AdaBelief,
              Descent, Momentum, Nesterov, ADAMW, # not in Flux PR
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

    opt = f(1e-2)
    state = Optimisers.setup(opt, model)

    # Train for 10 iterations, enforcing that loss is monotonically decreasing
    last_loss = Inf
    for idx in 1:10
      grads = loggradient(opt)(loss, model)
      state, model = Optimisers.update!(state, model, grads...)
      if opt isa Union{Momentum, Nesterov}
        @test_skip loss(model) < last_loss
      else
        @test loss(model) < last_loss
      end
      last_loss = loss(model)
    end

    # Repeat with StaticArrays
    static_model = (w = SA[1.0 + 0im],)
    static_state = Optimisers.setup(opt, static_model)
    function static_loss(m)
      x = hcat(SA[1.0 + im])
      sum(abs2.(m.w * x .- conj(x)))
    end
    last_loss = Inf
    for idx in 1:10
      grads = gradient(static_loss, static_model)
      static_state, static_model = Optimisers.update!(static_state, static_model, grads...)
      if opt isa Union{Momentum, Nesterov}
        @test_skip loss(static_model) < last_loss
      else
        @test loss(static_model) < last_loss
      end
      last_loss = loss(static_model)
    end
  end
end

#=

_plot(:ADAM)  # nice
_plot!(:RADAM)
_plot!(:OADAM)

_plot!(:RMSProp) 
_plot!(:NADAM)
_plot!(:ADADelta)  # barely declines

=#
