using Optimisers
using ChainRulesCore, Functors, StaticArrays, Zygote
using LinearAlgebra, Statistics, Test, Random
using Optimisers: @.., @lazy

Random.seed!(1)

struct Foo; x; y; end
Functors.@functor Foo
Optimisers.trainable(x::Foo) = (x.y, x.x)

struct TwoThirds a; b; c; end
Functors.@functor TwoThirds (a, c)
Optimisers.trainable(x::TwoThirds) = (a = x.a,)

@testset verbose=true "Optimisers.jl" begin
  @testset verbose=true "Features" begin

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

      g4 = Tangent{typeof(m)}(g...)
      s4, m4 = Optimisers.update!(s, ([1.0, 2.0],), g4)
      @test m4[1] ≈ [1,2] .- 0.1 .* [25, 33]
    end

    @testset "gradient clipping" begin
      m = (α = ([0.0], sin), γ = rand(3))
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

    @testset "OptimiserChain" begin
      x = [1, 10, 100.0]; dx = [1, 2, 3.0];
      @test Optimisers.update(Optimisers.setup(WeightDecay(0.1), x), x, dx)[2] ≈ [1-0.1-1, 10-1-2, 100-10-3]
      @test Optimisers.update(Optimisers.setup(ClipGrad(2), x), x, dx)[2] ≈ [1-1, 10-2, 100-2]

      o2 = OptimiserChain(ClipGrad(2), WeightDecay(0.1))
      @test Optimisers.update(Optimisers.setup(o2, x), x, dx)[2] ≈ [1-0.1-1, 10-1-2, 100-10-2]

      o2n = OptimiserChain(OptimiserChain(ClipGrad(2), WeightDecay(0.1)))  # nested
      @test Optimisers.update(Optimisers.setup(o2n, x), x, dx)[2] ≈ [1-0.1-1, 10-1-2, 100-10-2]

      o2r = OptimiserChain(WeightDecay(0.1), ClipGrad(2))  # reversed
      @test Optimisers.update(Optimisers.setup(o2r, x), x, dx)[2] != [1-0.1-1, 10-2, 100-2]

      # Trivial cases
      o1 = OptimiserChain(Descent(0.1))
      @test Optimisers.update(Optimisers.setup(o1, x), x, dx)[2] ≈ [0.9, 9.8, 99.7]

      o0 = OptimiserChain()
      @test Optimisers.update(Optimisers.setup(o0, x), x, dx)[2] ≈ [1-1,10-2,100-3]
    end

    @testset "trainable subset" begin
      @info "ignore these warnings about trainable, testing the old path"
      # Foo has an old-style tuple trainable, both elements
      mf = Foo([1.0, 2.0], (a = sin, b = [3.0, 4.0], c = 5))
      sf = Optimisers.setup(Descent(0.1), mf)
      gf = (x = nothing, y = (a = nothing, b = [1,1], c = 1))
      _, mf2 = Optimisers.update(sf, mf, gf)
      @test mf2.x == [1,2]
      @test mf2.y == (a = sin, b = [2.9, 3.9], c = 5)

      gf3 = Tangent{typeof(mf)}(; x = NoTangent(), y = Tangent{typeof(mf.y)}(; a = NoTangent(), b = [1,1], c = 1))
      _, mf3 = Optimisers.update(sf, mf, gf3)  # the same, but with ChainRules types
      @test mf3.x == [1,2]
      @test mf3.y == (a = sin, b = [2.9, 3.9], c = 5)

      # TwoThirds has functor a,c only, and trainable a only
      mt = TwoThirds(Float32[1,2], Float32[3,4], Float32[5,6])
      mt10 = fmap(x -> 10x, mt)
      @test mt10.a == [10, 20]
      @test mt10.b == [3, 4]
      @test mt10.c == [50, 60]
      st = Optimisers.setup(Momentum(0.1, 0.9), mt)
      gt = gradient(m -> sum(abs2, m.a) + 100sum(abs2, m.b), mt)
      _, mtup = Optimisers.update(st, mt, gt...)
      @test mtup.a ≈ [0.8, 1.6]
      @test mtup.b == [3, 4]
      @test mtup.c == [5, 6]

      # Various kinds of missing branches together:
      m = Foo(
          TwoThirds(Foo(1.0, Float32[2,3,4]), 5.0, Float32[6,7]),
          TwoThirds((p = Float32[1,2,3],), sin, (q = 4.0, r = cos,)),
          )
      s = Optimisers.setup(Momentum(0.1, 0.9), m)
      g = gradient(m -> sum(abs2, m.x.a.y) + m.x.b^2 + log(m.y.c.q), m)
      @test Optimisers.update!(s, m, g...)[2] isa Foo
    end

    @testset "eltype preservation" begin
      m = (Float16[1,2], Float32[3,4])
      s1 = Optimisers.setup(Descent(0.1), m)
      s2, m2 = Optimisers.update(s1, m, m)
      @test eltype(m2[1]) == Float16  # because update copies & calls update!
      @test eltype(m2[2]) == Float32

      staticm = (SA{Float16}[1,2], SA{Float32}[3,4])
      s3 = Optimisers.setup(Descent(0.1), staticm)
      s4, m4 = Optimisers.update(s3, staticm, staticm)
      @test eltype(m4[1]) == Float16  # because of explicit broadcast in subtract!
      @test eltype(m4[2]) == Float32
    end
    
    @testset "adjusyting parameters" begin
      # Simple momentum:
      m = (α = ([0.0], sin), γ = Float32[4,3,2])
      s = Optimisers.setup(Momentum(0.1, 0.9), m)
      s1, m1 = Optimisers.update(s, m, (α = nothing, γ = [1,10,100],))
      @test m.γ .- m1.γ ≈ [0.1, 1, 10]
      @test s1.γ.rule.eta == 0.1
      @test s1.γ.state ≈ [0.1, 1, 10]
  
      s2 = Optimisers.adjust(s1, 0.2)
      @test s2.γ.rule.eta == 0.2
      @test s2.γ.rule.rho == 0.9
      @test s2.γ.state == s1.γ.state
      @test s2.α[1].rule.eta == 0.2
      @test s2.α[1].state == s1.α[1].state
      
      s3 = Optimisers.adjust(s1; eta=0.3, rho=0.7)
      @test s3.γ.rule.eta == 0.3
      @test s3.γ.rule.rho == 0.7
      @test s3.γ.state == s1.γ.state
      @test s3.α[1].rule.rho == 0.7
      
      _, m3 = Optimisers.update(s3, m, (α = nothing, γ = [1,10,100],))
      @test !(m.γ .- m3.γ ≈ [1, 10, 100])

      @test s1 == Optimisers.adjust(s1, zeta = "this does nothing")
  
      # OptimiserChain
      sc = Optimisers.setup(OptimiserChain(ClipGrad(2), Adam()), m)
      sc1, mc1 = Optimisers.update(sc, m, (α = nothing, γ = [1,10,100],))
      @test sc1.γ.rule.opts[2].eta == 0.001f0
      @test sc1.γ.state[2][1] ≈ [0.1, 0.2, 0.2]

      sc2 = Optimisers.adjust(sc1, 0.2)
      @test sc2.γ.rule.opts[1].delta == 2 # unchanged
      @test sc2.γ.rule.opts[2].eta === 0.2f0
      @test sc2.γ.state[2][1] ≈ [0.1, 0.2, 0.2]
      
      sc2 = Optimisers.adjust(sc1; delta = 2.5)  # ClipGrad(2) does not store an Int, for this reason
      @test sc2.γ.rule.opts[1].delta == 2.5
      @test sc2.γ.rule.opts[2].eta === 0.001f0 # unchanged
      @test sc2.γ.state[2][1] ≈ [0.1, 0.2, 0.2]
    end

    @testset "forgotten gradient" begin
      x = [1.0, 2.0]
      sx = Optimisers.setup(Descent(), x)
      @test_throws MethodError Optimisers.update(sx, x)

      m = (x = x, y = sin)
      sm = Optimisers.setup(Descent(), m)
      @test_throws MethodError Optimisers.update(sm, m)
    end

    @testset "broadcasting macros" begin
      x = [1.0, 2.0]; y = [3,4]; z = [5,6]
      @test (@lazy x + y * z) isa Broadcast.Broadcasted
      bc = @lazy x + y * z
      @test (y .+ 2 .* bc) == [35,56]

      @test (@.. x = y * z) isa Array
      @test x == y .* z  # mutated
      r = 1.0:2.0  # immutable
      @test (@.. r = y * z) isa Array
      @test (@.. r = y * z) == y .* z
      @.. r = y * z
      @test r == y .* z  # attaches name r to result
    end

    @testset "tied weights" begin
      ok = (1.0:3.0, sin, "abc", :abc)
      m = (α = ok, β = rand(3), γ = ok)
      m1 = (rand(3), m, rand(3))
      @test Optimisers.setup(AdamW(), m1) isa Tuple
      m2 = (rand(3), m, rand(3), m, rand(3))  # illegal
      @test_throws ArgumentError Optimisers.setup(AdamW(), m2)
    end

  end
  @testset verbose=true "Destructure" begin
    include("destructure.jl")
  end
  @testset verbose=true "Optimisation Rules" begin
    include("rules.jl")
  end
end
