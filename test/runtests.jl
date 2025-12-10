using Optimisers
using ChainRulesCore, Functors, StaticArrays, Zygote, EnzymeCore
using LinearAlgebra, Statistics, Test, Random
using Optimisers: @.., @lazy
using Base.Broadcast: broadcasted, instantiate, Broadcasted

Random.seed!(1)

# Fake "models" for testing

struct Foo; x; y; end
Functors.@functor Foo
Optimisers.trainable(x::Foo) = (; x.y, x.x)

struct TwoThirds a; b; c; end
Functors.@functor TwoThirds (a, c)
Optimisers.trainable(x::TwoThirds) = (a = x.a,)

mutable struct MutTwo; x; y; end
Functors.@functor MutTwo

# Simple rules for testing

struct DummyHigherOrder <: AbstractRule end
Optimisers.init(::DummyHigherOrder, x::AbstractArray) =
  (ones(eltype(x), size(x)), zero(x))
dummy_update_rule(st, p, dx, dx2) = @. p - (st[1] * dx + st[2] * dx2)
function Optimisers.apply!(::DummyHigherOrder, state, x, dx, dx2)
  a, b = state
  @.. dx = a * dx + b * dx2
  return (a .+ 1, b .+ 1), dx
end

struct BiRule <: Optimisers.AbstractRule end
Optimisers.init(o::BiRule, x::AbstractArray) = nothing
function Optimisers.apply!(o::BiRule, state, x, dx, dx2)
  dx == dx2 || error("expected 1st & 2nd gradients to agree")
  return state, dx
end


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
      @test Optimisers.maywrite(m[1])
      @test m2[1] ≈ [1,2] .- 0.1 .* [25, 33]

      s3, m3 = Optimisers.update!(s, m, g)
      @test objectid(m3[1]) == mid
      @test m3[1] ≈ [1,2] .- 0.1 .* [25, 33]

      g4 = Tangent{typeof(m)}(g...)
      s4, m4 = Optimisers.update!(s, ([1.0, 2.0],), g4)
      @test m4[1] ≈ [1,2] .- 0.1 .* [25, 33]
      
      o5 = Momentum(0.1)
      s5 = Optimisers.setup(o5, m)
      
      s6, m6 = Optimisers.update(s5, m, g)
      @test s6[1].state ≈ [2.5, 3.3]
      @test s5[1].state == [0, 0]  # not mutated -- wrong on v0.2.9

      s7, m7 = Optimisers.update!(s5, m, g)
      @test s7[1].state === s5[1].state  # same array
      @test s7[1] === s5[1]  # same Leaf
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
      # This error is thrown by apply! due to NaN input.
      @test_throws DomainError Optimisers.update(s2, m, (α = ([0.1], nothing), γ = [1,10,NaN],))

      s3 = Optimisers.setup(ClipNorm(5, 1; throw=false), m)
      _, m3 = Optimisers.update(s3, m, (α = ([0.1], nothing), γ = [1,10,100],))
      @test only(m.α[1] .- m3.α[1]) ≈ 0.1
      @test norm(m.γ .- m3.γ, 1) ≈ 5
      _, m3n = Optimisers.update!(s3, m, (α = nothing, γ = [1,10,Inf],))
      @test isnan(m3n.γ[3])
    end

    @testset "Dict support" begin
      @testset "simple dict" begin
        d = Dict(:a => [1.0,2.0], :b => [3.0,4.0], :c => 1)
        s = Optimisers.setup(AdamW(0.1), d)
        @test s isa Dict{Symbol, <:Any}
        @test s[:a] isa Optimisers.Leaf
        @test s[:b] isa Optimisers.Leaf
        @test s[:c] === ()
        loss(model) = sum(abs2, model[:a])
        g = gradient(loss, d)[1]
        s2, d2 = Optimisers.update(s, d, g)
        @test s2 isa Dict{Symbol, <:Any}
        @test d2 isa Dict{Symbol, <:Any}
        @test d2[:a] ≈ [0.9, 1.9]
        @test d2[:b] ≈ [3, 4]
        @test d2[:c] ≈ 1
      end

      @testset "nested dict" begin
        d = Dict(1 => [1.0,2.0], 2 => Dict("a" => (; c=[3.0,4.0]), "b" => 1))
        s = Optimisers.setup(AdamW(0.1), d)
        @test s[2]["a"].c isa Optimisers.Leaf
        g = gradient(d -> sum(d[2]["a"].c), d)[1]
        s2, d2 = Optimisers.update(s, d, g)
        @test d2[2]["a"].c ≈ [2.9, 3.9]
        @test d2[1] ≈ [1, 2]
        @test d2[2]["b"] ≈ 1
      end
    end

    @testset "OptimiserChain" begin
      x = [1, 10, 100.0]; dx = [1, 2, 3.0];
      @test Optimisers.update(Optimisers.setup(WeightDecay(0.1), x), x, dx)[2] ≈ [1-0.1-1, 10-1-2, 100-10-3]
      @test Optimisers.update(Optimisers.setup(SignDecay(0.1), x), x, dx)[2] ≈ [1-0.1-1, 10-0.1-2, 100-0.1-3]
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

      # L1 norm via sign
      xm = [1, -10, 100.0]; dxm = [3, 2, -1];
      @test Optimisers.update(Optimisers.setup(SignDecay(0.1), xm), xm, dxm)[2] ≈ [1-0.1-3, -10+0.1-2, 100-0.1+1]
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

      # Rprop re-creates its state arrays, check they don't get widened:
      s5 = Optimisers.setup(Rprop(0.1), m)  # Float64 rule
      grad64 = ([1.0,2.0], SA[3.0,4.0])  # Float64 gradients
      s6, m6 = Optimisers.update(s5, m, grad64)
      @test eltype(m6[1]) == Float16
      @test eltype(m6[2]) == Float32
      @test eltype(s6[1].state[1]) == Float16
      @test eltype(s6[1].state[2]) == Float16
      @test eltype(s6[2].state[1]) == Float32
      @test eltype(s6[2].state[2]) == Float32
    end

    @testset "adjusting parameters, out-of-place" begin
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
      @test sc1.γ.rule.opts[2].eta == 0.001
      @test sc1.γ.state[2][1] ≈ [0.1, 0.2, 0.2]

      sc2 = Optimisers.adjust(sc1, 0.2)
      @test sc2.γ.rule.opts[1].delta == 2 # unchanged
      @test sc2.γ.rule.opts[2].eta == 0.2
      @test sc2.γ.state[2][1] ≈ [0.1, 0.2, 0.2]

      sc2 = Optimisers.adjust(sc1; delta = 2.5)  # ClipGrad(2) does not store an Int, for this reason
      @test sc2.γ.rule.opts[1].delta == 2.5
      @test sc2.γ.rule.opts[2].eta == 0.001 # unchanged
      @test sc2.γ.state[2][1] ≈ [0.1, 0.2, 0.2]

      # MixedPrecision
      mp = Optimisers.setup(MixedPrecision(Momentum(0.1, 0.9)), m)
      mp1, _ = Optimisers.update(mp, m, (α = nothing, γ = [1,10,100],))
      @test mp1.γ.rule.rule.eta == 0.1
      @test mp1.γ.state[2] ≈ [0.1, 1, 10]

      mp2 = Optimisers.adjust(mp1, 0.2)
      @test mp2.γ.rule.rule.eta == 0.2
      @test mp2.γ.rule.rule.rho == 0.9

      mp3 = Optimisers.adjust(mp1; eta=0.3, rho=0.7)
      @test mp3.γ.rule.rule.eta == 0.3
      @test mp3.γ.rule.rule.rho == 0.7
    end

    @testset "adjusting parameters, in-place" begin
      # Simple momentum:
      m = (α = ([0.0], sin), γ = Float32[4,3,2])
      s = Optimisers.setup(Momentum(0.1, 0.9), m)
      s1, m1 = Optimisers.update(s, m, (α = nothing, γ = [1,10,100],))
      @test m.γ .- m1.γ ≈ [0.1, 1, 10]
      @test s1.γ.rule.eta ≈ 0.1
      @test s1.γ.state ≈ [0.1, 1, 10]

      Optimisers.adjust!(s1, 0.2)
      @test s1.γ.rule.eta ≈ 0.2
      @test s1.γ.rule.rho ≈ 0.9
      @test s1.γ.state ≈ [0.1, 1, 10]
      @test s1.α[1].rule.eta ≈ 0.2

      Optimisers.adjust!(s1; eta=0.3, rho=0.7)
      @test s1.γ.rule.eta ≈ 0.3
      @test s1.γ.rule.rho ≈ 0.7
      @test s1.γ.state ≈ [0.1, 1, 10]
      @test s1.α[1].rule.rho ≈ 0.7

      _, m3 = Optimisers.update(s1, m, (α = nothing, γ = [1,10,100],))
      @test !(m.γ .- m3.γ ≈ [1, 10, 100])

      Optimisers.adjust!(s1, zeta = "this does nothing")
      @test s1.γ.rule.eta ≈ 0.3

      # OptimiserChain
      sc = Optimisers.setup(OptimiserChain(ClipGrad(2), Adam()), m)
      sc1, mc1 = Optimisers.update(sc, m, (α = nothing, γ = [1,10,100],))
      @test sc1.γ.rule.opts[2].eta ≈ 0.001f0
      @test sc1.γ.state[2][1] ≈ [0.1, 0.2, 0.2]

      Optimisers.adjust!(sc1, 0.2)
      @test sc1.γ.rule.opts[1].delta == 2 # unchanged
      @test sc1.γ.rule.opts[2].eta ≈ 0.2
      @test sc1.γ.state[2][1] ≈ [0.1, 0.2, 0.2]

      Optimisers.adjust!(sc1; delta = 2.5)  # ClipGrad(2) does not store an Int, for this reason
      @test sc1.γ.rule.opts[1].delta ≈ 2.5
      @test sc1.γ.rule.opts[2].eta ≈ 0.2 # unchanged
      @test sc1.γ.state[2][1] ≈ [0.1, 0.2, 0.2]

      # MixedPrecision
      mp = Optimisers.setup(MixedPrecision(Momentum(0.1, 0.9)), m)
      mp1, _ = Optimisers.update(mp, m, (α = nothing, γ = [1,10,100],))
      @test mp1.γ.rule.rule.eta == 0.1
      @test mp1.γ.state[2] ≈ [0.1, 1, 10]

      Optimisers.adjust!(mp1, 0.2)
      @test mp1.γ.rule.rule.eta == 0.2
      @test mp1.γ.rule.rule.rho == 0.9

      Optimisers.adjust!(mp1; eta=0.3, rho=0.7)
      @test mp1.γ.rule.rule.eta == 0.3
      @test mp1.γ.rule.rule.rho == 0.7
    end

    @testset "freeze/thaw" begin
      m = (x=[1.0, 2.0], y=([3.0, 4.0], sin));
      st = Optimisers.setup(Descent(0.1), m);
      Optimisers.freeze!(st.y)
      st, m = Optimisers.update(st, m, (x=[1,10], y=([100,1000], nothing)));
      @test m.x ≈ [0.9, 1.0]
      @test m.y[1] ≈ [3, 4]

      st = Optimisers.adjust(st, 0.2)
      Optimisers.thaw!(st)
      st, m = Optimisers.update(st, m, (x=[1,10], y=([100,1000], nothing)));
      @test m.y[1] ≈ [-17.0, -196.0]
      @test m.x ≈ [0.7, -1.0]

      @test_throws ArgumentError Optimisers.freeze!(m)
      @test_throws ArgumentError Optimisers.thaw!(m)
    end

    @testset "keyword arguments" begin
      @test Nesterov(rho=0.8, eta=0.1) === Nesterov(0.1, 0.8)
      @test AdamW(lambda=0.3, eta=0.1) == AdamW(0.1, (0.9, 0.999), 0.3, 1.0e-8)
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
      @testset "tuples" begin
         twice = [1,2.0]
         mtup = (twice, (copy(twice), twice)) # (tied (not tied, tied))

         # simplest rule for which opt(g1) + opt(g2) != opt(g1 + g2)
         stup = Optimisers.setup(Momentum(0.1), mtup)
         gtup = ([3,3], ([10,10], [7,7])) # (g1, (g1 + g2, g2))

         snew, mnew = Optimisers.update(stup, mtup, gtup)
         @test mnew[1] ≈ mnew[2][1]  # gradient was accumulated
         @test mnew[2][2] === mnew[1]  # and tie is not broken

         st3, mt3 = Optimisers.update(stup, mtup, ([3,3], nothing))
         @test mt3[1] ≈ [1,2] - 0.1 * [3,3]
         @test mt3[2][2] === mt3[1]

         st4, mt4 = Optimisers.update(stup, mtup, (nothing, ([5,5], [7,7])))
         @test mt4[1] ≈ [1,2] - 0.1 * [7,7]
       end

       @testset "named" begin
         thrice = [3f0]
         model = (a = (x = thrice, y = Float32[4,5,6], z = true), b = ((m = (0, 1, thrice),),), c = (x = Float32[7,8], y = thrice))
         tree = Optimisers.setup(Momentum(0.1, 0.9), model)
         @test model.a.x === model.b[1].m[3] == model.c.y

         loss(x::Array) = sum(abs2, x)
         loss(x::Number) = x^3
         loss(m) = sum(2 * loss(x) for x in m)
         gradient(loss, model)
         _, m2 = Optimisers.update(tree, model, gradient(loss, model)...)
         @test m2.a.x === m2.b[1].m[3] == m2.c.y

         loss3(m) = sum(x isa Tuple ? 0 : 2 * loss(x) for x in m)
         gradient(loss3, model)  # truncates the b limb
         _, m3 = Optimisers.update(tree, model, gradient(loss3, model)...)
         @test m3.a.x === m3.b[1].m[3] == m3.c.y
       end

       @testset "transpose" begin
         mat = [1 2 3; 4 5 6.0]
         bidir = (m = mat, f = log, t = transpose(mat), v = [7, 8, 9.0])
         bigrad, _ = gradient((m, x) -> sum(abs2, m.m * (m.f).(m.t*x .+ m.v)), bidir, [1, 0.1])
         @test bigrad.t isa Matrix  # not a Transpose, that's the point here

         state = Optimisers.setup(Descent(0.1), bidir)
         @test state.t.parent === state.m  # successfully tied

         s2, b2 = Optimisers.update(state, bidir, bigrad)
         @test b2.t.parent === b2.m  # tie restored
         @test b2.m ≈ bidir.m - 0.1 * (bigrad.m + transpose(bigrad.t))  # grad accumulated

         state = Optimisers.setup(OptimiserChain(ClipGrad(10), Descent(0.1), ClipGrad(10)), bidir)
         s2, b2 = Optimisers.update(state, bidir, bigrad)
         @test b2.t.parent === b2.m
         @test b2.m ≈ bidir.m - 0.1 * clamp.((bigrad.m + transpose(bigrad.t)), -10, 10)

         # Similar, but now "primary" field is the transposed one:
         tri = (a = transpose(mat), b = mat, c = transpose(mat), d = 4.0)
         trigrad = gradient(m -> sum(abs2, m.a * (m.b * (m.c * [0.1, 1] .+ m.d) .- m.d)), tri)[1]
         stri = Optimisers.setup(Descent(0.1), tri)
         s3, t3 = Optimisers.update(stri, tri, trigrad)
         @test t3.a.parent === t3.b === t3.c.parent
         @test t3.a ≈ tri.a - 0.1 * (trigrad.a + trigrad.b' + trigrad.c)

         g4 = (a = Broadcast.broadcasted(+, mat', 1), b = nothing, c = @thunk(mat' .+ 1), d = nothing)
         # Error: no constructors for type Any
         @test_broken s4, t4 = Optimisers.update(stri, tri, g4)
       end

       @testset "artificial" begin
         # Interpret shared Leaf as implying shared parameters, even if this did not arise from shared arrays.
         # No API for setting this at the moment, but can construct one by hand:
         model = (a = SA[1,2.0], b = SA[1, 2.0], c = SA[1, 2.0], d = SA[1, 2.0])
         auto = Optimisers.setup(Momentum(0.1), model)
         @test auto.a !== auto.b  # not tied just by value

         trick = (a = auto.a, b = auto.a, c = auto.c, d= auto.d)  # makes a & b tied
  
         trick2, model2 = Optimisers.update(trick, model, (a=[3,3], b=[7,7], c=[3,3], d=[10, 10]))
         trick3, model3 = Optimisers.update(trick2, model2, (a=[3,3], b=[7,7], c=[3,3], d=[10, 10]))
         
         @test model3.a ≈ model3.b ≈ model3.d  # same as having the gradients added
         @test !(model3.a ≈ model3.c)
         @test trick3.a === trick3.b  # leaves remain shared
       end

       @testset "mutable containers" begin
         tmp = MutTwo([1.0], [2.0])
         model = (a=tmp, b=tmp, c=MutTwo(tmp.x, tmp.y))
         state = Optimisers.setup(Momentum(), model)

         @test model.a === model.b
         @test model.a !== model.c  # fields are identified, but struct is not

         @test state.a.x === state.b.x
         @test state.a === state.b
         @test state.a === state.c  # unavoidable, but means we can't use leaf ID alone

         mgrad = (a=(x=[1], y=[10]), b=(x=[100], y=[1000]), c=(x=[1/3], y=[1/30]))
         state2, model2 = Optimisers.update(state, model, mgrad)

         @test model2.a === model2.b  # tie of MutTwo structs is restored
         @test model2.a !== model2.c  # but a new tie is not created
      end
    end  # tied weights

    @testset "2nd-order interface" begin
      @testset "BiRule" begin
        m = (α = ([1.0], sin), γ = Float32[4,3,2])

        # Special rule which requires this:
        s = Optimisers.setup(BiRule(), m)
        g = (α = ([0.1], ZeroTangent()), γ = [1,10,100],)
        s1, m1 = Optimisers.update(s, m, g, g)
        @test m1.α[1] ≈ [0.9]
        @test_throws Exception Optimisers.update(s, m, g, map(x->2 .* x, g))

        # Ordinary rule which doesn't need it:
        s2 = Optimisers.setup(Adam(), m)
        s3, m3 = Optimisers.update(s2, m, g)
        s4, m4 = Optimisers.update(s2, m, g, g)
        @test m3.γ ≈ m4.γ
      end

      @testset "DummyHigherOrder" begin
        w, b = rand(3, 4), rand(3)

        o = DummyHigherOrder()
        psin = (w, b)
        dxs = map(x -> rand(size(x)...), psin)
        dx2s = map(x -> rand(size(x)...), psin)
        stin = Optimisers.setup(o, psin)
        stout, psout = Optimisers.update(stin, psin, dxs, dx2s)

        # hardcoded rule behavior for dummy rule
        @test psout[1] == dummy_update_rule(stin[1].state, psin[1], dxs[1], dx2s[1])
        @test psout[2] == dummy_update_rule(stin[2].state, psin[2], dxs[2], dx2s[2])
        @test stout[1].state[1] == stin[1].state[1] .+ 1
        @test stout[2].state[2] == stin[2].state[2] .+ 1

        # error if only given one derivative
        @test_throws MethodError Optimisers.update(stin, psin, dxs)

        # first-order rules compose with second-order
        ochain = OptimiserChain(Descent(0.1), o)
        stin = Optimisers.setup(ochain, psin)
        stout, psout = Optimisers.update(stin, psin, dxs, dx2s)
        @test psout[1] == dummy_update_rule(stin[1].state[2], psin[1], 0.1 * dxs[1], dx2s[1])
        @test psout[2] == dummy_update_rule(stin[2].state[2], psin[2], 0.1 * dxs[2], dx2s[2])
      end
    end  # 2nd-order

    @testset "subtract! handles Zero" begin
      x = rand(3)
      y = Optimisers.subtract!(x, ChainRulesCore.ZeroTangent())
      @test y === x
      y = Optimisers.subtract!(x, nothing)
      @test y === x
    end

    @testset "_norm(dx, p) works" begin
      bc = instantiate(broadcasted(+, randn(Float32, 10), randn(Float32, 10)'));
      arr = collect(bc)
      bc2 = instantiate(broadcasted(+, [1, 0, -3, 4], 0))
      arr2 = collect(bc2)
      for p in (-Inf, -3, -1, 0, 0.5, 1, 1.5, 2, 3f0, Inf32)
        @test Optimisers._norm(bc, p) ≈ norm(arr, p)
        @test Optimisers._norm(bc, p) isa Float32
        @test Optimisers._norm(bc2, p) ≈ norm(arr2, p)
        @test Optimisers._norm(bc2, p) isa Float64
      end
    end

    @testset "Enzyme Duplicated" begin
      x_dx = Duplicated(Float16[1,2,3], Float16[1,0,-4])
      st = Optimisers.setup(Momentum(1/9), x_dx)  # acts only on x not on dx
      @test st isa Optimisers.Leaf
      @test nothing === Optimisers.update!(st, x_dx)  # mutates both arguments
      @test x_dx.val ≈ Float16[0.8887, 2.0, 3.445]

      shared = [1.0]
      model = (x=shared, y=shared)
      grad = deepcopy(model) # Enzyme produces something like this, grad.x === grad.y, already accumulated.
      dup = Duplicated(model, model)
      st2 = Optimisers.setup(Descent(0.1), model)
      Optimisers.update!(st2, dup)
      @test model.x ≈ [0.9]
      shared .= 1
      Optimisers.update!(st2, model, grad)
      model.x ≈ [0.8]  # This is wrong, but don't make it a test.
      # Ideally, perhaps the 3-arg update! could notice that grad.x===grad.y, and not accumulate the gradient in this case?

      @test_throws ArgumentError Optimisers.setup(Adam(), (; a=[1,2,3.], b=x_dx))  # Duplicated deep inside is not allowed
    end
  end
  @testset verbose=true "Destructure" begin
    include("destructure.jl")
  end
  @testset verbose=true "Trainables" begin
    include("trainables.jl")
  end
  @testset verbose=true "Optimisation Rules" begin
    include("rules.jl")
  end
  @testset verbose=true "interface" begin
    include("interface.jl")
  end
  if VERSION ≤ v"1.12-" && !Sys.iswindows()
    @testset verbose=true "Reactant" begin
      include("reactant.jl")
    end
  end
end
