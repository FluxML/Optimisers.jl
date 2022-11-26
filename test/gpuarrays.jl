using Optimisers
using ChainRulesCore, Zygote
using Test

import CUDA
if CUDA.functional()
  using CUDA  # exports CuArray, etc
  CUDA.allowscalar(false)
else
  @info "CUDA not functional, testing with JLArrays instead"
  using JLArrays
  JLArrays.allowscalar(false)

  cu = jl
  CuArray{T,N} = JLArray{T,N}
end
@test cu(rand(3)) .+ 1 isa CuArray

@testset "very basics" begin
  m = (cu([1.0, 2.0]),)
  mid = objectid(m[1])
  g = (cu([25, 33]),)
  o = Descent(0.1f0)
  s = Optimisers.setup(o, m)
  
  s2, m2 = Optimisers.update(s, m, g)
  @test Array(m[1]) == 1:2  # not mutated
  @test m2[1] isa CuArray
  @test Array(m2[1]) ≈ [1,2] .- 0.1 .* [25, 33]  atol=1e-6

  s3, m3 = Optimisers.update!(s, m, g)
  @test objectid(m3[1]) == mid
  @test Array(m3[1]) ≈ [1,2] .- 0.1 .* [25, 33]  atol=1e-6

  g4 = Tangent{typeof(m)}(g...)
  s4, m4 = Optimisers.update!(s, (cu([1.0, 2.0]),), g4)
  @test Array(m4[1]) ≈ [1,2] .- 0.1 .* [25, 33]  atol=1e-6
end

@testset "basic mixed" begin
  # Works trivially as every element of the tree is either here or there
  m = (device = cu([1.0, 2.0]), host = [3.0, 4.0], neither = (5, 6, sin))
  s = Optimisers.setup(ADAM(0.1), m)
  @test s.device.state[1] isa CuArray
  @test s.host.state[1] isa Array

  g = (device = cu([1, 0.1]), host = [1, 10], neither = nothing)
  s2, m2 = Optimisers.update(s, m, g)
  
  @test m2.device isa CuArray
  @test Array(m2.device) ≈ [0.9, 1.9]  atol=1e-6
  
  @test m2.host isa Array
  @test m2.host ≈ [2.9, 3.9]
end

RULES = [
  # Just a selection:
  Descent(), ADAM(), RMSProp(), NADAM(),
  # A few chained combinations:
  OptimiserChain(WeightDecay(), ADAM(0.001)),
  OptimiserChain(ClipNorm(), ADAM(0.001)),
  OptimiserChain(ClipGrad(0.5), Momentum()),
]

name(o) = typeof(o).name.name  # just for printing testset headings
name(o::OptimiserChain) = join(name.(o.opts), " → ")

@testset "rules: simple sum" begin
  @testset "$(name(o))" for o in RULES
    m = cu(shuffle!(reshape(1:64, 8, 8) .+ 0.0))
    s = Optimisers.setup(o, m)
    for _ in 1:10
      g = Zygote.gradient(x -> sum(abs2, x + x'), m)[1]
      s, m = Optimisers.update!(s, m, g)
    end
    @test sum(m) < sum(1:64)
  end
end

@testset "destructure GPU" begin
  m = (x = cu(Float32[1,2,3]), y = (0, 99), z = cu(Float32[4,5]))
  v, re = destructure(m)
  @test v isa CuArray
  @test re(2v).x isa CuArray

  dm = gradient(m -> sum(abs2, destructure(m)[1]), m)[1]
  @test dm.z isa CuArray
  dv = gradient(v -> sum(abs2, re(v).z), cu([10, 20, 30, 40, 50.0]))[1]
  @test dv isa CuArray
end

@testset "destructure mixed" begin
  # Not sure what should happen here!
  m_c1 = (x = cu(Float32[1,2,3]), y = Float32[4,5])
  v, re = destructure(m_c1)
  @test re(2v).x isa CuArray
  @test_broken re(2v).y isa Array
  
  m_c2 = (x = Float32[1,2,3], y = cu(Float32[4,5]))
  @test_skip destructure(m_c2)  # ERROR: Scalar indexing
end
