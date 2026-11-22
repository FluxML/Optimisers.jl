# JLArrays stand in for GPU arrays, which must never be indexed one element at a time:
using JLArrays, MLDataDevices
JLArrays.allowscalar(false)

@testset "very basics" begin
  m = (jl([1.0, 2.0]),)
  mid = objectid(m[1])
  g = (jl([25, 33]),)
  o = Descent(0.1f0)
  s = Optimisers.setup(o, m)

  s2, m2 = Optimisers.update(s, m, g)
  @test Array(m[1]) == 1:2  # not mutated
  @test m2[1] isa JLArray
  @test Array(m2[1]) ≈ [1,2] .- 0.1 .* [25, 33]  atol=1e-6

  s3, m3 = Optimisers.update!(s, m, g)
  @test objectid(m3[1]) == mid
  @test Array(m3[1]) ≈ [1,2] .- 0.1 .* [25, 33]  atol=1e-6

  g4 = Tangent{typeof(m)}(g...)
  s4, m4 = Optimisers.update!(s, (jl([1.0, 2.0]),), g4)
  @test Array(m4[1]) ≈ [1,2] .- 0.1 .* [25, 33]  atol=1e-6
end

@testset "basic mixed" begin
  # Works trivially as every element of the tree is either here or there
  m = (device = jl([1.0, 2.0]), host = [3.0, 4.0], neither = (5, 6, sin))
  s = Optimisers.setup(Adam(0.1), m)
  @test s.device.state[1] isa JLArray
  @test s.host.state[1] isa Array

  g = (device = jl([1, 0.1]), host = [1, 10], neither = nothing)
  s2, m2 = Optimisers.update(s, m, g)
  @test m2.device isa JLArray
  @test Array(m2.device) ≈ [0.9, 1.9]  atol=1e-6
  @test m2.host isa Array
  @test m2.host ≈ [2.9, 3.9]
end

GPU_RULES = [
  Descent(), Adam(), RMSProp(), NAdam(),
  OptimiserChain(WeightDecay(), Adam(0.001)),
  OptimiserChain(ClipNorm(), Adam(0.001)),
  OptimiserChain(ClipGrad(0.5), Momentum()),
]

@testset "rules: simple sum" begin
  @testset "$(typeof(o).name.name)" for o in GPU_RULES
    m = jl(shuffle!(reshape(1:64, 8, 8) .+ 0.0))
    s = Optimisers.setup(o, m)
    for _ in 1:10
      g = Zygote.gradient(x -> sum(abs2, x + x'), m)[1]
      s, m = Optimisers.update!(s, m, g)
    end
    @test sum(m) < sum(1:64)
  end
end

@testset "destructure" begin
  m = (x = jl(Float32[1, 2, 3]), y = (0, 99), z = jl(Float32[4, 5]))
  v, re = destructure(m)
  @test v isa JLArray{Float32}
  @test re(2v).x isa JLArray{Float32}
  @test Array(re(2v).z) == [8, 10]

  dm = gradient(m -> sum(abs2, destructure(m)[1]), m)[1]
  @test dm.z isa JLArray
  dv = gradient(v -> sum(abs2, re(v).z), jl([10, 20, 30, 40, 50f0]))[1]
  @test dv isa JLArray
  @test Array(dv) == [0, 0, 0, 80, 100]

  # the precision is the model's, the device is that of p
  m64 = @test_logs (:warn,) re(jl(Float64[1, 2, 3, 4, 5]))
  @test m64.x isa JLArray{Float32}
  @test (@test_logs re(jl([1, 2, 3, 4, 5]))).z isa JLArray{Float32}

  # rebuilding a CPU model from a device vector gives a device model
  vc, rec = destructure((x = Float32[1, 2], y = Float32[3 4]))
  @test rec(jl(vc)).y isa JLArray{Float32, 2}

  va, rea = destructure(m; alias = true)
  ma = rea(va)
  @test Array(ma.z) == [4, 5]
  @test sum(ma.x) == 6

  m′ = deepcopy(m)
  v!, re! = destructure!(m′)
  @test re!(2v!) === m′
  @test Array(m′.z) == [8, 10]
end

# MLDataDevices sees JLArrays as CPU arrays, so this pretends to be on a CUDA GPU:
struct FakeGPUArray{T, N} <: AbstractArray{T, N}
  data::Array{T, N}
end
Base.size(x::FakeGPUArray) = size(x.data)
Base.getindex(x::FakeGPUArray, i::Int...) = x.data[i...]
MLDataDevices.Internal.get_device_type(::FakeGPUArray) = MLDataDevices.CUDADevice

@testset "destructure on mixed devices, issue 69" begin
  @test get_device_type(FakeGPUArray([1.0])) == MLDataDevices.CUDADevice
  mixed = (a = FakeGPUArray([1.0, 2.0]), b = [3.0])
  @test_throws "on one device" destructure(mixed)
  @test_throws "on one device" destructure(mixed; alias = true)
  @test_throws "on one device" destructure!(mixed)
  @test_throws "on one device" destructure!(zeros(2), (a = FakeGPUArray([1.0, 2.0]),))
  @test destructure((a = FakeGPUArray([1.0, 2.0]), b = FakeGPUArray([3.0])))[1] == 1:3
end
