using ForwardDiff

m1 = collect(1:3.0)
m2 = (collect(1:3.0), collect(4:6.0))
m3 = (x = m1, y = sin, z = collect(4:6.0))

m4 = (x = m1, y = m1, z = collect(4:6.0))  # tied
m5 = (a = (m3, true), b = (m1, false), c = (m4, true))
m6 = (a = m1, b = [4.0 + im], c = m1)

m7 = TwoThirds((sin, collect(1:3.0)), (cos, collect(4:6.0)), (tan, collect(7:9.0)))
m8 = [Foo(m1, m1), (a = true, b = Foo([4.0], false), c = ()), [[5.0]]]

mat = Float32[4 6; 5 7]
m9 = (a = m1, b = mat, c = [mat, m1])

@testset "flatten & rebuild" begin
  @test destructure(m1)[1] isa Vector{Float64}
  @test destructure(m1)[1] == 1:3
  @test destructure(m2)[1] == 1:6
  @test destructure(m3)[1] == 1:6
  @test destructure(m4)[1] == 1:6
  @test destructure(m5)[1] == vcat(1:6, 4:6)
  @test destructure(m6)[1] == vcat(1:3, 4 + im)
  @test destructure(m9)[1] == 1:7

  @test destructure(m1)[2](7:9) == [7,8,9]
  @test destructure(m2)[2](4:9) == ([4,5,6], [7,8,9])
  @test destructure(m3)[2](4:9) == (x = [4,5,6], y = sin, z = [7,8,9])
  m4′ = destructure(m4)[2](4:9)
  @test m4′ == (x = [4,5,6], y = [4,5,6], z = [7,8,9])
  @test m4′.x === m4′.y
  m5′ = destructure(m5)[2](reverse(1:9))
  @test m5′.a[1].x === m5′.b[1]
  @test m5′.b[2] === false
  m6′ = destructure(m6)[2]((4:7) .+ (1:4) .* im)
  @test m6′.a isa Vector{Float64}
  @test m6′.a == 4:6
  @test m6′.a === m6′.c
  @test m6′.b == [7 + 4im]

  # struct, trainable
  @test destructure(m7)[1] == 1:3
  m7′ = destructure(m7)[2]([10,20,30])
  @test m7′.a == (sin, [10,20,30])
  @test m7′.b == (cos, [4,5,6])
  @test m7′.c == (tan, [7,8,9])

  @test destructure(m8)[1] == 1:5
  m8′ = destructure(m8)[2](1:5)
  @test m8′[1].x === m8′[1].y
  @test m8′[2].b.y === false
  @test m8′[3][1] == [5.0]

  m9′ = destructure(m9)[2](10:10:70)
  @test m9′.b === m9′.c[1]
  @test m9′.b isa Matrix{Float32}

  # errors
  @test_throws Exception destructure(m7)[2]([10,20])
  @test_throws Exception destructure(m7)[2]([10,20,30,40])
end

@testset "gradient of flatten" begin
  @test gradient(m -> destructure(m)[1][1], m1)[1] == [1,0,0]
  @test gradient(m -> destructure(m)[1][2], m2)[1] == ([0,1,0], [0,0,0])
  @test gradient(m -> destructure(m)[1][3], (m1, m1))[1] == ([0,0,1], nothing)
  @test gradient(m -> destructure(m)[1][1], m3)[1] == (x = [1,0,0], y = nothing, z = [0,0,0])
  @test gradient(m -> destructure(m)[1][2], m4)[1] == (x = [0,1,0], y = nothing, z = [0,0,0])

  g5 = gradient(m -> destructure(m)[1][3], m5)[1]
  @test g5.a[1].x == [0,0,1]
  @test g5.a[2] === nothing

  g6 = gradient(m -> imag(destructure(m)[1][4]), m6)[1]
  @test g6.a == [0,0,0]
  @test g6.a isa Vector{Float64}
  @test g6.b == [0+im]

  g8 = gradient(m -> sum(abs2, destructure(m)[1]), m8)[1]
  @test g8[1].x == [2,4,6]
  @test g8[2].b.x == [8]
  @test g8[3] == [[10.0]]

  g9 = gradient(m -> sum(sqrt, destructure(m)[1]), m9)[1]
  @test g9.c === nothing

  @testset "second derivative" begin
    @test gradient([1,2,3.0]) do v
      sum(abs2, gradient(m -> sum(abs2, destructure(m)[1]), (v, [4,5,6.0]))[1][1])
    end[1] ≈ [8,16,24]

    @test gradient([1,2,3.0]) do v
      sum(gradient(m -> sum(destructure(m)[1])^3, (v, [4,5,6.0]))[1][1])
    end[1] == [378, 378, 378]

    @test gradient([1,2,3.0]) do v
      sum(abs2, gradient(m -> sum(abs2, destructure(m)[1]), (x = v, y = sin, z = [4,5,6.0]))[1][1])
    end[1] ≈ [8,16,24]
  end
end

@testset "gradient of rebuild" begin
  re1 = destructure(m1)[2]
  @test gradient(x -> re1(x)[1], rand(3))[1] == [1,0,0]
  re2 = destructure(m2)[2]
  @test gradient(x -> re2(x)[1][2], rand(6))[1] == [0,1,0,0,0,0]
  re3 = destructure(m3)[2]
  @test gradient(x -> re3(x).x[3], rand(6))[1] == [0,0,1,0,0,0]
  @test gradient(x -> re3(x).z[1], rand(6))[1] == [0,0,0,1,0,0]

  re4 = destructure(m4)[2]
  @test gradient(x -> re4(x).x[1], rand(6))[1] == [1,0,0,0,0,0]
  @test gradient(x -> re4(x).y[2], rand(6))[1] == [0,1,0,0,0,0]
  @test gradient(rand(6)) do x
    m = re4(x)
    m.x[1] + 2*m.y[2] + 3*m.z[3]
  end[1] == [1,2,0, 0,0,3]

  re7 = destructure(m7)[2]
  @test gradient(x -> re7(x).a[2][3], rand(3))[1] == [0,0,1]
  @test gradient(x -> re7(x).b[2][2], rand(3))[1] == [0,0,0]
  @test gradient(x -> re7(x).c[2][1], rand(3))[1] == [0,0,0]

  v8, re8 = destructure(m8)
  @test gradient(x -> sum(abs2, re8(x)[1].y), v8)[1] == [2,4,6,0,0]
  @test gradient(x -> only(sum(re8(x)[3]))^2, v8)[1] == [0,0,0,0,10]

  re9 = destructure(m9)[2]
  @test gradient(x -> sum(abs2, re9(x).c[1]), 1:7)[1] == [0,0,0, 8,10,12,14]

  @testset "second derivative" begin
    @test gradient(collect(1:6.0)) do y
      sum(abs2, gradient(x -> sum(abs2, re2(x)[1]), y)[1])
    end[1] ≈ [8,16,24,0,0,0]

    @test gradient(collect(1:6.0)) do y
      sum(abs2, gradient(x -> sum(abs2, re3(x).z), y)[1])
    end[1] ≈ [0,0,0,32,40,48]

    re3t = destructure((x = m1, y = sin, z = m1))[2]  # tied
    @test gradient(collect(1:3.0)) do y
      sum(abs2, gradient(x -> sum(abs2, re3t(x).z), y)[1])
    end[1] ≈ [8,16,24]
  end
end

@testset "Flux issue 1826" begin
  v, re = destructure((x=[1,2.0], y=[3,4,5.0]))
  @test gradient(zero(v)) do w
    m = re(w)
    5 * sum(m.x) + 7 * sum(m[2])  # uses both x and y
  end == ([5.0, 5.0, 7.0, 7.0, 7.0],)
  # This, using only x, was broken on Flux:
  @test gradient(w -> sum(re(w).x), zero(v)) == ([1.0, 1.0, 0.0, 0.0, 0.0],)

  sh = [7,7.0];
  v, re = destructure((x=sh, y=[3.0,4.0], z=sh))  # shared array in the model
  @test v == [7, 7, 3, 4]
  @test re([1,10,100,1000]) == (x = [1, 10], y = [100, 1000], z = [1, 10])

  @test gradient(zero(v)) do w
    m = re(w)
    3 * sum(m.x) + 13 * sum(m.z)  # no dependence on y, but two distinct gradient arrays
  end == ([16, 16, 0, 0],)  # Flux gave ([3.0, 3.0, 13.0, 13.0],)

  @test gradient(zero(v)) do w
    m = re(w)
    4(sum(m.x) + sum(m.z))  # now two gradients are ===, so it eliminates one
  end == ([8,8,0,0],)

  @test gradient(zero(v)) do w
    m = re(w)
    4(sum(m.x) + sum(m.y)) + 13*sum(m.z)  # again two gradients are ===, so it eliminates one
  end == ([17,17,4,4],)  # Flux gave ([4.0, 4.0, 13.0, 13.0],)
end

@testset "DiffEqFlux issue 699" begin
  # The gradient of `re` is a vector into which we accumulate contributions, and the issue
  # is that one contribution may have a wider type than `v`, especially for `Dual` numbers.
  v, re = destructure((x=Float32[1,2], y=Float32[3,4,5]))
  _, bk = Zygote.pullback(re, ones(Float32, 5))
  # Testing with `Complex` isn't ideal, but this was an error on 0.2.1.
  # If some upgrade inserts ProjectTo, this will fail, and can be changed:
  @test bk((x=[1.0,im], y=nothing)) == ([1,im,0,0,0],)
  
  @test bk((x=nothing, y=[10,20,30]))[1] isa Vector{Float32}  # despite some ZeroTangent
  @test bk((x=nothing, y=nothing)) == ([0,0,0,0,0],) 
  @test bk((x=nothing, y=@thunk [1,2,3] .* 10.0)) == ([0,0,10,20,30],)
  @test bk((x=[1.2, 3.4], y=Float32[5,6,7])) == ([1.2, 3.4, 5, 6, 7],)
end

#=

# Adapted from https://github.com/SciML/DiffEqFlux.jl/pull/699#issuecomment-1092846657
using ForwardDiff, Zygote, Flux, Optimisers, Test

y = Float32[0.8564646, 0.21083355]
p = randn(Float32, 27);
t = 1.5f0
λ = [ForwardDiff.Dual(0.87135935, 1, 0, 0, 0, 0, 0), ForwardDiff.Dual(1.5225363, 0, 1, 0, 0, 0, 0)]

model = Chain(x -> x .^ 3,
    Dense(2 => 5, tanh),
    Dense(5 => 2))

p,re = Optimisers.destructure(model)
f(u, p, t) = re(p)(u)
_dy, back = Zygote.pullback(y, p) do u, p
    vec(f(u, p, t))
end
tmp1, tmp2 = back(λ);
tmp1
@test tmp2 isa Vector{<:ForwardDiff.Dual}

=#

@testset "empty, issue 67" begin
    m0 = (nothing, missing, isempty)
    @test destructure(m0)[1] isa Vector{<:Real}
    v0, re0 = destructure(m0)
    @test re0(Float32[]) === m0
    @test_throws DimensionMismatch re0([1])

    # This is an elaborate way of checking that it doesn't cause promotions, even of small floats:
    m01 = [(x=nothing, y=0), (x=Float16[1, 2], y=Float16[3])]
    v01, _ = destructure(m01)
    v012 = vcat(destructure(m01[1])[1], destructure(m01[2])[1])
    @test v01 == v012
    @test v012 isa Vector{Float16}

    y, bk = Zygote.pullback(x -> sum(destructure(x)[1]), ("a", :beta))
    @test bk(1.0) == (nothing,)
    # Zygote regards 3,4 as differentiable, but Optimisers does not regard them as parameters:
    y, bk = Zygote.pullback(x -> sum(destructure(x)[1]), (3, 4))
    @test bk(1.0) == (nothing,)
end

# A functor whose children are not its fields, like Flux's Chain once was:
struct Splat{T}; layers::T; Splat(ls...) = new{typeof(ls)}(ls); end
Functors.functor(::Type{<:Splat}, x) = x.layers, ls -> Splat(ls...)

struct TLayer{M,V}; w::M; b::V; end
struct TChain{T<:Tuple}; layers::T; end

@testset "custom functor, issue 62" begin
  sk = Splat([1.0, 2.0], (x = 3, y = [4.0, 5.0]))
  v, re = destructure(sk)
  @test v == [1, 2, 4, 5]
  sk2 = re(10v)
  @test sk2 isa Splat
  @test sk2.layers[1] == [10, 20]
  @test sk2.layers[2] == (x = 3, y = [40, 50])
  @test gradient(p -> sum(abs2, re(p).layers[2].y), v)[1] == [0, 0, 8, 10]
  # The gradient of destructure w.r.t. sk would need Splat's fields, and so would the second derivative of re:
  @test_throws ArgumentError gradient(m -> sum(destructure(m)[1]), sk)
  dx = (layers = (nothing, (x = nothing, y = [1.0, 1.0])),)  # Zygote's gradient w.r.t. re(v)
  @test_throws ArgumentError Optimisers._tangent_like(dx, sk, re.offsets, v, nothing)
end

@testset "non-trainable children get no gradient, issue 72" begin
  m = TwoThirds([1.0, 2.0], [3.0], [4.0, 5.0])
  @test gradient(x -> destructure(x)[1][2]^2, m)[1] == (a = [0.0, 4.0], b = nothing, c = nothing)
end

@testset "Dict, issue 154" begin
  d = Dict(:a => Dict(:b => [1.0], :c => [2.0, 3.0]), :d => [4.0])
  v, re = destructure(d)
  @test sort(v) == 1:4
  d2 = re(10v)
  @test d2 isa Dict
  @test d2[:a][:b] == [10] && d2[:a][:c] == [20, 30] && d2[:d] == [40]
  g = gradient(p -> sum(re(p)[:a][:c]) + 3 * sum(re(p)[:d]), v)[1]
  @test g == [o == 2 || o == 3 ? 1 : o == 4 ? 3 : 0 for o in v]
  gd = gradient(x -> sum(abs2, destructure(x)[1]), d)[1]
  @test gd[:a][:c] == [4, 6] && gd[:d] == [8]
  # Zygote's nested AD would drop the second derivative of re, so it is an error:
  @test_throws ArgumentError Optimisers._tangent_like(Dict(:d => [1.0]), d, re.offsets, v, nothing)
end

@testset "Transpose and Adjoint" begin
  @test gradient(m -> sum(destructure(m)[1]), (a = transpose([1.0 2.0; 3.0 4.0]),))[1].a == ones(2, 2)
  @test gradient(m -> sum(destructure(m)[1]), (a = [1.0 2.0 3.0]',))[1].a == ones(3, 1)
  W = [1.0 2.0; 3.0 4.0; 5.0 6.0]
  m = (enc = TLayer(W, [0.0, 0.0, 0.0]), dec = TLayer(transpose(W), [0.0, 0.0]))  # tied
  v, re = destructure(m)
  @test length(v) == 11
  m2 = re(v .+ 1)
  @test m2.dec.w.parent === m2.enc.w
  g = gradient(x -> sum(destructure(x)[1]), m)[1]
  @test g.enc.w == ones(3, 2)
  @test g.dec.w === nothing  # only the first tied position gets the gradient
  @test gradient(p -> sum(re(p).dec.w), v)[1] == vcat(ones(6), zeros(5))
end

@testset "type stability of rebuild" begin
  # Julia 1.10 cannot infer Functors' default reconstructor y -> S(y...), a closure over S::UnionAll
  if VERSION >= v"1.11"
    m1 = TChain((TLayer(rand(3, 2), rand(3)), TLayer(rand(Float32, 1, 3), rand(Float32, 1)), sin))
    v1, re1 = destructure(m1)
    @test @inferred(re1(v1)) isa typeof(m1)
    @test @inferred(re1(Float32.(v1))) isa typeof(m1)

    l = TLayer(rand(2, 2), rand(2))
    m2 = TChain((l, TLayer(rand(2, 2), rand(2)), l))  # tied arrays
    v2, re2 = destructure(m2)
    @test length(v2) == 12
    m2′ = @inferred re2(v2)
    @test m2′.layers[1].w === m2′.layers[3].w
  end

  m3 = (a = rand(2), b = (c = rand(2, 2), d = sin, e = (rand(Float32, 1),)), f = 1)
  v3, re3 = destructure(m3)
  @test @inferred(re3(v3)) isa typeof(m3)
  @test @inferred(re3(ForwardDiff.Dual.(v3, 1.0))).b.e[1] isa Vector{<:ForwardDiff.Dual{Nothing, Float32}}
end

@testset "reverse over reverse, gradient penalty" begin
  v, re = destructure((w = [1.0 2.0; 3.0 4.0],))
  x = [1.0, 1.0]
  g = gradient(q -> sum(abs2, gradient(xx -> sum(re(q).w * xx), x)[1]), v)[1]
  @test g ≈ [8, 8, 12, 12]  # sum(abs2, sum(w, dims=1)), w = reshape(q, 2, 2)
end

@testset "Restructure with 3 arguments" begin
  v, re = destructure((x = [1.0, 2.0], y = [3.0]))
  re3 = Optimisers.Restructure(re.model, re.offsets, re.length)
  @test re3([4, 5, 6]) == (x = [4.0, 5.0], y = [6.0])
end

@testset "destructure!" begin
  m3′ = deepcopy(m3)
  v3, re3! = destructure!(m3′)
  @test v3 == 1:6
  @test @inferred(re3!(4:9)) === m3′
  @test m3′ == (x = [4,5,6], y = sin, z = [7,8,9])
  @test m3 == (x = 1:3, y = sin, z = 4:6)
  @test re3! isa Optimisers.Restructure{<:Any, <:Any, :inplace}
  @test sprint(show, re3!) == "Restructure!(NamedTuple, ..., 6)"
  @test length(re3!) == 6
  # Functors' reconstruction, used by e.g. Flux's gpu, keeps it in-place:
  @test fmap(x -> x isa AbstractArray ? 2x : x, re3!) isa Optimisers.Restructure{<:Any, <:Any, :inplace}

  m7′ = deepcopy(m7)
  v7, re7! = destructure!(m7′)
  @test v7 == 1:3
  re7!([10,20,30])
  @test m7′.a == (sin, [10,20,30])
  @test m7′.b == (cos, [4,5,6])
  @test m7′.c == (tan, [7,8,9])

  # tied arrays, and arrays of arrays
  m8′ = deepcopy(m8)
  destructure!(m8′)[2](10:14)
  @test m8′[1].x === m8′[1].y == [10, 11, 12]
  @test m8′[2].b.x == [13]
  @test m8′[3][1] == [14]
  m9′ = deepcopy(m9)
  destructure!(m9′)[2](10:10:70)
  @test m9′.a === m9′.c[2] == [10, 20, 30]
  @test m9′.b === m9′.c[1] == [40 60; 50 70]

  sk = Splat([1.0, 2.0], (x = 3, y = [4.0, 5.0]))  # issue 62
  destructure!(sk)[2]([10, 20, 30, 40])
  @test sk.layers == ([10, 20], (x = 3, y = [30, 40]))

  d = Dict(:a => Dict(:b => [1.0], :c => [2.0, 3.0]), :d => [4.0])  # issue 154
  vd, red! = destructure!(d)
  @test red!(10vd) === d
  @test d[:a][:b] == [10] && d[:a][:c] == [20, 30] && d[:d] == [40]

  W = [1.0 2.0; 3.0 4.0; 5.0 6.0]
  t = (enc = TLayer(W, zeros(3)), dec = TLayer(transpose(W), zeros(2)))
  destructure!(t)[2](1:11)
  @test t.enc.w === W == [1 4; 2 5; 3 6]
  @test t.dec.w.parent === W
  @test t.dec.b == [10, 11]

  a = [1.0, 2.0]
  @test destructure!(a)[2]([3, 4]) === a == [3, 4]
  @test destructure!((sin, cos))[2](Bool[]) == (sin, cos)

  # writing the model into an existing vector
  v = zeros(6)
  @test destructure!(v, m3)[1] === v
  @test v == 1:6
  m9′′ = deepcopy(m9)
  v9, re9! = destructure!(zeros(Float32, 7), m9′′)
  @test v9 isa Vector{Float32} && v9 == 1:7
  @test_logs re9!(v9 .+ 1)  # less precise than the model, so no warning
  @test m9′′.c[1] == mat .+ 1
  @test m9′′.a isa Vector{Float64}

  # errors
  @test_throws DimensionMismatch re7!([10, 20])
  @test_throws DimensionMismatch re7!([10, 20, 30, 40])
  @test_throws DimensionMismatch destructure!(zeros(5), m3)
  @test_throws ArgumentError destructure!((a = SA[1.0, 2.0], b = [3.0]))
  @test_throws ArgumentError destructure!(zeros(3), (a = SA[1.0, 2.0], b = [3.0]))
  @test_throws ArgumentError destructure!((a = Diagonal([1.0, 2.0]),))
  @test_throws ArgumentError destructure!([1.0, 2.0])[2]([1, 2im])
end

@testset "gradient of rebuild!" begin
  re1! = destructure!(deepcopy(m1))[2]
  @test gradient(x -> re1!(x)[1], rand(3))[1] == [1,0,0]

  re3! = destructure!(deepcopy(m3))[2]
  @test gradient(x -> re3!(x).x[3], rand(6))[1] == [0,0,1,0,0,0]
  @test gradient(x -> re3!(x).z[1], rand(6))[1] == [0,0,0,1,0,0]

  re4! = destructure!(deepcopy(m4))[2]
  @test gradient(rand(6)) do x
    m = re4!(x)
    m.x[1] + 2*m.y[2] + 3*m.z[3]
  end[1] == [1,2,0, 0,0,3]

  w = (W = [1.0 2.0; 3.0 4.0], b = [0.0, 0.0])
  p, rew! = destructure!(w)
  q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  loss(m) = sum(abs2, m.W * [1, 1] .+ m.b)
  @test gradient(q -> loss(rew!(q)), q)[1] == gradient(q -> loss(destructure(w)[2](q)), q)[1]

  @test gradient(m -> destructure!(m)[1][1], deepcopy(m3))[1] == (x = [1,0,0], y = nothing, z = [0,0,0])
end

@testset "destructure with alias" begin
  v3, re3 = destructure(m3; alias = true)
  @test v3 == 1:6
  p = collect(4.0:9.0)
  m = @inferred re3(p)
  @test m == (x = [4,5,6], y = sin, z = [7,8,9])
  p[1] = 0
  @test m.x == [0, 5, 6]
  m.z[1] = 10
  @test p[4] == 10
  # views cannot convert, so p must have the model's element type
  @test_throws "needs a vector of Float64" re3(Float32.(p))
  @test_throws "needs a vector of Float64" re3(collect(1:6))
  @test_throws "no complex parameters" re3(ComplexF64.(p))
  @test eltype(re3(ForwardDiff.Dual.(p, 1.0)).x) == ForwardDiff.Dual{Nothing, Float64, 1}
  @test_throws "needs a vector of ForwardDiff.Dual{Nothing, Float64, 1}" re3(ForwardDiff.Dual.(Float32.(p), 1f0))
  @test_throws "one element type" destructure(m9; alias = true)
  @test re3 isa Optimisers.Restructure{<:Any, <:Any, :alias}
  @test sprint(show, re3) == "Restructure(NamedTuple, ..., 6; alias = true)"
  @test length(re3) == 6
  @test fmap(x -> x isa AbstractArray ? 2x : x, re3) isa Optimisers.Restructure{<:Any, <:Any, :alias}

  m7a = destructure(m7; alias = true)[2]([10.0, 20, 30])
  @test m7a.a == (sin, [10, 20, 30])
  @test m7a.b === m7.b

  # tied arrays are views of the same part of p, also arrays of arrays
  m8a = destructure(m8; alias = true)[2](collect(10.0:14.0))
  @test m8a[1].x === m8a[1].y == [10, 11, 12]
  @test m8a[2].b.x == [13]
  @test m8a[3][1] == [14]
  mat64 = Float64.(mat)
  m9a = destructure((a = m1, b = mat64, c = [mat64, m1]); alias = true)[2](collect(10.0:10:70))
  @test m9a.a === m9a.c[2] == [10, 20, 30]
  @test m9a.b === m9a.c[1] == [40 60; 50 70]

  W = [1.0 2.0; 3.0 4.0; 5.0 6.0]
  t = (enc = TLayer(W, zeros(3)), dec = TLayer(transpose(W), zeros(2)))
  q = collect(1.0:11.0)
  ta = destructure(t; alias = true)[2](q)
  @test ta.dec.w.parent === ta.enc.w == [1 4; 2 5; 3 6]
  @test Base.mightalias(ta.enc.w, q)

  d = Dict(:a => Dict(:b => [1.0], :c => [2.0, 3.0]), :d => [4.0])  # issue 154
  vd, red = destructure(d; alias = true)
  da = red(vd)
  vd .*= 10
  @test da[:a][:b] == [10] && da[:a][:c] == [20, 30] && da[:d] == [40]

  sk = Splat([1.0, 2.0], (x = 3, y = [4.0, 5.0]))  # issue 62
  @test destructure(sk; alias = true)[2]([10.0, 20, 30, 40]).layers == ([10, 20], (x = 3, y = [30, 40]))

  # a model made of views can be destructured again
  v3a, re3a = destructure(re3(p); alias = true)
  @test v3a == p
  @test re3a(v3) == m3

  @test destructure([1.0, 2.0]; alias = true)[2]([3.0, 4.0]) == [3, 4]
  @test destructure((sin, cos); alias = true)[2](Bool[]) == (sin, cos)

  @test_throws DimensionMismatch re3(rand(5))
  @test_throws ArgumentError destructure((a = SA[1.0, 2.0], b = [3.0]); alias = true)
  @test_throws ArgumentError destructure((a = Diagonal([1.0, 2.0]),); alias = true)
end

@testset "gradient of rebuild with alias" begin
  re1 = destructure(m1; alias = true)[2]
  @test gradient(x -> re1(x)[1], rand(3))[1] == [1,0,0]

  re4 = destructure(m4; alias = true)[2]
  @test gradient(rand(6)) do x
    m = re4(x)
    m.x[1] + 2*m.y[2] + 3*m.z[3]
  end[1] == [1,2,0, 0,0,3]

  W = [1.0 2.0; 3.0 4.0; 5.0 6.0]
  t = (enc = TLayer(W, zeros(3)), dec = TLayer(transpose(W), zeros(2)))
  q = rand(11)
  loss(re) = q -> (m = re(q); sum(m.dec.w * tanh.(m.enc.w * [1, 2] .+ m.enc.b) .+ m.dec.b))
  @test gradient(loss(destructure(t; alias = true)[2]), q)[1] ≈ gradient(loss(destructure(t)[2]), q)[1]

  @test gradient(m -> destructure(m; alias = true)[1][1], m3)[1] == (x = [1,0,0], y = nothing, z = [0,0,0])
end

@testset "element types, issue 95" begin
  m = (x = Float32[1, 2], y = TwoThirds(Float32[3 4], Float32[5], Float32[6]))  # y.c is not trainable
  v, re = destructure(m)
  @test v isa Vector{Float32}
  @test (@test_logs re(v .+ 1)).x isa Vector{Float32}
  m64 = @test_logs (:warn, r"precision Float64, for a model whose parameters are Float32") re(Float64.(v))
  @test m64.x isa Vector{Float32}
  @test m64.y.a isa Matrix{Float32}
  @test m64.y.c === m.y.c  # shared with the model
  @test (@test_logs re(1:4)).y.a isa Matrix{Float32}
  @test (@test_logs re(Float16.(v))).x isa Vector{Float32}  # no precision is lost

  # Dual numbers take the model's precision, so ForwardDiff and Zygote see the same model
  d = ForwardDiff.Dual.(Float64.(v), 1.0)
  @test (@test_logs (:warn, r"precision Float64") re(d)).x isa Vector{ForwardDiff.Dual{Nothing, Float32, 1}}
  d32 = ForwardDiff.Dual.(v, 1f0)
  @test eltype((@test_logs re(d32)).x) == eltype(d32)
  loss(p) = sum(abs2, re(p).x) + sum(re(p).y.a)
  p = Float64.(v) .+ 0.1
  gf = @test_logs (:warn,) match_mode=:any ForwardDiff.gradient(loss, p)
  gz = @test_logs (:warn,) match_mode=:any Zygote.gradient(loss, p)[1]
  @test gf == gz == [2 .* Float32.(p[1:2]); 1; 1]  # computed in Float32
  h = @test_logs (:warn,) match_mode=:any ForwardDiff.hessian(q -> sum(abs2, re(q).x), p)  # nested Duals
  @test h == Diagonal([2, 2, 0, 0])

  # real parameters stay real, complex ones complex
  @test_throws "no complex parameters" re(ComplexF32.(v))
  mc = (x = [1.0, 2.0], y = [3.0 + 4.0im])
  vc, rec = destructure(mc)
  @test vc isa Vector{ComplexF64}
  @test rec(2vc) == (x = [2.0, 4.0], y = [6.0 + 8.0im])
  @test rec([1 + 9im, 2, 3]).x == [1, 2]
  mc32 = @test_logs rec(ComplexF32.(vc))
  @test mc32.x isa Vector{Float64} && mc32.y isa Vector{ComplexF64}
  @test rec([1.0, 2.0, 3.0]).y isa Vector{ComplexF64}

  # structured arrays keep their structure, and precision
  @testset "$(nameof(typeof(w)))" for w in (Diagonal(Float32[1, 2]), Symmetric(Float32[1 2; 2 3]),
                                           UpperTriangular(Float32[1 2; 0 3]), SA{Float32}[1, 2])
    vw, rew = destructure((; w))
    @test typeof(@test_logs((:warn,), rew(Float64.(vw))).w) == typeof(w)
    @test rew(vw).w == w
  end

  # mixed precision is kept, without a warning
  mm = (a = Float16[1, 2], b = Float32[3])
  vm, rem = destructure(mm)
  @test vm isa Vector{Float32}
  @test (@test_logs rem(vm)).a isa Vector{Float16}

  # Functors' reconstruction finds the model's new precision
  re64 = fmap(y -> y isa AbstractArray{Float32} ? Float64.(y) : y, re)
  @test (@test_logs re64(Float64.(v))).x isa Vector{Float64}
  @test (@test_logs Optimisers.Restructure(re64.model, re.offsets, re.length)(Float64.(v))).x isa Vector{Float64}

  # in place: converts, and warns about lost precision too
  m′ = deepcopy(m)
  re! = destructure!(m′)[2]
  @test (@test_logs (:warn, r"Restructure! got numbers of precision Float64") re!(Float64.(v))).x isa Vector{Float32}
  @test_throws "no complex parameters" re!(ComplexF32.(v))

  # with alias, the vector must have the model's element type
  rea = destructure(m; alias = true)[2]
  @test (@test_logs rea(v)).x isa SubArray{Float32}
  @test_throws "needs a vector of Float32" rea(Float64.(v))
  @test_throws "needs a vector of Float32" rea(Float16.(v))
  @test_throws "no complex parameters" rea(ComplexF32.(v))
  @test eltype(rea(d32).x) == eltype(d32)
  @test_throws ArgumentError rea(d)
  @test_throws "one element type" destructure(mm; alias = true)
end
