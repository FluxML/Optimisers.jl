
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
    # With Diffractor, non-leaf _grad!(x, dx, off, flat::AbstractVector) gets double-wrapped dx:
    # off = (0, 3), dx = Tangent{Tangent{Tuple{Vector{Float64}, Vector{Float64}}, ...
    # until you add explicit double-unwrap: base(dx::Tangent{<:Tangent}) = backing(dx).backing
    # With Zygote, instead:
    # dx = Tangent{Any}(backing = Tangent{Any}([4.0, 8.0, 12.0], ZeroTangent()),)

    @test gradient([1,2,3.0]) do v
      sum(gradient(m -> sum(destructure(m)[1])^3, (v, [4,5,6.0]))[1][1])
    end[1] == [378, 378, 378]

    @test_broken gradient([1,2,3.0]) do v
      sum(abs2, gradient(m -> sum(abs2, destructure(m)[1]), (x = v, y = sin, z = [4,5,6.0]))[1][1])
    end[1] ≈ [8,16,24]
    # Zygote error in (::typeof(∂(canonicalize)))(Δ::NamedTuple{(:backing,), Tuple{NamedTuple{(:x, :y, :z)
    # Diffractor error in perform_optic_transform
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
    @test_broken gradient(collect(1:6.0)) do y
      sum(abs2, gradient(x -> sum(abs2, re2(x)[1]), y)[1])
    end[1] ≈ [8,16,24,0,0,0]
    # ERROR: Need an adjoint for constructor ChainRulesCore.Tangent{Any, Tuple{Vector{Float64}, ChainRulesCore.ZeroTangent}}. Gradient is of type Tuple{Vector{Float64}, Vector{Float64}}
    # with Zygote, which can be fixed by:
    # Zygote.@adjoint Tangent{T,B}(x::Tuple) where {T,B<:Tuple} = Tangent{T,B}(x), dx -> (dx,)

    @test_broken gradient(collect(1:6.0)) do y
      sum(abs2, gradient(x -> sum(abs2, re3(x).z), y)[1])
    end[1] ≈ [0,0,0,32,40,48]
    # Not fixed by this:
    # Zygote.@adjoint Tangent{T,B}(x::NamedTuple) where {T,B<:NamedTuple} = Tangent{T,B}(x), dx -> (dx,)
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
  v, re = destructure((x=[1,2.0], y=[3,4,5.0]))
  _, bk = Zygote.pullback(re, ones(5))
  # Testing with `Complex` isn't ideal, but this was an error on 0.2.1.
  # If some upgrade inserts ProjectTo, this will fail, and can be changed:
  @test bk((x=[1.0,im], y=nothing)) == ([1,im,0,0,0],)
  
  @test bk((x=nothing, y=[10,20,30]))[1] isa Vector{Float64}  # despite some ZeroTangent
  @test bk((x=nothing, y=nothing)) == (nothing,)  # don't reduce over empty list of eltypes
  @test bk((x=nothing, y=@thunk [1,2,3.0] .* 10)) == ([0,0,10,20,30],)
end

#=

# Adapted from https://github.com/SciML/DiffEqFlux.jl/pull/699#issuecomment-1092846657
using ForwardDiff, Zygote, Flux, Optimisers, Test

y = Float32[0.8564646, 0.21083355]
p = randn(Float32, 252);
t = 1.5f0
λ = [ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float32}}(0.87135935, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float32}}(1.5225363, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

model = Chain(x -> x .^ 3,
    Dense(2 => 50, tanh),
    Dense(50 => 2))

p,re = Optimisers.destructure(model)
f(u, p, t) = re(p)(u)
_dy, back = Zygote.pullback(y, p) do u, p
    vec(f(u, p, t))
end
tmp1, tmp2 = back(λ);
tmp1
@test tmp2 isa Vector{<:ForwardDiff.Dual}

=#
