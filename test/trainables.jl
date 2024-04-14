
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

@testset "trainables" begin
    ps = trainables(m1)
    @test ps isa Vector
    @test length(ps) == 1
    @test ps[1] == m1

    ps = trainables(m2)
    @test ps isa Vector
    @test length(ps) == 2
    @test ps[1] == m2[1]
    @test ps[2] == m2[2]

    ps = trainables(m3)
    @test length(ps) == 2
    @test ps[1] == 1:3
    @test ps[2] == 4:6

    ps = trainables(m4)
    @test length(ps) == 2
    @test ps[1] == 1:3
    @test ps[2] == 4:6
    
    ps = trainables(m5)
    @test length(ps) == 3
    @test ps[1] == 1:3
    @test ps[2] == 4:6
    @test ps[3] == 4:6

    ps = trainables(m6)
    @test length(ps) == 2
    @test ps[1] == 1:3
    @test ps[2] == ComplexF64[4.0 + 1.0im]

    ps = trainables(m7)
    @test length(ps) == 1
    @test ps[1] == [1.0, 2.0, 3.0]

    ps = trainables(m8)
    @test length(ps) == 3
    @test ps[1] == 1:3
    @test ps[2] == [4.0]
    @test ps[3] == [5.0]

    ps = trainables(m9)
    @test length(ps) == 2
    @test ps[1] == 1:3
    @test ps[2] == mat

    @testset "gradient" begin
        loss(m) = sum([sum(abs2, p) for p in trainables(m)])
        g = gradient(loss, m1)[1]
        @test g == [2.0, 4.0, 6.0]

        g = gradient(loss, m2)[1]
        @test g == ([2.0, 4.0, 6.0], [8.0, 10.0, 12.0])

        g = gradient(loss, m3)[1]
        @test g.x == [2.0, 4.0, 6.0]
        @test g.y === nothing
        @test g.z == [8.0, 10.0, 12.0]

        g = gradient(loss, m4)[1]
        @test g == (x = [2.0, 4.0, 6.0], y = [2.0, 4.0, 6.0], z = [8.0, 10.0, 12.0])
        g.x === g.y # shared gradient for shared weights

        g = gradient(loss, m5)[1]
        @test g == (a = ((x = [2.0, 4.0, 6.0], y = nothing, z = [8.0, 10.0, 12.0]), nothing), b = ([2.0, 4.0, 6.0], nothing), c = ((x = [2.0, 4.0, 6.0], y = [2.0, 4.0, 6.0], z = [8.0, 10.0, 12.0]), nothing))
        
        g = gradient(loss, m6)[1]
        @test g == (a = [2.0, 4.0, 6.0], b = ComplexF64[8.0 + 2.0im], c = [2.0, 4.0, 6.0])
        
        g = gradient(loss, m7)[1]
        @test g == (a = (nothing, [2.0, 4.0, 6.0]), b = nothing, c = nothing)

        g = gradient(loss, m8)[1]
        @test g[1] == (x = [2.0, 4.0, 6.0], y = [2.0, 4.0, 6.0])
        @test g[2] == (a = nothing, b = (x = [8.0], y = nothing), c = nothing)
        @test g[3] == [[10.0]]

        g = gradient(loss, m9)[1]
        @test g == (a = [2.0, 4.0, 6.0], b = Float32[8.0 12.0; 10.0 14.0], c = Array[Float32[8.0 12.0; 10.0 14.0], [2.0, 4.0, 6.0]])
    end

    @testset "dict" begin
        d = Dict(:a => rand(2), :b => ones(2))
        ps = trainables(d)
        @test length(ps) == 2
        @test ps[1] == d[:a]
        @test ps[2] == d[:b]

        g = gradient(d -> sum(trainables(d)[1].^2) /2 + sum(trainables(d)[2]), d)[1]
        @test g[:a] == d[:a]
        @test_broken g[:b] == [1.0, 1.0]
    end

    @testset "second order derivatives" begin
        struct DenseLayer
            w
            b
        end

        Functors.@functor DenseLayer

        loss(m) = sum([sum(abs2, p) for p in trainables(m)])

        model = DenseLayer([1. 2.; 3. 4.], [0., 0.])

        g = gradient(m -> loss(gradient(loss, m)), model)[1]
        @test g.w == [8.0 16.0; 24.0 32.0]
        @test g.b == [0.0, 0.0]
    end

    @testset "trainables(x, path=true)" begin
        loss(m) = sum(abs2, trainables(m, path=true)[1][2])

        ps = trainables(m4, path=true)
        @test length(ps) == 2
        @test ps[1] == (KeyPath(:x,), [1.0, 2.0, 3.0])
        @test ps[2] == (KeyPath(:z,), [4.0, 5.0, 6.0])

        g = gradient(loss, m4)[1]
        @test g.x == [2.0, 4.0, 6.0]
        @test g.y == [2.0, 4.0, 6.0]
        @test g.z === nothing
    end
end 

@testset "trainables_nt" begin

    @testset "nt & rebuild" begin
        @test trainables_nt(m1)[1] isa Vector{Float64}
        @test trainables_nt(m1)[1] == 1:3
        @test trainables_nt(m2)[1] == (_1 = [1.0, 2.0, 3.0], _2 = [4.0, 5.0, 6.0])
        @test trainables_nt(m3)[1] == (x = [1.0, 2.0, 3.0], y = NamedTuple(), z = [4.0, 5.0, 6.0])
        @test trainables_nt(m4)[1] == (x = [1.0, 2.0, 3.0], y = [1.0, 2.0, 3.0], z = [4.0, 5.0, 6.0])
        @test trainables_nt(m5)[1] == (a = (_1 = (x = [1.0, 2.0, 3.0], y = NamedTuple(), z = [4.0, 5.0, 6.0]), _2 = NamedTuple()), b = (_1 = [1.0, 2.0, 3.0], _2 = NamedTuple()), c = (_1 = (x = [1.0, 2.0, 3.0], y = [1.0, 2.0, 3.0], z = [4.0, 5.0, 6.0]), _2 = NamedTuple()))
        @test trainables_nt(m6)[1] == (a = [1.0, 2.0, 3.0], b = ComplexF64[4.0 + 1.0im], c = [1.0, 2.0, 3.0])
        nt9 = trainables_nt(m9)[1] 
        @test nt9 == (a = [1.0, 2.0, 3.0], b = Float32[4.0 6.0; 5.0 7.0], c = (_1 = Float32[4.0 6.0; 5.0 7.0], _2 = [1.0, 2.0, 3.0]))
        @test nt9.a === m9.a     # no copies
        @test nt9.a === nt9.c._2  # keeps shared references

        @test trainables_nt(m1)[2](7:9) == [7,8,9]
        @test trainables_nt(m2)[2]((_1 = 4:6, _2 = 7:9)) == ([4,5,6], [7,8,9])
        # reconstruction doesn't need the full input
        @test trainables_nt(m2)[2]((; _2 = 7:9)) == ([1,2,3], [7,8,9])
        @test trainables_nt(m3)[2]((x =4:6, z=7:9)) == (x = [4,5,6], y = sin, z = [7,8,9])
        a = [4,5,6]
        Δ = (x=a, y=a, z=7:9)
        m4′ = trainables_nt(m4)[2](Δ)
        @test m4′ == (x = [4,5,6], y = [4,5,6], z = [7,8,9])
        @test m4′.x === m4′.y  # shared references are preserved

        # struct, partially trainable
        a = [10,20,30]
        @test trainables_nt(m7)[1] == (a = (_1 = NamedTuple(), _2 = [1.0, 2.0, 3.0]),)
        m7′ = trainables_nt(m7)[2]((; a = (; _2 = a)))
        @test m7′.a == (sin, a)
        @test m7′.b == (cos, [4,5,6])
        @test m7′.c == (tan, [7,8,9])
        @test m7′.a[2] === a # no copies

        @test trainables_nt(m8)[1] == (_1 = (y = [1.0, 2.0, 3.0], x = [1.0, 2.0, 3.0]), _2 = (a = NamedTuple(), b = (y = NamedTuple(), x = [4.0]), c = NamedTuple()), _3 = (_1 = [5.0],))
        a = [7, 8, 9]
        m8′ = trainables_nt(m8)[2]((; _1 = (; x = a)))
        @test m8′[1].x === a
        @test m8′[1].y == [1.0, 2.0, 3.0] # tie is broken
        @test m8′[2].b.y === false
        @test m8′[3][1] == [5.0]
    end

    @testset "re(ComponentArrays)" begin
        model = TwoThirds((a=[1.0,2.0], b=([3.,4.0], [5.0 6.0; 7.0 8.0]), c=[11,12]), [9.0], [10.0])
        ps, re = trainables_nt(model)
        @test ps == (a = (a = [1.0, 2.0], b = (_1 = [3.0, 4.0], _2 = [5.0 6.0; 7.0 8.0]), c = NamedTuple()),)
        v = ComponentVector(ps)
        v2 = 2v
        model′ = re(v2)
        @test model′.a.a == 2*model.a.a
        @test model′.a.b[1] == 2*model.a.b[1]
        @test model′.a.b[2] == 2*model.a.b[2]
        @test model′.a.c == model.a.c
        @test model′.b == model.b
        @test model′.c == model.c
        @test model′.a.b[1] === v2.a.b._1 # no copies
        @test model′.a.a === v2.a.a
    end
end

# @testset "gradient of flatten" begin
#   @test gradient(m -> trainables_nt(m)[1][1], m1)[1] == [1,0,0]
#   @test gradient(m -> trainables_nt(m)[1][2], m2)[1] == ([0,1,0], [0,0,0])
#   @test gradient(m -> trainables_nt(m)[1][3], (m1, m1))[1] == ([0,0,1], nothing)
#   @test gradient(m -> trainables_nt(m)[1][1], m3)[1] == (x = [1,0,0], y = nothing, z = [0,0,0])
#   @test gradient(m -> trainables_nt(m)[1][2], m4)[1] == (x = [0,1,0], y = nothing, z = [0,0,0])

#   g5 = gradient(m -> trainables_nt(m)[1][3], m5)[1]
#   @test g5.a[1].x == [0,0,1]
#   @test g5.a[2] === nothing

#   g6 = gradient(m -> imag(trainables_nt(m)[1][4]), m6)[1]
#   @test g6.a == [0,0,0]
#   @test g6.a isa Vector{Float64}
#   @test g6.b == [0+im]

#   g8 = gradient(m -> sum(abs2, trainables_nt(m)[1]), m8)[1]
#   @test g8[1].x == [2,4,6]
#   @test g8[2].b.x == [8]
#   @test g8[3] == [[10.0]]

#   g9 = gradient(m -> sum(sqrt, trainables_nt(m)[1]), m9)[1]
#   @test g9.c === nothing

#   @testset "second derivative" begin
#     @test gradient([1,2,3.0]) do v
#       sum(abs2, gradient(m -> sum(abs2, trainables_nt(m)[1]), (v, [4,5,6.0]))[1][1])
#     end[1] ≈ [8,16,24]
#     # With Diffractor, non-leaf _grad!(x, dx, off, flat::AbstractVector) gets double-wrapped dx:
#     # off = (0, 3), dx = Tangent{Tangent{Tuple{Vector{Float64}, Vector{Float64}}, ...
#     # until you add explicit double-unwrap: base(dx::Tangent{<:Tangent}) = backing(dx).backing
#     # With Zygote, instead:
#     # dx = Tangent{Any}(backing = Tangent{Any}([4.0, 8.0, 12.0], ZeroTangent()),)

#     @test gradient([1,2,3.0]) do v
#       sum(gradient(m -> sum(trainables_nt(m)[1])^3, (v, [4,5,6.0]))[1][1])
#     end[1] == [378, 378, 378]

#     VERSION >= v"1.10" && @test gradient([1,2,3.0]) do v
#       sum(abs2, gradient(m -> sum(abs2, trainables_nt(m)[1]), (x = v, y = sin, z = [4,5,6.0]))[1][1])
#     end[1] ≈ [8,16,24]
#     # Zygote error in (::typeof(∂(canonicalize)))(Δ::NamedTuple{(:backing,), Tuple{NamedTuple{(:x, :y, :z)
#     # Diffractor error in perform_optic_transform
#   end
  
#   false && @testset "using Yota" begin
#     @test Yota_gradient(m -> trainables_nt(m)[1][1], m1)[1] == [1,0,0]
#     @test Yota_gradient(m -> trainables_nt(m)[1][2], m2)[1] == ([0,1,0], [0,0,0])
#     @test Yota_gradient(m -> trainables_nt(m)[1][3], (m1, m1))[1] == ([0,0,1], nothing)
#     @test Yota_gradient(m -> trainables_nt(m)[1][1], m3)[1] == (x = [1,0,0], y = nothing, z = [0,0,0])
#     @test Yota_gradient(m -> trainables_nt(m)[1][2], m4)[1] == (x = [0,1,0], y = nothing, z = [0,0,0])

#     g5 = Yota_gradient(m -> trainables_nt(m)[1][3], m5)[1]
#     @test g5.a[1].x == [0,0,1]
#     @test g5.a[2] === nothing

#     g6 = Yota_gradient(m -> imag(trainables_nt(m)[1][4]), m6)[1]
#     @test g6.a == [0,0,0]
#     @test g6.a isa Vector{Float64}
#     @test g6.b == [0+im]

#     g8 = Yota_gradient(m -> sum(abs2, trainables_nt(m)[1]), m8)[1]
#     @test g8[1].x == [2,4,6]
#     @test g8[2].b.x == [8]
#     @test g8[3] == [[10.0]]

#     g9 = Yota_gradient(m -> sum(sqrt, trainables_nt(m)[1]), m9)[1]
#     @test g9.c === nothing
#   end
# end

# @testset "gradient of rebuild" begin
#   re1 = trainables_nt(m1)[2]
#   @test gradient(x -> re1(x)[1], rand(3))[1] == [1,0,0]
#   re2 = trainables_nt(m2)[2]
#   @test gradient(x -> re2(x)[1][2], rand(6))[1] == [0,1,0,0,0,0]
#   re3 = trainables_nt(m3)[2]
#   @test gradient(x -> re3(x).x[3], rand(6))[1] == [0,0,1,0,0,0]
#   @test gradient(x -> re3(x).z[1], rand(6))[1] == [0,0,0,1,0,0]

#   re4 = trainables_nt(m4)[2]
#   @test gradient(x -> re4(x).x[1], rand(6))[1] == [1,0,0,0,0,0]
#   @test gradient(x -> re4(x).y[2], rand(6))[1] == [0,1,0,0,0,0]
#   @test gradient(rand(6)) do x
#     m = re4(x)
#     m.x[1] + 2*m.y[2] + 3*m.z[3]
#   end[1] == [1,2,0, 0,0,3]

#   re7 = trainables_nt(m7)[2]
#   @test gradient(x -> re7(x).a[2][3], rand(3))[1] == [0,0,1]
#   @test gradient(x -> re7(x).b[2][2], rand(3))[1] == [0,0,0]
#   @test gradient(x -> re7(x).c[2][1], rand(3))[1] == [0,0,0]

#   v8, re8 = trainables_nt(m8)
#   @test gradient(x -> sum(abs2, re8(x)[1].y), v8)[1] == [2,4,6,0,0]
#   @test gradient(x -> only(sum(re8(x)[3]))^2, v8)[1] == [0,0,0,0,10]

#   re9 = trainables_nt(m9)[2]
#   @test gradient(x -> sum(abs2, re9(x).c[1]), 1:7)[1] == [0,0,0, 8,10,12,14]

#   @testset "second derivative" begin
#     @test_broken gradient(collect(1:6.0)) do y
#       sum(abs2, gradient(x -> sum(abs2, re2(x)[1]), y)[1])
#     end[1] ≈ [8,16,24,0,0,0]
#     # ERROR: Need an adjoint for constructor ChainRulesCore.Tangent{Any, Tuple{Vector{Float64}, ChainRulesCore.ZeroTangent}}. Gradient is of type Tuple{Vector{Float64}, Vector{Float64}}
#     # with Zygote, which can be fixed by:
#     # Zygote.@adjoint Tangent{T,B}(x::Tuple) where {T,B<:Tuple} = Tangent{T,B}(x), dx -> (dx,)

#     @test_broken gradient(collect(1:6.0)) do y
#       sum(abs2, gradient(x -> sum(abs2, re3(x).z), y)[1])
#     end[1] ≈ [0,0,0,32,40,48]
#     # Not fixed by this:
#     # Zygote.@adjoint Tangent{T,B}(x::NamedTuple) where {T,B<:NamedTuple} = Tangent{T,B}(x), dx -> (dx,)
#   end
  
#   false && @testset "using Yota" begin
#     re1 = trainables_nt(m1)[2]
#     @test Yota_gradient(x -> re1(x)[1], rand(3))[1] == [1,0,0]
#     re2 = trainables_nt(m2)[2]
#     @test Yota_gradient(x -> re2(x)[1][2], rand(6))[1] == [0,1,0,0,0,0]
#     re3 = trainables_nt(m3)[2]
#     @test Yota_gradient(x -> re3(x).x[3], rand(6))[1] == [0,0,1,0,0,0]
#     @test Yota_gradient(x -> re3(x).z[1], rand(6))[1] == [0,0,0,1,0,0]

#     re4 = trainables_nt(m4)[2]
#     @test Yota_gradient(x -> re4(x).x[1], rand(6))[1] == [1,0,0,0,0,0]
#     @test Yota_gradient(x -> re4(x).y[2], rand(6))[1] == [0,1,0,0,0,0]
#     @test Yota_gradient(rand(6)) do x
#       m = re4(x)
#       m.x[1] + 2*m.y[2] + 3*m.z[3]
#     end[1] == [1,2,0, 0,0,3]

#     re7 = trainables_nt(m7)[2]
#     @test Yota_gradient(x -> re7(x).a[2][3], rand(3))[1] == [0,0,1]
#     @test Yota_gradient(x -> re7(x).b[2][2], rand(3))[1] == [0,0,0]
#     @test Yota_gradient(x -> re7(x).c[2][1], rand(3))[1] == [0,0,0]

#     v8, re8 = trainables_nt(m8)
#     @test Yota_gradient(x -> sum(abs2, re8(x)[1].y), v8)[1] == [2,4,6,0,0]
#     @test Yota_gradient(x -> only(sum(re8(x)[3]))^2, v8)[1] == [0,0,0,0,10]

#     re9 = trainables_nt(m9)[2]
#     @test Yota_gradient(x -> sum(abs2, re9(x).c[1]), 1:7)[1] == [0,0,0, 8,10,12,14]
#   end
# end

# @testset "Flux issue 1826" begin
#   v, re = trainables_nt((x=[1,2.0], y=[3,4,5.0]))
#   @test gradient(zero(v)) do w
#     m = re(w)
#     5 * sum(m.x) + 7 * sum(m[2])  # uses both x and y
#   end == ([5.0, 5.0, 7.0, 7.0, 7.0],)
#   # This, using only x, was broken on Flux:
#   @test gradient(w -> sum(re(w).x), zero(v)) == ([1.0, 1.0, 0.0, 0.0, 0.0],)

#   sh = [7,7.0];
#   v, re = trainables_nt((x=sh, y=[3.0,4.0], z=sh))  # shared array in the model
#   @test v == [7, 7, 3, 4]
#   @test re([1,10,100,1000]) == (x = [1, 10], y = [100, 1000], z = [1, 10])

#   @test gradient(zero(v)) do w
#     m = re(w)
#     3 * sum(m.x) + 13 * sum(m.z)  # no dependence on y, but two distinct gradient arrays
#   end == ([16, 16, 0, 0],)  # Flux gave ([3.0, 3.0, 13.0, 13.0],)

#   @test gradient(zero(v)) do w
#     m = re(w)
#     4(sum(m.x) + sum(m.z))  # now two gradients are ===, so it eliminates one
#   end == ([8,8,0,0],)

#   @test gradient(zero(v)) do w
#     m = re(w)
#     4(sum(m.x) + sum(m.y)) + 13*sum(m.z)  # again two gradients are ===, so it eliminates one
#   end == ([17,17,4,4],)  # Flux gave ([4.0, 4.0, 13.0, 13.0],)
# end

# @testset "DiffEqFlux issue 699" begin
#   # The gradient of `re` is a vector into which we accumulate contributions, and the issue
#   # is that one contribution may have a wider type than `v`, especially for `Dual` numbers.
#   v, re = destructure((x=Float32[1,2], y=Float32[3,4,5]))
#   _, bk = Zygote.pullback(re, ones(Float32, 5))
#   # Testing with `Complex` isn't ideal, but this was an error on 0.2.1.
#   # If some upgrade inserts ProjectTo, this will fail, and can be changed:
#   @test bk((x=[1.0,im], y=nothing)) == ([1,im,0,0,0],)
  
#   @test bk((x=nothing, y=[10,20,30]))[1] isa Vector{Float32}  # despite some ZeroTangent
#   @test bk((x=nothing, y=nothing)) == ([0,0,0,0,0],) 
#   @test bk((x=nothing, y=@thunk [1,2,3] .* 10.0)) == ([0,0,10,20,30],)
#   @test bk((x=[1.2, 3.4], y=Float32[5,6,7])) == ([1.2, 3.4, 5, 6, 7],)
# end

# #=

# # Adapted from https://github.com/SciML/DiffEqFlux.jl/pull/699#issuecomment-1092846657
# using ForwardDiff, Zygote, Flux, Optimisers, Test

# y = Float32[0.8564646, 0.21083355]
# p = randn(Float32, 27);
# t = 1.5f0
# λ = [ForwardDiff.Dual(0.87135935, 1, 0, 0, 0, 0, 0), ForwardDiff.Dual(1.5225363, 0, 1, 0, 0, 0, 0)]

# model = Chain(x -> x .^ 3,
#     Dense(2 => 5, tanh),
#     Dense(5 => 2))

# p,re = Optimisers.destructure(model)
# f(u, p, t) = re(p)(u)
# _dy, back = Zygote.pullback(y, p) do u, p
#     vec(f(u, p, t))
# end
# tmp1, tmp2 = back(λ);
# tmp1
# @test tmp2 isa Vector{<:ForwardDiff.Dual}

# =#

# @testset "empty, issue 67" begin
#     m0 = (nothing, missing, isempty)
#     @test destructure(m0)[1] isa Vector{<:Real}
#     v0, re0 = destructure(m0)
#     @test re0(Float32[]) === m0
#     @test_throws DimensionMismatch re0([1])

#     # This is an elaborate way of checking that it doesn't cause promotions, even of small floats:
#     m01 = [(x=nothing, y=0), (x=Float16[1, 2], y=Float16[3])]
#     v01, _ = destructure(m01)
#     v012 = vcat(destructure(m01[1])[1], destructure(m01[2])[1])
#     @test v01 == v012
#     @test v012 isa Vector{Float16}

#     y, bk = Zygote.pullback(x -> sum(destructure(x)[1]), ("a", :beta))
#     @test bk(1.0) == (nothing,)
#     # Zygote regards 3,4 as differentiable, but Optimisers does not regard them as parameters:
#     y, bk = Zygote.pullback(x -> sum(destructure(x)[1]), (3, 4))
#     @test bk(1.0) == (nothing,)
# end


using Flux, Optimisers
using ComponentArrays
using Test


model0 = Chain(
  Dense(784, 32, relu),
  Dense(32, 10))

ps, re = trainables_nt(model0)
@test ps.layers._1.weight === model0[1].weight
model1 = re(ps)
@test model1[1].weight === ps.layers._1.weight

v = ComponentVector(ps)
v2 = 2 * v
model2 = re(v2)
@test model2[1].weight === v2.layers._1.weight

g = gradient(model0) do m
    ps, re = trainables_nt(m)
    return sum(ps.layers._1.weight)
end[1]
@test eltype(g.layers[1].weight) == Float32
@test g.layers[1].weight ≈ ones(Float32, 32, 784)
@test g.layers[1].bias === nothing
@test g.layers[2] === nothing


# # TODO
# - [] Name?
# - [] Should the named tuple contain NamedTuple() leaves?
# - [] Optimize performance and improve type stability
