
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
end

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
