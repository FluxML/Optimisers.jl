
m1 = collect(1:3.0)
m2 = (collect(1:3.0), collect(4:6.0))
m3 = (x = m1, y = sin, z = collect(4:6.0))
m4 = (x = m1, y = m1, z = collect(4:6.0))
m5 = (a = (m3, true), b = (m1, false), c = (m4, true))
m6 = (a = m1, b = [4.0 + im], c = m1)
m7 = TwoThirds((sin, collect(1:3.0)), (cos, collect(4:6.0)), (tan, collect(7:9.0)))

@testset "flatten & rebuild" begin
  @test destructure(m1)[1] isa Vector{Float64}
  @test destructure(m1)[1] == 1:3
  @test destructure(m2)[1] == 1:6
  @test destructure(m3)[1] == 1:6
  @test destructure(m4)[1] == 1:6
  @test destructure(m5)[1] == vcat(1:6, 4:6)
  @test destructure(m6)[1] == vcat(1:3, 4 + im)

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

  @test destructure(m7)[1] == 1:3
  m7′ = destructure(m7)[2]([10,20,30])
  @test m7′.a == (sin, [10,20,30])
  @test m7′.b == (cos, [4,5,6])
  @test m7′.c == (tan, [7,8,9])

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
