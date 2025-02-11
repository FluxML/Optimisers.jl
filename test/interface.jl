@testset "@def" begin 
    Optimisers.@def struct DummyRule
        a = 1
        b1 = 1.5
        b2 = 2.5f0
        c = (1.0, 2.0)
    end

    # no args
    r = DummyRule()
    @test typeof(r.a) == Int
    @test typeof(r.b1) == Float64
    @test typeof(r.b2) == Float32
    @test typeof(r.c) == Tuple{Float64, Float64}
    
    # some positional args    
    r = DummyRule(2, 2, 4.5)
    @test r.a == 2
    @test r.b1 == 2
    @test r.b2 == 4.5
    @test typeof(r.b1) == Float64 # int promoted to float
    @test typeof(r.b2) == Float64 # Float64 not converted to Float32
    @test r.c == (1.0, 2.0)
end