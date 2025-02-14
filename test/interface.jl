@testset "@def" begin 
    Optimisers.@def struct DummyRule
        eta = 1.0
        b1 = 1
        b2 = 2.5f0
        c = (1.0, 2.0)
    end

    # no args
    r = DummyRule()
    @test r.eta === 1.0
    @test r.b1 === 1
    @test r.b2 === 2.5f0
    @test r.c === (1.0, 2.0)

    # some positional args    
    r = DummyRule(2, 2, 4.5)
    @test r.eta === 2.0 # int promoted to float
    @test r.b1 === 2
    @test r.b2 === 4.5  # Float64 not converted to Float32
    @test r.c === (1.0, 2.0)

    # errors due to type mismatch
    @test_throws ArgumentError DummyRule(0.1+im)
    @test_throws ArgumentError DummyRule(b2=0.1+im)
    @test_throws ArgumentError DummyRule(0.1, :beta)
    @test_throws ArgumentError DummyRule(c=0.9)
    @test_throws ArgumentError DummyRule(c=(0.9, 0.99im))
    @test_throws ArgumentError Optimisers.adjust(DummyRule(), c=0.9)
end
