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

@testset "Leaf show" begin
    lim(x) = sprint(show, x; context = :limit => true)
    ℓ = Optimisers.Leaf(Adam(), (collect(1f0:100f0), zeros(Float32, 5, 4), (0.9f0, 0.99f0)))

    # without :limit (print, repr, logging), nothing is cut
    @test occursin("100.0", repr(ℓ))
    @test count(==(','), repr(ℓ)) > 100

    # with :limit (REPL display), arrays keep a few entries, and brackets stay balanced
    str = lim(ℓ)
    @test occursin("Float32[1.0, 2.0, 3.0, …]", str)
    @test occursin("Float32[0.0, 0.0, 0.0, …]", str)
    @test occursin("(0.9, 0.99)", str)
    @test length(str) < length(repr(ℓ)) ÷ 2
    @test count(==('('), str) == count(==(')'), str)
    @test count(==('['), str) == count(==(']'), str)

    # short states, `nothing`, rule chains, and frozen leaves
    small = Optimisers.setup(Momentum(), [1.0, 2.0, 3.0])
    @test lim(small) == repr(small)
    @test lim(Optimisers.setup(Descent(), [1.0, 2.0, 3.0, 4.0])) == repr(Optimisers.setup(Descent(), [1.0, 2.0, 3.0, 4.0]))
    chain = Optimisers.setup(OptimiserChain(ClipGrad(), Adam()), ones(Float32, 10))
    @test occursin("(nothing, (Float32[0.0, 0.0, 0.0, …]", lim(chain))
    frozen = Optimisers.setup(Adam(), ones(4)); Optimisers.freeze!(frozen)
    @test endswith(lim(frozen), ", frozen = true)")

    # inside a model state
    st = Optimisers.setup(Adam(), (a = ones(100), b = (ones(2), ones(2))))
    @test count("Leaf(", lim(st)) == 3
    @test length(lim(st)) < length(repr(st))
end
