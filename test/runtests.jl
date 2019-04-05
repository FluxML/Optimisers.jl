using Optimisers, Test

@testset "Optimisers" begin

rule = Optimisers.Descent(0.1)

x = [1.0]
x̄ = [1.0]

x′, state = Optimisers.update(rule, x, x̄)

@test x′ == [0.9]

Optimisers.update!(rule, x, x̄)

@test x == [0.9]

end
