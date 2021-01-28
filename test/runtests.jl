using Optimisers, Test
using Zygote, Random
using Statistics

@testset "Optimisers" begin
  Random.seed!(84)
  w′ = (α = rand(3, 3), β = rand(3, 3))
  @testset for o in (Descent(), Momentum(), Nesterov(), RMSProp(), ADAM())
    w = (α = rand(3, 3), β = rand(3, 3))
    st = init(o, w)
    loss(x, y) = mean((x.α .* x.β .- y.α .* y.β) .^ 2)
    l = loss(w, w′)
    for i = 1:10^4
      gs = gradient(x -> loss(x, w′), w)
      w, st = o(w, gs..., st)
    end
    @test loss(w, w′) < 0.01
  end

end
