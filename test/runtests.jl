using Optimisers, Test
using Zygote, Random
using Statistics

@testset "Optimisers" begin
  Random.seed!(84)
  w′ = rand(3,3)
  @testset for o in (Descent(0.1), Momentum(0.01, 0.9), Nesterov(0.001, 0.9), RMSProp(0.001, 0.9),
                     ADAM(0.001, (0.9, 0.99)))
    w = rand(3,3)
    st = Optimisers.init(o,w)
    loss(x, y) = mean((x .- y) .^ 2)
    l = loss(w, w′)
    for i = 1:10^4
      gs = gradient(x -> loss(x,w′), w)
      w, st = o(w, gs..., st)
    end
    @test loss(w, w′) < 0.01
  end

end
