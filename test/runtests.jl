using Optimisers, Test
using Zygote, Random
using Statistics

@testset "Optimisers" begin
  Random.seed!(84)
  w′ = rand(3,3)
  @testset for o in (Descent(0.1),
                     Momentum(0.01, 0.9),
                     Nesterov(0.001, 0.9),
                     RMSProp(0.001, 0.9),
                     ADAM(0.001, (0.9, 0.99)),
                     RADAM(0.001, (0.9, 0.999)),
                     AdaMax(0.001, (0.9, 0.999)),
                     OADAM(0.001, (0.5, 0.9)),
                     ADAGrad(0.1),
                     ADADelta(0.9),
                     AMSGrad(0.001, (0.9, 0.999)),
                     NADAM(0.001, (0.9, 0.999)),
                     AdaBelief(0.001, (0.9, 0.999)),
                     )
    w = rand(3,3)
    st = Optimisers.init(o,w)
    loss(x, y) = mean((x .- y) .^ 2)
    l = loss(w, w′)
    for i = 1:10^4
      gs = gradient(x -> loss(x,w′), w)
      # w, st = o(w, gs..., st)
      w, st = Optimisers.update(o, w, gs..., st)
    end
    @test loss(w, w′) < 0.01
  end

end
