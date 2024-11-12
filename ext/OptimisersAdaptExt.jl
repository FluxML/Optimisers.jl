module OptimisersAdaptExt

import Adapt
import Optimisers: Leaf

function Adapt.adapt_structure(to, leaf::Leaf)
  @warn """`Optimisers.Leaf` object does not support device transfer via
  `Adapt.jl`. This could lead to incorrect gradient updates. Avoid this by
  calling `Flux.gpu/cpu` or `MLDataDevices.cpu_device()/gpu_device()` on the
  optimiser state object.
  """ maxlog=1

  rule = Adapt.adapt(to, leaf.rule)
  state = Adapt.adapt(to, leaf.state)

  Leaf(rule, state, leaf.frozen)
end
	
end
