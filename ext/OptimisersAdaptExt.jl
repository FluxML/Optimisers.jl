module OptimisersAdaptExt

import Adapt
import Optimisers: Leaf

function Adapt.adapt_structure(to, leaf::Leaf)
	@warn """`Optimisers.Leaf` object does not support device transfer via
	`Adapt.jl`. Avoid this by calling `Flux.gpu/cpu` or
	`MLDataDevices.cpu_device()/gpu_device()` on the optimiser state object.
	See below GitHub issue for more details.
	https://github.com/FluxML/Optimisers.jl/issues/179 """ maxlog=1

	rule = Adapt.adapt_structure(to, leaf.rule)
	state = Adapt.adapt_structure(to, leaf.state)

	Leaf(rule, state, leaf.frozen)
end
	
end
