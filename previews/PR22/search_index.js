var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"Optimisers.Descent\nOptimisers.Momentum\nOptimisers.Nesterov\nOptimisers.RMSProp\nOptimisers.ADAM\nOptimisers.RADAM\nOptimisers.AdaMax\nOptimisers.OADAM\nOptimisers.ADAGrad\nOptimisers.ADADelta\nOptimisers.AMSGrad\nOptimisers.NADAM\nOptimisers.ADAMW\nOptimisers.AdaBelief\nOptimisers.weightDecay\nOptimisers.OptimiserChain","category":"page"},{"location":"api/#Optimisers.Descent","page":"API","title":"Optimisers.Descent","text":"Descent(η = 1f-1)\n\nClassic gradient descent optimiser with learning rate η. For each parameter p and its gradient dp, this runs p -= η*dp.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.Momentum","page":"API","title":"Optimisers.Momentum","text":"Momentum(η = 1f-2, ρ = 9f-1)\n\nGradient descent optimizer with learning rate η and momentum ρ.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nMomentum (ρ): Controls the acceleration of gradient descent in the                 prominent direction, in effect dampening oscillations.\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.Nesterov","page":"API","title":"Optimisers.Nesterov","text":"Nesterov(η = 1f-3, ρ = 9f-1)\n\nGradient descent optimizer with learning rate η and Nesterov momentum ρ.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nNesterov momentum (ρ): Controls the acceleration of gradient descent in the                          prominent direction, in effect dampening oscillations.\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.RMSProp","page":"API","title":"Optimisers.RMSProp","text":"RMSProp(η = 1f-3, ρ = 9f-1, ϵ = eps(typeof(η)))\n\nOptimizer using the RMSProp algorithm. Often a good choice for recurrent networks. Parameters other than learning rate generally don't need tuning.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nMomentum (ρ): Controls the acceleration of gradient descent in the                 prominent direction, in effect dampening oscillations.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.ADAM","page":"API","title":"Optimisers.ADAM","text":"ADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))\n\nADAM optimiser.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.RADAM","page":"API","title":"Optimisers.RADAM","text":"RADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))\n\nRectified ADAM optimizer.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.AdaMax","page":"API","title":"Optimisers.AdaMax","text":"AdaMax(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))\n\nAdaMax is a variant of ADAM based on the ∞-norm.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.OADAM","page":"API","title":"Optimisers.OADAM","text":"OADAM(η = 1f-3, β = (5f-1, 9f-1), ϵ = eps(typeof(η)))\n\nOADAM (Optimistic ADAM) is a variant of ADAM adding an \"optimistic\" term suitable for adversarial training.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.ADAGrad","page":"API","title":"Optimisers.ADAGrad","text":"ADAGrad(η = 1f-1, ϵ = eps(typeof(η)))\n\nADAGrad optimizer. It has parameter specific learning rates based on how frequently it is updated. Parameters don't need tuning.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.ADADelta","page":"API","title":"Optimisers.ADADelta","text":"ADADelta(ρ = 9f-1, ϵ = eps(typeof(ρ)))\n\nADADelta is a version of ADAGrad adapting its learning rate based on a window of past gradient updates. Parameters don't need tuning.\n\nParameters\n\nRho (ρ): Factor by which the gradient is decayed at each time step.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.AMSGrad","page":"API","title":"Optimisers.AMSGrad","text":"AMSGrad(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))\n\nThe AMSGrad version of the ADAM optimiser. Parameters don't need tuning.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.NADAM","page":"API","title":"Optimisers.NADAM","text":"NADAM(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))\n\nNADAM is a Nesterov variant of ADAM. Parameters don't need tuning.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.ADAMW","page":"API","title":"Optimisers.ADAMW","text":"ADAMW(η = 1f-3, β = (9f-1, 9.99f-1), γ = 0, ϵ = eps(typeof(η)))\n\nADAMW is a variant of ADAM fixing (as in repairing) its weight decay regularization.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nWeight decay (γ): Decay applied to weights during optimisation.\nMachine epsilon (ϵ): Constant to prevent division by zero                        (no need to change default)\n\n\n\n\n\n","category":"function"},{"location":"api/#Optimisers.AdaBelief","page":"API","title":"Optimisers.AdaBelief","text":"AdaBelief(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = eps(typeof(η)))\n\nThe AdaBelief optimiser is a variant of the well-known ADAM optimiser.\n\nParameters\n\nLearning rate (η): Amount by which gradients are discounted before updating                      the weights.\nDecay of momentums (β::Tuple): Exponential decay for the first (β1) and the                                  second (β2) momentum estimate.\nMachine epsilon (ϵ::Float32): Constant to prevent division by zero                                 (no need to change default)\n\n\n\n\n\n","category":"type"},{"location":"api/#Optimisers.OptimiserChain","page":"API","title":"Optimisers.OptimiserChain","text":"OptimiserChain(opts...)\n\nCompose a chain (sequence) of optimisers so that each opt in opts updates the gradient in the order specified.\n\n\n\n\n\n","category":"type"},{"location":"#Optimisers.jl","page":"Home","title":"Optimisers.jl","text":"","category":"section"}]
}
