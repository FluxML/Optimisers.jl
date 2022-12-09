using Optimisers
using Zygote

loss(model) = sum(abs2, model[:a])


d = Dict(:a => [1.0,2.0], :b => [3.0,4.0], :c => 1)
opt = Optimisers.setup(AdamW(0.1), d)
@assert opt isa Dict{Symbol, <:Any}
@assert opt[:a] isa Optimisers.Leaf
# @assert opt[:b] isa Optimisers.Leaf
# @assert opt[:c] === ()
g = gradient(loss, d)[1]
opt, d = Optimisers.update(opt, d, g) # still broken

loss(model) = sum(abs2, model[:a])

nt = (a = [1.0,2.0], b = [3.0,4.0], c = 1)
gradient(loss, nt)[1]

d = Dict(:a => [1.0,2.0], :b => [3.0,4.0], :c => 1)
gradient(loss, d)[1]

