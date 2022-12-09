using Optimisers
using Zygote

# d = Dict(:a => [1.0,2.0], :b => [3.0,4.0], :c => 1)
d = (a = [1.0,2.0], b = [3.0,4.0], c = 1)

s = Optimisers.setup(AdamW(0.1), d)
# s isa Dict{Symbol, <:Any}
# s[:a] isa Optimisers.Leaf
# s[:b] isa Optimisers.Leaf
# s[:c] === ()
loss(model) = sum(abs2, model[:a])
g = gradient(loss, d)[1]
g = (; a = [2.0, 4.0])
# s2, d2 = Optimisers.update(s, d, g)
Optimisers.update(s, d, g) # still broken

loss(model) = sum(abs2, model[:a])

nt = (a = [1.0,2.0], b = [3.0,4.0], c = 1)
gradient(loss, nt)[1]

d = Dict(:a => [1.0,2.0], :b => [3.0,4.0], :c => 1)
gradient(loss, d)[1]

