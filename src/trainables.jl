using BenchmarkTools
using Optimisers
using Functors
using Zygote, Flux
using ChainRulesCore

function trainables1(x)
    arrays = AbstractArray[]
    exclude(x) = Optimisers.isnumeric(x)
    fmap(x; exclude, walk = Optimisers._TrainableStructWalk()) do y
        push!(arrays, y)
        return y
    end
    return arrays
end

function ∇trainables1(x, Δ)
    exclude(x) = Optimisers.isnumeric(x)
    i = 0
    return fmapstructure(x; exclude, walk = Optimisers._TrainableStructWalk()) do _
                return Δ[i+=1]
           end
end


function ChainRulesCore.rrule(::typeof(trainables1), x)
    y = trainables1(x)
    trainables_back(Δ) = (NoTangent(), ∇trainables1(x, unthunk(Δ)))
    return y, trainables_back
end

############

using Functors: AbstractWalk, _map, _values, execute, ExcludeWalk

struct TrainableWalk2 <: AbstractWalk end

function (walk::TrainableWalk2)(recurse, x, ys...)
    x_children = Optimisers.trainable(x)
    ys_children = map(Optimisers.trainable, ys)
    res = map(recurse, x_children, ys_children...)
    return reduce(vcat, values(res),init=[])
end

function trainables2(x)
    exclude(x) = Optimisers.isnumeric(x) && Functors.isleaf(x)
    return execute(ExcludeWalk(TrainableWalk2(), x ->[x], exclude), x)
end


struct TrainableWalk3 <: AbstractWalk end

function (walk::TrainableWalk3)(recurse, x, ys...)
    x_children = Optimisers.trainable(x)
    ys_children = map(Optimisers.trainable, ys)
    res = map(recurse, x_children, ys_children...)
    return vcat(values(res)...)
end

function trainables3(x)
    exclude(x) = Optimisers.isnumeric(x)
    return execute(ExcludeWalk(TrainableWalk3(), x ->[x], exclude), x)
end


function floss(ps)
    sum([sum(abs2, p) for p in ps])
end

using Flux

function perf()
    m = Chain(Dense(128 => 128, relu), 
              Dense(128 => 128, relu),
              BatchNorm(128),
              x -> x^2,
              Dense(128 => 128, relu), 
              Dense(128 => 128, relu))
              
    println("trainables1")
    @btime floss(trainables1($m))
    println("trainables2")
    @btime floss(trainables2($m))
    println("trainables3")
    @btime floss(trainables3($m))
    println()

    println("gradient trainables1")
    @btime gradient(m -> floss(trainables1(m)), $m)
    println("gradient trainables2")
    @btime gradient(m -> floss(trainables2(m)), $m)
    println("gradient trainables3")
    @btime gradient(m -> floss(trainables3(m)), $m)

    nothing
end

Zygote.refresh()
perf()


m = Chain(Dense(128 => 128, relu), 
              Dense(128 => 128, relu),
              BatchNorm(128),
              x -> x^2,
              Dense(128 => 128, relu), 
              Dense(128 => 128, relu))
              
floss(trainables1(m))
g1 = gradient(m -> floss(trainables1(m)), m)[1]
g2 = gradient(m -> floss(trainables2(m)), m)[1]
@test g1.layers[1].weight ≈ g2.layers[1].weight
@test g1.layers[1].weight ≈ g2.layers[1].weight
@test g1.layers[3].μ === nothing
@test g2.layers[3].μ === nothing
