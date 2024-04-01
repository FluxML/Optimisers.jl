using BenchmarkTools
using Optimisers
using Functors
using Zygote, Flux

function trainables1(x)
    Optimisers.isnumeric(x) && return [x]
    arrays = AbstractArray[]
    exclude(x) = Optimisers.isnumeric(x) && Functors.isleaf(x)
    fmap(x; exclude, walk = Optimisers._TrainableStructWalk()) do y
        push!(arrays, y)
        return y
    end
    return arrays
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
    sum([sum(p) for p in ps])
end

using Flux

function perf()
    m = Chain(Dense(128 => 128, relu), 
              Dense(128 => 128, relu),
              BatchNorm(128), Dense(3 => 2), x -> x^2)
              Dense(128 => 128, relu), 
              Dense(128 => 128, relu)
              
    println("trainables1")
    @btime trainables1($m)
    println("trainables2")
    @btime trainables2($m)
    println("trainables3")
    @btime trainables3($m)
    println()


    # gradient(m -> floss(trainables1(m)), #m) # non differentiable since mutating
    println("gradient trainables2")
    @btime gradient(m -> floss(trainables2(m)), $m)
    println("gradient trainables3")
    @btime gradient(m -> floss(trainables3(m)), $m)
end

Zygote.refresh()
perf()