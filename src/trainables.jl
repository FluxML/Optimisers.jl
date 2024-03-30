

function trainables1(x)
    isnumeric(x) && return [x]
    arrays = AbstractArray[]
    fmap(x; exclude = isnumeric, walk = _TrainableStructWalk()) do y
        push!(arrays, y)
        return y
    end
    return arrays
end

############

using Functors: AbstractWalk, _map, _values, execute, ExcludeWalk

struct TrainableWalk2 <: AbstractWalk end

function (walk::TrainableWalk2)(recurse, x, ys...)
    x_children = _values(Optimisers.trainable(x))
    ys_children = map(Optimisers.trainable, ys)
    res = _map(recurse, x_children, ys_children...)
    @show _values(res)
    return _values(res)
end

function trainables2(x)
    exclude(x) = Optimisers.isnumeric(x) && Functors.isleaf(x)
    return execute(ExcludeWalk(TrainableWalk2(), x -> x, exclude), x)
end

using Flux

m = Chain(Dense(2 => 3, relu), BatchNorm(3), Dense(3 => 2))
trainables2(m)