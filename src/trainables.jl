
"""
    trainables(x, path = false)

Return an iterable over all the trainable parameters in `x`, that is all the numerical
arrays (see [`isnumeric`](@ref Optimisers.isnumeric)) which are reachable through [`trainable`](@ref Optimisers.trainable).

Parameters appearing multiple times in the model (tied weights) will be present only once in the output.

If `path = false`, the output is a list of numerical arrays.

If `path = true`, the output is a list of `(KeyPath, AbstractArray)` pairs, where [`KeyPath`](@ref) is a type
representing the path to the array in the original structure.

See also [`destructure`](@ref) for a similar operation that returns a single flat vector instead.

# Examples

```jldoctest
julia> struct MyLayer
         w
         b
       end

julia> Functors.@functor MyLayer

julia> Optimisers.trainable(x::MyLayer) = (; w = x.w,) # only w is trainable in this example

julia> x = MyLayer([1.0,2.0,3.0], [4.0,5.0,6.0]);

julia> trainables(x)
1-element Vector{AbstractArray}:
 [1.0, 2.0, 3.0]

julia> x = MyLayer((a=[1.0,2.0], b=[3.0]), [4.0,5.0,6.0]);

julia> trainables(x) # collects nested parameters
2-element Vector{AbstractArray}:
 [1.0, 2.0]
 [3.0]
```

```jldoctest
julia> x = (a = [1.0,2.0], b = (Dict("c" => [3.0, 4.0], "d" => 5.0), [6.0,7.0]));

julia> for (kp, y) in trainables(x, path = true)
           println(kp, " => ", y)
       end
KeyPath(:a,) => [1.0, 2.0]
KeyPath(:b, 1, "c") => [3.0, 4.0]
KeyPath(:b, 2) => [6.0, 7.0]

julia> getkeypath(x, KeyPath(:b, 1, "c"))
2-element Vector{Float64}:
 3.0
 4.0
```
"""
function trainables(x; path = false)
    if path
        return _trainables_with_path(x)
    else
        return _trainables(x)
    end
end


function _trainables(x)
    arrays = AbstractArray[]
    fmap(x; exclude = isnumeric, walk = TrainableStructWalk()) do y
        push!(arrays, y)
        return y
    end
    return arrays
end

function ∇trainables(x, Δ)
    i = 0
    return fmapstructure(x; exclude = isnumeric, walk = TrainableStructWalk()) do _
                return Δ[i+=1]
           end
end

function ChainRulesCore.rrule(::typeof(_trainables), x)
    y = trainables(x)
    trainables_back(Δ) = (NoTangent(), ∇trainables(x, unthunk(Δ)))
    return y, trainables_back
end

function _trainables_with_path(x)
    named_params = []
    exclude(kp, x) = isnumeric(x)
    fmap_with_path(x; exclude, walk = TrainableStructWalkWithPath()) do kp, y
        push!(named_params, (kp, y))
        return y
    end
    return named_params
end

struct TrainableStructWalkWithPath <: AbstractWalk end

function (::TrainableStructWalkWithPath)(recurse, kp::KeyPath, x)
    x_children = trainable(x)
    kps = mapkey(c -> KeyPath(kp, c), x_children)
    return mapvalue(recurse, kps, x_children)
end

function ChainRulesCore.rrule(::typeof(_trainables_with_path), x)
    y = _trainables_with_path(x)
    trainables_with_path_back(Δ) = (NoTangent(), ∇trainables_with_path(x, unthunk(Δ)))
    return y, trainables_with_path_back
end

function ∇trainables_with_path(x, Δ)
    i = 0
    return fmapstructure(x; exclude = isnumeric, walk = TrainableStructWalk()) do _
                Δi = Δ[i+=1]
                if isnothing(Δi)
                    return nothing
                else
                    return Δi[2]
                end
           end
end
