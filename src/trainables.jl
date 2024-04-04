
"""
    trainables(x)

Return a list over all the trainable parameters in `x`, that is all the numerical
arrays (see [`isnumeric`](@ref Optimisers.isnumeric)) which are reachable through [`trainable`](@ref Optimisers.trainable).

Parameters appearing multiple times in the model (tied weights) will be present only once in the output.

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
"""
function trainables(x)
    arrays = AbstractArray[]
    exclude(x) = Optimisers.isnumeric(x)
    fmap(x; exclude, walk = Optimisers.TrainableStructWalk()) do y
        push!(arrays, y)
        return y
    end
    return arrays
end

function ∇trainables(x, Δ)
    exclude(x) = Optimisers.isnumeric(x)
    i = 0
    return fmapstructure(x; exclude, walk = TrainableStructWalk()) do _
                return Δ[i+=1]
           end
end

function ChainRulesCore.rrule(::typeof(trainables), x)
    y = trainables(x)
    trainables_back(Δ) = (NoTangent(), ∇trainables(x, unthunk(Δ)))
    return y, trainables_back
end
