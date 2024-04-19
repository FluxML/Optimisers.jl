
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
    return fmap(x; exclude = isnumeric, walk = TrainableStructWalk()) do _
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
    return fmap(x; exclude = isnumeric, walk = TrainableStructWalk()) do _
                Δi = Δ[i+=1]
                if isnothing(Δi)
                    return nothing
                else
                    return Δi[2]
                end
           end
end


### trainables_nt ######################

"""
    trainables_nt(model) -> ps, re

Return a pair `(ps, re)` where `ps` is a nested named tuple with the same structure as 
the trainable part of `model` and with leaves the trainable parameters.

Parameters are not copied, but the returned `ps` is a view into the original model.

The `re` is a function that reconstructs a model from the parameters, 
i.e. `re(ps)` is the same as the origin `model` but with the trainable parameters replaced by `ps`.

# Examples

```jldoctest
julia> using Flux, Optimisers

julia> model = Chain(Dense(784, 32, relu), Dense(32, 10));

julia> ps, re = trainables_nt(model);

julia> ps.layers._1.weight === model.layers[1].weight
true
```

```jldoctest

julia> v = ComponentVector(ps)

julia> model2 = re(2 * v) 
Chain(
  Dense(784 => 32, relu),               # 25_120 parameters
  Dense(32 => 10),                      # 330 parameters
)                   # Total: 4 arrays, 25_450 parameters, 100.281 KiB.
```
"""
function trainables_nt(model)
    ps = _trainables_nt(model)
    re = RestructureFromNT(model)
    return ps, re
end

function _trainables_nt(x)
    walknt = TrainableNamedTupleWalk()
    ps = fmap(identity, x; exclude=isnumeric, walk=walknt, cache=nothing)
    return ps
end

function ChainRulesCore.rrule(::typeof(_trainables_nt), model)
    ps = _trainables_nt(model)
    function _trainables_nt_back(Δps)
        walk = TrainableNamedTupleBackWalk()
        Δmodel = fmap(model, Δps; exclude=isnumeric, walk, cache=nothing) do x, Δ
                    return Δ
                end
        return (NoTangent(), Δmodel)
    end
    return ps, _trainables_nt_back
end


struct TrainableNamedTupleWalk <: AbstractWalk end

function (::TrainableNamedTupleWalk)(recurse, x)
    ch = trainable(x)
    y = map(recurse, make_named_tuple(ch))
    return y
end

struct TrainableNamedTupleBackWalk <: AbstractWalk end

function (::TrainableNamedTupleBackWalk)(recurse, model, Δps)
    # @show 1 typeof(model) typeof(Δps)
    ch = trainable(model)
    Δ = unmake_named_tuple(ch, Δps)
    # @show 2 typeof(ch) typeof(Δ)
    Δ === nothing && return nothing
    Δ === ZeroTangent() && return ZeroTangent()
    y = mapvalue(recurse, ch, Δ)
    # @show 3 typeof(model) typeof(ch) typeof(Δ) typeof(y)
    return y
end


struct RestructureFromNT{T}
    x::T
end

(re::RestructureFromNT)(ps) = restructure_from_nt(re.x, ps)

function restructure_from_nt(model, ps)
    walk = RestructureFromNamedTupleWalk()
    return fmap(model, ps; exclude=isnumeric, walk, cache=nothing) do x, p
                return p
            end
end

struct RestructureFromNamedTupleWalk <: AbstractWalk end

function (::RestructureFromNamedTupleWalk)(recurse, x, nt)
    children, re = functor(x)
    newchildren = map_commons(recurse, children, nt)
    return re(newchildren)
end

function ChainRulesCore.rrule(::typeof(restructure_from_nt), x, ps)
    model = restructure_from_nt(x, ps)
    proj_ps = ProjectTo(ps)

    function restructure_from_nt_back(Δmodel_raw)
        Δmodel = unthunk(Δmodel_raw)
        walk = RestructureFromNamedTupleBackWalk()
        function exclude(x)
            @show "exclude" x isnumeric(x)
            # i += 1
            # return i > 1
            return isnumeric(x)
        end
        Δps = fmap(ps, Δmodel; exclude, walk, cache=nothing) do p, Δ
                    @show "fmap" Δ p

                    return Δ
                end
        @show "rrule" Δmodel x ps Δps
        @show  typeof(Δmodel) typeof(ps) typeof(Δps)
        Δps = (_1=ones(3), _2=zeros(3))
        Δpst = Tangent{typeof(Δps)}(; Δps...)
        # pR
        return (NoTangent(), NoTangent(), proj_ps(Δpst))
    end
    return model, restructure_from_nt_back
end

struct RestructureFromNamedTupleBackWalk <: AbstractWalk end

function (::RestructureFromNamedTupleBackWalk)(recurse, ps, Δmodel)
    @show 1 typeof(Δmodel) typeof(ps)
    Δm = make_named_tuple(Δmodel)
    @show 2 typeof(Δm) ps Δm
    # Δm isa Float64 && return Δm
    # Δm isa Array && return Δm
    # ps isa Float64 && return ps
    # ps isa Array && return ps
    # return nothing
    Δm === nothing && return nothing
    Δm === ZeroTangent() && return ZeroTangent()
    y = mapvalue(recurse, ps, Δm)
    @show 3 typeof(Δmodel) typeof(Δm) typeof(y)
    return y
end

function map_commons(f, x::NamedTuple{xkeys}, y) where {xkeys}
    ykeys = propertynames(y)
    vals = map(k -> k in ykeys ? f(x[k], getproperty(y, k)) : x[k], xkeys)
    return NamedTuple{xkeys}(vals)
end

function map_commons(f, x::Tuple, y)
    ykeys = propertynames(y)
    vals = ntuple(length(x)) do i 
                k = Symbol("_", i)
                k in ykeys ? f(x[i], getproperty(y, k)) : x[i]
            end
    return vals
end

function map_commons(f, x::Vector, y)
    ykeys = propertynames(y)
    vals = map(1:length(x)) do i 
                k = Symbol("_", i)
                k in ykeys ? f(x[i], getproperty(y, k)) : x[i]
            end
    return vals
end

make_named_tuple(x) = x
make_named_tuple(x::AbstractDict{Symbol}) = NamedTuple(x)
make_named_tuple(x::AbstractDict) = NamedTuple(Symbol("_", k) => v for (k, v) in pairs(x))
make_named_tuple(x::Tuple) = NamedTuple{ntuple(i -> Symbol("_",i), length(x))}(x)
make_named_tuple(x::Vector) = NamedTuple{ntuple(i -> Symbol("_",i), length(x))}(x)

make_named_tuple(x::Tangent{<:Any,<:NamedTuple}) = x
make_named_tuple(x::Tangent{<:Any,<:AbstractDict{Symbol}}) = NamedTuple(x)
make_named_tuple(x::Tangent{<:Any,<:AbstractDict}) = NamedTuple(Symbol("_", k) => v for (k, v) in pairs(x))
make_named_tuple(x::Tangent{<:Any,<:Tuple}) = NamedTuple{ntuple(i -> Symbol("_",i), length(x))}(x)
make_named_tuple(x::Tangent{<:Any,<:Vector}) = NamedTuple{ntuple(i -> Symbol("_",i), length(x))}(x)


unmake_named_tuple(x::NamedTuple, ps) = ps

function unmake_named_tuple(x::Tuple, ps)
    return ntuple(length(x)) do i
                ps[Symbol("_", i)]
            end
end

function unmake_named_tuple(x::Vector, ps)
    return map(1:length(x)) do i
                ps[Symbol("_", i)]
            end   
end
