###
### Reading trainable parameters without specializing on their type
###

"""
    Optimisers.nospecialize_trainable(x) -> Union{Nothing, Functors.ChildList}

The same children as [`trainable`](@ref Optimisers.trainable)`(x)`, as a `Functors.ChildList`
whose `keys` and `values` are `Vector{Any}`s, or `nothing` when `trainable(x) === ()`.

This is compiled once for all types that use the default `trainable`, for which it calls
`Functors.nospecialize_children`. Types with their own `trainable` method get it called.
It is meant for code that only reads a model, such as printing it, and each call is slower than `trainable`.

# Example
```jldoctest
julia> struct Layer; w; b; end

julia> Optimisers.trainable(l::Layer) = (; w = l.w)

julia> c = Optimisers.nospecialize_trainable(Layer([1.0, 2.0], [3.0]));

julia> keys(c), values(c)
(Any[:w], Any[[1.0, 2.0]])
```
"""
Base.@nospecializeinfer function nospecialize_trainable(@nospecialize(x))
  _kinds(typeof(x)).trainable && return Functors.nospecialize_children(x)
  return _childlist(trainable(x))
end

"""
    Optimisers.nospecialize_trainables(x) -> Vector{AbstractArray}

The same arrays as [`trainables`](@ref Optimisers.trainables)`(x)`, in the same order.

This does not compile an `fmap` for each type in `x`. Nodes using the default `functor` and
`trainable` methods compile nothing, and those with their own methods get them called.
It is meant for code that only reads a model, such as counting its parameters when printing it.
There is no gradient rule for it, so use `trainables` inside a loss function.
"""
Base.@nospecializeinfer function nospecialize_trainables(@nospecialize(x))
  arrays = AbstractArray[]
  _nospecialize_trainables!(arrays, Base.IdSet{Any}(), x)
  return arrays
end

# The walk of `trainables`: `fmap` with `exclude = isnumeric`, `TrainableStructWalk` and the cache.
Base.@nospecializeinfer function _nospecialize_trainables!(arrays::Vector{AbstractArray}, seen::Base.IdSet{Any}, @nospecialize(x))
  if _usecache(x)
    x in seen && return
    push!(seen, x)
  end
  if isnumeric(x)
    push!(arrays, x)
  elseif _kinds(typeof(x)).walk
    ch = Functors.nospecialize_children(x)
    if ch !== nothing
      for c in values(ch)
        _nospecialize_trainables!(arrays, seen, c)
      end
    end
  else
    tr = _trainable(x)
    if tr isa Union{Tuple, NamedTuple}
      for i in 1:nfields(tr)
        _nospecialize_trainables!(arrays, seen, getfield(tr, i))
      end
    else
      for c in values(tr)
        _nospecialize_trainables!(arrays, seen, c)
      end
    end
  end
  return
end

# Which nodes the cache of `fmap` remembers, see `Functors.usecache` and `Functors.anymutable`:
Base.@nospecializeinfer function _usecache(@nospecialize(x))::Bool
  return Functors.nospecialize_isleaf(x) ? _anymutable(x) : ismutable(x)
end

Base.@nospecializeinfer function _anymutable(@nospecialize(x))::Bool
  T = typeof(x)
  ismutabletype(T) && return true
  for i in 1:fieldcount(T)
    fieldtype(T, i) == T && continue
    isdefined(x, i) && _anymutable(getfield(x, i)) && return true
  end
  return false
end

Base.@nospecializeinfer function _childlist(@nospecialize(tr))
  tr === () && return nothing
  if tr isa Union{Tuple, NamedTuple}
    n = nfields(tr)
    ks = Vector{Any}(undef, n)
    vs = Vector{Any}(undef, n)
    for i in 1:n
      ks[i] = tr isa NamedTuple ? fieldname(typeof(tr), i) : i
      vs[i] = getfield(tr, i)
    end
    return Functors.ChildList(ks, vs)
  else
    return Functors.ChildList(vec(collect(Any, keys(tr))), vec(collect(Any, values(tr))))
  end
end

# Calling `trainable(x)` would compile `functor(::Type{T}, ::T)` for each new `T`. So `which`
# tells us when the default methods apply, and the result is cached for each `T`.
struct _Kinds
  trainable::Bool  # trainable(x) == children(x)
  walk::Bool       # and _trainable(x) also has the values of children(x), in the same order
end

const _DEFAULT_TRAINABLE = which(trainable, Tuple{Any})
const _DEFAULT_CHILDREN = which(children, Tuple{Any})
const _DEFAULT_FUNCTOR1 = which(functor, Tuple{Any})

const _KINDS = IdDict{Any, _Kinds}()
const _KINDS_WORLD = Ref(typemax(UInt))
const _KINDS_LOCK = ReentrantLock()

# Forgets everything whenever the world age changes, as new methods may have been defined.
Base.@nospecializeinfer function _kinds(@nospecialize(T::Type))
  world = Base.get_world_counter()
  lock(_KINDS_LOCK)
  try
    if _KINDS_WORLD[] != world
      empty!(_KINDS)
      _KINDS_WORLD[] = world
    end
    k = get(_KINDS, T, nothing)
    if k === nothing
      k = _KINDS[T] = _find_kinds(T)
    end
    return k::_Kinds
  finally
    unlock(_KINDS_LOCK)
  end
end

Base.@nospecializeinfer function _find_kinds(@nospecialize(T::Type))
  trainable_default = _which(trainable, Tuple{T}) === _DEFAULT_TRAINABLE &&
                      _which(children, Tuple{T}) === _DEFAULT_CHILDREN
  trainable_default || return _Kinds(false, false)
  # Functors' own `functor` methods return `x` itself, its fields, a NamedTuple or a Vector,
  # for which `_trainable(x) == trainable(x)`. Not dictionaries, which `merge` may reorder.
  f2 = _which(functor, Tuple{Type{T}, T})
  walk = !(T <: AbstractDict) && _which(functor, Tuple{T}) === _DEFAULT_FUNCTOR1 &&
         f2 !== nothing && f2.module === Functors
  return _Kinds(true, walk)
end

# `nothing` when there is no unique method, e.g. an ambiguity. Then calling the function reports the error.
Base.@nospecializeinfer function _which(@nospecialize(f), @nospecialize(tt::Type))
  try
    return which(f, tt)
  catch err
    err isa ErrorException || err isa MethodError || rethrow()
    return nothing
  end
end
