###
### Reading trainable parameters without specializing on their type
###

"""
    Optimisers.nospecialize_trainable(x) -> Union{Nothing, Functors.ChildList}

The same children as [`trainable`](@ref Optimisers.trainable)`(x)`, as a `Functors.ChildList`
whose `keys` and `values` are `Vector{Any}`s, or `nothing` when `trainable(x) === ()`.

For types that use the default `trainable`, this calls `Functors.nospecialize_children`, which
compiles nothing for types that use the default `functor`, nor for `Tuple`s, `NamedTuple`s and most arrays.
Other types, such as those with their own `trainable`, `functor` or `children` methods
(including those from `@functor`), get these methods called, which compiles them as usual.
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

This does not compile an `fmap` for each type in `x`. It compiles nothing for nodes whose types use
the default `functor`, `trainable`, `isleaf` and `isnumeric` methods, nor for `Tuple`s, `NamedTuple`s
and most arrays. Other nodes, such as array wrappers like `transpose` which have their own `functor`,
get their own methods called, which compiles them as usual.
It is meant for code that only reads a model, such as counting its parameters when printing it,
and each call is slower than `trainables`.
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
  k = _kinds(typeof(x))
  if _isnumeric(k, x)
    push!(arrays, x)
    return
  end
  ch = k.walk ? Functors.nospecialize_children(x) : _childlist(_trainable(x))
  ch === nothing && return
  for c in values(ch)
    _nospecialize_trainables!(arrays, seen, c)
  end
  return
end

# Which nodes the cache of `fmap` remembers, `isleaf(x) ? anymutable(x) : ismutable(x)`,
# see `Functors.usecache`. Mutable `x` are also `anymutable`, so they need no `isleaf`.
Base.@nospecializeinfer function _usecache(@nospecialize(x))::Bool
  return ismutable(x) || (Functors.nospecialize_isleaf(x) && _anymutable(x))
end

Base.@nospecializeinfer function _anymutable(@nospecialize(x))::Bool
  T = typeof(x)
  ismutabletype(T) && return true
  isbitstype(T) && return false
  for i in 1:fieldcount(T)
    fieldtype(T, i) === T && continue
    isdefined(x, i) && _anymutable(getfield(x, i)) && return true
  end
  return false
end

# Calling `isnumeric(x)` from here costs a slow dynamic dispatch, and compiles it for each array type.
Base.@nospecializeinfer function _isnumeric(k, @nospecialize(x))::Bool
  k.numeric == _NOT_NUMERIC && return false
  k.numeric == _NUMERIC_IF_LEAF && return Functors.nospecialize_isleaf(x)
  return isnumeric(x)
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
@enum _NumericKind::UInt8 begin
  _NOT_NUMERIC     # isnumeric(x) == false
  _NUMERIC_IF_LEAF # isnumeric(x) == isleaf(x)
  _CALL_ISNUMERIC  # anything else: call isnumeric(x)
end

struct _Kinds
  numeric::_NumericKind
  trainable::Bool  # trainable(x) == children(x)
  walk::Bool       # and _trainable(x) also has the values of children(x), in the same order
end

const _DEFAULT_TRAINABLE = which(trainable, Tuple{Any})
const _DEFAULT_CHILDREN = which(children, Tuple{Any})
const _DEFAULT_FUNCTOR1 = which(functor, Tuple{Any})
const _NUMERIC_KINDS = IdDict{Any, _NumericKind}(
  which(isnumeric, Tuple{Any}) => _NOT_NUMERIC,
  which(isnumeric, Tuple{Vector{Int}}) => _NOT_NUMERIC,
  which(isnumeric, Tuple{Vector{Float64}}) => _NUMERIC_IF_LEAF,
)

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
  numeric = get(_NUMERIC_KINDS, _which(isnumeric, Tuple{T}), _CALL_ISNUMERIC)
  trainable_default = _which(trainable, Tuple{T}) === _DEFAULT_TRAINABLE &&
                      _which(children, Tuple{T}) === _DEFAULT_CHILDREN
  trainable_default || return _Kinds(numeric, false, false)
  # Functors' own `functor` methods return `x` itself, its fields, a NamedTuple or a Vector,
  # for which `_trainable(x) == trainable(x)`. Not dictionaries, which `merge` may reorder.
  f2 = _which(functor, Tuple{Type{T}, T})
  walk = !(T <: AbstractDict) && _which(functor, Tuple{T}) === _DEFAULT_FUNCTOR1 &&
         f2 !== nothing && f2.module === Functors
  return _Kinds(numeric, true, walk)
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
