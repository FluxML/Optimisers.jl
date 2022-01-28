
struct Leaf{R,S}
  rule::R
  state::S
end
function Base.show(io::IO, ℓ::Leaf)
  ioc = IOContext(io, :compact => true)
  print(io, "Leaf("); show(ioc, ℓ.rule); print(io, ", "); show(ioc, ℓ.state); print(io, ")")
end

function setup(r, x)
  if isnumeric(x)
    return Leaf(r, init(r, x))
  elseif isleaf(x)
    return nothing
  else
    x′, _ = functor(x)
    return map(xᵢ -> setup(r, xᵢ), x′)
  end
end

subtract!(x, x̄) = iswriteable(x) ? (x .= x .- x̄) : (x .- x̄)

function update!(ℓ::Leaf, x, x̄s...)
  if all(isnothing, x̄s)
    return ℓ, x
  else
    r = ℓ.rule
    s′, x̄′ = apply!(r, ℓ.state, x, x̄s...)
    return Leaf(r, s′), subtract!(x, x̄′)
  end
end

function update!(tree, x, x̄s...)
  if all(isnothing, x̄s)
    return tree, x
  else
    x̄s′ = map(x̄ -> functor(typeof(x), x̄)[1], x̄s)
    x′, re = functor(typeof(x), x)
    xtree = map((stᵢ, xᵢ, x̄sᵢ...) -> update!(stᵢ, xᵢ, x̄sᵢ...), tree, x′, x̄s′...)
    return map(first, xtree), re(map(last, xtree))
  end
end

function update(o, state, x, x̄s...)
  state′ = fmap(copy, state; exclude = iswriteable)
  x′ = fmap(copy, x; exclude = iswriteable)
  update!(o, state′, x′, x̄s...)
end

# default all rules to first order calls
apply!(o, state, x, dx, dxs...) = apply!(o, state, x, dx)

isnumeric(x::AbstractArray{<:Number}) = isleaf(x)  # isleaf to allow for e.g. transposed shared weights
isnumeric(x::AbstractArray{<:Bool}) = false  # convention of ChainRules is that Bool is non-differentiable
isnumeric(x) = false

iswriteable(::DenseArray{<:AbstractFloat}) = true  # more elaborate versions are possible, wait until needed?
iswriteable(_) = false

"""
    @.. x = x + y
    @.. x + y / z

Magic broadcasting macro, for use in `apply!` rules:
* Applied to assignment `x = ...` it is like `@.` unless `!iswriteable(x)`,
  in which case it ignores `x`, and applies `@.` on the right.
* Applied to other expressions, it broadcasts like `@.` but does not materialise,
  returning a `Broadcasted` object for later use.
"""
macro var".."(ex)
  if Meta.isexpr(ex, :(=))
    dst = esc(ex.args[1])
    src = esc(Broadcast.__dot__(ex.args[2]))
    :(if $iswriteable($dst)
        $dst .= $src
      else
        $src
      end)
  else
    bc = esc(Broadcast.__dot__(ex))
    :($lazy.($bc))
  end
end

function lazy end
Broadcast.broadcasted(::typeof(lazy), x) = Lazy(x)
struct Lazy{T}; bc::T; end
Broadcast.materialize(x::Lazy) = Broadcast.instantiate(x.bc)
