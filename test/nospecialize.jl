using Optimisers: nospecialize_trainable, nospecialize_trainables, trainable
using Logging: with_logger, NullLogger

struct NSPlain; a; b; end
mutable struct NSMutable; a; b; end
struct NSEmpty end
struct NSSubset; a; b; c; end
Functors.@functor NSSubset (a, c)
struct NSTrainable; a; b; end
Optimisers.trainable(x::NSTrainable) = (; a = x.a)
struct NSOldTuple; a; b; end
Optimisers.trainable(x::NSOldTuple) = (x.a,)
struct NSChildren; a; b; end
Functors.children(x::NSChildren) = (x.a,)
struct NSIsLeaf; a; end
Functors.isleaf(::NSIsLeaf) = true
struct NSLeaf; a; end
Functors.@leaf NSLeaf
struct NSArray <: AbstractVector{Float64}; parent::Vector{Float64}; end
Base.size(x::NSArray) = size(x.parent)
Base.getindex(x::NSArray, i::Int) = x.parent[i]
Optimisers.isnumeric(::NSArray) = false
struct NSDictLike; d; end
Functors.functor(::Type{<:NSDictLike}, x) = x.d, y -> NSDictLike(y)
struct NSLater; a; b; end
struct NSFresh{T}; a::T; b; end

function nospecialize_agrees(x)
  ps, ref = nospecialize_trainables(x), trainables(x)
  length(ps) == length(ref) && all(map(===, ps, ref)) || return false
  c, tr = nospecialize_trainable(x), trainable(x)
  tr === () && return c === nothing
  return c isa Functors.ChildList &&
         isequal(keys(c), vec(collect(Any, keys(tr)))) &&
         all(map(===, values(c), vec(collect(Any, values(tr)))))
end

@testset "nospecialize_trainables agrees with trainables" begin
  W, v = [1.0 2.0; 3.0 4.0], [5.0, 6.0]
  lay = NSMutable(W, v)
  dict = Dict(i => [Float64(i)] for i in 1:40)  # merge in _trainable may reorder it
  objs = Any[
    W, v, 1.0, "s", nothing, sin, Int, (), (;), [1, 2], [1.0 + im], 1:3.0,
    (W, v), (a = W, b = v, c = sin), (a = W, b = W), [W, v, W], reshape(Any[1, v, W, 2], 2, 2),
    (a = W, b = transpose(W), c = W', d = PermutedDimsArray(W, (2, 1))),
    (a = 1:3.0, b = 1:3.0), (a = SA[1.0, 2.0], b = SA[1.0, 2.0]), (a = view(W, :, 1), b = view(W, :, 1)),
    Dict("c" => [3.0, 4.0], "d" => 5.0, "e" => W), dict, NSDictLike(dict),
    Set([[1.0], [2.0]]), Base.Fix1(*, W), enumerate([W, v]), x -> W * x,
    NSPlain(W, v), NSPlain(sin, (W, v)), lay, (lay, lay), NSPlain(lay, NSPlain(lay, v)),
    (NSPlain(W, v), NSPlain(W, v)), NSEmpty(), NSSubset(W, v, [7.0]), NSTrainable(W, v),
    NSOldTuple(W, v), NSChildren(W, v), NSIsLeaf(W), (NSIsLeaf(W), NSIsLeaf(W)), NSLeaf(W),
    NSArray([1.0]), Foo(W, v), TwoThirds(W, v, [7.0]), MutTwo(W, v),
    (a = (NSPlain(W, v), true), b = [lay, Dict(:k => lay)], c = NSTrainable(lay, W)),
  ]
  disagree = with_logger(NullLogger()) do  # NSOldTuple warns
    filter(!nospecialize_agrees, objs)
  end
  @test disagree == []
end

@testset "nospecialize_trainables notices methods defined after a first call" begin
  x = NSLater([1.0], [2.0])
  @test keys(nospecialize_trainable(x)) == [:a, :b]
  @test length(nospecialize_trainables(x)) == 2
  @eval Optimisers.trainable(x::NSLater) = (; b = x.b)
  @test keys(Base.invokelatest(nospecialize_trainable, x)) == [:b]
  @test Base.invokelatest(nospecialize_trainables, x) == [[2.0]]
end

@testset "nospecialize_trainables compiles nothing for a type with the default methods" begin
  x = (NSFresh([1.0], NSFresh(:a, [2.0])), [3.0])
  @test nospecialize_trainables(x) == [[1.0], [2.0], [3.0]]
  @test keys(nospecialize_trainable(x[1])) == [:a, :b]
  # `which` may create a MethodInstance, but without compiled code:
  compiled_for_NSFresh(f) = any(methods(f)) do m
    any(Base.specializations(m)) do mi
      mi !== nothing && isdefined(mi, :cache) && occursin("NSFresh", string(mi.specTypes))
    end
  end
  for f in (Optimisers.trainable, Optimisers._trainable, Optimisers.isnumeric, Functors.functor,
            Functors.children, Functors.isleaf, Functors.anymutable)
    @test !compiled_for_NSFresh(f)
  end
end
