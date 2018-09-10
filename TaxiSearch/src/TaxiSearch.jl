module TaxiSearch
using Reexport, DelimitedFiles
import LightGraphs; const LG = LightGraphs
import Laplacians; const Lap = Laplacians
import FunctionWrappers; const Fn = FunctionWrappers.FunctionWrapper
import PyCall
using Reel, GraphPlot
const Itr = Iterators
const T = Tuple

@reexport using PyPlot, Distributions, LinearAlgebra, SparseArrays, Random

export LG, Lap,
       randPol, greedyPol, greedyPolProp, hotPtrs, greedyF, greedyBF,
       iterated, fixedPoint, neighborMin, scorePolicy, ptrPolicy, randStep,
       routeDiff, nonNan, toDist, RoadNet, randM, justCycles,
       getGraph, mkP, toContinuous, plotG, changeLen

const Graph = SparseMatrixCSC{Float64,Int}
const Policy = Union{Graph, Matrix{Float64}}

"Read a csv where each row is an edge"
function getGraph(fname)
  df = readdlm(fname, ',')
  is = Int.(df[:,2])
  js = Int.(df[:,1])
  g = sparse(is .+ 1, js .+ 1, 1.0)
  g .+ g'
end

"Road graph with associated properties"
struct RoadNet
  g::Graph
  lg::LG.AbstractGraph{Int}
  lam::Vector{Float64}
  xs::Vector{Float64}
  ys::Vector{Float64}
  dists::Matrix{Float64}
  M::Policy
  parents::Matrix{Int}
  len::Union{Vector{Float64},Nothing}
end

function randM(n)
  M = rand(n,n)
  M -= spdiagm(0=>diag(M));
  M ./= sum(M;dims= 2);
  M
end

function RoadNet(n::Int)
  lg = LG.Grid([n,n])
  M = randM(LG.nv(lg))
  RoadNet(lg, M, nothing, Lap.grid2coords(n)...)
end

function RoadNet(lg::LG.AbstractGraph{Int}, M, len, xs, ys)
  g = copy(LG.adjacency_matrix(lg, Float64)')
  p = rand(Exponential(0.1), size(g)[1])
  RoadNet(g, lg, M, len, p, xs, ys)
end

function RoadNet(g, lg, M, len, p, xs, ys)
  dmat = len == nothing ? LG.weights(lg) : mkDistMat(g, len)
  fwState = LG.floyd_warshall_shortest_paths(lg, dmat)
  RoadNet(g, lg, p,
    xs, ys, Float64.(fwState.dists), M, fwState.parents, len)
end

RoadNet(g::Graph, M, len, p, xs, ys) = RoadNet(g, LG.DiGraph(g'), M, len, p, xs, ys)

function changeLen(net::RoadNet, l::Vector{Float64})
  dmat = mkDistMat(net.g, l)
  fwState = LG.floyd_warshall_shortest_paths(net.lg, dmat)
  RoadNet(net.g, net.lg, net.lam, net.xs, net.ys,
    Float64.(fwState.dists), net.M, fwState.parents, l)
end

"What fraction if the routes are the same?"
routeDiff(a, b) = norm(a - b, 0) / length(a)

"Normalize a probability distribution"
toDist(x) = x ./ sum(x)

"Filter only non nan values"
nonNan(x) = x[.!isnan.(x)]

"Policy that takes the shortest path to the highest prob node"
function hotPtrs(net::RoadNet)
  m = argmax(net.lam)
  ns = LG.neighbors(net.lg, m)
  n = ns[argmax(net.lam[ns])]
  pol = net.parents[m,:] # only for undirected
  pol[m] = n
  pol
end

randPol(net) = scorePolicy(ones(length(net.lam)), net.g)
greedyPol(net) = greedyPol(net, net.lam)
greedyPol(net, score) = ptrPolicy(neighborMin(net.g, 1.0 ./ score)[2])
greedyPolProp(net) = scorePolicy(net.lam, net.g)

# We actually need to know what others are doing
greedyF(net) = x::Vector{Int}-> scorePolicy(net.lam ./ (x .+ 1), net.g)
greedyBF(net) = x::Vector{Int}-> ptrPolicy(neighborMin(net.g,
  collect(zip(net.lam, 1.0 / Float64.(x .+ 1))))[2])


"Keeps only the cycles of a graph"
justCycles(g::Graph) = justCycles(LG.DiGraph(g'))
function justCycles(g::LG.AbstractGraph)
  cg = LG.DiGraph{Int}()
  LG.add_vertices!(cg, LG.nv(g))
  for c in LG.simplecycles(g)
    for i in 1:(length(c) - 1)
      LG.add_edge!(cg, (c[i], c[i+1]))
    end
    LG.add_edge!(cg, (c[end], c[1]))
  end
  cg
end

"Applies `f` `n` times to an initial vector"
function iterated(f, n::Int, initial::Vector{X})::Vector{X} where {X}
  for i in 1:n
    println("Iteration ", i)
    initial = f(initial)
  end
  initial
end

"Applies `f` to an initial vector until nothing changes"
function fixedPoint(f, initial::Vector{X})::Vector{X} where {X}
  i = 0
  while true
    newval = f(initial)
    if all(abs.(newval - initial) .< 1e-5) # isapprox(newval, initial)
      println("Converged in ", i, " steps")
      return newval
    end
    if i % 10000 == 9999
      println("Progress: ", mean(abs.(newval .- initial)))
    end
    i += 1
    initial = newval
  end
end

function scorePolicy(scores::Vector{X}, A::Graph)::Graph where {X}
  r = spdiagm(0=>scores) * A
  r * spdiagm(0=> 1.0 ./ reshape(sum(r;dims=1), :))
end

function ptrPolicy(ptrs::Vector{Int})::Graph
  roads = length(ptrs)
  sparse(ptrs, 1:roads, ones(roads), roads, roads)
end

function neighborMin(A::Graph,v::Vector{Float64})::
    Tuple{Vector{Float64},Vector{Int}}
  w = zeros(Int,A.m)
  c = fill(Inf64, A.m)
  for vi in 1:A.m
      for ind in A.colptr[vi]:(A.colptr[vi+1]-1)
          nbr = A.rowval[ind]
          newval =v[nbr]
          if newval <= c[vi]
              c[vi] = newval
              w[vi] = nbr
          end
      end
  end
  (c, w)
end

function neighborMin(A::Graph,v::Vector{NTuple{N,Float64}}) where {N}
  w = zeros(Int,A.m)
  c = fill(NTuple{N,Float64}(I.cycle(Inf64)), A.m)
  for vi in 1:A.m
      for ind in A.colptr[vi]:(A.colptr[vi+1]-1)
          nbr = A.rowval[ind]
          newval =v[nbr]
          if newval <= c[vi]
              c[vi] = newval
              w[vi] = nbr
          end
      end
  end
  (c, w)
end

"Expected waiting time for a single taxi"
function a1Min(A, lam)
  vals = fixedPoint(x->lam .+ (1 .- lam) .* (1 .+ neighborMin(A, x)[1]), 1 ./ lam)
  ptrs = neighborMin(A, vals)[2]
  pol = ptrPolicy(ptrs)
  (pol, vals, ptrs)
end

"Make a distance matrix assuming all edges out of node i have length p[i]"
function mkDistMat(A::Graph, p::Vector{Float64})
  nzval = collect(Iterators.flatten(
    [p[vi] for ind in A.colptr[vi]:(A.colptr[vi+1]-1)] for vi in 1:A.m))
  SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, nzval)
end

"Make a rate vector"
mkP(a, n) = a ./ (n .* maximum(a))

"Approximate continuous params from discrete ones"
toContinuous(p) = -log(-p + 1)

"Sample from a policy"
function randStep(m::Graph, loc::Int)::Int
  inds = m.colptr[loc]:(m.colptr[loc+1]-1)
  wsample(m.rowval[inds], m.nzval[inds])
end
@views randStep(m::Matrix{Float64}, loc::Int)::Int = wsample(m[:,loc])

plotG(net::RoadNet, args...; kwargs...) = plotG(net.lg, net, args...; kwargs...)
plotG(g::Graph, net::RoadNet, args...; kwargs...) = plotG(LG.DiGraph(g'), net, args...; kwargs...)
plotG(g::LG.DiGraph, net::RoadNet, sizes::Vector{Float64}=net.lam; kwargs...) =
  gplot(g, net.xs, net.ys; arrowlengthfrac=0.06, nodesize=sizes, kwargs...)
plotG(g::LG.Graph, net::RoadNet, sizes::Vector{Float64}=net.lam; kwargs...) =
  gplot(g, net.xs, net.ys; nodesize=sizes, kwargs...)

include("simulation.jl")
include("waitsum.jl")

end # module
