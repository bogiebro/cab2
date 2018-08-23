module TaxiSearch
using Reexport
import LightGraphs; LG = LightGraphs
import Laplacians; Lap = Laplacians
using Reel
I = Iterators

@reexport using Plots, GraphPlot, Distributions,
  LinearAlgebra, SparseArrays, Random

export LG, Lap, getGraph, routeDiff, todist, hotVals,
  randPol, greedyPol, greedyPolProp, greedyF, justCycles,
  iterated, fixedPoint, scorePolicy, ptrPolicy, neighborMin,
  mkDistMat, mkP, PolSpec, uni, solo, ratios, SimStat,
  TaxiEmptyFrac, NodeSOLFrac, MaxCars, NodeWaitTimes,
  Vis, replay, competB, withRho, indivB, indivE,
  benchH, benchS, benchL, nonNan, agg, id, prToLam, c

Graph = SparseMatrixCSC{Float64,Int}

"Read a csv where each row is an edge"
function getGraph(fname)
  df = readcsv(fname, Int)
  is = df[:,2]
  js = df[:,1]
  g = sparse(is .+ 1, js .+ 1, 1.0)
  g .+ g'
end

"What fraction if the routes are the same?"
routeDiff(a, b) = norm(a - b, 0) / length(a)

"Normalize a probability distribution"
todist(x) = x ./ sum(x)

"Filter only non nan values"
nonNan(x) = x[.!isnan.(x)]

"Policy that takes the shortest path to the highest prob node"
function hotVals(g::LG.SimpleDiGraph{Int}, lam::Vector{Float64})
  m = argmax(lam)
  pol = LG.bellman_ford_shortest_paths(reverse(g), m).parents
  ns = LG.neighbors(g, m)
  n = ns[argmax(lam[ns])]
  pol[m] = n
  pol
end

# Undirected version takes floyd warshall state
function hotVals(g::LG.SimpleGraph{Int}, st, lam)
  m = argmax(lam)
  ns = LG.neighbors(g, m)
  n = ns[argmax(lam[ns])]
  pol = st[m,:]
  pol[m] = n
  pol
end

randPol(A, lam) = scorePolicy(ones(length(lam)), A)
greedyPol(A, lam) = ptrPolicy(neighborMin(A, 1 ./ lam)[2])
greedyPolProp(A, lam) = scorePolicy(lam, A)

greedyF(A, lam) = x-> scorePolicy(lam ./ (x .+ 1), A)

#ptrPolicy(neighborMin(A, collect(
  #zip(p, Float64.(x))))[2])

"Keeps only the cycles of a graph"
function justCycles(g)
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
  r * spdiagm(0=> 1 ./ reshape(sum(r;dims=1), :))
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

function neighborMin(A,v)
  w = zeros(Int,A.m)
  c = fill(Tuple(fill(Inf64, length(v[1]))), A.m)
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

c(x) = _->x

# Make a distance matrix assuming all edges out of node i
# have length p[i]
function mkDistMat(A::Graph, p::Vector{Float64})
  nzval = collect(Iterators.flatten(
    [p[vi] for ind in A.colptr[vi]:(A.colptr[vi+1]-1)] for vi in 1:A.m))
  SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, nzval)
end


# Make a rate vector
mkP(a, n) = a ./ (n .* maximum(a))

# Approximate exponential params from Bernoulli params
prToLam(p) = -log(-p + 1)

include("simulation.jl")

end # module
