using Distributions, GraphPlot, Compose, Colors, MAT, Reel
import StatsBase: sample
import Plots: histogram, histogram!, plot, plot!
import LightGraphs; LG = LightGraphs
import Laplacians; Lap = Laplacians
import JLD: @save, @load
Plots.gr()

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

"Policy that takes the shortest path to the highest prob node"
function hotVals(g::LG.SimpleDiGraph{Int}, lam::Vector{Float64})
  m = indmax(lam)
  pol = LG.bellman_ford_shortest_paths(reverse(g), m).parents
  ns = LG.neighbors(g, m)
  n = ns[indmax(lam[ns])]
  pol[m] = n
  pol
end

# Undirected version takes floyd warshall state
function hotVals(g::LG.SimpleGraph{Int}, st, lam)
  m = indmax(lam)
  ns = LG.neighbors(g, m)
  n = ns[indmax(lam[ns])]
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
function iterated{X}(f, n::Int, initial::Vector{X})::Vector{X}
  for i in 1:n
    println("Iteration ", i)
    initial = f(initial)
  end
  initial
end

"Applies `f` to an initial vector until nothing changes"
function fixedPoint{X}(f, initial::Vector{X})::Vector{X}
  i = 0
  while true
    newval = f(initial)
    if all(abs.(newval - initial) .< 1e-5) # isapprox(newval, initial)
      println("Converged in ", i, " steps")
      return newval
    end
    if i % 1000 == 999
      println("Progress: ", mean(abs.(newval .- initial)))
    end
    i += 1
    initial = newval
  end
end

function scorePolicy{X}(scores::Vector{X}, A::Graph)::Graph
  r = spdiagm(scores) * A
  r * spdiagm(1./ reshape(sum(r, 1), :))
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

function sampleDest(M, dists, ix)
  dest = wsample(M[:,ix])
  dest, dists[ix, dest]
end

function updatePath!(path, timeDist)
  nr = length(path)
  for (ri, r) in enumerate(path)
    push!(timeDist[r], nr - ri + 1)
  end
  empty!(path)
end

function indivBDist(lam, M, dists, pol; limit::Int=800, n::Int=1000)
  srand(1234)
  timeDist = [Int[] for _ in 1:(size(M)[1])]
  for i in 1:n
    loc = rand(1:length(lam))
    t = 0
    path = [loc]
    while t < limit
      t += 1
      if rand() < lam[loc]
        updatePath!(path, timeDist)
        loc, dist = sampleDest(M, dists, loc)
        t += Int(dist)
      else
        loc = wsample(full(pol[:, loc]))
        push!(path, loc)
      end
    end
  end
  timeDist
end

function indivB(lam, M, dists, pol; limit::Int=800, n::Int=1000)
  srand(1234)
  waitTimes = zeros(n)
  Threads.@threads for i in 1:n
    loc = rand(1:length(lam))
    waitTime = 0
    t = 0
    while t < limit
      waitTime += 1
      t += 1
      if rand() < lam[loc]
        loc, dist = sampleDest(M, dists, loc)
        t += Int(dist)
      else
        loc = wsample(full(pol[:, loc]))
      end
    end
    waitTimes[i] = waitTime / t
  end
  waitTimes
end

function indivE(lam, M, dists, len, pol; limit::Float64=800.0, n::Int=1000)
  srand(1234)
  waitTimes = zeros(n)
  exps = Exponential.(1 ./ lam)
  Threads.@threads for i in 1:n
    loc = rand(1:length(lam))
    waitTime = 0.0
    t = 0.0
    while t < limit
      a = rand(exps[loc])
      if a <= len[loc]
        waitTime += a
        t += a
        loc, dist = sampleDest(M, dists, loc)
        t += dist
      else
        waitTime += len[loc]
        t += len[loc]
        loc = wsample(full(pol[:, loc]))
      end
    end
    waitTimes[i] = waitTime / t
  end
  waitTimes
end

function bench(f, args...; pols...)
  h = histogram(normalize=true, xlabel="fraction of time empty")
  for (k,p) in pols
    result = f(args..., p) 
    println(k, " ", mean(result))
    histogram!(h, result, alpha=0.2, label=k, normalize=true)
  end
  flush(STDOUT)
  h
end

function benchS(f,lab, args...; pols...)
  h = plot(line=:scatter, xlabel=lab)
  for (k,p) in pols
    result = f(args..., p) 
    println(k, " ", mean(result))
    plot!(h, result, line=:scatter, alpha=0.3, label=k)
  end
  flush(STDOUT)
  h
end

function benchL(f, args...; pols...)
  h = plot(xlabel="time")
  for (k,p) in pols
    result = f(args..., p) 
    println(k, " ", mean(result))
    plot!(h, result, alpha=0.5, label=k)
  end
  flush(STDOUT)
  h
end

fst(x::Tuple) = x[1]
fst(x) = x

nonNan(x) = x[!isnan.(x)]
waitWith(f, distFn) = (x...)-> nonNan(f.(fst(distFn(x...))))
solWith(distFn) = (x...)-> nonNan(distFn(x...)[2])
concentrations(distFn) = (x...)-> distFn(x...)[3]

function rhoToLocs(rho)
  locs = zeros(Int, sum(rho))
  n = 1
  for (i,r) in enumerate(rho)
    for _ in 1:r
      locs[n] = i
      n+=1
    end
  end
  locs
end

function locsToRho(locs, nLocs)::Vector{Int}
  rho = zeros(Int, nLocs)
  for l in locs
    if l > 0 rho[l] += 1 end
  end
  rho
end

function competBDist(lam, M, dists, density, p; steps::Int=1000)
  srand(1234)
  rho0 = trunc.(Int, min.(25.0, randexp(size(M)[1]) .* density))
  poissons = Poisson.(lam)
  nLocs = length(rho0)
  locs = rhoToLocs(rho0)
  n = length(locs)
  routes = [Int[] for _ in 1:n]
  concentrations = zeros(Int, steps)
  waitDists = [Int[] for _ in 1:nLocs]
  sol = zeros(Int, nLocs)
  generated = zeros(Int, nLocs)
  timers = zeros(Int, n)
  for i in 1:steps
    passengers = rand.(poissons)
    generated .+= passengers
    rho = locsToRho(locs, nLocs)
    concentrations[i] = maximum(rho)
    pol = p(rho)
    for taxi in randperm(n)
      if timers[taxi] == 0
        locs[taxi] = abs(locs[taxi])
        if passengers[locs[taxi]] > 0
          passengers[locs[taxi]] -= 1
          updatePath!(routes[taxi], waitDists)
          dest, dist = sampleDest(M, dists, locs[taxi])
          locs[taxi] = -dest
          timers[taxi] = dist
        else
          locs[taxi] = wsample(full(pol[:, locs[taxi]]))
          push!(routes[taxi], locs[taxi])
        end
      else timers[taxi] -= 1 end
    end
    sol .+= passengers
  end
  (waitDists, sol ./ generated, concentrations)
end

function competB(lam, M, dists, density, p)
  srand(1234)
  rho0 = trunc.(Int, min.(25.0, randexp(size(M)[1]) .* density))
  competBCore(lam, M, dists, p, rho0)
end

function competBCore(lam, M, dists, p, rho0;
    chan=Nullable{Channel{Vector{Int}}}(), steps::Int=1000)
  poissons = Poisson.(lam)
  nLocs = length(rho0)
  locs = rhoToLocs(rho0)
  n = length(locs)
  waitingTimes = zeros(Int, n)
  timers = zeros(Int, n)
  for i in 1:steps
    passengers = rand.(poissons)
    rho = locsToRho(locs, nLocs)
    if chan.hasvalue put!(chan.value, rho) end
    pol = p(rho)
    for taxi in randperm(n)
      if timers[taxi] == 0
        locs[taxi] = abs(locs[taxi])
        waitingTimes[taxi] += 1
        if passengers[locs[taxi]] > 0
          passengers[locs[taxi]] -= 1
          dest, dist = sampleDest(M, dists, locs[taxi])
          locs[taxi] = -dest
          timers[taxi] = dist
        else
          locs[taxi] = wsample(full(pol[:, locs[taxi]]))
        end
      else timers[taxi] -= 1 end
    end
  end
  if chan.hasvalue close(chan.value) end
  waitingTimes ./ steps
end

Reel.extension(::MIME{Symbol("text/html")}) = "html"
Reel.set_output_type("gif") 
function replay(g, xs, ys, M, lam, dists, pol, rho0)
  chan = Channel{Vector{Int}}(0);
  @schedule competBCore(lam, M, dists, pol, rho0; chan=Nullable(chan), steps=10)
  roll([gplot(g, xs, ys, nodelabel=rho, nodesize=lam) for rho in chan],
    fps=1)
end
