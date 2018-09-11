export uni, solo, ratios,
       maxCars, nodeSOLFrac, taxiEmptyFrac, nodeWaitTimes,
       indivB, indivE, competP, withNTaxis, withTaxiDensity,
       bench, S, P, F, id, oneEach, PolFn

# Manipulating heterogenous policies

"Policies taken by each taxi"
struct PolSpec{X}
  pols::Vector{X}
  offsets::Vector{Int}
end

"PolSpec constructors, given nTaxis"
struct MkPolSpec{X}
  f::Fn{PolSpec{X}, T{Int}}
end

"A policy that depends on rho"
const PolFn = Fn{Int,T{Vector{Int}, Int}}

c(x::Graph) = PolFn((_,loc)-> randStep(x, loc))

"Assume policies are constant and uniform unless otherwise specified"
MkPolSpec{X}(p::MkPolSpec{X}) where {X} = p
MkPolSpec{Graph}(p::Graph) = uni(p)
MkPolSpec{PolFn}(p::Graph) = uni(c(p))
MkPolSpec{PolFn}(p::PolFn) = uni(p)
MkPolSpec{PolFn}(p::MkPolSpec{Graph}) = MkPolSpec{PolFn}(x-> begin
  polSpec = p.f(x)
  PolSpec(map(c, polSpec.pols), polSpec.offsets)
end)

"Get the policy index for a given taxi"
@inline polIdx(p::PolSpec, x::Int)::Int = findfirst(y->x<=y, p.offsets) - 1

"Get the policy for a given taxi"
@inline polMat(p::PolSpec, x::Int) = p.pols[polIdx(p, x)]

"Everyone follows uniform policy f"
uni(f::X) where {X} = MkPolSpec{X}(n-> PolSpec([f], [0, n]))

"One guy outsmarts everyone"
solo(f::X, g::X) where {X} = MkPolSpec{X}(n-> PolSpec([f, g], [0, 1, n]))

"Assign policies from given weights"
ratios(fs::Vector{X}, ws::Vector{Float64}) where {X} =
  MkPolSpec(n-> PolSpec(fs, cumsum(collect(I.flatten((0, ws)))) * n)) 


# Statistics collection

"Stats that must be collected separately for each policy"
abstract type PolStat end
updateFound!(s::PolStat, t::Int) = nothing
updateSearch!(s::PolStat, t::Int, l::Int, w::Float64) = nothing

"Stats that must be collected globally for all policies"
abstract type GblStat end
updateGenerated!(s::GblStat, p::Vector{Int}) = nothing
updateLeftover!(s::GblStat, p::Vector{Int}) = nothing
updateRho!(s::GblStat, i::Int, p::Vector{Int}) = nothing

"Construct a stat from (nTaxis, limit, net)"
struct MkStat{X} f::Fn{X, T{Int,Int,RoadNet}} end

"Generalized 1d statistics type"
struct SimVec{X,Y}
  pol::Vector{X}
  gbl::Vector{X}
end

"Generalized nd statistics type"
struct SimArray{X,Y}
  pol::Array{X}
  gbl::Array{X}
end

const SimStat = SimVec{Vector{PolStat}, GblStat}
const S = SimVec{MkStat{PolStat}, MkStat{GblStat}}
const P = SimArray{Symbol, Symbol}
const F = SimArray{Function,Function}
const SimRes = SimVec{Vector{Pair{DataType,Any}}, Pair{DataType,Any}}

"Construct a SimStat from an S and necessary params"
function initStats(nTaxis::Int, limit::Int, net::RoadNet, p::PolSpec, mkStats::S)
  Random.seed!(1234)
  SimStat([[s.f(o2 - o1, limit, net) for s in mkStats.pol]
    for (o1,o2) in zip(p.offsets, Itr.rest(p.offsets, 2))],
    [s.f(nTaxis, limit, net) for s in mkStats.gbl])
end

"Construct a SimRes from a SimStat"
finalizeStats(stats, limit) =
  SimRes([[typeof(s)=>finalize(s, limit) for s in sp] for sp in stats.pol],
    [typeof(s)=>finalize(s, limit) for s in stats.gbl])

"Add to the statistics about a given taxi"
function runPolStats(f, x::Int, p::PolSpec, stats::SimStat, args...)
  ix = polIdx(p,x)
  for s in stats.pol[ix] f(s, x - p.offsets[ix], args...) end
end

"Add to global statistics"
function runGblStats(f, stats::SimStat, args...)
  for s in stats.gbl f(s, args...) end
end

"Tracks the empty time of each taxi"
struct TaxiEmptyFrac <: PolStat emptyTime::Vector{Float64} end

const taxiEmptyFrac = MkStat{PolStat}((nTaxis::Int, _::Int, net::RoadNet)->
  TaxiEmptyFrac(zeros(nTaxis)))
updateSearch!(s::TaxiEmptyFrac, t::Int, l::Int, w::Float64) = s.emptyTime[t] += w;
finalize(s::TaxiEmptyFrac, steps) = s.emptyTime ./ steps
xlabel(::Type{TaxiEmptyFrac}) = "taxi"
ylabel(::Type{TaxiEmptyFrac}) = "fraction of time empty"

"Records the wait times of each node"
struct NodeWaitTimes <: PolStat
  waitTimes::Vector{Vector{Float64}}
  routes::Vector{Vector{Pair{Int,Int}}}
end

const nodeWaitTimes = MkStat{PolStat}((nTaxis::Int, _::Int, net::RoadNet)-> NodeWaitTimes(
   [Int[] for _ in 1:length(net.lam)], [Pair{Int,Int}[] for _ in 1:nTaxis]))
updateFound!(s::NodeWaitTimes, t::Int) = updatePath!(s.routes[t], s.waitTimes)
updateSearch!(s::NodeWaitTimes, t::Int, l::Int, w::Float64) = push!(s.routes[t], l=>w);
finalize(s::NodeWaitTimes, steps) = s.waitTimes
xlabel(::Type{NodeWaitTimes}) = "node"
ylabel(::Type{NodeWaitTimes}) = "waiting time"

"Tracks the fraction of time each node generates an unserved passenger"
struct NodeSOLFrac <: GblStat
  sol::Vector{Int}
  generated::Vector{Int}
end

const nodeSOLFrac = MkStat{GblStat}((_::Int, _::Int, net::RoadNet)->
  NodeSOLFrac(zeros(Int, length(net.lam)), zeros(Int, length(net.lam))))
updateGenerated!(s::NodeSOLFrac, p::Vector{Int}) = s.generated .+= p;
updateLeftover!(s::NodeSOLFrac, p::Vector{Int}) = s.sol .+= p;
finalize(s::NodeSOLFrac, steps) = nonNan(s.sol ./ s.generated)
xlabel(::Type{NodeSOLFrac}) = "node"
ylabel(::Type{NodeSOLFrac}) = "fraction of time without taxi"

"Tracks the highest taxi density at each timestep"
struct MaxCars <: GblStat cars::Vector{Int} end

const maxCars = MkStat{GblStat}((_::Int, lim::Int, net::RoadNet)-> MaxCars(zeros(Int, lim)))
updateRho!(s::MaxCars, i::Int, rho::Vector{Int}) = s.cars[i] = maximum(rho);
finalize(s::MaxCars, steps) = s.cars
xlabel(::Type{MaxCars}) = "time"
ylabel(::Type{MaxCars}) = "highest taxi density"

#=
"Creates a visualization of the simulation"
struct Vis <: GblStat
  frames::Reel.Frames{MIME{Symbol("image/png")}}
  f::Function
end

Reel.extension(::MIME{Symbol("text/html")}) = "html"
Reel.set_output_type("gif") 

maker(::Type{Vis}) = MkGblStat((net::RoadNet, steps::Int)->
  Vis(Frames(MIME("image/png"), fps=1), rho->
    gplot(net.lg, net.xs, net.ys, nodelabel=rho, nodesize=lam)))
updateRho!(s::Vis, i::Int, rho::Vector{Int}) = push!(frames, s.f(rho))
finalize(s::Vis) = s.frames
=#


# Utility functions

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

function sampleDest(M, dists, ix)
  dest = randStep(M, ix)
  dest, dists[ix, dest]
end

function updatePath!(path, timeDist)
  nr = sum(p[2] for p in path)
  for (r,w) in path
    push!(timeDist[r], nr)
    nr -= w
  end
  empty!(path);
end


# Simulation functions

function competP(locs::Vector{Int}, net::RoadNet, mkStats, pn, limit::Int=1000)
  poissons = Poisson.(net.lam)
  nLocs = length(net.lam)
  nTaxis = length(locs)
  pf = MkPolSpec{PolFn}(pn).f(nTaxis)
  timers = zeros(Int, nTaxis)
  stats = initStats(nTaxis, limit, net, pf, mkStats)
  rho = locsToRho(locs, nLocs)
  for i in 1:limit
    passengers = [rand(p) for p in poissons]
    runGblStats(updateGenerated!, stats, passengers)
    runGblStats(updateRho!, stats, i, rho) 
    for taxi in randperm(nTaxis)
      if timers[taxi] == 0
        locs[taxi] = abs(locs[taxi])
        rho[locs[taxi]] -= 1
        if passengers[locs[taxi]] > 0
          passengers[locs[taxi]] -= 1
          runPolStats(updateSearch!, taxi, pf, stats, locs[taxi], 1.0)
          runPolStats(updateFound!, taxi, pf, stats)
          dest, dist = sampleDest(net.M, net.dists, locs[taxi])
          locs[taxi] = -dest
          timers[taxi] = dist
        else
          runPolStats(updateSearch!, taxi, pf, stats, locs[taxi], 1.0)
          locs[taxi] = polMat(pf, taxi)(rho, locs[taxi])
          rho[locs[taxi]] += 1
        end
      else timers[taxi] -= 1 end
    end
    runGblStats(updateLeftover!, stats, passengers)
  end
  finalizeStats(stats, limit)
end

"Start with a given average taxi density per node"
withTaxiDensity(density, f) = (net::RoadNet, args...)-> begin
  rho0 = trunc.(Int, min.(25.0, randexp(length(net.lam)) .* density))
  f(rhoToLocs(rho0), net, args...)
end

"Start with a taxi on each node"
oneEach(f) = (net::RoadNet, args...)->
  f(collect(1:length(net.lam)), net, args...)

"Start with a given number of taxis total, distributed randomly"
withNTaxis(nTaxis, f) = (net::RoadNet, args...)->
  f(rand(1:length(net.lam), nTaxis), net, args...)

"Agents don't interact, probabilities are Bernoulli" 
function indivB(locs::Vector{Int}, net::RoadNet, mkStats, pn, limit::Int=1000)
  nTaxis = length(locs)
  p = MkPolSpec{Graph}(pn).f(nTaxis)
  nLocs = length(net.lam)
  stats = initStats(nTaxis, limit, net, p, mkStats)
  for i in 1:nTaxis
    loc = locs[i]
    t = 0
    while t < limit
      t += 1
      runPolStats(updateSearch!, i, p, stats, loc, 1.0)
      if rand() < net.lam[loc]
        runPolStats(updateFound!, i, p, stats)
        loc, dist = sampleDest(net.M, net.dists, loc)
        t += Int(dist)
      else
        loc = randStep(polMat(p, i), loc)
      end
    end
  end
  finalizeStats(stats, limit)
end

"Agents don't interact, probabilities are exponential"
function indivE(locs::Vector{Int}, net::RoadNet, mkStats, pn, limit::Int=1000)
  nLocs = length(net.lam)
  nTaxis = length(locs)
  p = MkPolSpec{Graph}(pn).f(nTaxis)
  stats = initStats(nTaxis, limit, net, p, mkStats)
  exps = Exponential.(1.0 ./ net.lam)
  for i in 1:nTaxis
    loc = rand(1:nLocs)
    t = 0.0
    while t < limit
      a = min(rand(exps[loc]), limit - t)
      if a < net.len[loc]
        runPolStats(updateSearch!, i, p, stats, loc, a)
        runPolStats(updateFound!, i, p, stats)
        loc, dist = sampleDest(net.M, net.dists, loc)
        t += (a + dist)
      else
        runPolStats(updateSearch!, i, p, stats, loc, net.len[loc])
        t += net.len[loc]
        loc = randStep(polMat(p, i), loc)
      end
    end
  end
  finalizeStats(stats, limit)
end


# Benchmarking functions

"Start a plot"
function mkPlot(ty, stat, nrows, ncols, i)
  a = plt[:subplot](nrows, ncols, i)
  if ty == :hist
    plt[:xlabel](ylabel(stat))
  else
    plt[:xlabel](xlabel(stat))
    plt[:ylabel](ylabel(stat))
  end
  a
end

"Add a series to a plot"
function addPlot!(ty, h, k, val::Vector{Float64})
 if ty == :hist
   h[:hist](val, alpha=0.2, density=true, label=k, ec="gray")
 elseif ty == :scatter
   h[:plot](val, "o", alpha=0.3, label=k)
 elseif ty == :line
   h[:plot](val, alpha=0.3, label=k)
 end
end

"Plot the result of a specific policy"
function plotPol(hs, plts, fns, res, str)
  if !isempty(res)
    fnRes = map.(fns, getindex.(res, 2))
    for (i, r) in enumerate(fnRes)
      println(str, "stat $i => $(mean(r))")
    end
    tuples = tuple.(fnRes, getindex.(res, 1), plts, hs)
    n = length(tuples)
    nrow = ncol = 1
    if n > 1
      ncol = 2
      nrow = div(n, 2)
    end
    h2s = []
    for (i,(r, stat, plt, h)) in enumerate(tuples)
      h2 = (h == nothing ? mkPlot(plt, stat, nrow, ncol, i) : h)
      push!(h2s, h2)
      addPlot!(plt, h2, str, r)
    end
    return h2s
  end
  hs
end

id(x) = x

"Run a simulation with different policies and plot them"
function bench(f, net::RoadNet, stats::S, plts::P, fns::F, args...; pols...)
  hs = nothing
  for (k,p) in pols
    res = f(net, stats, p, args...)
    hs = plotPol(hs, plts.gbl, fns.gbl, res.gbl, "$k gbl ")
    for (i, pRes) in enumerate(res.pol)
      hs = plotPol(hs, plts.pol, fns.pol, pRes, "$k pol $i ")
    end
  end
  flush(stdout)
  for h in hs h[:legend]() end
  plt[:tight_layout]()
  plt[:show]();
end

