export uni, solo, ratios,
       MaxCars, NodeSOLFrac, TaxiEmptyFrac, NodeWaitTimes,
       indivB, indivE, competB, withNTaxis, withTaxiDensity,
       bench, Agg

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
const PolFn = Fn{Graph,T{Vector{Int}}}

c(x::Graph) = PolFn(_->x)

"Assume policies are constant and uniform unless otherwise specified"
MkPolSpec{X}(p::MkPolSpec{X}) where {X} = p
MkPolSpec{Graph}(p::Graph) = uni(p)
MkPolSpec{PolFn}(p::Graph) = uni(c(p))
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

"Turn a rho-dependent polspec into an independent one"
applySpec(p::PolSpec{PolFn}, rho::Vector{Int})::PolSpec{Graph} =
  PolSpec([f(rho) for f in p.pols], p.offsets)
applySpec(p::PolSpec{Graph}, rho::Vector{Int})::PolSpec{Graph} = p


# Statistics collection

"Statistics collected during simulation for a single policy"
abstract type PolStat end
updateFound!(s::PolStat, t::Int) = nothing
updateSearch!(s::PolStat, t::Int, l::Int, w::Float64) = nothing

"Overall statistics collected during simulation"
abstract type GblStat end
updateGenerated!(s::GblStat, p::Vector{Int}) = nothing
updateLeftover!(s::GblStat, p::Vector{Int}) = nothing
updateRho!(s::GblStat, i::Int, p::Vector{Int}) = nothing

"Construct a stat from (nTaxis, net)"
struct MkPolStat f::Fn{PolStat, T{Int,RoadNet}} end

"Construct a stat from (net, steps)"
struct MkGblStat f::Fn{GblStat, T{RoadNet,Int}} end

mutable struct MkSimStats
  pol::Vector{MkPolStat}
  gbl::Vector{MkGblStat}
end

addStat!(ps::MkSimStats, f::MkPolStat) = push!(ps.pol, f);
addStat!(ps::MkSimStats, f::MkGblStat) = push!(ps.gbl, f);

MkSimStats(f::DataType) = MkSimStats([f])
function MkSimStats(fs::Vector{DataType})
  stats = MkSimStats([],[])
  for f in fs addStat!(stats, maker(f)); end
  stats
end


struct SimStats
  pol::Vector{Vector{PolStat}}
  gbl::Vector{GblStat}
end

struct SimRes
  pol::Vector{Vector{Any}}
  gbl::Vector{Any}
end

function runPolStat(f, x::Int, p::PolSpec, stats::SimStats, args...)
  ix = polIdx(p,x)
  for s in stats.pol[ix] f(s, x - p.offsets[ix], args...) end
end

function runGblStat(f, stats::SimStats, args...)
  for s in stats.gbl f(s, args...) end
end

"Tracks the empty time of each taxi"
struct TaxiEmptyFrac <: PolStat emptyTime::Vector{Float64} end

# there are 799 following one policy, and one following the other
# we're not doing the updateSearch thing right
# we initialize with the correct number of taxis
# but then, when people use an id, they must subtract the previous offset (or 0)
# Might be easiest to always make the first offset 0

maker(::Type{TaxiEmptyFrac}) = MkPolStat((nTaxis::Int, net::RoadNet)-> TaxiEmptyFrac(zeros(nTaxis)))
updateSearch!(s::TaxiEmptyFrac, t::Int, l::Int, w::Float64) = s.emptyTime[t] += w;
finalize(s::TaxiEmptyFrac, steps) = s.emptyTime ./ steps
xlabel(::Type{TaxiEmptyFrac}) = "taxi"
ylabel(::Type{TaxiEmptyFrac}) = "fraction of time empty"

"Records the wait times of each node"
struct NodeWaitTimes <: PolStat
  waitTimes::Vector{Vector{Float64}}
  routes::Vector{Vector{Pair{Int,Int}}}
end

maker(::Type{NodeWaitTimes}) = MkPolStat((nTaxis::Int, net::RoadNet)-> NodeWaitTimes(
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

maker(::Type{NodeSOLFrac}) = MkGblStat((net::RoadNet, steps::Int)->
  NodeSOLFrac(zeros(Int, length(net.lam)), zeros(Int, length(net.lam))))
updateGenerated!(s::NodeSOLFrac, p::Vector{Int}) = s.generated .+= p;
updateLeftover!(s::NodeSOLFrac, p::Vector{Int}) = s.sol .+= p;
finalize(s::NodeSOLFrac) = nonNan(s.sol ./ s.generated)
xlabel(::Type{NodeSOLFrac}) = "node"
ylabel(::Type{NodeSOLFrac}) = "fraction of time without taxi"

"Tracks the highest taxi density at each timestep"
struct MaxCars <: GblStat cars::Vector{Int} end

maker(::Type{MaxCars}) = MkGblStat((net::RoadNet, steps::Int)-> MaxCars(zeros(Int, steps)))
updateRho!(s::MaxCars, i::Int, rho::Vector{Int}) = s.cars[i] = maximum(rho);
finalize(s::MaxCars) = s.cars
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

function initStats(nTaxis::Int, net::RoadNet, limit::Int, p, stats)
  mkStats = MkSimStats(stats)
  SimStats([[s.f(o2 - o1, net) for s in mkStats.pol]
    for (o1,o2) in zip(p.offsets, Itr.rest(p.offsets, 2))],
    [s.f(net, limit) for s in mkStats.gbl])
end

finalizeStats(stats, limit) =
  SimRes([[finalize(s, limit) for s in sp] for sp in stats.pol],
    [finalize(s) for s in stats.gbl])


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

function competB(locs::Vector{Int}, net::RoadNet, mkStats, pn, limit::Int=1000)
  poissons = Poisson.(net.lam)
  nLocs = length(net.lam)
  nTaxis = length(locs)
  pf = MkPolSpec{PolFn}(pn).f(nTaxis)
  timers = zeros(Int, nTaxis)
  stats = initStats(nTaxis, net, limit, pf, mkStats)
  for i in 1:limit
    passengers = rand.(poissons)
    runGblStat(updateGenerated!, stats, passengers)
    rho = locsToRho(locs, nLocs)
    runGblStat(updateRho!, stats, i, rho) 
    p = applySpec(pf, rho)
    for taxi in randperm(nTaxis)
      if timers[taxi] == 0
        locs[taxi] = abs(locs[taxi])
        if passengers[locs[taxi]] > 0
          passengers[locs[taxi]] -= 1
          runPolStat(updateSearch!, taxi, p, stats, locs[taxi], 1.0)
          runPolStat(updateFound!, taxi, p, stats)
          dest, dist = sampleDest(net.M, net.dists, locs[taxi])
          locs[taxi] = -dest
          timers[taxi] = dist
        else
          runPolStat(updateSearch!, taxi, p, stats, locs[taxi], 1.0)
          locs[taxi] = randStep(polMat(p, taxi), locs[taxi])
        end
      else timers[taxi] -= 1 end
    end
    runGblStat(updateLeftover!, stats, passengers)
  end
  finalizeStats(stats, limit)
end

"Start with a given average taxi density per node"
withTaxiDensity(density, f) = (net::RoadNet, args...)-> begin
  rho0 = trunc.(Int, min.(25.0, randexp(length(net.lam)) .* density))
  f(rhoToLocs(rho0), net, args...)
end

"Start with a given number of taxis total, distributed randomly"
withNTaxis(nTaxis, f) = (net::RoadNet, args...)->
  f(rand(1:length(net.lam), nTaxis), net, args...)

"Agents don't interact, probabilities are Bernoulli" 
function indivB(locs::Vector{Int}, net::RoadNet, mkStats, pn, limit::Int=1000)
  nTaxis = length(locs)
  p = MkPolSpec{Graph}(pn).f(nTaxis)
  nLocs = length(net.lam)
  stats = initStats(nTaxis, net, limit, p, mkStats)
  for i in 1:nTaxis
    loc = locs[i]
    t = 0
    while t < limit
      t += 1
      if rand() < net.lam[loc]
        runPolStat(updateSearch!, i, p, stats, loc, 1.0)
        runPolStat(updateFound!, i, p, stats)
        loc, dist = sampleDest(net.M, net.dists, loc)
        t += Int(dist)
      else
        runPolStat(updateSearch!, i, p, stats, loc, 1.0)
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
  stats = initStats(nTaxis, net, limit, p, mkStats)
  exps = Exponential.(1.0 ./ net.lam)
  for i in 1:nTaxis
    loc = rand(1:nLocs)
    t = 0.0
    while t < limit
      a = min(rand(exps[loc]), limit - t)
      if a < net.len[loc]
        runPolStat(updateSearch!, i, p, stats, loc, a)
        runPolStat(updateFound!, i, p, stats)
        loc, dist = sampleDest(net.M, net.dists, loc)
        t += (a + dist)
      else
        runPolStat(updateSearch!, i, p, stats, loc, net.len[loc])
        t += net.len[loc]
        loc = randStep(polMat(p, i), loc)
      end
    end
  end
  finalizeStats(stats, limit)
end


# Benchmarking functions

"Start a plot"
function mkPlot(ty, stat)
  if ty == :hist
    histogram(xlabel=ylabel(stat))
  elseif ty == :scatter
    plot(line=:scatter, xlabel=xlabel(stat), ylabel=ylabel(stat)) 
  elseif ty == :line
    plot(xlabel=xlabel(stat), ylabel=ylabel(stat)) 
  end
end

"Add a series to a plot"
function addPlot!(ty, h, k, val)
 @assert !isempty(val)
 if ty == :hist
   if length(val) == 1
     histogram!(h, val, alpha=0.2, normalize=true, label=k, nbins=1)
   else
     histogram!(h, val, alpha=0.2, normalize=true, label=k)
   end
 elseif ty == :scatter
   plot!(h, val, alpha=0.3, line=:scatter, label=k)
 elseif ty == :line
   plot!(h, val, alpha=0.3, label=k)
 end
end

struct Agg
  f::Function
  s::Any
end
Base.broadcastable(a::Agg) = Ref(a)

unagg(x) = x
unagg(a::Agg) = a.s
unagg(v::Vector) = [unagg(a) for a in v]

agg(x, y) = y
agg(a::Agg, y) = nonNan(collect(map(a.f, y)))

# This is doing the work twice! Do better!

"Plot the result of a specific policy"
function plotPol(hs, plts, stats, res, str)
  if !isempty(res)
    tuples = tuple.(res, plts, stats, hs)
    h2s = []
    for (i,(svec, plt, stat, h)) in enumerate(tuples)
      r = agg(stat, svec)
      h2 = (h == nothing ? mkPlot(plt, unagg(stat)) : h)
      println(str, "stat $i ", mean(r), " of ", length(r))
      push!(h2s, addPlot!(plt, h2, str, r))
    end
    return h2s
  end
  hs
end

"Run a simulation with different policies and plot them"
function bench(f, plts, net, stats, args...; pols...)
  hs = nothing
  for (k,p) in pols
    Random.seed!(1234)
    res = f(net, unagg(stats), p, args...)
    flush(stdout)
    hs = plotPol(hs, plts, stats, res.gbl, "$k gbl ")
    for (i, pRes) in enumerate(res.pol)
      hs = plotPol(hs, plts, stats, pRes, "$k pol $i ")
    end
  end
  flush(stdout)
  plot(hs...)
end

