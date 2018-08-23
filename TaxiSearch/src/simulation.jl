# Statistics collection

"Policies taken by each taxi"
struct PolSpec{X}
  pols::Vector{X}
  offsets::Vector{Int}
end

"Get the policy index for a given taxi"
@inline polIdx(p,x) = findfirst(y->x<=y, p.offsets)

"Get the policy for a given taxi"
@inline polMat(p::PolSpec, x) = p.pols[polIdx(p, x)]

"Everyone follows uniform policy f"
uni(f) = n-> PolSpec([f], [n])

"One guy outsmarts everyone"
solo(f, g) = n-> PolSpec([f, g], [1, n])

"Assign policies from given weights"
ratios(fs, ws) = n-> PolSpec(fs, cumsum(ws) * n)

"Turn a rho-dependent polspec into an independent one"
applySpec(p::PolSpec{Function}, rho::Vector{Int})::PolSpec{Graph} =
  PolSpec([f(rho) for f in p.pols], p.offsets)

"Statistics collected during simulation for a single policy"
abstract type SimPolStat end
updateFound!(s::SimPolStat, t::Int) = nothing
updateSearch!(s::SimPolStat, t::Int, l::Int, w::Float64) = nothing

"Overall statistics collected during simulation"
abstract type SimGblStat end
updateGenerated!(s::SimGblStat, p::Vector{Int}) = nothing
updateLeftover!(s::SimGblStat, p::Vector{Int}) = nothing
updateRho!(s::SimGblStat, i::Int, p::Vector{Int}) = nothing

@inline function runPolStat(f, x, p, stats, args...)
  for s in stats f(s[polIdx(p,x)], x, args...) end
end

@inline function runGblStat(f, stats, args...)
  for s in stats f(s, args...) end
end

"Tracks the empty time of each taxi"
struct TaxiEmptyFrac <: SimPolStat
  emptyTime::Vector{Float64}
end

TaxiEmptyFrac(nTaxis::Int, nLocs::Int) = TaxiEmptyFrac(zeros(nTaxis))

updateSearch!(s::TaxiEmptyFrac, t::Int, l::Int, w::Float64) = s.emptyTime[t] += w
finalize(s::TaxiEmptyFrac, steps) =  s.emptyTime ./ steps
xlabel(::Type{TaxiEmptyFrac}) = "taxi"
ylabel(::Type{TaxiEmptyFrac}) = "fraction of time empty"

"Records the wait times of each node"
struct NodeWaitTimes <: SimPolStat
  waitTimes::Vector{Vector{Float64}}
  routes::Vector{Vector{Pair{Int,Int}}}
end

NodeWaitTimes(nTaxis::Int, nLocs::Int) = NodeWaitTimes(
   [Int[] for _ in 1:nLocs], [Pair{Int,Int}[] for _ in 1:nTaxis])
updateFound!(s::NodeWaitTimes, t::Int) = updatePath!(s.routes[t], s.waitTimes)
updateSearch!(s::NodeWaitTimes, t::Int, l::Int, w::Float64) = push!(s.routes[t], l=>w)
finalize(s::NodeWaitTimes, steps) = s.waitTimes
xlabel(::Type{NodeWaitTimes}) = "node"
ylabel(::Type{NodeWaitTimes}) = "waiting time"

"Tracks the fraction of time each node generates an unserved passenger"
struct NodeSOLFrac <: SimGblStat
  sol::Vector{Int}
  generated::Vector{Int}
end

NodeSOLFrac(nLocs::Int, steps::Int) =
  NodeSOLFrac(zeros(Int, nLocs), zeros(Int, nLocs))
updateGenerated!(s::NodeSOLFrac, p::Vector{Int}) = s.generated += p
updateLeftover!(s::NodeSOLFrac, p::Vector{Int}) = s.sol += p
finalize(s::NodeSOLFrac) = nonNan(s.sol ./ s.generated)

"Tracks the highest taxi density at each timestep"
struct MaxCars <: SimGblStat
  cars::Vector{Int}
end

MaxCars(nLocs::Int, steps::Int) = MaxCars(zeros(Int, steps))
updateRho!(s::MaxCars, i::Int, rho::Vector{Int}) = s.cars[i] = maximum(rho)
finalize(s::MaxCars) = s.cars
xlabel(::Type{MaxCars}) = "time"

"Creates a visualization of the simulation"
struct Vis <: SimGblStat
  frames::Reel.Frames{MIME{Symbol("image/png")}}
  f::Function
end

Reel.extension(::MIME{Symbol("text/html")}) = "html"
Reel.set_output_type("gif") 

replay(g, xs, ys, lam) = (nLocs::Int, steps::Int)->
  Vis(Frames(MIME("image/png"), fps=1), rho->
    gplot(g, xs, ys, nodelabel=rho, nodesize=lam))
updateRho!(s::Vis, i::Int, rho::Vector{Int}) = push!(frames, f(rho))
finalize(s::Vis) = s.frames

"Observe a list in twos"
stagger(a, z) = zip(I.flatten(((z,),a)), a)

initStats(nTaxis, nLocs, limit, p, polStats, gblStats) =
  ([[s(o2 - o1, nLocs) for (o1,o2) in stagger(p.offsets, 0)] for s in polStats],
    [s(nLocs, limit) for s in gblStats])

finalizeStats(polStatVals, gblStatVals, limit) =
  ([[finalize(s, limit) for s in sp] for sp in polStatVals],
    [finalize(s) for s in gblStatVals])


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
  dest = wsample(M[:,ix])
  dest, dists[ix, dest]
end

function updatePath!(path, timeDist)
  nr = sum(p[2] for p in path)
  for (r,w) in path
    push!(timeDist[r], nr)
    nr -= w
  end
  empty!(path)
end

# Simulation functions

function competB(polStats, gblStats, rho0, lam, M, dists, pn, steps=1000)
  poissons = Poisson.(lam)
  nLocs = length(rho0)
  locs = rhoToLocs(rho0)
  nTaxis = length(locs)
  p = pn(nTaxis)
  timers = zeros(Int, n)
  polStatVals, gblStatVals = initStats(nTaxis, nLocs, limit, p, polStats, gblStats)
  for i in 1:steps
    passengers = rand.(poissons)
    runGblStat(updateGenerated!, gblStatVals, passengers)
    rho = locsToRho(locs, nLocs)
    runGblStat(updateRho!, gblStatVals, i, rho) 
    pol = applySpec(pf, rho)
    for taxi in randperm(nTaxis)
      if timers[taxi] == 0
        locs[taxi] = abs(locs[taxi])
        if passengers[locs[taxi]] > 0
          passengers[locs[taxi]] -= 1
          runPolStat(updateSearch!, taxi, p, polStatVals, locs[taxi], 1.0)
          runPolStat(updateFound!, taxi, p, polStatVals)
          dest, dist = sampleDest(M, dists, locs[taxi])
          locs[taxi] = -dest
          timers[taxi] = dist
        else
          runPolStat(updateSearch!, i, p, polStatVals, loc, 1.0)
          locs[taxi] = wsample(Array(polMat(pol, taxi)[:, locs[taxi]]))
        end
      else timers[taxi] -= 1 end
    end
    runGblStat(updateLeftover!, gblStatVals, passengers)
  end
  finalizeStats(polStatsVals, gblStatVals, limit)
end

"Use with compet functions to generate starting dist"
@inline function withRho(density, stats, f, lam, args...)
  rho0 = trunc.(Int, min.(25.0, randexp(length(lam)) .* density))
  f(stats, rho0, lam, args...)
end

"Agents don't interact, probabilities are Bernoulli" 
function indivB(polStats, _, lam, M, dists, pn, nTaxis::Int=1000, limit::Int=800)
  p = pn(nTaxis)
  nLocs = length(lam)
  polStatVals = initStats(nTaxis, nLocs, limit, p, polStats, [])[1]
  for i in 1:nTaxis
    loc = rand(1:nLocs)
    t = 0
    while t < limit
      t += 1
      if rand() < lam[loc]
        runPolStat(updateSearch!, i, p, polStatVals, loc, 1.0)
        runPolStat(updateFound!, i, p, polStatVals)
        loc, dist = sampleDest(M, dists, loc)
        t += Int(dist)
      else
        runPolStat(updateSearch!, i, p, polStatVals, loc, 1.0)
        loc = wsample(Array(polMat(p, i)[:, loc]))
      end
    end
  end
  finalizeStats(polStatVals, [], limit)
end

"Agents don't interact, probabilities are exponential"
function indivE(polStats, _, lam, M, dists, len, pn, nTaxis::Int=1000, limit::Float64=800.0)
  nLocs = length(lam)
  p = pn(nTaxis)
  polStatVals = initStats(nTaxis, nLocs, limit, p, polStats, [])[1]
  exps = Exponential.(1 ./ lam)
  for i in 1:nTaxis
    loc = rand(1:nLocs)
    t = 0.0
    while t < limit
      a = min(rand(exps[loc]), limit - t)
      if a < len[loc]
        runPolStat(updateSearch!, i, p, polStatVals, loc, a)
        runPolStat(updateFound!, i, p, polStatVals)
        loc, dist = sampleDest(M, dists, loc)
        t += (a + dist)
      else
        runPolStat(updateSearch!, i, p, polStatVals, loc, len[loc])
        t += len[loc]
        loc = wsample(Array(polMat(p, i)[:, loc]))
      end
    end
  end
  finalizeStats(polStatVals, [], limit)
end


# Benchmarking functions

"Get the first policy's stat's"
id(x) = x[1]

"Get f of the first policy's stats"
agg(f) = x-> nonNan(map(f, x[1]))

"Run a simulation for a single statistic"
getStat(f,stat::SimPolStat, args...) = f([stat],[], args...)[1]
getStat(f,stat::SimGblStat, args...) = f([],[stat], args...)[2]

"Start a plot"
function mkPlot(ty, stat)
  if ty == :hist
    histogram(normalize=true, xlabel=ylabel(stat))
  elseif ty == :scatter
    plot(line=:scatter, xlabel=xlabel(stat), ylabel=ylabel(stat)) 
  elseif ty == :line
    plot(xlabel=xlabel(stat), ylabel=ylabel(stat)) 
  end
end

"Add a series to a plot"
function addPlot!(ty, h, k, v)
  if ty == :hist
   histogram!(h, val, alpha=0.2, normalize=true, label=k)
 elseif ty == :scatter
   plot!(h, val, alpha=0.3, line=:scatter, label=k)
 elseif ty == :line
   plot!(h, val, alpha=0.3, label=k)
 end
end

"Run a simulation with different policies and plot them"
function bench(f, agg, plt, stat, args...; pols...)
  h = mkPlot(plt, stat)
  for (k,p) in pols
    Random.seed!(1234)
    result = agg(getStat(f,stat, args..., p))
    println(k, " ", mean(result))
    addPlot!(plt, h, k, result)
  end
  flush(stdout)
  h
end
