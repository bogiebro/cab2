using ResumableFunctions, DataStructures, Flux, Flux.Tracker, VisdomLog
export trainModel, easyStates, SampledRho, NStep, VdLog, TxtLog, Logger

# Training utilities

struct StopTraining <: Exception end

const RLState = Tuple{Vector{Int}, Vector{Int}, Int} # rho, locs, rhoSum

function trainModel(net; eps=0.9, nLocs=length(net.lam), logger=Logger(),
    sampler=SampledRho(nLocs=nLocs), model=NN(nLocs), alg=NStep())
  lamSampler = Poisson.(net.lam)
  try
    for (episode, rlState::RLState) in enumerate(sampler)
      startEpisode!(model, episode)
      showLog(logger, episode)
      reward = 0
      while step!(alg, model, rlState, logger)
        reward += rlState[3]
        rlState = pickup(rlState, lamSampler)
        rlState = moveTaxis(rlState, net, model, eps)
      end
      report!(logger, :reward, reward)
      reset!(alg)
    end
  catch e
    if isa(e, StopTraining) return model end
    rethrow()
  end
end

# Taxi simulation

function addTaxi(a, i)
  a2 = copy(a)
  a2[i] += 1
  a2
end

function greedyAction(net, ρ, ρSum, model, l)
  inds = net.g.colptr[l]:(net.g.colptr[l+1]-1)
  best = argmin(predict.(model, ρSum, addTaxi(ρ, net.g.rowval[i]) for i in inds))
  net.g.rowval[inds[best]]
end

function moveTaxis((oldρ, oldLocs, ρSum), net, model, eps)
  ρ = copy(oldρ)
  locs = copy(oldLocs)
  for t in 1:length(locs)
    l = locs[t]
    if l <= 0 continue end
    ρ[l] -= 1
    l2 = rand() >= eps ? greedyAction(net, ρ, ρSum, model, l) :
      net.g.rowval[rand(net.g.colptr[l]:(net.g.colptr[l+1]-1))]
    locs[t] = l2
    ρ[l2] += 1
  end
  (ρ, locs, ρSum)
end

function hot(len, ixs)
  a = zeros(Int, len)
  for i in ixs
    a[i] += 1
  end
  a 
end

# this would be better with functional data structures
# Look into this! Finger tree?

function pickup((oldRho, oldLocs, rhoSum), poissons)
  locs = copy(oldLocs)
  rho = copy(oldRho)
  passengers = rand.(poissons)
  for taxi in randperm(length(locs))
    l = locs[taxi]
    if l > 0 && passengers[l] > 0
      passengers[l] -= 1
      rho[l] -= 1
      rhoSum -= 1
      locs[taxi] = -1
    end
  end
  (rho, locs, rhoSum)
end


# Samplers

@resumable function easyStates()
  for i in 1:16
    oneHot = hot(16, i)
    @yield (oneHot, rhoToLocs(oneHot), 1)
  end
  for i in 1:16
    for j in i:16
      twoHot = hot(16, [i,j])
      @yield (twoHot, rhoToLocs(twoHot), 2)
    end
  end 
end

"Randomly sample from a truncated normal"
@with_kw struct SampledRho
  nLocs::Int
  nDist = Truncated(Normal(0, 20), 1, 50)
end

Base.iterate(SampledRho, _) = Base.iterate(SampledRho)
Base.eltype(::Type{SampledRho}) = RLState
function Base.iterate(s::SampledRho)
  n = floor(Int, rand(s.nDist))
  locs = rand(1:s.nLocs, n)
  ((locsToRho(locs, s.nLocs), locs, n), nothing)
end


# Training algorithms

mutable struct NStep
  history::CircularBuffer{Pair{Vector{Int}, Float64}}
  seen::Set{Vector{Int}}
  curSum::Int
end
NStep(n=6) = NStep(CircularBuffer{Pair{Vector{Int}, Float64}}(n-1), Set{Vector{Int}}(), 0)

function reset!(alg::NStep)
  alg.curSum = 0
  empty!(alg.seen)
end

function step!(alg::NStep, model, (ρ, locs, ρSum)::RLState, logger)::Bool
  if ρSum == 0
    while !isempty(alg.history)
      ρ0, t = popfirst!(alg.history)
      if !(ρ0 in alg.seen)
        push!(alg.seen, ρ0)
        backup!(model, ρ0, alg.curSum, logger)
      end
      alg.curSum -= t
    end
    return false
  end
  if isfull(alg.history)
    ρ0 = alg.history[1][1]
    if !(ρ0 in alg.seen)
      push!(alg.seen, ρ0)
      backup!(model, ρ0, alg.curSum + predict(model, ρSum, ρ), logger)
    end
  end
  alg.curSum += ρSum
  if isfull(alg.history) alg.curSum -= alg.history[1][2] end
  push!(alg.history, ρ=>ρSum)
  true
end


# Models

abstract type Model end

struct NN <: Model
  q1
  q2
  opt!
  copyRate::Int
end
Base.broadcastable(a::NN) = Base.RefValue(a)

function NN(nLocs; lr=0.0005, copyRate=1000)
  model() = Chain(Dense(nLocs, 1), x->x[1])
  q1 = model()
  NN(q1, model(), ADAM(Flux.params(q1), lr), copyRate)
end

startEpisode!(model::NN, episode) = 
  if episode % model.copyRate == 0
    Flux.loadparams!(model.q2, Flux.params(model.q1))
  end

predict(model::NN, ρSum, ρ)= ρSum == 0 ? 0.0 : Flux.data(model.q2(ρ))

function backup!(model::NN, ρ, target, logger)
  loss = abs(model.q1(ρ) - target)
  Flux.back!(loss)
  inform!(logger, :gradW, model.q1.layers[1].W.grad)
  model.opt!()
  report!(logger, :loss, Flux.data(loss))
end


# Loggers

abstract type LogView end

@with_kw struct Logger
  freq::Int=1000
  reports::Dict{Symbol, Pair{Float64,Int}}=Dict{Symbol,Pair{Float64,Int}}()
  info::Dict{Symbol, Array{Float64}}=Dict{Symbol,Array{Float64}}()
  view::Vector{LogView} = [TxtLog()]
end
 
showLog(l, episode) = if episode % l.freq == (l.freq - 1)
  for (k,(v,_)) in l.reports
    for lv in l.view showReport(lv, k, v) end
  end
  for (k, v) in l.info
    for lv in l.view showInfo(lv, k, v) end
  end
  empty!(l.reports)
  empty!(l.info)
end

function inform!(l::Logger, k, v)
  if all(v .== 0.0)
    println("FUCK")
  end
  l.info[k] = v
end

function report!(l::Logger, s::Symbol, data)
  if haskey(l.reports, s)
    v, n = l.reports[s] 
    l.reports[s] = (v + ((Float64(data) - v) / n))=>( n + 1);
  else
    l.reports[s] = Float64(data)=>1;
  end
end

struct TxtLog <: LogView end
showReport(::TxtLog, k, v) = println("$k: $v");
showInfo(::TxtLog, k, v) = println("$k: $v")

struct VdLog <: LogView 
  vd::Visdom
end
VdLog(env::String) = VdLog(Visdom(env))

showReport(vd::VdLog, k, v) = VisdomLog.report(vd.vd, k, v);
showInfo(vd::VdLog, k, v) = VisdomLog.inform(vd.vd, k, v);


