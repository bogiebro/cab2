using DataStructures, Flux, VisdomLog
using Flux.Optimise: optimiser, adam, invdecay, descent, clip
using Flux.Tracker: grad, TrackedReal
using BSON: @save, @load
using Distributed: myid
export trainModel, NStep, VdLog, TxtLog, Logger, nn, nnStab, competTest,
  pretrain, scaleTest, mixedTest, hpSearch
pygui(false)

# should limit each hp search to a given number of steps
# also- not start new workers until old ones are dead. something
# xargs-like
function hpSearch(net, n)
  for i in 1:n
    lr=exp(rand(-5:-2))
    tr=2^(rand(0:7))
    decay=rand([0.0, 0.00001, 0.0001, 0.001])
    limit=2^(rand(3:10))
    copyRate=floor(Int, exp(rand(1:4)))
    nsteps = rand(1:10)
    @spawn trainModel(net; modelF=nnStab(copyRate=copyRate), lr=lr,
      limit=limit, decay=decay, trainRate=tr, alg=NStep(nsteps), views=[VdLog])
  end
end
                                                                                      

# Validating performance of trained models

function competTest(trained, net, episode, logView)
  nTaxis = ceil(Int, 0.2 * length(net.lam))
  trainedPol(ρ, l) = greedyAction(net, ρ, trained, l)
  fig = plt[:figure]()
  bench(withNTaxis(nTaxis, competP), net,
    S([taxiEmptyFrac, nodeWaitTimes], []),
    P([:hist, :scatter],[]), F([id, mean],[]), 1;
    rand=randPol(net), greedy=greedyPol(net), trained=PolFn(trainedPol))
  showReport(logView, Symbol("comparison $episode"), fig)
end

function maxCarTest(trained, net, episode, logView)
  nTaxis = ceil(Int, 0.2 * length(net.lam))
  trainedPol(ρ, l) = greedyAction(net, ρ, trained, l)[1]
  fig = plt[:figure]()
  bench(withNTaxis(nTaxis, competP), net,
    S([], [maxCars]),
    P([],[:line]), F([],[id]), 1;
    rand=randPol(net), greedy=greedyPol(net), trained=PolFn(trainedPol))
  showReport(logView, Symbol("maxcar $episode"), fig)
end

function nTaxiList(net)
  minTaxis = 2
  maxTaxis = length(net.lam)
  incr = max(2, (maxTaxis - minTaxis) / 100)
  minTaxis:incr:maxTaxis
end

function polGap(p, net, n)
  p1, p2 = withNTaxis(n, competP)(net, S([taxiEmptyFrac], []), p).pol
  mean(p1[1][2]) / mean(p2[1][2])
end

function mixedTest(trained, net, episode, logView)
  smarty = solo(trained, randPol(net));
  xs = nTaxiList(net)
  res = [polGap(smarty, net, n) for n in xs]
  showReport(logView, Symbol("mixed scaling $episode"), (xs, res))
end

meanEmptyFrac(n, net, p) = mean(withNTaxis(n, competP)(net, S([taxiEmptyFrac],[]), p).pol[1][1][2])

function scaleTest(net)
  legend = ["rand", "greedy", "trained"]
  xs = nTaxiList(net)
  randRes = [meanEmptyFrac(i, net, randPol(net)) for i in xs]
  greedyRes = [meanEmptyFrac(i, net, greedyPol(net)) for i in xs]
  (trained, episode, logView)-> begin
    trainedRes = [meanEmptyFrac(i, net, testA1Min) for i in xs]
    ys = [randRes' greedyRes' trainedRes']
    showReport(logView, Symbol("scaling $episode"), (xs, ys, legend));
  end
end


# Training utilities

joinDescr(args...) = foldl((x,y)-> y == nothing ? x : "$x $y", map(descr, args))
descr(::Dense) = "d"
descr(s::String) = s
descr(s::Any) = nothing

mutable struct RLState
  ρ::SparseVector{Int,Int}
  locs::Vector{Int}
  ρSum::Int
end

function trainLoop(gf, model, opt!, sampler, trDescr, logger,
    trainRate; saveFreq=204800, tests=[])
  f = Fn{Nothing, T{Int,RLState}}(gf)
  mkpath("checkpoints")
  bestLoss = Inf
  try
    for (episode, rlState::RLState) in enumerate(sampler)
      if episode % trainRate == 0 opt!() end
      f(episode, rlState)
      if episode % saveFreq == 1
        avgLoss = logger.reports[:loss][1]
        if avgLoss <= bestLoss 
          bestLoss = avgLoss
          @save "checkpoints/$trDescr $trainRate $episode" model opt!
          for t in tests t(model, net, episode, logger.view) end
        end
      end
      showLog(logger, episode)
    end
  catch e
    @save "checkpoints/$trDescr err" model opt!
    if isa(e, InterruptException) return (model, opt!, logger) end
    rethrow()
  end
end

function resumeTraining(net, model, opt!, trDescr; trainRate::Int=1, logRate::Int=800,
    views=[], nDist=MultiAgent, alg=NStep(), args...)
  invTrainRate = 1.0 / trainRate
  lamSampler = Poisson.(net.lam)
  sampler = Sampler(length(net.lam), nDist)
  trDescr = joinDescr(trDescr, "lf $logRate", sampler, model, alg)
  log = Logger(logRate, [v(trDescr) for v in views])
  trainLoop(model, opt!, sampler, trDescr, log, trainRate; args...) do episode, rlState
    waitTime = 0.0
    nTaxis = rlState.ρSum
    startEpisode!(model, episode, log)
    valEst = model(rlState.ρ)
    while step!(alg, model, rlState, valEst, log, invTrainRate)
      waitTime += rlState.ρSum
      rlState = pickup(rlState, lamSampler)
      rlState, valEst = moveTaxis(rlState, net, model)
    end
    report!(log, :waitTime, waitTime / nTaxis)
    reset!(alg)
  end
end  
 
function trainModel(net; modelF=nnStab(), lr::Float64=0.0005, limit::Float64=80.0, decay::Float64=0.0, args...)
  model = modelF(net)
  opt! = optimiser(model.params, p->adam(p; η=lr),
    p->invdecay(p, decay), p->descent(p,1), p->clip(p, limit))
  resumeTraining(net, model, opt!, "$(myid()) lr $lr"; args...)
end

function pretrain(net; views=[], lr::Float64=0.001, decay::Float64=0.0,
    trainRate::Int=800, limit::Float64=80.0, logRate::Int=800, args...)
  model = modelFn(net)
  opt! = optimiser(model.params, p->adam(p; η=lr), p->invdecay(p, decay),
    p->descent(p,1), p->clip(p, limit))
  invTrainRate = 1.0 / trainRate
  sampler = Sampler(length(net.lam), SingleAgent)
  trueVals = a1Min(net.g, net.lam)[2]
  trDescr = joinDescr("$(myid()) lr $lr lr $logRate", "pretrain", model)
  log = Logger(logRate, [v(trDescr) for v in views])
  trainLoop(model, opt!, sampler, trDescr, log, trainRate; args...) do episode::Int, s::RLState
    backup!(model, s.ρ, model(s.ρ), trueVals[s.ρ.nzind[1]], log, invTrainRate)
  end
end


# Taxi simulation

function addTaxi(a, i)
  a2 = copy(a)
  a2[i] += 1
  a2
end

function greedyAction(net, ρ, model, l)
  inds = net.g.colptr[l]:(net.g.colptr[l+1]-1)
  valEsts = model.(addTaxi(ρ, net.g.rowval[i]) for i in inds)
  best = argmin(valEsts)
  (net.g.rowval[inds[best]], valEsts[best])
end

function moveTaxis(st, net, model)::Tuple{RLState, TrackedReal{Float64}}
  st2 = deepcopy(st)
  l2Val = 0.0
  for (t,l) in enumerate(st2.locs)
    if l <= 0 continue end
    st2.ρ[l] -= 1
    l2, l2Val = greedyAction(net, st2.ρ, model, l)
    st2.locs[t] = l2
    st2.ρ[l2] += 1
  end
  (st2, l2Val)
end

function pickup(st, poissons)
  st2 = deepcopy(st)
  passengers = rand.(poissons)
  for taxi in randperm(length(st2.locs))
    l = st2.locs[taxi]
    if l > 0 && passengers[l] > 0
      passengers[l] -= 1
      st2.ρ[l] -= 1
      st2.ρSum -= 1
      st2.locs[taxi] = -1
    end
  end
  st2
end


# Samplers

struct SingleAgent end
Base.rand(::Type{SingleAgent}) = 1
agentStr(x::Type{SingleAgent}) = "s"

const MultiAgent = Truncated(Normal(0, 10), 1, 25)
agentStr(x) = "m"

"Randomly sample a rho"
struct Sampler{A}
  nLocs::Int
  nDist::A
end
descr(s::Sampler) = "$(agentStr(s.nDist)) ∈ $(s.nLocs)"

Base.iterate(s::Sampler, _) = Base.iterate(s)
Base.eltype(::Type{Sampler}) = RLState
function Base.iterate(s::Sampler)
  n = floor(Int, rand(s.nDist))
  locs = rand(1:s.nLocs, n)
  (RLState(locsToRho(locs, s.nLocs), locs, n), nothing)
end


# Training algorithms

abstract type Alg end

mutable struct NStep <: Alg
  history::CircularBuffer{Tuple{SparseVector{Int,Int}, Int, TrackedReal{Float64}}}
  seen::Set{SparseVector{Int,Int}}
  curSum::Int # sum of waiting times over all in history
end
NStep(n=4) = NStep(CircularBuffer{Tuple{SparseVector{Int,Int}, Int, TrackedReal{Float64}}}(n),
   Set{SparseVector{Int,Int}}(), 0)
descr(ns::NStep) = "TD$(ns.history.capacity)"

function reset!(alg::NStep)
  alg.curSum = 0
  empty!(alg.seen)
end

function step!(alg::NStep, model, st::RLState, valEst::TrackedReal{Float64}, logger, invTrainRate)::Bool
  if st.ρSum == 0
    while !isempty(alg.history)
      ρ0, t, est = popfirst!(alg.history)
      if !(ρ0 in alg.seen)
        push!(alg.seen, ρ0)
        backup!(model, ρ0, est, alg.curSum, logger, invTrainRate)
      end
      alg.curSum -= t
    end
    backup!(model, st.ρ, valEst, 0, logger, invTrainRate)
    return false
  end
  if isfull(alg.history)
    ρ0, t, est = alg.history[1]
    if !(ρ0 in alg.seen)
      push!(alg.seen, ρ0)
      backup!(model, ρ0, est, alg.curSum + predictT(model, st.ρ), logger, invTrainRate)
    end
    alg.curSum -= t
  end
  alg.curSum += st.ρSum
  push!(alg.history, (st.ρ, st.ρSum, valEst))
  true
end


# Neural net utilities

function denseNet(net::RoadNet)
  nLoc = length(net.lam)
  Chain(Dense(nLoc, 2 * nLoc, leakyrelu), Dense(2 * nLoc, 1), x->x[1])
end

regularize(x, ps) = sum(map(x->sum(abs.(x)), ps))

huber(a, delta) = Flux.data(a) <= delta ?  0.5 * a.^2 : delta * (abs.(a) - 0.5 * delta)

descr(c::Chain) = joinDescr(descr.(c.layers)...)

function updateTarget!(targetPs, behaviorPs, tau)
  for (prev, current) in zip(targetPs, behaviorPs)
    newval = Flux.data(prev) .* (1 - tau) + tau .* Flux.data(current)
    copyto!(Flux.data(prev), newval)
  end
end
  
function updateTarget!(targetPs, behaviorPs)
  for (prev, current) in zip(targetPs, behaviorPs)
    copyto!(Flux.data(prev), Flux.data(current))
  end
end


# Models

abstract type Model end
Base.broadcastable(a::Model) = Base.RefValue(a)
startEpisode!(model::Model, episode, logger) = nothing

function backup!(model, ρ::SparseVector{Int,Int}, predicted::TrackedReal{Float64},
    target, logger, invRate::Float64)
  loss = 0.5 * (predicted - target).^2
  Flux.back!(loss, invRate)
  report!(logger, :grad, vcat(map(x->grad(x)[:], model.params)...))
  report!(logger, :loss, Flux.data(loss))
end

struct NN{W,P} <: Model
  v::W
  params::P
end
(m::NN)(x) = m.v(x)
descr(n::NN) = "nn $(descr(n.v))"

nn(arch=denseNet) = net-> begin
  v = arch(net)
  NN(v, Flux.params(v))
end

fromSingle(net::RoadNet) = NN(Linear(a1Min(net.g, net.lam)[2]))

predictT(model::NN, ρ)= Flux.data(model(ρ))

struct NNStab{R,M,P} <: Model
  q1::M
  q2::M
  params::P
  q2params::P
  copyRate::R
end
(m::NNStab)(x) = m.q1(x)
descr(n::NNStab{R}) where {R} = "nn2 $(descr(n.q1)) $(n.copyRate)"

nnStab(;copyRate=3200, arch=denseNet) = net-> begin
  q1 = arch(net)
  q2 = arch(net)
  NNStab(q1, q2, Flux.params(q1), Flux.params(q2), copyRate)
end

function startEpisode!(model::NNStab{Int}, episode, logger)
  if episode % model.copyRate == 1
    updateTarget!(model.q2params, model.params)
  end
end

function startEpisode!(model::NNStab{Float64}, episode, logger)
  updateTarget!(model.q1params, model.params, model.copyRate)
end

predictT(model::NNStab, ρ)= Flux.data(model.q2(ρ))


# Loggers

abstract type LogView end

struct Logger
  freq::Int
  reports::Dict{Symbol, Pair{Any,Int}}
  view::Vector{LogView}
end
Logger(freq,views) = Logger(freq, Dict{Symbol,Pair{Any,Int}}(), views)

showLog(l, episode) = if episode % l.freq == 0
  for (k,(v,_)) in l.reports
    for lv in l.view showReport(lv, k, v) end
  end
  empty!(l.reports)
end

function report!(l::Logger, s::Symbol, data)
  if haskey(l.reports, s)
    v, n = l.reports[s] 
    l.reports[s] = (v .+ ((Float64.(data) .- v) ./ n))=>(n + 1);
  else
    l.reports[s] = Float64.(data)=>1;
  end
end

struct TxtLog <: LogView end
TxtLog(_) = TxtLog()
showReport(::TxtLog, k, v::Float64) = println("$k: $v");
showReport(::TxtLog, k, v) = nothing
showFig(::TxtLog, fig) = fig[:show]()

struct VdLog <: LogView 
  vd::Visdom
end
VdLog(env::String) = VdLog(Visdom(env))

showReport(vd::VdLog, k, v) = VisdomLog.report(vd.vd, k, v);
