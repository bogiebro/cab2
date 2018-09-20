using DataStructures, Flux, VisdomLog
export trainModel, easyStates, SampledRho, NStep, VdLog, TxtLog, Logger

struct Linear
  W
end
Linear(n::Integer) = Linear(param(randn(n)))
(m::Linear)(x) = dot(m.W, x)
Flux.@treelike Linear


# Training utilities

struct StopTraining <: Exception end

mutable struct RLState
  ρ::SparseVector{Int,Int}
  locs::Vector{Int}
  ρSum::Int
end

function trainModel(net; logger=Logger(), nDist=singleAgent, modelFn=nn(), alg=NStep())
  model = modelFn(net)
  sampler = Sampler(length(net.lam), nDist)
  lamSampler = Poisson.(net.lam)
  try
    for (episode, rlState::RLState) in enumerate(sampler)
      waitTime = 0.0
      nTaxis = rlState.ρSum
      startEpisode!(model, episode, logger)
      showLog(logger, episode)
      while step!(alg, model, rlState, logger)
        waitTime += rlState.ρSum
        rlState = pickup(rlState, lamSampler)
        rlState = moveTaxis(rlState, net, model)
      end
      report!(logger, :waitTime, waitTime / nTaxis)
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
  best = argmin(predictB.(model, addTaxi(ρ, net.g.rowval[i]) for i in inds))
  net.g.rowval[inds[best]]
end

function moveTaxis(st, net, model)
  st2 = deepcopy(st)
  for (t,l) in enumerate(st2.locs)
    if l <= 0 continue end
    st2.ρ[l] -= 1
    l2 = greedyAction(net, st2.ρ, st2.ρSum, model, l)
    st2.locs[t] = l2
    st2.ρ[l2] += 1
  end
  st2
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
const singleAgent = SingleAgent()
Base.rand(::SingleAgent) = 1

const multiAgent = Truncated(Normal(0, 10), 1, 25)

"Randomly sample a rho"
struct Sampler
  nLocs::Int
  nDist
end

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
  history::CircularBuffer{Pair{SparseVector{Int,Int}, Float64}}
  seen::Set{SparseVector{Int,Int}}
  curSum::Int
end
NStep(n=2) = NStep(CircularBuffer{Pair{SparseVector{Int,Int}, Float64}}(n-1),
   Set{Vector{Int}}(), 0)

function reset!(alg::NStep)
  alg.curSum = 0
  empty!(alg.seen)
end

function step!(alg::NStep, model, st::RLState, logger)::Bool
  @assert st.ρSum == sum(st.ρ)
  if st.ρSum == 0
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
      backup!(model, ρ0, alg.curSum + predictT(model, st.ρ), logger)
    end
  end
  alg.curSum += st.ρSum
  if isfull(alg.history) alg.curSum -= alg.history[1][2] end
  push!(alg.history, st.ρ=>st.ρSum)
  true
end


# Models

abstract type Model end
Base.broadcastable(a::Model) = Base.RefValue(a)

struct LinModel <: Model
  x
  opt!
  trainRate::Int
end

linModel(lr=0.001, trainRate=800) = net-> begin
  x = param(a1Min(net.g, net.lam)[2])
  LinModel(x, ADAM([x], lr), trainRate)
end

function startEpisode!(model::LinModel, episode, logger)
  if episode % model.trainRate == 0
    model.opt!()
  end
end

predictB(model::LinModel, ρ)= dot(Flux.data(model.x), ρ) 
predictT(model::LinModel, ρ)= dot(Flux.data(model.x), ρ)

function backup!(model::LinModel, ρ, target, logger)
  prediction = model.x' * ρ
  loss = 0.5 * (prediction - target).^2
  Flux.back!(loss, 1 / model.trainRate)
  report!(logger, :grad, model.x.grad)
  report!(logger, :loss, Flux.data(loss))
end

clipgrad!(nn,::Nothing) = nothing
function clipgrad!(nn, rng::Tuple{Float64,Float64})
  for p in Flux.params(nn)
    p.grad[p.grad .< rng[1]] .= rng[1]
    p.grad[p.grad .> rng[2]] .= rng[2];
  end
end
  
struct NN <: Model
  q1
  q2
  opt!
  params
  copyRate::Int
  trainRate::Int
  rng::Union{Nothing,Tuple{Float64,Float64}}
end

nn(;lr=0.0005, copyRate=3200, trainRate=800, rng=(-40.0,40.0)) = net-> begin
  nLoc = length(net.lam)
  model() = Chain(Dense(nLoc, 2 * nLoc, leakyrelu), Dense(2 * nLoc, 1), x->x[1])
  q1 = model()
  ps = Flux.params(q1)
  NN(q1, model(), ADAM(ps, lr), ps, copyRate, trainRate, rng)
end

function startEpisode!(model::NN, episode, logger)
  if episode % model.trainRate == 0
    model.opt!()
  end
  if episode % model.copyRate == 1
    println("Copying")
    Flux.loadparams!(model.q2, Flux.params(model.q1))
  end
end

predictB(model::NN, ρ)= Flux.data(model.q1(ρ))
predictT(model::NN, ρ)= Flux.data(model.q2(ρ))

norms(ps) = sum(map(x->sum(abs.(x)), ps)) / length(ps)

huber(a, delta) = Flux.data(a) <= delta ?  0.5 * a.^2 : delta * (abs.(a) - 0.5 * delta)

function backup!(model::NN, ρ, target, logger)
  loss = 0.5 * (model.q1(ρ) - target).^2 # + 0.001 * norms(model.params) 
  Flux.back!(loss, 1 / model.trainRate)
  #clipgrad!(model.q1, model.rng)
  report!(logger, :gradW1, model.q1.layers[1].W.grad)
  report!(logger, :gradW2, model.q1.layers[2].W.grad)
  report!(logger, :loss, Flux.data(loss))
  model.opt!()
end


# Loggers

abstract type LogView end

@with_kw struct Logger
  freq::Int=800
  reports::Dict{Symbol, Pair{Any,Int}}=Dict{Symbol,Pair{Any,Int}}()
  view::Vector{LogView} = [TxtLog()]
end
 
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
showReport(::TxtLog, k, v::Float64) = println("$k: $v");
showReport(::TxtLog, k, v) = nothing

struct VdLog <: LogView 
  vd::Visdom
end
VdLog(env::String) = VdLog(Visdom(env))

showReport(vd::VdLog, k, v) = VisdomLog.report(vd.vd, k, v);


