using ResumableFunctions, DataStructures, Flux
export nStepTabular, untilStopped, mcTabular, greedyAction

# Training utilities

struct StopTraining <: Exception end
    
@inline function untilStopped(f)
  model = nothing
  try model = f() catch e
    if isa(e, StopTraining)
      return model
    end
    rethrow()
  end
end


# Taxi simulation

function addTaxi(a, i)
  a2 = copy(a)
  a2[i] += 1
  a2
end

function greedyAction(net, rho, predict, l)
  inds = net.g.colptr[l]:(net.g.colptr[l+1]-1)
  best = argmin(predict.(addTaxi(rho, net.g.rowval[i]) for i in inds))
  net.g.rowval[inds[best]]
end

function moveTaxis(rho, locs, net, predict, eps)
  for t in 1:length(locs)
    l = locs[t]
    if l <= 0 continue end
    rho[l] -= 1
    l2 = rand() >= eps ? greedyAction(net, rho, predict, l) :
      net.g.rowval[rand(net.g.colptr[l]:(net.g.colptr[l+1]-1))]
    locs[t] = l2
    rho[l2] += 1
  end
  (rho, locs)
end

function hot(len, ixs)
  a = zeros(Int, len)
  for i in ixs
    a[i] += 1
  end
  a 
end

function pickup(rho, locs, poissons)
  passengers = rand.(poissons)
  for taxi in randperm(length(locs))
    l = locs[taxi]
    if l > 0 && passengers[l] > 0
      passengers[l] -= 1
      rho[l] -= 1
      locs[taxi] = -1
    end
  end
  (rho, locs)
end

# Tabular case

@resumable function easyStates()
  for i in 1:16
    @yield hot(16, i)
  end
  for i in 1:16
    for j in i:16
      @yield hot(16, [i,j])
    end
  end 
end


function mcTabular(net)
  lamSampler = Poisson.(net.lam)
  states = DefaultDict{Vector{Int}, Pair{Float64, Int}}(0.0=>1)
  predict(ρ) = states[ρ][1]
  for ρStart in easyStates()
    for attempt in 1:500
      ρ = copy(ρStart)
      locs = rhoToLocs(ρ)
      history = Vector{Tuple{Vector{Int}, Float64, Bool}}()
      while true
        seen = Set{Vector{Int}}()
        ρSum = sum(ρ)
        if ρSum == 0
          cumSum = 0
          for (k,v,b) in Itr.reverse(history)
            cumSum += v
            if !b
              oldVal, oldN = states[k]
              states[k] = (oldVal + (1.0 / oldN) * (cumSum - oldVal)) => (oldN + 1)
            end
          end
          break
        else
          push!(history, (ρ, ρSum, ρ in seen))
          push!(seen, ρ)
        end
        ρ, locs = pickup(ρ, locs, lamSampler)
        ρ, locs = moveTaxis(ρ, locs, net, predict, 0.9)
      end
    end
  end
  predict
end



# what if we start it with the one taxi prediction?
# and the default value is n times the one taxi prediction 
# do we not need to separate prediction from improvement phase?
function nStepTabular(net, n=6)
  lr = 0.001
  lamSampler = Poisson.(net.lam)
  states = DefaultDict{Vector{Int}, Float64}(0.0)
  oldStates = copy(states)
  predict(ρ) = oldStates[ρ]
  history = CircularBuffer{Pair{Vector{Int}, Float64}}(n - 1)
  for (i,ρ) in Itr.take(enumerate(Itr.cycle(easyStates())), 500000)
    tdSum = 0.0
    tdN = 0
    curSum = 0.0
    while true
      seen = Set{Vector{Int}}()
      ρSum = sum(ρ)
      if ρSum == 0
        while !isempty(history)
          st, t = popfirst!(history)
          if !(st in seen)
            push!(seen, st)
            td = curSum - states[st] 
            tdSum += abs(td)
            tdN += 1
            states[st] += lr * td
          end
          curSum -= t
        end
        break
      end
      if isfull(history)
        st = history[1][1]
        if !(st in seen)
          push!(seen, st)
          target = curSum + states[ρ]
          td = target - states[st]
          states[st] += lr * td
          tdSum += abs(td)
          tdN += 1
        end
      end
      curSum += ρSum
      if isfull(history) curSum -= history[1][2] end
      push!(history, ρ=>ρSum)
      ρ = moveTaxis(max.(0, ρ .- rand.(lamSampler)), net, predict)
    end
    if i % 2000 == 999 merge!(oldStates, states) end
    if i % 3000 == 2999 lr *= 0.9 end
    if i % 6000 == 0
      @info "Average td error $(tdSum / tdN)"
      tdSum = 0.0
      tdN = 0
    end
  end
  predict
end

# In the simulation:
# Every round, a turn order is established randomly. 
# Then each taxi takes their turn moving



function trainedModel(net; lr=0.0005, nSampler=Truncated(Normal(0, 20), 0, 50),
  copyRate=100)
  #vd = Visdom("loss", [:loss])
  k = 1
  nLocs = length(net.lam)
  model() = Chain(Dense(nLocs, 1), x->x[1])
  q1 = model()
  q2 = model()
  lamSampler = Poisson.(net.lam)
  opt = ADAM(Flux.params(q1), lr)
  history = CircularBuffer{Pair{Vector{Int}, Float64}}(n - 1)
  runningLoss = 0.0
   
  try
    for episode in Itr.countfrom()
      if episode % copyRate == 0
        Flux.loadparams!(q2, Flux.params(q1))
      end
      n = rand(nSampler)
      ρ = locsToRho(rand(1:nLocs,n), nLocs)
    
    
    
    ρEnd = moveTaxis(max.(0, ρ .- rand.(lamSampler)), net, q2)
    ρSum = sum(ρ)
    timeEst = (ρSum == 0) ? 0.0 : ρSum + Flux.data(q2(ρEnd))
    loss = Flux.mse(q1(ρ), timeEst)
    Flux.back!(loss)
    opt()
    runningLoss += Flux.data(loss)
    if i % 1000 == 999
      println("Running loss ", runningLoss / 1000.0)
      runningLoss = 0.0
      #VisdomLog.inform(vd, :loss, runningLoss / 100.0)
    end
  end
  q1
end

