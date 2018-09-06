using Flux
export trainedModel

neighborhoods(g::Graph, n::Int)::Graph = (g + speye(g))^n .> 0

struct DenseLocal
  channels::Vector{Pair{Any, Any}}
end

function DenseLocal(net, n)
  mask = neighborhoods(net.g, 5)
  DenseLocal([param(SparseMatrixCSC(mask.m, mask.n, mask.colptr, mask.rowval,
    0.01 * rand(length(mask.nzval)))) => param(sqrt(1 ./ net.lam)) for _ in 1:n])
end

(d::DenseLocal)(x) = vcat([W * x + b for (W,b) in d.channels]...)
@Flux.treelike DenseLocal

function addTaxi(a, i)
  a2 = copy(a)
  a2[i] += 1
  a2
end

function moveTaxis(rho, net, predict)
  nTaxis = sum(rho)
  locs = rhoToLocs(rho)
  for t in 1:nTaxis
    l = locs[t]
    rho[l] -= 1
    inds = net.g.colptr[l]:(net.g.colptr[l+1]-1)
    best = argmax(predict.(addTaxi(rho, net.g.rowval[i]) for i in inds))
    rho[net.g.rowval[inds[best]]] += 1
  end
  rho
end

# easiest: 2x2 grid, one person
# maybe the loss ever very high, so we don't notice convergence
# well a 4x4 net is not bad. Something is wrong. Investigate

function trainedModel(net)
  #vd = Visdom("loss", [:loss])
  k = 1
  nLocs = length(net.lam)
  # DenseLocal(net, 10), leakyrelu,
  #predict = Chain(Dense(nLocs, k * nLocs, leakyrelu),
    #Dense(k * nLocs, 50, leakyrelu),
    #Dense(50, 1), x-> max(0,x[1]))
  predict = Chain(Dense(nLocs, 1), x->x[1].^2)
  guess(x) = Flux.data(predict(x))
  lamSampler = Poisson.(net.lam)
  opt = ADAM(Flux.params(predict), 0.0003)
  
  # We're just failing terribly. what's going on?
  runningLoss = 0.0
  for i in 1:10000
    ρ = locsToRho(rand(1:nLocs,4), nLocs)
    for j in 1:10000
    timeEst = sum(ρ) + guess(moveTaxis(max.(0, ρ .- rand.(lamSampler)),
      net, guess))
    loss = Flux.mse(predict(ρ), timeEst)
    Flux.back!(loss)
    opt()
    runningLoss += Flux.data(loss)
    if j % 1000== 999
      println("Running loss ", runningLoss / 1000.0)
      runningLoss = 0.0
      #VisdomLog.inform(vd, :loss, runningLoss / 100.0)
    end
    end
  end
  predict
end

