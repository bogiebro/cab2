using MAT
export manhattan, testNet, manhattan_sg

"Read a csv where each row is an edge"
function getGraph(fname)
  df = readdlm(fname, ',')
  is = Int.(df[:,2])
  js = Int.(df[:,1])
  g = sparse(is .+ 1, js .+ 1, 1.0)
  g .+ g'
end

function manhattan()
  m = getGraph("data/manhattan.csv");
  trips = matread("data/manhattan-trips.mat");
  coords = readdlm("data/manhattan-coords.csv", ',', Float32);
  RoadNet(m, trips["eveM"], nothing, trips["eveP"][:,1], coords[:,1], coords[:,2]);
end

function manhattan_sg()
  g = getGraph("data/manhattan.csv");
  trips = matread("data/manhattan-trips.mat");
  coords = readdlm("data/manhattan-coords.csv", ',', Float32);
  ix = Bool.(neighborhoods(g, 10)[:,1])
  RoadNet(g[ix,ix], trips["eveM"][ix,ix], nothing, trips["eveP"][ix,1],
          coords[ix,1], coords[ix,2]);
end

function testNet()
  Random.seed!(1)
  RoadNet(10)
end
