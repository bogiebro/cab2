@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere using TaxiSearch
net = manhattan()
hpSearch(net, 10)
wait()
