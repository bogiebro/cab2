@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere using TaxiSearch
net = testNet()
hpSearch(net, 20)
