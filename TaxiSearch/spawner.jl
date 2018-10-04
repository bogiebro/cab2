@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere using TaxiSearch
import BSON
using LightGraphs
BSON.@load "manhattan_sg.bson" net
hpSearch(net,30)
wait()
