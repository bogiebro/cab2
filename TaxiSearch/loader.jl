import Pkg
Pkg.activate(".")
using BSON, LightGraphs
using Revise
using TaxiSearch

BSON.@load "manhattan_sg.bson" net
