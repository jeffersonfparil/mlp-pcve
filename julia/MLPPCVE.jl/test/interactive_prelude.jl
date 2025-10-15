using Pkg
Pkg.activate(".")
# try
#     Pkg.update()
# catch
#     nothing
# end
using MLPPCVE
using Random, LinearAlgebra, CUDA, JLD2
