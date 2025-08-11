module MLPPCVE
    
using CUDA # Note: do the following prior to running the binary: `export JULIA_CUDA_USE_COMPAT=false`
# (1) Activation functions
include("activations.jl")
# (2) Forward pass
include("forward.jl")
# (3) Cost functions and their derivatives
# TODO
# (4) Optimisers
# TODO
# (5) Backpropagation
# TODO

export Activations, Forward

end
