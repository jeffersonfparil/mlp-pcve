module MLPPCVE
    
using CUDA # Note: do the following prior to running the binary: `export JULIA_CUDA_USE_COMPAT=false`
# (1) Activation functions
include("activations.jl")
# (2) Cost functions and their derivatives
include("costs.jl")
# (3) Forward pass and backpropagation
include("forwardbackward.jl")
# (4) Optimisers
include("optimisers.jl")

export Activations, Costs, ForwardBackward, Optimisers

end
