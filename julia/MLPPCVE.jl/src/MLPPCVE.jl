module MLPPCVE

using Random, CUDA

# TODO: 
# - always compile CUDA with the graphics card you want to use the binary on
# - do the following prior to running the binary: `export JULIA_CUDA_USE_COMPAT=false;

# (0) Simulation functions
include("sim.jl")
# (1) Activation functions
include("activations.jl")
# (2) Cost functions and their derivatives
include("costs.jl")
# (3) Forward pass and backpropagation
include("forwardbackward.jl")
# (4) Optimisers
include("optimisers.jl")
# (5) Training and validation
include("traintest.jl")

export simulate
export sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative, leakyrelu, leakyrelu_derivative
export MSE, MSE_derivative, MAE, MAE_derivative, HL, HL_derivative
export init, initgradient, forwardpass, backpropagation!
export gradientdescent!, Adam!, AdamMax!
export train


end
