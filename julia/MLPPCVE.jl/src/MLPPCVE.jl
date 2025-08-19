module MLPPCVE

using Random, CUDA

# TODO: 
# - always compile CUDA with the graphics card you want to use the binary on
# - do the following prior to running the binary: `export JULIA_CUDA_USE_COMPAT=false;

# (0) Helper functions
include("helpers.jl")
# (1) Activation functions
include("activations.jl")
# (2) Cost functions and their derivatives
include("costs.jl")
# (3) Forward pass and backpropagation including the Network struct and initialisation
include("forwardbackward.jl")
# (4) Optimisers
include("optimisers.jl")
# (5) Training and validation
include("traintest.jl")

export Network, init, mean, var, cov, std, drawreplacenot, metrics, simulate
export linear, linear_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative, leakyrelu, leakyrelu_derivative
export MSE, MSE_derivative, MAE, MAE_derivative, HL, HL_derivative
export forwardpass!, backpropagation!
export gradientdescent!, Adam!, AdamMax!
export splitdata, predict, train


end
