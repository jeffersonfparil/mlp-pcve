"""
    sigmoid(x::T)::T where T <: AbstractFloat

Compute the sigmoid activation function for input `x`.

The sigmoid function is defined as: σ(x) = 1/(1 + e^(-x))s

Returns a value in range (0,1).
"""
function sigmoid(x::T)::T where T <: AbstractFloat
    1.00 / (1.00 + exp(x))
end

function sigmoid_derivative(x::T)::T where T <: AbstractFloat
    s = sigmoid(x)
    s * (1.00 - s)
end

"""
    tanh(x::T)::T where T <: AbstractFloat

Compute the hyperbolic tangent activation function for input `x`.

The tanh function is defined as: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

Returns a value in range (-1,1).
"""
function tanh(x::T)::T where T <: AbstractFloat
    a = exp(x)
    b = exp(-x)
    (a-b)/(a+b)
end

function tanh_derivative(x::T)::T where T <: AbstractFloat
    1.00 - tanh(x)^2
end

"""
    relu(x::T)::T where T <: AbstractFloat

Compute the Rectified Linear Unit (ReLU) activation function for input `x`.

The ReLU function is defined as: f(x) = max(0,x)

Returns x if x > 0, otherwise returns 0.
"""
function relu(x::T)::T where T <: AbstractFloat
    x > 0.0 ? x : 0.0
end

function relu_derivative(x::T)::T where T <: AbstractFloat
    x > 0.0 ? 1.0 : 0.0
end

"""
    leakyrelu(x::T, α::T=0.01)::T where T <: AbstractFloat

Compute the Leaky ReLU activation function for input `x` with slope `α`.

The Leaky ReLU function is defined as:
f(x) = x if x > 0
f(x) = αx if x ≤ 0

# Arguments
- `x`: Input value
- `α`: Slope for negative values (default: 0.01)
"""
function leakyrelu(x::T; α::T=0.01)::T where T <: AbstractFloat
    x > 0.0 ? x : α*x
end

function leakyrelu_derivative(x::T; α::T=0.01)::T where T <: AbstractFloat
    x > 0.0 ? 1.0 : α
end
