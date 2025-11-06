"""
    forwardpass!(Ω::Network{T}) where {T<:AbstractFloat} -> Nothing

Perform a forward pass through a neural network with dropout.

# Arguments
- `Ω::Network{T}`: Network structure containing weights, biases, activations, and other network parameters

# Details
The function:
1. Applies dropout masks randomly based on specified dropout rates
2. Computes linear transformations (W*A + b) for each layer
3. Applies activation functions between layers
4. Stores final output in Ω.ŷ

# Side Effects
- Modifies the network object `Ω` in-place, updating activations (`Ω.A`), 
  pre-activations (`Ω.S`), and predictions (`Ω.ŷ`)
- Uses CUDA for GPU acceleration
- Sets random seed from Ω.seed for reproducibility

# Returns
- `Nothing`: Function modifies the network in-place
"""
function forwardpass!(Ω::Network{T})::Nothing where {T<:AbstractFloat}
    # T = Float32; X = CuArray{T, 2}(rand(Bool, 1_000, 25)); Ω = init(X)
    Random.seed!(Ω.seed)
    for i = 1:Ω.n_hidden_layers
        # i = 1
        d = CuArray{T,2}(CUDA.rand(size(Ω.W[i], 1), 1) .> Ω.dropout_rates[i])
        Ω.S[i] .= ((Ω.W[i] .* d) * Ω.A[i]) .+ Ω.b[i]
        Ω.A[i+1] .= Ω.F.(Ω.S[i])
    end
    Ω.S[end] .= (Ω.W[end] * Ω.A[end]) .+ Ω.b[end]
    Ω.ŷ .= deepcopy(Ω.S[end])
    nothing
end

"""
    backpropagation!(Ω::Network{T}, y::CuArray{T,2}) where {T<:AbstractFloat}

Perform the backpropagation algorithm to compute gradients in a neural network.

# Arguments
- `Ω::Network{T}`: Network object containing weights, biases, activations, and other network parameters
- `y::CuArray{T,2}`: Target/ground truth values as a GPU array

# Details
Implements the backpropagation algorithm to compute gradients for network parameters:
1. Computes initial error at output layer
2. Propagates error backwards through the network using chain rule
3. Computes gradients for weights (∇W) and biases (∇b) at each layer

The function updates the following fields in the Network object:
- `Ω.∇W`: Weight gradients
- `Ω.∇b`: Bias gradients

# Implementation Notes
- Uses chain rule to compute gradients: ∂C/∂Wˡ = (∂C/∂Aᴸ) * (∂Aᴸ/∂Sˡ) * (∂Sˡ/∂Wˡ)
- Operates on GPU arrays (CuArray) for efficient computation
- Includes commented-out options for L2 regularization and batch normalization

# Returns
- `Nothing`: Function modifies the network in-place
"""
function backpropagation!(Ω::Network{T}, y::CuArray{T,2})::Nothing where {T<:AbstractFloat}
    # T = Float32; X = CuArray{T, 2}(rand(Bool, 1_000, 25)); Ω = init(X); y = CUDA.randn(1, size(X, 2))
    # Cost gradients with respect to (w.r.t.) the weights: ∂C/∂Wˡ = (∂C/∂Aᴸ) * (∂Aᴸ/∂Sˡ) * (∂Sˡ/∂Wˡ)
    # Starting with the output layer down to the first hidden layer
    ∂C_over_∂Aᴸ = Ω.∂C(Ω.ŷ, y) # cost derivative with respect to (w.r.t.) to the activations at the output layer 
    ∂Aˡ_over_∂Sˡ = 1.00 # activation derivative w.r.t. the sum of the weights (i.e. pre-acitvation values) at the output layer which is just 1.00 (linear activation) because this is a regression and not a classification network
    ∂C_over_∂Sᴸ = ∂C_over_∂Aᴸ .* ∂Aˡ_over_∂Sˡ # error for the output layer (cost derivative w.r.t. the sum of the weights via chain rule): element-wise product of the cost derivatives and activation derivatives
    Δ::Vector{CuArray{T,2}} = [∂C_over_∂Sᴸ]
    for i = 0:(Ω.n_hidden_layers-1)
        # i = 1
        ∂C_over_∂Aᴸ = Ω.W[end-i]' * Δ[end] # back-propagated (notice the transposed weights) cost derivative w.r.t. the activations at the current layer
        ∂Aˡ_over_∂Sˡ = Ω.∂F.(Ω.S[end-(i+1)]) # activation derivative w.r.t. the sum of the weights (since Ω.S[end] == Ω.ŷ then the previous pre-activations are Ω.S[end-1])
        ∂C_over_∂Sᴸ = ∂C_over_∂Aᴸ .* ∂Aˡ_over_∂Sˡ # chain rule-derived cost derivative w.r.t. the sum of the weights
        push!(Δ, ∂C_over_∂Sᴸ)
    end
    # Calculate the gradients per layer
    # We want ∂C/∂Wˡ = (∂C/∂Sˡ) * (∂Sˡ/∂Wˡ)
    # where: ∂Sˡ/∂Wˡ = Aˡ⁻¹, since: Sˡ = Wˡ*Aˡ⁻¹ + bˡ
    # Then ∂C/∂Wˡ = (∂C/∂Sˡ) * (Aˡ⁻¹)'
    Δ = reverse(Δ)
    # [size(δ) for δ in Δ]
    # [minimum(δ) for δ in Δ]; [maximum(δ) for δ in Δ]
    # λ = T(0.01)
    for i in eachindex(Δ)
        # i = 1
        Ω.∇W[i] .= (Δ[i] * Ω.A[i]') # outer-product of the error in hidden layer 1 (l_1 x n) and the transpose of the activation at 1 layer below (n x l_0) to yield a gradient matrix corresponding to the weights matrix (l_1 x l_0)
        Ω.∇b[i] .= view(sum(Δ[i], dims = 2), 1:size(Δ[i], 1), 1) # sum-up the errors across n samples in the current hidden layer to calculate the gradients for the bias
        # Ω.∇W[i] .= (Δ[i] * Ω.A[i]') ./ length(y)
        # Ω.∇b[i] .= view(sum(Δ[i], dims=2), 1:size(Δ[i], 1), 1) ./ length(y)
        # Ω.∇W[i] .= ((Δ[i] * Ω.A[i]') ./ length(y)) + (λ .* (Ω.W[i].^2))
        # Ω.∇b[i] .= (view(sum(Δ[i], dims=2), 1:size(Δ[i], 1), 1) ./ length(y)) + (λ .* (Ω.b[i].^2))
        # Ω.∇W[i] .= (Δ[i] * Ω.A[i]') + (λ .* (Ω.W[i].^2))
        # Ω.∇b[i] .= view(sum(Δ[i], dims=2), 1:size(Δ[i], 1), 1) + (λ .* (Ω.b[i].^2))
    end
    # [size(x) for x in Ω.∇W]
    # [minimum(x) for x in Ω.∇W]
    # [maximum(x) for x in Ω.∇W]
    nothing
end
