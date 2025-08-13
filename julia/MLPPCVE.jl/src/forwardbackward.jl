module ForwardBackward

using CUDA

# T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.leakyrelu
# FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F)
function forwardpass(;
    X::CuArray{T},
    W::Vector{CuArray{T}},
    B::Vector{CuArray{T}},
    F::Function,
)::Dict{String, Union{CuArray{T}, Vector{CuArray{T}}}} where T <:AbstractFloat
    # T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.leakyrelu
    S::Vector{CuArray{T}} = []
    A::Vector{CuArray{T}} = [X]
    for i in eachindex(W)
        s = (W[i] * A[i]) .+ B[i]
        push!(S, s)
        push!(A, F.(s))
    end
    Dict(
        "ŷ" => A[end], # prediction
        "S" => S, # weighted sums
        "A" => A[1:(end-1)], # activations (excludes the input layer)
    )
end

function initgradient(W_or_B::CuArray{T})::Vector{CuArray{T}} where T <: AbstractFloat
    # T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); C = Costs.MSE; ŷ = FP["ŷ"]; y = CUDA.randn(1, length(ŷ)); δF = Activations.relu_derivative; S = FP["weighted_sums"]; A = FP["activations"]
    # W_or_B = W
    # W_or_B = B
    [CuArray{T}(zeros(size(x))) for x in W_or_B]
end


# T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); C = Costs.MSE; ŷ = FP["ŷ"]; y = CUDA.randn(1, length(ŷ)); δF = Activations.relu_derivative; S = FP["weighted_sums"]; A = FP["activations"]
# BP = ForwardBackward.backpropagation(y=y, ŷ=ŷ, S=S, W=W, A=A, C=C, δF=δF)
function backpropagation(;
    # TODO: mutate instantiated gradients instead - for memory efficiency...
    y::CuArray{T}, 
    ŷ::CuArray{T},
    W::Vector{CuArray{T}},
    S::Vector{CuArray{T}},
    A::Vector{CuArray{T}},
    δC::Function,
    δF::Function,
)::Dict{String, Vector{CuArray{T}}} where T <: AbstractFloat
    # T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); δC = Costs.MSE_derivative; ŷ = FP["ŷ"]; y = CUDA.randn(1, length(ŷ)); δF = Activations.relu_derivative; S = FP["weighted_sums"]; A = FP["activations"]
    # Extract the errors via chain rule including last layer up until the first hidden layer (excludes the input layer)
    dC = δC(ŷ, y) # cost derivative at the output layer
    dA = δF.(S[end]) # activation derivative at the output layer
    ϵ = dC .* dA # error for the output layer: element-wise product of the cost derivatives and activation derivatives
    Δ::Vector{CuArray{T}} = [ϵ]
    for i in 1:(length(S)-1)
        # i = 1
        dC = W[end-(i-1)]' * Δ[end] # back-propagated (notice the transposed weights) cost derivative at the current layer
        dA = δF.(S[end-i]) # activation derivative at the current layer
        ϵ = dC .* dA # error for the current layer: element-wise product of the cost derivatives and activation derivatives
        push!(Δ, ϵ)
    end
    # Calculate the gradients per layer
    Δ = reverse(Δ)
    ∇W::Vector{CuArray{T}} = []
    ∇B::Vector{CuArray{T}} = []
    for i in eachindex(Δ)
        # i = 1
        push!(∇W, Δ[i] * A[i]') # outer-product of the error in hidden layer 1 (l_1 x n) and the transpose of the activation at 1 layer below (n x l_0) to yield a gradient matrix corresponding to the weights matrix (l_1 x l_0)
        push!(∇B, view(sum(Δ[i], dims=2), 1:size(Δ[i], 1), 1)) # sum-up the errors across n samples in the current hidden layer to calculate the gradients for the bias
    end
    Dict(
        "∇W" => ∇W, 
        "∇B" => ∇B, 
    )
end



end