function init(
    X::CuArray{T, 2};
    n_layers::Int64=2,
    n_hidden_nodes::Vector{Int64}=vcat([x * 100 for x in 1:n_layers], 1),
)::Tuple{Vector{CuArray{T, 2}}, Vector{CuArray{T, 1}}} where T <: AbstractFloat
    # T = Float32; X = CuArray{T, 2}(rand(Bool, 10, 412)); n_layers::Int64=2; n_hidden_nodes::Vector{Int64}=vcat([x * 100 for x in 1:n_layers], 1)
    p, _n = size(X)
    W::Vector{CuArray{T}} = [CUDA.randn(T, n_hidden_nodes[1], p)]
    b::Vector{CuArray{T}} = [CUDA.randn(T, n_hidden_nodes[1])]
    for i in 1:n_layers
        push!(W, CUDA.randn(T, n_hidden_nodes[i+1], n_hidden_nodes[i]))
        push!(b, CUDA.randn(T, n_hidden_nodes[i+1]))
    end
    (W, b)
end

function initgradient(b::Vector{CuArray{T, 1}})::Vector{CuArray{T, 1}} where T <: AbstractFloat
    [CuArray{T, 1}(zeros(size(x))) for x in b]
end

function initgradient(W::Vector{CuArray{T, 2}})::Vector{CuArray{T, 2}} where T <: AbstractFloat
    [CuArray{T, 2}(zeros(size(x))) for x in W]
end

function forwardpass(;
    X::CuArray{T, 2},
    W::Vector{CuArray{T, 2}},
    b::Vector{CuArray{T, 1}},
    F::Function,
)::Dict{String, Union{CuArray{T, 2}, Vector{CuArray{T, 2}}}} where T <:AbstractFloat
    S::Vector{CuArray{T, 2}} = []
    A::Vector{CuArray{T, 2}} = [X]
    for i in 1:(length(W)-1)
        # i = 1
        s = (W[i] * A[i]) .+ b[i]
        push!(S, s)
        push!(A, F.(s))
    end
    # Output layer
    s = (W[end] * A[end]) .+ b[end]
    push!(S, s)
    push!(A, s)
    # Output
    Dict(
        "ŷ" => A[end], # prediction
        "S" => S, # weighted sums
        "A" => A[1:(end-1)], # A (excludes the input layer)
    )
end

function backpropagation!(
    ∇W::Vector{CuArray{T, 2}},
    ∇b::Vector{CuArray{T, 1}};
    y::CuArray{T, 2}, 
    ŷ::CuArray{T, 2},
    W::Vector{CuArray{T, 2}},
    S::Vector{CuArray{T, 2}},
    A::Vector{CuArray{T, 2}},
    δC::Function,
    δF::Function,
)::Nothing where T <: AbstractFloat
    # Extract the errors via chain rule including last layer up until the first hidden layer (excludes the input layer)
    dC = δC(ŷ, y) # cost derivative at the output layer
    # dA = δF.(S[end]) # activation derivative at the output layer
    # ϵ = dC .* dA # error for the output layer: element-wise product of the cost derivatives and activation derivatives
    ϵ = dC
    Δ::Vector{CuArray{T, 2}} = [ϵ]
    for i in 1:(length(S)-1)
        # i = 1
        dC = W[end-(i-1)]' * Δ[end] # back-propagated (notice the transposed weights) cost derivative at the current layer
        dA = δF.(S[end-i]) # activation derivative at the current layer
        ϵ = dC .* dA # error for the current layer: element-wise product of the cost derivatives and activation derivatives
        push!(Δ, ϵ)
    end
    # Calculate the gradients per layer
    Δ = reverse(Δ)
    for i in eachindex(Δ)
        # i = 1
        ∇W[i] = Δ[i] * A[i]' # outer-product of the error in hidden layer 1 (l_1 x n) and the transpose of the activation at 1 layer below (n x l_0) to yield a gradient matrix corresponding to the weights matrix (l_1 x l_0)
        ∇b[i] = view(sum(Δ[i], dims=2), 1:size(Δ[i], 1), 1) # sum-up the errors across n samples in the current hidden layer to calculate the gradients for the bias
    end
    nothing
end
