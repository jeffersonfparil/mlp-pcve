function init(
    X::CuArray{T, 2};
    n_hidden_layers::Int64=2,
    n_hidden_nodes::Vector{Int64}=[x * 100 for x in 1:n_hidden_layers],
    F::Function=relu,
    δF::Function=relu_derivative,
    C::Function=MSE,
    δC::Function=MSE_derivative,
)::Network{T} where T <: AbstractFloat
    # X = CuArray{T, 2}(rand(Bool, 25, 1_000)); n_hidden_layers::Int64=2; n_hidden_nodes::Vector{Int64}=[x * 100 for x in 1:n_hidden_layers]; F::Function=relu; δF::Function=relu_derivative; C::Function=MSE; δC::Function=MSE_derivative
    p, n = size(X)
    n_total_layers = 1 + n_hidden_layers + 1
    n_total_nodes = vcat(collect(p), n_hidden_nodes, 1)
    W = []
    b = []
    for i in 1:(n_total_layers-1)
        push!(W, CUDA.randn(T, n_total_nodes[i+1], n_total_nodes[i]))
        push!(b, CUDA.randn(T, n_total_nodes[i+1]))
    end
    # [size(x) for x in W]
    # [size(x) for x in b]
    ŷ = CuArray{T, 2}(zeros(1, n))
    S = vcat([deepcopy(X)], [CuArray{T, 2}(zeros(size(W[i], 1), size(b[i], 1))) for i in 1:(n_total_layers-2)])
    A = vcat([deepcopy(X)], [CuArray{T, 2}(zeros(size(W[i], 1), size(b[i], 1))) for i in 1:(n_total_layers-2)])
    # [size(x) for x in S]
    # [size(x) for x in A]
    ∇W = [CuArray{T, 2}(zeros(size(x))) for x in W]
    ∇b = [CuArray{T, 1}(zeros(size(x))) for x in b]
    Network{T}(n_hidden_layers, n_hidden_nodes, W, b, ŷ, S, A, ∇W, ∇b, F, δF, C, δC)
end

function forwardpass!(Ω::Network{T})::Nothing where T <:AbstractFloat
    # T = Float32; X = CuArray{T, 2}(rand(Bool, 1_000, 25)); Ω = init(X)
    for i in 1:Ω.n_hidden_layers
        # i = 1
        Ω.S[i] = (Ω.W[i] * Ω.A[i]) .+ Ω.b[i]
        Ω.A[i+1] = Ω.F.(Ω.S[i])
    end
    Ω.S[end] = (Ω.W[end] * Ω.A[end]) .+ Ω.b[end]
    Ω.ŷ .= deepcopy(Ω.S[end])
    nothing
end

function backpropagation!(Ω::Network{T}, y::CuArray{T, 2})::Nothing where T <: AbstractFloat
    # T = Float32; X = CuArray{T, 2}(rand(Bool, 1_000, 25)); Ω = init(X); y = CUDA.randn(1, size(X, 2))
    # Extract the errors via chain rule including last layer up until the first hidden layer (excludes the input layer)
    dC = Ω.δC(Ω.ŷ, y) # cost derivative at the output layer
    # dA = Ω.δF.(Ω.S[end]) # activation derivative at the output layer
    # ϵ = dC .* dA # error for the output layer: element-wise product of the cost derivatives and activation derivatives
    ϵ = dC
    Δ::Vector{CuArray{T, 2}} = [ϵ]
    for i in 1:Ω.n_hidden_layers
        # i = 1
        dC = Ω.W[end-(i-1)]' * Δ[end] # back-propagated (notice the transposed weights) cost derivative at the current layer
        dA = Ω.δF.(Ω.S[end-i]) # activation derivative at the current layer
        ϵ = dC .* dA # error for the current layer: element-wise product of the cost derivatives and activation derivatives
        push!(Δ, ϵ)
    end
    # Calculate the gradients per layer
    Δ = reverse(Δ)
    for i in eachindex(Δ)
        # i = 2
        Ω.∇W[i] = Δ[i] * Ω.A[i]' # outer-product of the error in hidden layer 1 (l_1 x n) and the transpose of the activation at 1 layer below (n x l_0) to yield a gradient matrix corresponding to the weights matrix (l_1 x l_0)
        Ω.∇b[i] = view(sum(Δ[i], dims=2), 1:size(Δ[i], 1), 1) # sum-up the errors across n samples in the current hidden layer to calculate the gradients for the bias
    end
    nothing
end
