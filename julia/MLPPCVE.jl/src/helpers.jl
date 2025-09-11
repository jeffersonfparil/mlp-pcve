"""
A struct representing a feedforward neural network.

# Fields
- `n_hidden_layers::Int64`: Number of hidden layers.
- `n_hidden_nodes::Vector{Int64}`: Number of nodes in each hidden layer.
- `dropout_rates::Vector{T}`: Dropout rates for each hidden layer.
- `W::Vector{CuArray{T, 2}}`: Weight matrices for each layer.
- `b::Vector{CuArray{T, 1}}`: Bias vectors for each layer.
- `ŷ::CuArray{T, 2}`: Output predictions of the network.
- `S::Vector{CuArray{T, 2}}`: Pre-activation values (weighted sums).
- `A::Vector{CuArray{T, 2}}`: Activations for each layer including input.
- `∇W::Vector{CuArray{T, 2}}`: Gradients of weights.
- `∇b::Vector{CuArray{T, 1}}`: Gradients of biases.
- `F::Function`: Activation function.
- `∂F::Function`: Derivative of the activation function.
- `C::Function`: Cost function.
- `∂C::Function`: Derivative of the cost function.
- `seed::Int64`: Random seed for reproducibility.
"""
struct Network{T}
    n_hidden_layers::Int64 # number of hidden layers
    n_hidden_nodes::Vector{Int64} # number of nodes per hidden layer
    dropout_rates::Vector{T} # soft dropout rates per hidden layer
    W::Vector{CuArray{T,2}} # weights
    b::Vector{CuArray{T,1}} # biases
    ŷ::CuArray{T,2} # predictions
    S::Vector{CuArray{T,2}} # summed weights (i.e. prior to activation function)
    A::Vector{CuArray{T,2}} # activation function output including the input layer as the first element
    ∇W::Vector{CuArray{T,2}} # gradients of the weights
    ∇b::Vector{CuArray{T,1}} # gradients of the biases
    F::Function # activation function
    ∂F::Function # derivative of the activation function
    C::Function # cost function
    ∂C::Function # derivative of the cost function
    seed::Int64 # random seed for dropouts
end


"""
Computes the mean of a matrix along the specified dimension.

# Arguments
- `X::Matrix{T}`: Input matrix.
- `dims::Int64=1`: Dimension to compute the mean over (default: 1)).

# Returns
- `Matrix{T}`: Mean values along the specified dimension.

# Examples
```jldoctest
julia> X = rand(10, 42);

julia> sum(mean(X, dims=1) .> 0.0) == 42
true

julia> sum(mean(X, dims=2) .> 0.0) == 10
true
```
"""
function mean(X::Matrix{T}; dims::Int64 = 1)::Matrix{T} where {T<:AbstractFloat}
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") :
    nothing
    n = size(X, dims)
    sum(X, dims = dims) ./ n
end


"""
Computes the sample variance of a matrix along the specified dimension.

# Arguments
- `X::Matrix{T}`: Input matrix.
- `dims::Int64=1`: Dimension to compute the variance over (default: 1)).

# Returns
- `Matrix{T}`: Variance values along the specified dimension.
"""
function var(X::Matrix{T}; dims::Int64 = 1)::Matrix{T} where {T<:AbstractFloat}
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") :
    nothing
    n = size(X, dims)
    μ = sum(X, dims = dims) ./ n
    sum((X .- μ) .^ 2, dims = dims) ./ (n - 1)
end


"""
Computes the sample standard deviation of a matrix along the specified dimension.

# Arguments
- `X::Matrix{T}`: Input matrix.
- `dims::Int64=1`: Dimension to compute the standard deviation over (default: 1)).

# Returns
- `Matrix{T}`: Standard deviation values.
"""
function std(X::Matrix{T}; dims::Int64 = 1)::Matrix{T} where {T<:AbstractFloat}
    sqrt.(var(X, dims = dims))
end


"""
Computes the sample covariance between two matrices along the specified dimension.

# Arguments
- `X::Matrix{T}`: First input matrix.
- `Y::Matrix{T}`: Second input matrix.
- `dims::Int64=1`: Dimension to compute the covariance over (default: 1)).

# Returns
- `Matrix
"""
function cov(
    X::Matrix{T},
    Y::Matrix{T};
    dims::Int64 = 1,
)::Matrix{T} where {T<:AbstractFloat}
    # X = rand(10, 4); Y = rand(10, 4); dims = 2
    size(X) != size(Y) ? throw("The dimensions of the input matrices are not the same") :
    nothing
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") :
    nothing
    n = size(X, dims)
    μ_X = sum(X, dims = dims) ./ n
    μ_Y = sum(Y, dims = dims) ./ n
    sum((X .- μ_X) .* (Y .- μ_Y), dims = dims) ./ (n - 1)
end


"""
Generates samples from a standard normal distribution using the Box-Muller transform.

# Arguments
- `n::Int64`: Number of samples.
- `μ::T=0.0`: Mean of the distribution (default: 0.0).
- `σ::T=1.0`: Standard deviation of the distribution (default: 1.0).

# Returns
- `Vector{T}`: Vector of normally distributed samples.
"""
function sampleNormal(n::Int64; μ::T = 0.0, σ::T = 1.00)::Vector{T} where {T<:AbstractFloat}
    # n = 100; μ = 0.0; σ = 1.0;
    U = rand(n)
    V = rand(n)
    z = sqrt.(-2.00 .* log.(U)) .* cos.(2.00 * π .* V)
    T.((z .* σ) .+ μ)
end


"""
Draws `n` unique samples from `N` elements without replacement.

# Arguments
- `N::T`: Total number of elements.
- `n::T`: Number of samples to draw.

# Returns
- `Vector{T}`: Vector of unique sampled indices.
"""
function drawreplacenot(N::T, n::T)::Vector{T} where {T<:Integer}
    # N=1_000; n=12; T = Int64
    if N < n
        throw("Cannot draw $n samples from $N total elements.")
    end
    idx::Vector{T} = []
    m = n
    while m > 0
        append!(idx, T.(ceil.(N * rand(Float64, m))))
        idx = unique(idx)
        m = n - length(idx)
    end
    idx
end



"""
Initializes a neural network with specified architecture and parameters.

# Arguments
- `X::CuArray{T, 2}`: Input data.
- `n_hidden_layers::Int64=2`: Number of hidden layers (default: 2).
- `n_hidden_nodes::Vector{Int64}=repeat([256], n_hidden_layers)`: Nodes per hidden layer (default: [256, 256]).
- `dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers)`: Dropout rates per layer (default: [0.0, 0.0]).
- `F::Function=relu`: Activation function (default: `relu`).
- `∂F::Function=relu_derivative`: Derivative of activation function (default: `relu_derivative`).
- `C::Function=MSE`: Cost function (default: `MSE`).
- `∂C::Function=MSE_derivative`: Derivative of cost function (default: `MSE_derivative`).
- `seed::Int64=42`: Random seed (default: 42).

# Notes:
- Assumes the output layer has a single node (1-dimensional)

# Returns
- `Network{T}`: Initialized network.
"""
function init(
    X::CuArray{T,2};
    n_hidden_layers::Int64 = 2,
    n_hidden_nodes::Vector{Int64} = repeat([256], n_hidden_layers),
    dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
    F::Function = relu,
    ∂F::Function = relu_derivative,
    C::Function = MSE,
    ∂C::Function = MSE_derivative,
    seed::Int64 = 42,
)::Network{T} where {T<:AbstractFloat}
    # X = CuArray{T, 2}(rand(Bool, 25, 1_000)); n_hidden_layers::Int64=2; n_hidden_nodes::Vector{Int64}=repeat([256], n_hidden_layers); dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers); F::Function=relu; ∂F::Function=relu_derivative; C::Function=MSE; ∂C::Function=MSE_derivative; seed::Int64 = 42
    Random.seed!(seed)
    p, n = size(X)
    n_total_layers = 1 + n_hidden_layers + 1 # input layer + hidden layers + output layer
    n_total_nodes = vcat(collect(p), n_hidden_nodes, 1)
    W = []
    b = []
    for i = 1:(n_total_layers-1)
        # i = 1
        n_i = n_total_nodes[i+1]
        p_i = n_total_nodes[i]
        # Initialise the weights as: w ~ N(0, 0.1) and all biases to zero
        push!(
            W,
            CuArray{T,2}(
                reshape(sampleNormal(n_i * p_i, μ = T(0.0), σ = T(0.1)), n_i, p_i),
            ),
        )
        push!(b, CuArray{T,1}(zeros(n_i)))
        # UnicodePlots.histogram(reshape(Matrix(W[1]), n_i*p_i))
    end
    # [size(x) for x in W]
    # [size(x) for x in b]
    ŷ = CuArray{T,2}(zeros(1, n))
    A = vcat(
        [deepcopy(X)],
        [CuArray{T,2}(zeros(size(W[i], 2), n)) for i = 2:(n_total_layers-1)],
    )
    # [size(x) for x in A]
    S = vcat([
        CuArray{T,2}(zeros(size(W[i], 1), size(A[i], 2))) for i = 1:(n_total_layers-1)
    ])
    # [size(x) for x in S]
    ∇W = [CuArray{T,2}(zeros(size(x))) for x in W]
    ∇b = [CuArray{T,1}(zeros(size(x))) for x in b]
    Ω = Network{T}(
        n_hidden_layers,
        n_hidden_nodes,
        T.(dropout_rates),
        W,
        b,
        ŷ,
        S,
        A,
        ∇W,
        ∇b,
        F,
        ∂F,
        C,
        ∂C,
        seed,
    )
end


"""
Computes evaluation metrics between predicted and true values.

# Arguments
- `ŷ::CuArray{T, 2}`: Predicted values.
- `y::CuArray{T, 2}`: True values.

# Returns
- `Dict{String, T}`: Dictionary of metrics including MSE, RMSE, MAE, correlation, etc.
"""
function metrics(ŷ::CuArray{T,2}, y::CuArray{T,2})::Dict{String,T} where {T<:AbstractFloat}
    # T::Type = Float32
    # Xy = simulate(T=T)
    # Ω = init(Xy["X"]); forwardpass!(Ω)
    # y = Xy["y"]
    # ŷ = Ω.ŷ
    @assert length(ŷ) == length(y)
    n = length(y)
    ϵ = ŷ .- y
    mse = sum(ϵ .^ 2) / n
    rmse = sqrt(mse)
    mae = sum(abs.(ϵ)) / n
    σ_ŷy = cov(Matrix(ŷ), Matrix(y), dims = 2)[1, 1]
    σ_ŷ = std(Matrix(ŷ), dims = 2)[1, 1]
    σ_y = std(Matrix(y), dims = 2)[1, 1]
    μ_ŷ = mean(Matrix(ŷ), dims = 2)[1, 1]
    μ_y = mean(Matrix(y), dims = 2)[1, 1]
    ρ = σ_ŷy / (σ_ŷ * σ_y)
    ρ_lin = (2 * ρ * σ_y * σ_ŷ) / (σ_y^2 + σ_ŷ^2 + (μ_y - μ_ŷ)^2)
    R² = 1.00 - (sum(ϵ .^ 2) / sum((y .- μ_y) .^ 2))
    Dict(
        "mse" => mse,
        "rmse" => rmse,
        "mae" => mae,
        "σ_ŷy" => σ_ŷy,
        "σ_ŷ" => σ_ŷ,
        "σ_y" => σ_y,
        "μ_ŷ" => μ_ŷ,
        "μ_y" => μ_y,
        "ρ" => ρ,
        "ρ_lin" => ρ_lin,
        "R²" => R²,
    )
end



"""
Simulates input and output data for training a neural network.

# Keyword Arguments
- `T::Type=Float32`: Data type (default: `Float32`).
- `seed::Int64=42`: Random seed (default: 42).
- `n::Int64=1000`: Number of samples (default: 1000).
- `p::Int64=10`: Number of features (default: 10).
- `l::Int64=5`: Number of hidden layers (default: 5).
- `d::Float64=0.50`: Dropout rate (default: 0.50).
- `F::Function=relu`: Activation function (default: `relu`).
- `phen_type::String="non-linear"`: Phenotype type ("random", "linear", "non-linear"; default: "non-linear").
- `normalise_X::Bool=true`: Whether to normalize input features (default: true).
- `normalise_y::Bool=true`: Whether to normalize output (default: true).
- `h²::Float64=0.75`: Heritability for phenotype generation (default: 0.75).
- `p_unobserved::Int64=1`: Number of unobserved features (default: 1).

# Returns
- `Dict{String, CuArray{T, 2}}`: Dictionary containing input `X` and output `y`.
"""
function simulate(;
    T::Type = Float32,
    seed::Int64 = 42,
    n::Int64 = 1_000,
    p::Int64 = 10,
    l::Int64 = 5,
    d::Float64 = 0.50,
    F::Function = relu,
    phen_type::String = ["random", "linear", "non-linear"][3],
    normalise_X::Bool = true,
    normalise_y::Bool = true,
    h²::Float64 = 0.75,
    p_unobserved::Int64 = 1,
)::Dict{String,CuArray{T,2}}
    # T=Float32; seed = 42; n = 1_000; p = 10; l = 5; d = 0.50; phen_type = "non-linear"; p_unobserved = 1; h² = 0.75; normalise_X = true; normalise_y = true; F::Function = relu
    Random.seed!(seed)
    CUDA.seed!(seed)
    X = rand(Bool, n, p)
    ϕ = if phen_type == "random"
        rand(n, 1)
    elseif phen_type == "linear"
        β = rand(p, 1)
        σ²g = var(X * β, dims = 1)[1, 1]
        σ²e = σ²g * ((1 / h²) - 1.00)
        e = sampleNormal(n, μ = T(0.00), σ = T(sqrt(σ²e)))
        (X * β) .+ e
    elseif phen_type == "non-linear"
        X_true = CuArray{T,2}(Matrix(hcat(X, rand(Bool, n, p_unobserved))'))
        Ω = init(
            X_true,
            n_hidden_layers = l,
            F = F,
            dropout_rates = repeat([d], l),
            seed = seed,
        )
        for i = 1:Ω.n_hidden_layers
            # i = 1
            Ω.W[i] .= CUDA.randn(size(Ω.W[i], 1), size(Ω.W[i], 2))
            Ω.b[i] .= CUDA.randn(size(Ω.b[i], 1))
        end
        forwardpass!(Ω)
        Matrix(Ω.ŷ')
    else
        throw(
            "Please use `phen_type=\"random\"` or `phen_type=\"linear\"` or `phen_type=\"non-linear\"`",
        )
    end
    X = normalise_X ? (X .- mean(T.(X), dims = 1)) ./ std(T.(X), dims = 1) : X
    ϕ = normalise_y ? (ϕ .- mean(T.(ϕ), dims = 1)) ./ std(T.(ϕ), dims = 1) : X
    X = CuArray{T,2}(Matrix(X'))
    y = CuArray{T,2}(reshape(ϕ, (1, n)))
    Dict("X" => X, "y" => y)
end

# function simulateyieldtrial(;
#     T::Type = Float32,
#     seed::Int64 = 42,
#     n_entries::Int64 = 1_000,
#     n_replications::Int64 = 3,
#     n_sites::Int64 = 3,
#     n_seasons::Int64 = 4,
#     n_years::Int64 = 5,
# )

# end