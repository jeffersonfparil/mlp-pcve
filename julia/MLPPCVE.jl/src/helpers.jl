struct Network{T}
    n_hidden_layers::Int64 # number of hidden layers
    n_hidden_nodes::Vector{Int64} # number of nodes per hidden layer
    W::Vector{CuArray{T, 2}} # weights
    b::Vector{CuArray{T, 1}} # biases
    ŷ::CuArray{T, 2} # predictions
    S::Vector{CuArray{T, 2}} # summed weights (i.e. prior to activation function)
    A::Vector{CuArray{T, 2}} # activation function output including the input layer as the first element
    ∇W::Vector{CuArray{T, 2}} # gradients of the weights
    ∇b::Vector{CuArray{T, 1}} # gradients of the biases
    F::Function # activation function
    ∂F::Function # derivative of the activation function
    C::Function # cost function
    ∂C::Function # derivative of the cost function
end

function init(
    X::CuArray{T, 2};
    n_hidden_layers::Int64=2,
    n_hidden_nodes::Vector{Int64}=repeat([256], n_hidden_layers),
    F::Function=relu,
    ∂F::Function=relu_derivative,
    C::Function=MSE,
    ∂C::Function=MSE_derivative,
)::Network{T} where T <: AbstractFloat
    # X = CuArray{T, 2}(rand(Bool, 25, 1_000)); n_hidden_layers::Int64=2; n_hidden_nodes::Vector{Int64}=repeat([256], n_hidden_layers); F::Function=relu; ∂F::Function=relu_derivative; C::Function=MSE; ∂C::Function=MSE_derivative
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
    A = vcat([deepcopy(X)], [CuArray{T, 2}(zeros(size(W[i], 2), n)) for i in 2:(n_total_layers-1)])
    # [size(x) for x in A]
    S = vcat([CuArray{T, 2}(zeros(size(W[i], 1), size(A[i], 2))) for i in 1:(n_total_layers-1)])
    # [size(x) for x in S]
    ∇W = [CuArray{T, 2}(zeros(size(x))) for x in W]
    ∇b = [CuArray{T, 1}(zeros(size(x))) for x in b]
    Network{T}(n_hidden_layers, n_hidden_nodes, W, b, ŷ, S, A, ∇W, ∇b, F, ∂F, C, ∂C)
end

function mean(X::Matrix{T}; dims::Int64=1)::Matrix{T} where T <: AbstractFloat
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") : nothing
    n = size(X, dims)
    sum(X, dims=dims) ./ n
end

function var(X::Matrix{T}; dims::Int64=1)::Matrix{T} where T <: AbstractFloat
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") : nothing
    n = size(X, dims)
    μ = sum(X, dims=dims) ./ n
    sum((X .- μ).^2, dims=dims) ./ (n-1)
end

function std(X::Matrix{T}; dims::Int64=1)::Matrix{T} where T <: AbstractFloat
    sqrt.(var(X, dims=dims))
end

function cov(X::Matrix{T}, Y::Matrix{T}; dims::Int64=1)::Matrix{T} where T <: AbstractFloat
    # X = rand(10, 4); Y = rand(10, 4); dims = 2
    size(X) != size(Y) ? throw("The dimensions of the input matrices are not the same") : nothing
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") : nothing
    n = size(X, dims)
    μ_X = sum(X, dims=dims) ./ n
    μ_Y = sum(Y, dims=dims) ./ n
    sum((X .- μ_X).*(Y .- μ_Y), dims=dims) ./ (n-1)
end

function sampleNormal(n::Int64; μ::T=0.0, σ::T=1.00)::Vector{T} where T <: AbstractFloat
    # n = 100; μ = 0.0; σ = 1.0;
    U = rand(n)
    V = rand(n)
    z = sqrt.(-2.00 .* log.(U)) .* cos.(2.00*π .* V)
    T.((z .* σ) .+ μ)
end

function drawreplacenot(N::T, n::T)::Vector{T} where T <:Integer
    # N=1_000; n=12; T = Int64
    if N < n
        throw("Cannot draw $n samples from $N total elements.")
    end
    idx::Vector{T} = []
    m = n
    while m > 0
        append!(idx, T.(ceil.(N*rand(Float64, m))))
        idx = unique(idx)
        m = n - length(idx)
    end
    idx
end

function metrics(
    ŷ::CuArray{T, 2},
    y::CuArray{T, 2},
)::Dict{String, T} where T <: AbstractFloat
    # T::Type = Float32
    # Xy = simulate(T=T)
    # Ω = init(Xy["X"]); forwardpass!(Ω)
    # y = Xy["y"]
    @assert length(ŷ) == length(y)
    n = length(y)
    ϵ = ŷ .- y
    mse = sum(ϵ.^2) / n
    rmse = sqrt(mse)
    mae = sum(abs.(ϵ)) / n
    σ_ŷy = cov(Matrix(ŷ), Matrix(y), dims=2)[1,1]
    σ_ŷ = std(Matrix(ŷ), dims=2)[1,1]
    σ_y = std(Matrix(y), dims=2)[1,1]
    μ_ŷ = mean(Matrix(ŷ), dims=2)[1,1]
    μ_y = mean(Matrix(y), dims=2)[1,1]
    ρ = σ_ŷy / (σ_ŷ*σ_y)
    ρ_lin = (2 * ρ * σ_y * σ_ŷ) / (σ_y^2 + σ_ŷ^2 + (μ_y - μ_ŷ)^2)
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
        "ρ_lin"  => ρ_lin,
    )
end

function simulate(;
    T::Type = Float32,
    seed::Int64 = 42,
    n::Int64 = 1_000,
    p::Int64 = 10,
    l::Int64 = 5,
    F::Function = relu,
    phen_type::String = ["random", "linear", "non-linear"][3],
    normalise_X::Bool = true,
    normalise_y::Bool = true,
    h²::Float64 = 0.75,
    p_unobserved::Int64 = 1,
)::Dict{String, CuArray{T, 2}}
    # seed = 42; n = 1_000; p = 10; l = 5; phen_type = "non-linear"; p_unobserved = 1; h² = 0.75; normalise_X = true; normalise_y = true
    Random.seed!(seed); CUDA.seed!(seed); 
    X = rand(Bool, n, p)
    ϕ = if phen_type == "random"
        rand(n, 1)
    elseif phen_type == "linear"
        β = rand(p, 1)
        σ²g = var(X*β, dims=1)[1, 1]
        σ²e = σ²g*((1/h²) - 1.00)
        e = sampleNormal(n, μ=T(0.00), σ=T(sqrt(σ²e)))
        (X*β) .+ e
    elseif phen_type == "non-linear"
        X_true = CuArray{T, 2}(Matrix(hcat(X, rand(Bool, n, p_unobserved))'))
        Ω = init(X_true, n_hidden_layers=l, F=F)
        forwardpass!(Ω)
        y = Matrix(Ω.ŷ')
        y ./ 10.0^(length(string(maximum(abs.(y)))))
    else
        throw("Please use `phen_type=\"random\"` or `phen_type=\"linear\"` or `phen_type=\"non-linear\"`")
    end
    X = normalise_X ? (X .- mean(T.(X), dims=1)) ./ std(T.(X), dims=1) : X
    ϕ = normalise_y ? (ϕ .- mean(T.(ϕ), dims=1)) ./ std(T.(ϕ), dims=1) : X
    X = CuArray{T, 2}(Matrix(X'))
    y = CuArray{T, 2}(reshape(ϕ, (1, n)))
    Dict(
        "X" => X, 
        "y" => y,
    )
end
