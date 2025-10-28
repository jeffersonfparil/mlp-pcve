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
function sampleNormal(
    n::Int64;
    μ::T = 0.0,
    σ::T = 1.00,
    seed::Int64 = 42,
)::Vector{T} where {T<:AbstractFloat}
    # n = 100; μ = 0.0; σ = 1.0;
    rng = Random.seed!(seed)
    U = rand(rng, n)
    V = rand(rng, n)
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
A struct representing a feed-forward neural network.

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

# Note
See `init(::CuArray{T,2}; ...)` function for network initialization.
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
    init(
        X::CuArray{T,2};
        n_hidden_layers::Int64 = 2,
        n_hidden_nodes::Vector{Int64} = repeat([256], n_hidden_layers),
        dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
        F::Function = relu,
        ∂F::Function = relu_derivative,
        C::Function = MSE,
        ∂C::Function = MSE_derivative,
        y::Union{Nothing,CuArray{T,2}} = nothing,
        seed::Int64 = 42,
    )::Network{T} where {T<:AbstractFloat}

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
- `y::Union{Nothing, CuArray{T,2}}=nothing`: Optional target values for OLS initialization.
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
    y::Union{Nothing,CuArray{T,2}} = nothing,
    seed::Int64 = 42,
)::Network{T} where {T<:AbstractFloat}
    # T = Float32; X = CUDA.randn(1_000, 100); n_hidden_layers::Int64=2; n_hidden_nodes::Vector{Int64}=repeat([256], n_hidden_layers); dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers); F::Function=relu; ∂F::Function=relu_derivative; C::Function=MSE; ∂C::Function=MSE_derivative; seed::Int64 = 42
    # T = Float32; X = CUDA.randn(1_000, 100); n_hidden_layers::Int64=0; n_hidden_nodes::Vector{Int64}=repeat([0], n_hidden_layers); dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers); F::Function=relu; ∂F::Function=relu_derivative; C::Function=MSE; ∂C::Function=MSE_derivative; seed::Int64 = 42
    # y::Union{Nothing, CuArray{T,2}} = nothing
    # y::Union{Nothing, CuArray{T,2}} = CUDA.randn(1, 100)
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
        w = if !isnothing(y) && (i == 1)
            # Force the first layer to have the OLS solution
            β = pinv(X * X') * (X*y')[:, 1]
            CuArray{T,2}(reshape(repeat(β, outer = n_i), p_i, n_i)')
        else
            CuArray{T,2}(reshape(sampleNormal(n_i * p_i, μ = T(0.0), σ = T(0.1)), n_i, p_i))
        end
        push!(W, w)
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
    Network{T}(
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

# Find the factors of N which results in the closest positive integer of the square-root
function squarest(N::Int64)::Tuple{Int64, Int64}
    selections = collect(1:Int(ceil(sqrt(N))))
    n_rows = selections[[N % x for x in selections].==0.0][end]
    n_cols = Int(N / n_rows)
    (n_rows, n_cols)
end

"""
    levenshteindistance(a::String, b::String)::Int64

Calculate the Levenshtein distance (edit distance) between two strings.

The Levenshtein distance is a measure of the minimum number of single-character edits 
(insertions, deletions, or substitutions) required to change one string into another.

# Arguments
- `a::String`: First input string
- `b::String`: Second input string

# Returns
- `Int64`: The minimum number of edits needed to transform string `a` into string `b`

# Examples
```jldoctest; setup = :(using GenomicBreedingIO)
julia> levenshteindistance("populations", "populations")
0

julia> levenshteindistance("populations", "poplation")
2

julia> levenshteindistance("populations", "entry")
3
```
"""
function levenshteindistance(a::String, b::String)::Int64
    # a = "populations"; b = "entry";
    n::Int64 = length(a)
    m::Int64 = length(b)
    d::Matrix{Int64} = fill(0, n, m)
    for j = 2:m
        d[1, j] = j - 1
    end
    cost::Int64 = 0
    deletion::Int64 = 0
    insertion::Int64 = 0
    substitution::Int64 = 0
    for j = 2:m
        for i = 2:n
            if a[i] == b[j]
                cost = 0
            else
                cost = 1
            end
            deletion = d[i-1, j] + 1
            insertion = d[i, j-1] + 1
            substitution = d[i-1, j-1] + cost
            d[i, j] = minimum([deletion, insertion, substitution])
        end
    end
    d[end, end]
end

"""
    isfuzzymatch(a::String, b::String; threshold::Float64=0.3)::Bool

Determines if two strings approximately match each other using Levenshtein distance.

The function compares two strings and returns `true` if they are considered similar enough
based on the Levenshtein edit distance and a threshold value. The threshold is applied as
a fraction of the length of the shorter string. Additionally, the function normalizes specific
string inputs (e.g., `"#chr"` is replaced with `"chrom"`) before comparison.

# Arguments
- `a::String`: First string to compare
- `b::String`: Second string to compare
- `threshold::Float64=0.3`: Maximum allowed edit distance as a fraction of the shorter string length

# Returns
- `Bool`: `true` if the strings match within the threshold, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingIO)
julia> isfuzzymatch("populations", "populations")
true

julia> isfuzzymatch("populations", "poplation")
true

julia> isfuzzymatch("populations", "entry")
false
```
"""
function isfuzzymatch(a::String, b::String; threshold::Float64 = 0.3)::Bool
    # a = "populations"; b = "populatins"; threshold = 0.3;
    a = if (a == "#chr")
        "chrom"
    else
        a
    end
    b = if (b == "#chr")
        "chrom"
    else
        b
    end
    n::Int64 = length(a)
    m::Int64 = length(b)
    dist::Int64 = levenshteindistance(a, b)
    dist < Int64(round(minimum([n, m]) * threshold))
end

mutable struct Trial
    X::Matrix{Float64}
    Y::Matrix{Float64}
    A::Matrix{String}
    labels_X::Vector{String}
    labels_Y::Vector{String}
    labels_A::Vector{String}
end

# TODO: add random seed for repeatability

# Simulate explanatory variables
# trial = inittrial()
function inittrial(;
    n_entries::Int64 = 120,
    n_populations::Int64 = 3,
    n_replications::Int64 = 3,
    n_plots::Int64 = n_entries * n_replications,
    n_blocks::Int64 = squarest(n_plots)[1],
    n_plots_per_block::Int64 = squarest(n_plots)[2],
    n_block_rows::Int64 = squarest(n_blocks)[1],
    n_block_cols::Int64 = squarest(n_blocks)[2],
    n_rows_per_block::Int64 = squarest(n_plots_per_block)[1],
    n_cols_per_block::Int64 = squarest(n_plots_per_block)[2],
    n_rows::Int64 = n_block_rows * n_rows_per_block,
    n_cols::Int64 = n_block_cols * n_cols_per_block,
    years::Vector{String} = ["2023_2024", "2024_2025", "2025_2026"],
    seasons::Vector{String} = ["Autumn", "Winter", "EarlySpring", "LateSpring", "Summer"],
    harvests::Vector{String} = ["1", "2"],
    sites::Vector{String} = ["Control", "HighN", "Drought", "Waterlogging", "HighTemp"],
    entries::Vector{String} = ["entry_$(string(i, pad=length(string(n_entries))))" for i = 1:n_entries],
    populations_cor_entries::Vector{String} = [["population_$(string(i, pad=length(string(n_populations))))" for i = 1:n_populations][Int(ceil(rand() * n_populations))] for _ = 1:n_entries],
    populations::Vector{String} = ["population_$(string(i, pad=length(string(n_populations))))" for i = 1:n_populations],
    replications::Vector{String} = ["replication_$(string(i, pad=length(string(n_replications))))" for i = 1:n_replications],
    blocks::Vector{String} = ["block_$(string(i, pad=length(string(n_blocks))))" for i = 1:n_blocks],
    rows::Vector{String} = ["row_$(string(i, pad=length(string(n_rows))))" for i = 1:n_rows],
    cols::Vector{String} = ["col_$(string(i, pad=length(string(n_cols))))" for i = 1:n_cols],
)::Trial
    if (n_rows * n_cols) != n_plots
        throw("The number of rows * cols must equal the number of plots.")
    end
    labels_A::Vector{String} = [
        "years",
        "seasons",
        "harvests",
        "sites",
        "replications",
        "entries",
        "populations",
        "blocks",
        "rows",
        "cols",
    ]
    explanatory_variables = Dict(
        "years" => years,
        "seasons" => seasons,
        "harvests" => harvests,
        "sites" => sites,
        "replications" => replications,
        "entries" => entries,
        "populations" => populations,
        "blocks" => blocks,
        "rows" => rows,
        "cols" => cols,
    )
    labels_X::Vector{String} = []
    for k in labels_A
        v = explanatory_variables[k]
        [push!(labels_X, string(k, "|", x)) for x in v]
    end
    n =
        length(years) *
        length(seasons) *
        length(harvests) *
        length(sites) *
        length(entries) *
        length(replications)
    p = length(labels_X)
    X = zeros(Float64, n, p)
    idx_row = 1
    for idx_years in findall(.!isnothing.(match.(Regex("^years"), labels_X)))
        for idx_seasons in findall(.!isnothing.(match.(Regex("^seasons"), labels_X)))
            for idx_harvests in findall(.!isnothing.(match.(Regex("^harvests"), labels_X)))
                for idx_sites in findall(.!isnothing.(match.(Regex("^sites"), labels_X)))
                    # Replications, entries, populations and the higher order grouping variables above
                    idx_row_sub = idx_row
                    for idx_replications in
                        findall(.!isnothing.(match.(Regex("^replications"), labels_X)))
                        for idx_entries in findall(.!isnothing.(match.(Regex("^entries"), labels_X)))
                            # println("idx_row=$idx_row; idx_row_sub=$idx_row_sub; idx_years=$idx_years; idx_seasons=$idx_seasons; idx_harvests=$idx_harvests; idx_sites=$idx_sites; idx_replications=$idx_replications; idx_entries=$idx_entries")
                            idx_replications_last = maximum(
                                findall(
                                    .!isnothing.(match.(Regex("^replications"), labels_X)),
                                ),
                            )
                            idx_entries_last = maximum(
                                findall(.!isnothing.(match.(Regex("^entries"), labels_X))),
                            )
                            idx_populations =
                                idx_entries_last + findall(
                                    populations .==
                                    populations_cor_entries[idx_entries-idx_replications_last],
                                )[1]
                            idx_cols = [
                                idx_years,
                                idx_seasons,
                                idx_harvests,
                                idx_sites,
                                idx_replications,
                                idx_entries,
                                idx_populations,
                            ]
                            X[idx_row, idx_cols] .= 1.0
                            idx_row += 1
                        end
                    end
                    # Blocks
                    idx_row = idx_row_sub
                    rep_blocks = Int64(
                        length(replications) *
                        length(entries) /
                        length(blocks),
                    )
                    for idx_cols in repeat(
                        findall(.!isnothing.(match.(Regex("^blocks"), labels_X))),
                        inner = rep_blocks,
                    )
                        X[idx_row, idx_cols] = 1.0
                        idx_row += 1
                    end
                    # Rows
                    idx_row = idx_row_sub
                    rep_rows = Int64(
                        length(replications) *
                        length(entries) /
                        length(rows),
                    )
                    for idx_cols in repeat(
                        findall(.!isnothing.(match.(Regex("^rows"), labels_X))),
                        inner = rep_rows,
                    )
                        X[idx_row, idx_cols] = 1.0
                        idx_row += 1
                    end
                    # Cols
                    idx_row = idx_row_sub
                    rep_cols = Int64(
                        length(replications) *
                        length(entries) /
                        length(cols),
                    )
                    for idx_cols in repeat(
                        findall(.!isnothing.(match.(Regex("^cols"), labels_X))),
                        inner = rep_cols,
                    )
                        X[idx_row, idx_cols] = 1.0
                        idx_row += 1
                    end
                end
            end
        end
    end
    # Design matrix to human-readable string matrix
    k = length(labels_A)
    A::Matrix{String} = reshape(repeat([""], n * k), n, k)
    for (k, v) in explanatory_variables
        # k = Symbol(String.(keys(explanatory_variables))[1]); v = explanatory_variables[k]
        # println("k=$k; v=$v")
        idx_A = findfirst(labels_A .== String(k))
        idx_X = findall(.!isnothing.(match.(Regex(string(k)), labels_X)))
        A[:, idx_A] = [v[findfirst(X[i, idx_X] .== 1.0)] for i = 1:n]
    end
    # Output
    Trial(
        X,
        fill(NaN64, 1, 1),
        A,
        labels_X,
        [""],
        labels_A,
    )
end

# Linear model
# trial = inittrial();
# simulatelineartrait!(trial)
function simulatelineartrait!(
    trial::Trial;
    linear_effects = Dict(
        "years" => (μ = 0.0, σ = 1.0),
        "seasons" => (μ = 0.0, σ = 1.0),
        "harvests" => (μ = 0.0, σ = 1.0),
        "sites" => (μ = 0.0, σ = 1.0),
        "entries" => (μ = 0.0, σ = 1.0),
        "populations" => (μ = 0.0, σ = 1.0),
        "replications" => (μ = 0.0, σ = 0.01),
        "blocks" => (μ = 0.0, σ = 0.01),
        "rows" => (μ = 0.0, σ = 0.01),
        "cols" => (μ = 0.0, σ = 0.01),
    ),
    seed::Int64 = 42,
)::Nothing
    n, p = size(trial.X)
    β::Vector{Float64} = zeros(Float64, p)
    for (k, v) in linear_effects
        # k = String.(keys(linear_effects))[1]; v = linear_effects[k]
        idx = findall(.!isnothing.(match.(Regex("^$(k)"), trial.labels_X)))
        β[idx] = sampleNormal(
            length(idx),
            μ = linear_effects[k].μ,
            σ = linear_effects[k].σ,
            seed = seed,
        )
    end
    y = trial.X * β
    y = (y .- mean(hcat(y))) ./ std(hcat(y))
    if (size(trial.Y) == (1,1)) && isnan(trial.Y[1,1])
        trial.Y = y
        trial.labels_Y = ["tmp_1"]
    else
        trial.Y = hcat(trial.Y, y)
        push!(trial.labels_Y, "tmp_$(size(trial.Y, 2))")
    end
    nothing
end

# Non-linear model
# trial = inittrial();
# simulatenonlineartrait!(trial)
function simulatenonlineartrait!(
    trial::Trial;
    n_hidden_layers::Int64 = 5,
    F::Function = relu,
    dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
    seed::Int64 = 123,
)::Nothing
    Ω = init(
        CuArray(Matrix(trial.X')),
        n_hidden_layers = n_hidden_layers,
        F = F,
        dropout_rates = dropout_rates,
        seed = seed,
    )
    rng = CUDA.seed!(seed)
    for i = 1:Ω.n_hidden_layers
        # i = 1
        Ω.W[i] = CUDA.randn(size(Ω.W[i], 1), size(Ω.W[i], 2))
        Ω.b[i] = CUDA.randn(size(Ω.b[i], 1))
    end
    forwardpass!(Ω)
    y = Matrix(Ω.ŷ')
    y = (y .- mean(y)) ./ std(y)
    if (size(trial.Y) == (1,1)) && isnan(trial.Y[1,1])
        trial.Y = y
        trial.labels_Y = ["tmp_1"]
    else
        trial.Y = hcat(trial.Y, y)
        push!(trial.labels_Y, "tmp_$(size(trial.Y, 2))")
    end
    nothing
end

# trial = simulatetrial()
function simulatetrial(; 
    n_entries::Int64 = 120,
    n_populations::Int64 = 3,
    n_replications::Int64 = 3,
    n_plots::Int64 = n_entries * n_replications,
    n_blocks::Int64 = squarest(n_plots)[1],
    n_plots_per_block::Int64 = squarest(n_plots)[2],
    n_block_rows::Int64 = squarest(n_blocks)[1],
    n_block_cols::Int64 = squarest(n_blocks)[2],
    n_rows_per_block::Int64 = squarest(n_plots_per_block)[1],
    n_cols_per_block::Int64 = squarest(n_plots_per_block)[2],
    n_rows::Int64 = n_block_rows * n_rows_per_block,
    n_cols::Int64 = n_block_cols * n_cols_per_block,
    years::Vector{String} = ["2023_2024", "2024_2025", "2025_2026"],
    seasons::Vector{String} = ["Autumn", "Winter", "EarlySpring", "LateSpring", "Summer"],
    harvests::Vector{String} = ["1", "2"],
    sites::Vector{String} = ["Control", "HighN", "Drought", "Waterlogging", "HighTemp"],
    entries::Vector{String} = ["entry_$(string(i, pad=length(string(n_entries))))" for i = 1:n_entries],
    populations_cor_entries::Vector{String} = [["population_$(string(i, pad=length(string(n_populations))))" for i = 1:n_populations][Int(ceil(rand() * n_populations))] for _ = 1:n_entries],
    populations::Vector{String} = ["population_$(string(i, pad=length(string(n_populations))))" for i = 1:n_populations],
    replications::Vector{String} = ["replication_$(string(i, pad=length(string(n_replications))))" for i = 1:n_replications],
    blocks::Vector{String} = ["block_$(string(i, pad=length(string(n_blocks))))" for i = 1:n_blocks],
    rows::Vector{String} = ["row_$(string(i, pad=length(string(n_rows))))" for i = 1:n_rows],
    cols::Vector{String} = ["col_$(string(i, pad=length(string(n_cols))))" for i = 1:n_cols],
    trait_specs::Dict{String} = Dict(
        "trait_1" => Dict(
            "linear" => true,
            "linear_effects" => Dict(
                "years" => (μ = 0.0, σ = 1.0),
                "seasons" => (μ = 0.0, σ = 1.0),
                "harvests" => (μ = 0.0, σ = 1.0),
                "sites" => (μ = 0.0, σ = 1.0),
                "entries" => (μ = 0.0, σ = 1.0),
                "populations" => (μ = 0.0, σ = 1.0),
                "replications" => (μ = 0.0, σ = 0.001),
                "blocks" => (μ = 0.0, σ = 0.001),
                "rows" => (μ = 0.0, σ = 0.001),
                "cols" => (μ = 0.0, σ = 0.001),
            ),
        ),
        "trait_2" => Dict(
            "linear" => false,
            "n_hidden_layers" => 1,
            "F" => relu,
            "dropout_rates" => repeat([0.0], 1),
        ),
        "trait_3" => Dict(
            "linear" => false,
            "n_hidden_layers" => 2,
            "F" => leakyrelu,
            "dropout_rates" => repeat([0.0], 2),
        ),
        "trait_4" => Dict(
            "linear" => false,
            "n_hidden_layers" => 3,
            "F" => leakyrelu,
            "dropout_rates" => repeat([0.0], 3),
        ),
        "trait_5" => Dict(
            "linear" => false,
            "n_hidden_layers" => 5,
            "F" => leakyrelu,
            "dropout_rates" => repeat([0.0], 5),
        ),
    ),
    verbose::Bool = true,
)::Trial
    trial = inittrial(
        n_entries=n_entries,
        n_populations=n_populations,
        n_replications=n_replications,
        n_plots=n_plots,
        n_blocks=n_blocks,
        n_plots_per_block=n_plots_per_block,
        n_block_rows=n_block_rows,
        n_block_cols=n_block_cols,
        n_rows_per_block=n_rows_per_block,
        n_cols_per_block=n_cols_per_block,
        n_rows=n_rows,
        n_cols=n_cols,
        years=years,
        seasons=seasons,
        harvests=harvests,
        sites=sites,
        entries=entries,
        populations_cor_entries=populations_cor_entries,
        populations=populations,
        replications=replications,
        blocks=blocks,
        rows=rows,
        cols=cols,
    )
    for (trait, specs) in sort(trait_specs)
        # trait = string.(keys(sort(trait_specs)))[1]; specs = trait_specs[trait]
        if verbose
            println("Simulating $trait ...")
        end
        if specs["linear"]
            simulatelineartrait!(
                trial,
                linear_effects = specs["linear_effects"],
            )
        else
            simulatenonlineartrait!(
                trial,
                n_hidden_layers = specs["n_hidden_layers"],
                F = specs["F"],
                dropout_rates = specs["dropout_rates"],
            )
        end
        trial.labels_Y[end] = trait
    end
    trial
end