"""
    splitdata(Xy::Dict{String,CuArray{T,2}}; n_batches::Int64 = 10, seed::Int64 = 42)::Dict{String,CuArray{T,2}} where {T<:AbstractFloat}

Split input data into training batches and a validation set.

# Arguments
- `Xy::Dict{String,CuArray{T,2}}`: Dictionary containing input features 'X' and target values 'y' as CuArrays
- `n_batches::Int64=10`: Number of batches to split the data into (default: 10)
- `seed::Int64=42`: Random seed for reproducibility (default: 42)

# Returns
- `Dict{String,CuArray{T,2}}`: Dictionary containing split data with keys:
    - "X_batch_i", "y_batch_i" for i in 1:(n_batches-1) for training batches
    - "X_validation", "y_validation" for the validation set

# Notes
- The last batch is always used as the validation set
- Data is randomly shuffled before splitting
- Batch sizes are approximately equal

# Throws
- `ArgumentError`: If number of samples is less than or equal to n_batches
- `ArgumentError`: If number of samples is less than 2
"""
function splitdata(
    Xy::Dict{String,CuArray{T,2}};
    n_batches::Int64 = 10, # with the last batch used for validation
    seed::Int64 = 42,
)::Dict{String,CuArray{T,2}} where {T<:AbstractFloat}
    # T::Type = Float32
    # Xy = simulate(T=T)
    # n_batches::Int64 = 10
    # seed::Int64 = 42
    Random.seed!(seed)
    p, n = size(Xy["X"])
    if n <= n_batches
        throw(
            ArgumentError(
                "Number of samples (n=$n) must be greater than the number of batches (n_batches=$n_batches)",
            ),
        )
    end
    if n < 2
        throw(
            ArgumentError(
                "Number of samples (n=$n) must be at least 2 with the first batch used for training and the second for validation",
            ),
        )
    end
    batch_size = Int64(round(n / n_batches))
    idx_rand = shuffle(1:n)
    out::Dict{String,CuArray{T,2}} = Dict()
    for i = 1:n_batches
        # i = 1
        idx = idx_rand[((i-1)*batch_size+1):min(i * batch_size, n)]
        id = if i < n_batches
            "batch_$i"
        else
            "validation"
        end
        out["X_$id"] = CuArray{T,2}(view(Xy["X"], 1:p, idx))
        out["y_$id"] = CuArray{T,2}(view(Xy["y"], 1:1, idx))
    end
    out
end

"""
    predict(Ω::Network{T}, X::CuArray{T,2})::CuArray{T,2} where {T<:AbstractFloat}

Perform forward propagation through the neural network to generate predictions.

# Arguments
- `Ω::Network{T}`: Neural network structure containing weights, biases and activation function
- `X::CuArray{T,2}`: Input data matrix on GPU where T is an AbstractFloat type

# Returns
- `CuArray{T,2}`: Predicted output matrix on GPU

# Details
For networks with no hidden layers (linear regression), applies:
    y = Wx + b

For networks with hidden layers, iteratively applies:
    h_i = F(W_i * h_{i-1} + b_i)
where F is the activation function, and h_0 = x

The final layer always applies a linear transformation without activation:
    y = W_final * h_final + b_final
"""
function predict(Ω::Network{T}, X::CuArray{T,2})::CuArray{T,2} where {T<:AbstractFloat}
    # No hidden layer (i.e. linear regression)
    if Ω.n_hidden_layers == 0
        return (Ω.W[end] * X) .+ Ω.b[end]
    end
    # With hidden layer/s
    a = Ω.F.((Ω.W[1] * X) .+ Ω.b[1])
    for i = 2:Ω.n_hidden_layers
        a = Ω.F.((Ω.W[i] * a) .+ Ω.b[i])
    end
    (Ω.W[end] * a) .+ Ω.b[end]
end

"""
    train(Xy::Dict{String,CuArray{T,2}}; kwargs...)::Dict{String,Union{Network{T},Vector{T},Dict{String,T}}} where {T<:AbstractFloat}

Train a neural network using the provided data and specified parameters.

# Arguments
- `Xy::Dict{String,CuArray{T,2}}`: Dictionary containing input features ("X") and target values ("y") as CuArrays

# Keywords
- `n_batches::Union{Int64,Nothing}=nothing`: Number of batches for training; auto-calculated if nothing
- `fit_full::Bool=false`: If true, fits model on full data without validation set
- `n_hidden_layers::Int64=3`: Number of hidden layers in the network
- `n_hidden_nodes::Vector{Int64}`: Number of nodes in each hidden layer
- `dropout_rates::Vector{Float64}`: Dropout rates for each hidden layer
- `F::Function=relu`: Activation function
- `∂F::Function=relu_derivative`: Derivative of activation function
- `C::Function=MSE`: Cost function
- `∂C::Function=MSE_derivative`: Derivative of cost function
- `n_epochs::Int64=10_000`: Maximum number of training epochs
- `n_burnin_epochs::Int64=100`: Number of initial epochs before early stopping
- `n_patient_epochs::Int64=5`: Number of epochs to wait for improvement before early stopping
- `optimiser::String="Adam"`: Optimization algorithm ("GD", "Adam", or "AdamMax")
- `η::Float64=0.001`: Learning rate
- `β₁::Float64=0.900`: First moment decay rate for Adam/AdamMax
- `β₂::Float64=0.999`: Second moment decay rate for Adam/AdamMax
- `ϵ::Float64=1e-8`: Small constant for numerical stability
- `seed::Int64=42`: Random seed for reproducibility
- `verbose::Bool=true`: Whether to print training progress

# Returns
Dictionary containing:
- `"Ω"`: Trained network
- `"y_true"`: True target values
- `"y_pred"`: Predicted values
- `"loss_training"`: Training loss history
- `"loss_validation"`: Validation loss history
- `"metrics_training"`: Training metrics
- `"metrics_validation"`: Validation metrics
- `"metrics_training_ols"`: OLS training metrics
- `"metrics_validation_ols"`: OLS validation metrics

# Notes
- Input features X should be standardized (μ=0, σ=1)
- Uses GPU acceleration with CUDA
- Implements early stopping based on both training and validation loss
- Includes OLS as a reference model

# Examples
```julia
dl = train(simulate())
dl = train(simulate(), n_hidden_layers=2)
dl = train(simulate(), fit_full=true)
dl = train(simulate(), n_batches=5)
dl = train(simulate(), F=leakyrelu)
dl = train(simulate(), F=tanh)
dl = train(simulate(), F=sigmoid)
```
"""
function train(
    Xy::Dict{String,CuArray{T,2}}; # It it recommended that X be standardised (μ=0, and σ=1)
    n_batches::Union{Int64,Nothing} = nothing, # if nothing, it will be calculated based on available GPU memory
    fit_full::Bool = false, # if true, the model will be fit on the full data (no validation set)
    n_hidden_layers::Int64 = 3,
    n_hidden_nodes::Vector{Int64} = repeat([size(Xy["X"], 1)], n_hidden_layers),
    dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
    F::Function = relu,
    ∂F::Function = relu_derivative,
    C::Function = MSE,
    ∂C::Function = MSE_derivative,
    n_epochs::Int64 = 10_000,
    n_burnin_epochs::Int64 = 100,
    n_patient_epochs::Int64 = 5,
    optimiser::String = ["GD", "Adam", "AdamMax"][2],
    η::Float64 = 0.001,
    β₁::Float64 = 0.900,
    β₂::Float64 = 0.999,
    ϵ::Float64 = 1e-8,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Dict{String,Union{Network{T},Vector{T},Dict{String,T}}} where {T<:AbstractFloat}
    # Xy = simulate(seed=1)
    # # Xy = simulate(phen_type="linear", h²=1.00)
    # # Xy = simulate(n=10_000, l=1_000)
    # n_batches::Union{Int64, Nothing} = nothing
    # fit_full::Bool = true
    # n_hidden_layers::Int64 = 1
    # n_hidden_nodes::Vector{Int64} = repeat([size(Xy["X"], 1)], n_hidden_layers)
    # dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers)
    # F::Function = relu
    # ∂F::Function = relu_derivative
    # C::Function = MSE
    # ∂C::Function = MSE_derivative
    # n_epochs::Int64 = 10_000
    # n_burnin_epochs::Int64 = 100
    # n_patient_epochs::Int64 = 10
    # optimiser::String = ["GD", "Adam", "AdamMax"][2]
    # η::Float64 = 0.001
    # β₁::Float64 = 0.900
    # β₂::Float64 = 0.999
    # ϵ::Float64 = 1e-8
    # seed::Int64 = 42
    # verbose::Bool = true
    # T = typeof(Matrix(Xy["y"])[1,1])
    # Check inputs
    if size(Xy["X"], 2) != size(Xy["y"], 2)
        throw(
            ArgumentError(
                "Number of samples in X ($(size(Xy["X"], 2))) must be equal to the number of samples in y ($(size(Xy["y"], 2)))",
            ),
        )
    end
    # Calculating the optimum number of batches based on available GPU memory
    n_batches = if isnothing(n_batches)
        requirement_bytes = sum([
            prod(size(Xy["X"])),
            4 *
            sum(n_hidden_layers * n_hidden_nodes * size(Xy["X"], 2)) *
            (@allocated T(0.0)),
            2 * sum(n_hidden_layers * n_hidden_nodes * (@allocated T(0.0))),
            size(Xy["X"], 2) * (@allocated T(0.0)),
        ])
        maximum(vcat([2], Int64.(round.(requirement_bytes ./ CUDA.memory_info()))...))
    else
        n_batches
    end
    # Split into validation and training sets and initialise the networks for each as well as the target outputs
    D, n_batches = if !fit_full
        verbose ?
        println("Using $n_batches batches (with the last batch used for validation)") :
        nothing
        if n_batches < 2
            throw(
                ArgumentError(
                    "Number of batches (n_batches=$n_batches) must be at least 2 with the first batch used for training and the second for validation",
                ),
            )
        end
        (splitdata(Xy, n_batches = n_batches, seed = seed), n_batches)
    else
        println(
            "Using $n_batches batches (to fit the model on the full data, i.e. no validation set, set fit_full=true)",
        )
        D = splitdata(Xy, n_batches = n_batches, seed = seed)
        X_tmp = D["X_validation"]
        y_tmp = D["y_validation"]
        delete!(D, "X_validation")
        delete!(D, "y_validation")
        D["X_batch_$n_batches"] = X_tmp
        D["y_batch_$n_batches"] = y_tmp
        (D, n_batches+1)
    end
    # Initialise the model using the training data only
    Ω = init(
        D["X_batch_1"],
        n_hidden_layers = n_hidden_layers,
        n_hidden_nodes = n_hidden_nodes,
        dropout_rates = dropout_rates,
        F = F,
        ∂F = ∂F,
        C = C,
        ∂C = ∂C,
        y = D["y_batch_1"], # OLS initialisation
        seed = seed,
    )
    state::Dict{String,Any} = if optimiser == "GD"
        Dict()
    else
        Dict(
            "β₁" => T(β₁),
            "β₂" => T(β₂),
            "ϵ" => T(ϵ),
            "t" => T(0.0),
            "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
            "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
            "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
            "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
        )
    end
    epochs::Vector{Int64} = []
    loss_training::Vector{T} = []
    loss_validation::Vector{T} = []
    y_true = hcat([D["y_batch_$j"] for j = 1:(n_batches-1)]...)
    for i = 1:n_epochs
        # i = 1
        if verbose
            p = Int(round(100 * i / n_epochs))
            print("\r$(repeat("░", p)) | $p% training")
        end
        for j = 1:(n_batches-1)
            # j = 1
            if n_batches > 2
                Ω.A[1] .= D["X_batch_$j"] # nil cost compared to the forward pass and back-propagation below - so we are keeping the struct as is for now and here in the Julia implementation
            end
            forwardpass!(Ω)
            backpropagation!(Ω, D["y_batch_$j"])
            if optimiser == "GD"
                gradientdescent!(Ω, η = T(η))
            elseif optimiser == "Adam"
                Adam!(Ω, η = T(η), state = state)
            elseif optimiser == "AdamMax"
                AdamMax!(Ω, η = T(η), state = state)
            else
                throw("Invalid optimiser: $optimiser")
            end
        end
        # Metrics for the training and validations sets
        # NOTE: do not use the forward pass because it may have dropouts
        # y_true = hcat([D["y_batch_$j"] for j = 1:(n_batches-1)]...)
        y_pred = hcat([predict(Ω, D["X_batch_$j"]) for j = 1:(n_batches-1)]...)
        Θ_training = metrics(y_pred, y_true) 
        Θ_validation = if "y_validation" ∈ keys(D)
            metrics(predict(Ω, D["X_validation"]), D["y_validation"])
        else
            Dict("mse" => NaN)
        end
        push!(epochs, i)
        push!(loss_training, Θ_training["mse"])
        push!(loss_validation, Θ_validation["mse"])
        # After burn-in epoches: stop early if validation loss does not improve after n_patient_epochs
        if i > n_burnin_epochs
            if (i > n_patient_epochs) &&
               ((loss_validation[end] - loss_validation[(end-n_patient_epochs)+1]) >= 0.0)
                if verbose
                    println(
                        "\nEarly stopping after $(length(loss_training)) epochs because validation loss is not improving for the past $n_patient_epochs epochs",
                    )
                end
                break
            end
            if (i >= n_patient_epochs) &&
               ((loss_training[end] - loss_training[(end-n_patient_epochs)+1]) >= 0.0)
                if verbose
                    println(
                        "\nEarly stopping after $(length(loss_training)) epochs because training loss is not improving for the past $n_patient_epochs epochs",
                    )
                end
                break
            end
        end
    end
    # Memory clean-up
    CUDA.reclaim()
    # if verbose
    #     println("")
    #     CUDA.pool_status()
    # end
    # DL fit
    metrics_training = begin
        y_true = hcat([D["y_batch_$j"] for j = 1:(n_batches-1)]...)
        y_pred = hcat([predict(Ω, D["X_batch_$j"]) for j = 1:(n_batches-1)]...)
        metrics(y_pred, y_true) # NOTE: do not use the forward pass because it may have dropouts
    end
    metrics_validation = if "y_validation" ∈ keys(D)
        metrics(predict(Ω, D["X_validation"]), D["y_validation"])
    else
        Dict("mse" => T(NaN))
    end
    # Ordinary least squares fit as reference
    metrics_training_ols = begin
        A = hcat(
            ones(sum([size(D["X_batch_$i"], 2) for i = 1:(n_batches-1)])),
            Matrix(hcat([D["X_batch_$i"] for i = 1:(n_batches-1)]...))',
        )
        b̂ = try
            inv(A' * A) * (A' * Matrix(hcat([D["y_batch_$i"] for i = 1:(n_batches-1)]...))')
        catch
            pinv(A' * A) * (A' * Matrix(hcat([D["y_batch_$i"] for i = 1:(n_batches-1)]...))')
        end
        ŷ_training = A * b̂
        metrics(
            CuArray{T,2}(Matrix(ŷ_training')),
            hcat([D["y_batch_$i"] for i = 1:(n_batches-1)]...),
        )
    end
    metrics_validation_ols = if "y_validation" ∈ keys(D)
        B = hcat(ones(size(Matrix(D["X_validation"])', 1)), Matrix(D["X_validation"])')
        ŷ_validation = B * b̂
        metrics(CuArray{T,2}(Matrix(ŷ_validation')), D["y_validation"])
    else
        Dict("mse" => T(NaN))
    end
    # Output
    Dict(
        "Ω" => Ω,
        "y_true" => Vector(y_true[1, :]),
        "y_pred" => Vector(y_pred[1, :]),
        "loss_training" => loss_training,
        "loss_validation" => loss_validation,
        "metrics_training" => metrics_training,
        "metrics_validation" => metrics_validation,
        "metrics_training_ols" => metrics_training_ols,
        "metrics_validation_ols" => metrics_validation_ols,
    )
end

"""
    optim(Xy::Dict{String,CuArray{T,2}}; kwargs...) where {T<:AbstractFloat} -> Dict{String,Dict{String}}

Perform hyperparameter optimization for a neural network using grid search.

# Arguments
- `Xy::Dict{String,CuArray{T,2}}`: Dictionary containing input features "X" and target values "y" as CuArrays

# Keywords
- `n_batches::Int64=10`: Number of batches for training/validation split (minimum 2)
- `opt_n_hidden_layers::Vector{Int64}=collect(1:3)`: Vector of hidden layer counts to try
- `opt_n_nodes_per_hidden_layer::Vector{Int64}=[size(Xy["X"],1)-i for i in [0,1]]`: Vector of node counts per hidden layer to try
- `opt_dropout_per_hidden_layer::Vector{Float64}=[0.0]`: Vector of dropout rates to try
- `opt_F_∂F::Vector{Dict{Symbol,Function}}`: Vector of activation function/derivative pairs to try
- `opt_C_∂C::Vector{Dict{Symbol,Function}}`: Vector of cost function/derivative pairs to try 
- `opt_n_epochs::Vector{Int64}=[1_000,10_000]`: Vector of epoch counts to try
- `opt_n_burnin_epochs::Vector{Int64}=[100]`: Vector of burn-in epoch counts to try
- `opt_n_patient_epochs::Vector{Int64}=[5]`: Vector of patience epoch counts to try
- `opt_optimisers::Vector{String}=["Adam"]`: Vector of optimizers to try
- `η::Float64=0.001`: Learning rate
- `β₁::Float64=0.900`: First moment decay rate for Adam
- `β₂::Float64=0.999`: Second moment decay rate for Adam  
- `ϵ::Float64=1e-8`: Small constant for numerical stability
- `n_threads::Int64=1`: Number of threads for parallel optimization
- `seed::Int64=42`: Random seed

# Returns
Dictionary containing:
- "Full_fit": Results from training with best parameters
- "Hyperparameter_search_space": Full parameter search space and results

# Examples
```julia
n_batches::Int64 = 2
D = splitdata(simulate(n=62_123, p=197, l=5), n_batches=n_batches);
Xy::Dict{String, CuArray{typeof(Vector(view(D["X_validation"], 1, 1:1))[1]), 2}} = Dict("X" => hcat([D["X_batch_$i"] for i in 1:(n_batches-1)]...),"y" => hcat([D["y_batch_$i"] for i in 1:(n_batches-1)]...),);
@time dl_opt = optim(
    Xy, 
    n_batches=n_batches,
    opt_n_hidden_layers=collect(0:5),
    opt_n_nodes_per_hidden_layer=[size(Xy["X"], 1)-i for i in [0]],
    opt_dropout_per_hidden_layer=[0.0],
    opt_F_∂F=[Dict(:F => relu, :∂F => relu_derivative), Dict(:F => leakyrelu, :∂F => leakyrelu_derivative)],
    opt_C_∂C=[Dict(:C => MSE, :∂C => MSE_derivative)],
    opt_n_epochs=[10_000],
    opt_n_burnin_epochs=[100, 1_000],
    opt_n_patient_epochs=[5, 10],
    opt_optimisers=["Adam"],
    n_threads=n_batches,
)
ŷ = predict(dl_opt["Full_fit"]["Ω"], D["X_validation"]);
y_training = vcat([Matrix(D["y_batch_$i"])[1, :] for i in 1:(n_batches-1)]...);
X_training = hcat(ones(length(y_training)), hcat([Matrix(D["X_batch_$i"])' for i in 1:(n_batches-1)]...));
b_hat = X_training \\ y_training;
y_hat::CuArray{Float32, 2} = CuArray{typeof(b_hat[1]), 2}(hcat(hcat(ones(size(D["X_validation"], 2)), Matrix(D["X_validation"])') * b_hat)');
metrics_mlp = metrics(ŷ, D["y_validation"])
metrics_ols = metrics(y_hat, D["y_validation"])
(metrics_mlp["ρ"] > metrics_ols["ρ"]) && (metrics_mlp["R²"] > metrics_ols["R²"]) && (metrics_mlp["rmse"] < metrics_ols["rmse"])
```
"""
function optim(
    Xy::Dict{String,CuArray{T,2}};
    n_batches::Int64 = 10,
    opt_n_hidden_layers::Vector{Int64} = collect(1:3),
    opt_n_nodes_per_hidden_layer::Vector{Int64} = [size(Xy["X"], 1) - i for i in [0, 1]],
    opt_dropout_per_hidden_layer::Vector{Float64} = [0.0],
    opt_F_∂F::Vector{Dict{Symbol,Function}} = [
        Dict(:F => relu, :∂F => relu_derivative),
        Dict(:F => leakyrelu, :∂F => leakyrelu_derivative),
    ],
    opt_C_∂C::Vector{Dict{Symbol,Function}} = [Dict(:C => MSE, :∂C => MSE_derivative)],
    opt_n_epochs::Vector{Int64} = [1_000, 10_000],
    opt_n_burnin_epochs::Vector{Int64} = [100],
    opt_n_patient_epochs::Vector{Int64} = [5],
    opt_optimisers::Vector{String} = ["Adam"],
    η::Float64 = 0.001,
    β₁::Float64 = 0.900,
    β₂::Float64 = 0.999,
    ϵ::Float64 = 1e-8,
    n_threads::Int64 = 1,
    seed::Int64 = 42,
)::Dict{String,Dict{String}} where {T<:AbstractFloat}
    # Xy::Dict{String, CuArray{Float32, 2}} = simulate()
    # n_batches::Int64 = 2
    # opt_n_hidden_layers::Vector{Int64} = collect(1:2)
    # opt_n_nodes_per_hidden_layer::Vector{Int64} = [size(Xy["X"], 1)-i for i in [0, 1]]
    # opt_dropout_per_hidden_layer::Vector{Float64} = [0.0]
    # opt_F_∂F::Vector{Dict{Symbol, Function}} = [Dict(:F => relu, :∂F => relu_derivative), Dict(:F => leakyrelu, :∂F => leakyrelu_derivative)]
    # opt_C_∂C::Vector{Dict{Symbol, Function}} = [Dict(:C => MSE, :∂C => MSE_derivative)]
    # opt_n_epochs::Vector{Int64} = [1_000, 10_000]
    # opt_n_burnin_epochs::Vector{Int64} = [100]
    # opt_n_patient_epochs::Vector{Float64} = [5]
    # opt_optimisers::Vector{String} = ["Adam"]
    # η::Float64 = 0.001
    # β₁::Float64 = 0.900
    # β₂::Float64 = 0.999
    # ϵ::Float64 = 1e-8
    # n_threads::Int64 = 2
    # seed::Int64 = 42
    #
    if n_batches < 2
        throw(
            ArgumentError(
                "Number of batches (n_batches=$n_batches) must be at least 2 with the first batch used for training and the second for validation",
            ),
        )
    end
    # Sort by increasing complexity so we can jump when accuracy decreases (excluding activation and cost functions)
    opt_n_hidden_layers = sort(opt_n_hidden_layers)
    opt_n_nodes_per_hidden_layer = sort(opt_n_nodes_per_hidden_layer)
    opt_dropout_per_hidden_layer = sort(opt_dropout_per_hidden_layer)
    # opt_F_∂F = sort(opt_F_∂F)
    # opt_C_∂C = sort(opt_C_∂C)
    opt_n_epochs = sort(opt_n_epochs)
    opt_n_burnin_epochs = sort(opt_n_burnin_epochs)
    opt_n_patient_epochs = sort(opt_n_patient_epochs)
    # opt_optimisers = sort(opt_optimisers)
    # Prepare parameter space
    par_n_hidden_layers::Vector{Int64} = []
    par_n_nodes_per_hidden_layer::Vector{Int64} = []
    par_dropout_per_hidden_layer::Vector{Float64} = []
    par_F::Vector{Function} = []
    par_∂F::Vector{Function} = []
    par_C::Vector{Function} = []
    par_∂C::Vector{Function} = []
    par_n_epochs::Vector{Int64} = []
    par_n_burnin_epochs::Vector{Int64} = []
    par_n_patient_epochs::Vector{Int64} = []
    par_optimisers::Vector{String} = []
    for i in eachindex(opt_n_hidden_layers)
        for j in eachindex(opt_n_nodes_per_hidden_layer)
            for k in eachindex(opt_dropout_per_hidden_layer)
                for l in eachindex(opt_F_∂F)
                    for m in eachindex(opt_C_∂C)
                        for n in eachindex(opt_n_epochs)
                            for o in eachindex(opt_n_patient_epochs)
                                for p in eachindex(opt_n_burnin_epochs)
                                    for q in eachindex(opt_optimisers)
                                        push!(par_n_hidden_layers, opt_n_hidden_layers[i])
                                        push!(
                                            par_n_nodes_per_hidden_layer,
                                            opt_n_nodes_per_hidden_layer[j],
                                        )
                                        push!(
                                            par_dropout_per_hidden_layer,
                                            opt_dropout_per_hidden_layer[k],
                                        )
                                        push!(par_F, opt_F_∂F[l][:F])
                                        push!(par_∂F, opt_F_∂F[l][:∂F])
                                        push!(par_C, opt_C_∂C[m][:C])
                                        push!(par_∂C, opt_C_∂C[m][:∂C])
                                        push!(par_n_epochs, opt_n_epochs[n])
                                        push!(par_n_patient_epochs, opt_n_patient_epochs[o])
                                        push!(par_n_burnin_epochs, opt_n_burnin_epochs[p])
                                        push!(par_optimisers, opt_optimisers[q])
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    # Optimise
    P = length(par_n_hidden_layers)
    idx_per_thread::Vector{Vector{Int64}} = []
    for i = 1:n_threads
        push!(idx_per_thread, collect(i:n_threads:P))
    end
    actual_n_epochs::Vector{Int64} = repeat([0], P)
    mse::Vector{Float64} = repeat([NaN], P)
    println("Optimising along $P sets of parameters via grid search:")
    Threads.@threads for idx in idx_per_thread
        for i in idx
            # i = 1
            dl = train(
                Xy,
                fit_full = false,
                n_batches = n_batches,
                n_hidden_layers = par_n_hidden_layers[i],
                n_hidden_nodes = repeat(
                    [par_n_nodes_per_hidden_layer[i]],
                    par_n_hidden_layers[i],
                ),
                dropout_rates = repeat(
                    [par_dropout_per_hidden_layer[i]],
                    par_n_hidden_layers[i],
                ),
                F = par_F[i],
                ∂F = par_∂F[i],
                C = par_C[i],
                ∂C = par_∂C[i],
                n_epochs = par_n_epochs[i],
                n_burnin_epochs = par_n_burnin_epochs[i],
                n_patient_epochs = par_n_patient_epochs[i],
                optimiser = par_optimisers[i],
                η = η,
                β₁ = β₁,
                β₂ = β₂,
                ϵ = ϵ,
                seed = seed,
                verbose = true, # setting this to true so that the user can still still some progress per training run
            )
            actual_n_epochs[i] = length(dl["loss_training"])
            mse[i] = Float64(dl["metrics_validation"]["mse"])
            p = Int(round(100 * sum(.!isnan.(mse)) / P))
            print("\r$(repeat("█", p)) | $p% hyperparameter optimisation.\n")
        end
    end
    println("")
    # Fit using the best set of parameters
    println("Fitting using the best set of parameters:")
    idx = findall(.!isnan.(mse) .&& (mse .== minimum(mse[.!isnan.(mse)])))[1]
    println(
        join(
            [
                "\t‣ n_hidden_layers=$(par_n_hidden_layers[idx])",
                "n_hidden_nodes=$(repeat([par_n_nodes_per_hidden_layer[idx]], par_n_hidden_layers[idx]))",
                "dropout_rates=$(repeat([par_dropout_per_hidden_layer[idx]], par_n_hidden_layers[idx]))",
                "F=$(string(par_F[idx]))",
                "∂F=$(string(par_∂F[idx]))",
                "C=$(string(par_C[idx]))",
                "∂C=$(string(par_∂C[idx]))",
                "n_epochs=$(actual_n_epochs[idx])", # using actual number of epochs used from the optimisation step
                "n_burnin_epochs=$(par_n_burnin_epochs[idx])",
                "n_patient_epochs=$(par_n_patient_epochs[idx])",
                "optimiser=$(par_optimisers[idx])",
            ],
            "\n\t‣ ",
        ),
    )
    dl_opt = train(
        Xy,
        fit_full = false,
        n_hidden_layers = par_n_hidden_layers[idx],
        n_hidden_nodes = repeat(
            [par_n_nodes_per_hidden_layer[idx]],
            par_n_hidden_layers[idx],
        ),
        dropout_rates = repeat(
            [par_dropout_per_hidden_layer[idx]],
            par_n_hidden_layers[idx],
        ),
        F = par_F[idx],
        ∂F = par_∂F[idx],
        C = par_C[idx],
        ∂C = par_∂C[idx],
        n_epochs = actual_n_epochs[idx], # using actual number of epochs used from the optimisation step
        n_burnin_epochs = par_n_burnin_epochs[idx],
        n_patient_epochs = par_n_patient_epochs[idx],
        optimiser = par_optimisers[idx],
        η = η,
        β₁ = β₁,
        β₂ = β₂,
        ϵ = ϵ,
        seed = seed,
        verbose = true,
    )
    println("Total number of epochs ran: $(length(dl_opt["loss_training"]))")
    println("Number of burn-in epochs ran: $(par_n_burnin_epochs[idx])")
    println(
        "Number of patient epochs used for early-stopping: $(par_n_patient_epochs[idx])",
    )
    println("Hidden layers: $(dl_opt["Ω"].n_hidden_layers)")
    println("Nodes per hidden layer: $(dl_opt["Ω"].n_hidden_nodes)")
    println("Dropout rates per hidden layer: $(dl_opt["Ω"].dropout_rates)")
    # println("Weights: $(dl_opt["Ω"].W)")
    # println("Biases: $(dl_opt["Ω"].b)")
    println("Activation function: $(dl_opt["Ω"].F)")
    println("Activation function derivative: $(dl_opt["Ω"].∂F)")
    println("Cost function: $(dl_opt["Ω"].C)")
    println("Cost function derivative: $(dl_opt["Ω"].∂C)")
    println("Randomisation seed: $(dl_opt["Ω"].seed)")
    # Output
    Dict(
        "Full_fit" => dl_opt,
        "Hyperparameter_search_space" => Dict(
            "n_hidden_layers" => par_n_hidden_layers,
            "n_nodes_per_hidden_layer" => par_n_nodes_per_hidden_layer,
            "dropout_per_hidden_layer" => par_dropout_per_hidden_layer,
            "F" => par_F,
            "∂F" => par_∂F,
            "C" => par_C,
            "∂C" => par_∂C,
            "n_epochs" => par_n_epochs,
            "n_patient_epochs" => par_n_patient_epochs,
            "optimisers" => par_optimisers,
            "mse" => mse,
        ),
    )
end
