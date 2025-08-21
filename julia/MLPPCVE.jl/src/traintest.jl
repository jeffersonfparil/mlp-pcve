function splitdata(
    Xy::Dict{String, CuArray{T, 2}};
    ν::Float64 = 0.10,
    seed::Int64 = 42,
)::Dict{String, CuArray{T, 2}} where T <:AbstractFloat
    # T::Type = Float32
    # Xy = simulate(T=T)
    # ν::T = 0.10
    # seed::Int64 = 42
    Random.seed!(seed)
    p, n = size(Xy["X"])
    n_validation, n_training = begin
        n_validation = Int64(round(ν*n))
        n_training = n - n_validation
        if n_training == 0
            (n_validation - 1, 1)
        else
            (n_validation, n_training)
        end
    end
    idx_validation = drawreplacenot(n, n_validation)
    idx_training = findall([!(x ∈ idx_validation) for x in 1:n])
    Dict(
        "X_training" => CuArray{T, 2}(view(Xy["X"], 1:p, idx_training)),
        "y_training" => CuArray{T, 2}(view(Xy["y"], 1:1, idx_training)), 
        "X_validation" => CuArray{T, 2}(view(Xy["X"], 1:p, idx_validation)),
        "y_validation" => CuArray{T, 2}(view(Xy["y"], 1:1, idx_validation)),
    )
end

function predict(Ω::Network{T}, X::CuArray{T, 2})::CuArray{T, 2} where T <:AbstractFloat
    a = Ω.F.((Ω.W[1] * X) .+ Ω.b[1])
    for i in 2:Ω.n_hidden_layers
        a = Ω.F.((Ω.W[i] * a) .+ Ω.b[i])
    end
    (Ω.W[end] * a) .+ Ω.b[end]
end

# train(simulate(), n_epochs=1_000)
function train(
    Xy::Dict{String, CuArray{T, 2}}; # It it recommended that X be standardised (μ=0, and σ=1)
    ν::Float64 = 0.10, # proportion of the dataset set aside for validation during training for early stopping
    n_hidden_layers::Int64 = 3,
    n_hidden_nodes::Vector{Int64} = repeat([size(Xy["X"], 1)], n_hidden_layers),
    dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers),
    F::Function = relu,
    ∂F::Function = relu_derivative,
    C::Function = MSE,
    ∂C::Function = MSE_derivative,
    n_epochs::Int64 = 10_000,
    n_patient_epochs::Int64 = 1_000,
    optimiser::String = ["GD", "Adam", "AdamMax"][2],
    η::Float64 = 0.001,
    β₁::Float64 = 0.900,
    β₂::Float64 = 0.999,
    ϵ::Float64 = 1e-8,
    t::Float64 = 0.0,
    seed::Int64 = 42,
)::Dict{String, Union{Network{T}, Vector{T}, Dict{String, T}}} where T <: AbstractFloat
    # Xy = simulate(seed=1)
    # # Xy = simulate(phen_type="linear", h²=1.00)
    # # Xy = simulate(n=10_000, l=1_000)
    # ν::Float64 = 0.10
    # n_hidden_layers::Int64 = 3
    # n_hidden_nodes::Vector{Int64} = repeat([size(Xy["X"], 1)], n_hidden_layers)
    # dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers)
    # F::Function = relu
    # ∂F::Function = relu_derivative
    # C::Function = MSE
    # ∂C::Function = MSE_derivative
    # n_epochs::Int64 = 10_000
    # n_patient_epochs::Int64 = 5_000
    # optimiser::String = ["GD", "Adam", "AdamMax"][2]
    # η::Float64 = 0.001
    # β₁::Float64 = 0.900
    # β₂::Float64 = 0.999
    # ϵ::Float64 = 1e-8
    # t::Float64 = 0.0
    # seed::Int64 = 42
    # T = typeof(Matrix(Xy["y"])[1,1])
    # Split into validation and training sets and initialise the networks for each as well as the target outputs
    D = splitdata(Xy, ν=Float64(ν), seed=seed)
    # Initialise the model using the training data only
    Ω = init(
        D["X_training"],
        n_hidden_layers=n_hidden_layers,
        n_hidden_nodes=n_hidden_nodes,
        dropout_rates=dropout_rates,
        F=F,
        ∂F=∂F,
        C=C,
        ∂C=∂C,
        seed=seed,
    )
    state::Dict{String, Any} = if optimiser == "GD"
        Dict()
    else
        Dict(
            "β₁" => T(β₁), "β₂" => T(β₂), "ϵ" => T(ϵ), "t" => T(t),
            "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
            "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
            "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
            "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
        )
    end
    
    epochs::Vector{Int64} = []
    loss_training::Vector{T} = []
    loss_validation::Vector{T} = []
    for i in 1:n_epochs
        # i = 1
        p = Int(round(100*i/n_epochs))
        print("\r$(repeat("█", p)) | $p% ")
        forwardpass!(Ω)
        backpropagation!(Ω, D["y_training"])
        if optimiser == "GD"
            gradientdescent!(Ω, η=T(η))
        elseif optimiser == "Adam"
            Adam!(Ω, η=T(η), state=state)
        elseif optimiser == "AdamMax"
            AdamMax!(Ω, η=T(η), state=state)
        else
            throw("Invalid optimiser: $optimiser")
        end
        Θ_training = metrics(predict(Ω, D["X_training"]), D["y_training"]) # NOTE: do not use the forward pass because it may have dropouts
        Θ_validation = if length(D["y_validation"]) > 0
            metrics(predict(Ω, D["X_validation"]), D["y_validation"])
        else
            Dict("mse" => NaN)
        end
        push!(epochs, i)
        push!(loss_training, Θ_training["mse"])
        push!(loss_validation, Θ_validation["mse"])
        # Stop early if validation loss does not improve within within n_patient_epochs
        if (i >= n_patient_epochs) && ((loss_validation[end] - loss_validation[(end-n_patient_epochs)+1]) >= 0.0)
            println("Early stopping because validation loss is not improving for the past $n_patient_epochs epochs")
            break
        end
    end
    # Memory clean-up
    CUDA.reclaim()
    CUDA.pool_status()

    # # NOTE: do not use the forward pass because it may have dropouts
    # # metrics(Ω.ŷ, D["y_training"]) 
    # metrics(predict(Ω, D["X_training"]), D["y_training"])
    # metrics(predict(Ω, D["X_validation"]), D["y_validation"])
    # # using UnicodePlots
    # UnicodePlots.scatterplot(epochs, loss_training)
    # UnicodePlots.scatterplot(Matrix(Ω.ŷ)[1, :], Matrix(D["y_training"])[1, :])
    # UnicodePlots.scatterplot(epochs, loss_validation)
    # UnicodePlots.scatterplot(Matrix(predict(Ω, D["X_validation"]))[1, :], Matrix(D["y_validation"])[1, :])
    # for i in 1:Ω.n_hidden_layers
    #     println("Hidden layer $i: first column are biases and the rest are the weights")
    #     display(UnicodePlots.heatmap(hcat(Vector(Ω.b[i]), Matrix(Ω.W[i]))))
    # end

    # # OLS
    # A = hcat(ones(size(Matrix(D["X_training"])', 1)), Matrix(D["X_training"])');
    # # A = Matrix(D["X_training"])';
    # b̂ = inv(A'*A)*(A'*Matrix(D["y_training"])');
    # ŷ = A * b̂;
    # metrics(CuArray{T, 2}(Matrix(ŷ')), D["y_training"])
    # UnicodePlots.scatterplot(ŷ[:, 1], Matrix(D["y_training"])[1, :])
    # B = hcat(ones(size(Matrix(D["X_validation"])', 1)), Matrix(D["X_validation"])');
    # # B = Matrix(D["X_validation"])';
    # ŷ = B * b̂;
    # metrics(CuArray{T, 2}(Matrix(ŷ')), D["y_validation"])
    # UnicodePlots.scatterplot(ŷ[:, 1], Matrix(D["y_validation"])[1, :])

    # Output
    # TODO: first use a non-zero ν to find the optimum n_epochs; then using this n_epochs set ν to zero to get the final model fit using the entire dataset
    Dict(
        "Ω" => Ω,
        "loss_training" => loss_training,
        "loss_validation" => loss_validation,
        "metrics_training" => metrics(predict(Ω, D["X_training"]), D["y_training"]),
        "metrics_validation" => metrics(predict(Ω, D["X_validation"]), D["y_validation"]),
    )
end