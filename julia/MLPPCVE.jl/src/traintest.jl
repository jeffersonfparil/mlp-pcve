function splitinit!(
    Xy::Dict{String, CuArray{T, 2}};
    ν::T = 0.10,
    n_hidden_layers::Int64 = 2,
    n_hidden_nodes::Vector{Int64} = vcat([x * 100 for x in 1:n_hidden_layers], 1),
    F::Function = relu,
    δF::Function = relu_derivative,
    C::Function = MSE,
    δC::Function = MSE_derivative,
    seed::Int64 = 42,
)::Tuple{Network{T}, Network{T}, CuArray{T, 2}, CuArray{T, 2}} where T <:AbstractFloat
    # T::Type = Float32
    # Xy = simulate(T=T)
    # ν::T = 0.10
    # n_hidden_layers::Int64 = 2
    # n_hidden_nodes::Vector{Int64} = vcat([x * 100 for x in 1:n_hidden_layers], 1)
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
    Ω_validation = if length(idx_validation) == 0 
        init(CuArray{T, 2}(view(Xy["X"], 1:1, idx_validation)), n_hidden_layers=0)
    else
        init(
            CuArray{T, 2}(view(Xy["X"], 1:p, idx_validation)),
            n_hidden_layers=n_hidden_layers,
            n_hidden_nodes=n_hidden_nodes,
            F=F,
            δF=δF,
            C=C,
            δC=δC,
        )
    end
    Ω_training = init(
        CuArray{T, 2}(view(Xy["X"], 1:p, idx_training)),
        n_hidden_layers=n_hidden_layers,
        n_hidden_nodes=n_hidden_nodes,
        F=F,
        δF=δF,
        C=C,
        δC=δC,
    )
    y_validation = CuArray{T, 2}(view(Xy["y"], 1:1, idx_validation))
    y_training = CuArray{T, 2}(view(Xy["y"], 1:1, idx_training))
    # empty!(Xy)
    (Ω_validation, Ω_training, y_validation, y_training)
end

# train(simulate(), n_epochs=1_000)
function train(
    Xy::Dict{String, CuArray{T, 2}}; # It it recommended that X be standardised (μ=0, and σ=1)
    ν::T = 0.10, # proportion of the dataset set aside for validation during training for early stopping
    n_hidden_layers::Int64 = 2,
    n_hidden_nodes::Vector{Int64} = vcat([x * 100 for x in 1:n_hidden_layers], 1),
    F::Function = relu,
    δF::Function = relu_derivative,
    C::Function = MSE,
    δC::Function = MSE_derivative,
    n_epochs::Int64 = 10_000,
    n_patient_epochs::Int64 = 1_000,
    optimiser::String = ["GD", "Adam", "AdamMax"][2],
    η::T = 0.001,
    β₁::T = 0.900,
    β₂::T = 0.999,
    ϵ::T = 1e-7,
    t::T = 0.0,
    seed::Int64 = 42,
)::Tuple{Network{T}, Vector{T}, Dict{String, T}} where T <: AbstractFloat
    # T::Type = Float32; Xy = simulate(T=T); ν::T = 0.10; n_hidden_layers::Int64 = 2; n_hidden_nodes::Vector{Int64} = vcat([x * 100 for x in 1:n_hidden_layers], 1); F::Function = relu; δF::Function = relu_derivative; C::Function = MSE; δC::Function = MSE_derivative; n_epochs::Int64 = 10_000; n_patient_epochs::Int64 = 1_000; optimiser::String = ["GD", "Adam", "AdamMax"][2]; η::T = 0.001; β₁ = T(0.900); β₂ = T(0.999); ϵ = T(1e-7); t = T(0.0); seed::Int64 = 42
    # Split into validatin and training sets and initialise the networks for each as well as the target outputs
    Ω_validation, Ω_training, y_validation, y_training = splitinit!(
        Xy,
        ν=ν,
        n_hidden_layers=n_hidden_layers,
        n_hidden_nodes=n_hidden_nodes,
        F=F,
        δF=δF,
        C=C,
        δC=δC,
        seed=seed,
    )
    state::Dict{String, Any} = if optimiser == "GD"
        Dict()
    else
        Dict(
            "β₁" => β₁, "β₂" => β₂, "ϵ" => ϵ, "t" => t,
            "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω_training.W],
            "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω_training.W],
            "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω_training.b],
            "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω_training.b],
        )
    end

    epochs::Vector{Int64} = []
    loss_training::Vector{T} = []
    loss_validation::Vector{T} = []
    for i in 1:n_epochs
        # i = 1
        @show i
        forwardpass!(Ω_training)
        backpropagation!(Ω_training, y_training)
        if optimiser == "GD"
            gradientdescent!(Ω_training, η=η)
        elseif optimiser == "Adam"
            Adam!(Ω_training, η=η, state=state)
        elseif optimiser == "Adam"
            AdamMax!(Ω_training, η=η, state=state)
        else
            throw("Invalid optimiser: $optimiser")
        end
        if length(Ω_validation.ŷ) > 0
            [Ω_validation.W[i] .= deepcopy(Ω_training.W[i]) for i in eachindex(Ω_training.W)]
            [Ω_validation.b[i] .= deepcopy(Ω_training.b[i]) for i in eachindex(Ω_training.b)]
            forwardpass!(Ω_validation)
        end
        Θ_training = metrics(Ω_training, y_training)
        Θ_validation = metrics(Ω_validation, y_validation)
        push!(epochs, i)
        push!(loss_training, Θ_training["mse"])
        push!(loss_validation, Θ_validation["mse"])
        # Stop early if validation loss increases or 
        if (
            (i > n_patient_epochs) && 
            (
                (loss_validation[end] > maximum(loss_validation)) || 
                ((length(Ω_validation.ŷ) > 0) && isnan(loss_validation[end]))
            )
        )
            break
        end
    end

    # using UnicodePlots
    # display(UnicodePlots.scatterplot(epochs, loss_training)); display(UnicodePlots.scatterplot(Matrix(Ω_training.ŷ)[1, :], Matrix(y_training)[1, :]))
    # display(UnicodePlots.scatterplot(epochs, loss_validation)); display(UnicodePlots.scatterplot(Matrix(Ω_validation.ŷ)[1, :], Matrix(y_validation)[1, :]))
    
    # Output
    # TODO: first use a non-zero ν to find the optimum n_epochs; then using this n_epochs set ν to zero to get the final model fit using the entire dataset
    (
        "Ω" => Ω_training,
        "loss" => loss_training,
        "metrics" => metrics(Ω_training, y_training),
    )
end