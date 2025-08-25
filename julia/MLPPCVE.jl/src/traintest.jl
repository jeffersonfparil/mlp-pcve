function splitdata(
    Xy::Dict{String,CuArray{T,2}};
    ν::Float64 = 0.10,
    seed::Int64 = 42,
)::Dict{String,CuArray{T,2}} where {T<:AbstractFloat}
    # T::Type = Float32
    # Xy = simulate(T=T)
    # ν::T = 0.10
    # seed::Int64 = 42
    Random.seed!(seed)
    p, n = size(Xy["X"])
    n_validation, n_training = begin
        n_validation = Int64(round(ν * n))
        n_training = n - n_validation
        if n_training == 0
            (n_validation - 1, 1)
        else
            (n_validation, n_training)
        end
    end
    idx_validation = drawreplacenot(n, n_validation)
    idx_training = findall([!(x ∈ idx_validation) for x = 1:n])
    Dict(
        "X_training" => CuArray{T,2}(view(Xy["X"], 1:p, idx_training)),
        "y_training" => CuArray{T,2}(view(Xy["y"], 1:1, idx_training)),
        "X_validation" => CuArray{T,2}(view(Xy["X"], 1:p, idx_validation)),
        "y_validation" => CuArray{T,2}(view(Xy["y"], 1:1, idx_validation)),
    )
end

function predict(Ω::Network{T}, X::CuArray{T,2})::CuArray{T,2} where {T<:AbstractFloat}
    a = Ω.F.((Ω.W[1] * X) .+ Ω.b[1])
    for i = 2:Ω.n_hidden_layers
        a = Ω.F.((Ω.W[i] * a) .+ Ω.b[i])
    end
    (Ω.W[end] * a) .+ Ω.b[end]
end

# dl = train(simulate())
# dl = train(simulate(), F=leakyrelu)
# dl = train(simulate(), F=sigmoid)
# dl = train(simulate(), F=tanh)
# dl = train(simulate(n=10_000, l=100))
function train(
    Xy::Dict{String,CuArray{T,2}}; # It it recommended that X be standardised (μ=0, and σ=1)
    ν::Float64 = 0.10, # proportion of the dataset set aside for validation during training for early stopping
    n_hidden_layers::Int64 = 3,
    n_hidden_nodes::Vector{Int64} = repeat([size(Xy["X"], 1)], n_hidden_layers),
    dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
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
    verbose::Bool = true,
)::Dict{String,Union{Network{T},Vector{T},Dict{String,T}}} where {T<:AbstractFloat}
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
    D = splitdata(Xy, ν = Float64(ν), seed = seed)
    # Initialise the model using the training data only
    Ω = init(
        D["X_training"],
        n_hidden_layers = n_hidden_layers,
        n_hidden_nodes = n_hidden_nodes,
        dropout_rates = dropout_rates,
        F = F,
        ∂F = ∂F,
        C = C,
        ∂C = ∂C,
        seed = seed,
    )
    state::Dict{String,Any} = if optimiser == "GD"
        Dict()
    else
        Dict(
            "β₁" => T(β₁),
            "β₂" => T(β₂),
            "ϵ" => T(ϵ),
            "t" => T(t),
            "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
            "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
            "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
            "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
        )
    end

    epochs::Vector{Int64} = []
    loss_training::Vector{T} = []
    loss_validation::Vector{T} = []
    for i = 1:n_epochs
        # i = 1
        if verbose
            p = Int(round(100 * i / n_epochs))
            print("\r$(repeat("█", p)) | $p% ")
        end
        forwardpass!(Ω)
        backpropagation!(Ω, D["y_training"])
        if optimiser == "GD"
            gradientdescent!(Ω, η = T(η))
        elseif optimiser == "Adam"
            Adam!(Ω, η = T(η), state = state)
        elseif optimiser == "AdamMax"
            AdamMax!(Ω, η = T(η), state = state)
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
        if (i >= n_patient_epochs) &&
           ((loss_validation[end] - loss_validation[(end-n_patient_epochs)+1]) >= 0.0)
            if verbose
                println(
                    "\nEarly stopping after $(length(loss_training)) epochs because validation loss is not improving for the past $n_patient_epochs epochs",
                )
            end
            break
        end
    end
    # Memory clean-up
    CUDA.reclaim()
    if verbose
        println("")
        CUDA.pool_status()
    end
    # Ordinary least squares fit as reference
    A = hcat(ones(size(Matrix(D["X_training"])', 1)), Matrix(D["X_training"])')
    b̂ = inv(A' * A) * (A' * Matrix(D["y_training"])')
    ŷ_training = A * b̂
    B = hcat(ones(size(Matrix(D["X_validation"])', 1)), Matrix(D["X_validation"])')
    ŷ_validation = B * b̂
    metrics_training_ols = metrics(CuArray{T,2}(Matrix(ŷ_training')), D["y_training"])
    metrics_validation_ols = metrics(CuArray{T,2}(Matrix(ŷ_validation')), D["y_validation"])
    # Output
    Dict(
        "Ω" => Ω,
        "loss_training" => loss_training,
        "loss_validation" => loss_validation,
        "metrics_training" => metrics(predict(Ω, D["X_training"]), D["y_training"]),
        "metrics_validation" => metrics(predict(Ω, D["X_validation"]), D["y_validation"]),
        "metrics_training_ols" => metrics_training_ols,
        "metrics_validation_ols" => metrics_validation_ols,
    )
end

# D = splitdata(simulate(n=10_000), ν=0.25);
# Xy::Dict{String, CuArray{typeof(Vector(view(D["X_training"], 1, 1:1))[1]), 2}} = Dict("X"=>D["X_training"], "y"=>D["y_training"]);
# dl_opt = optim(Xy, opt_n_hidden_layers=[2,3], opt_n_epochs=[10_000], opt_frac_patient_epochs=[0.25]);
# ŷ = predict(dl_opt["Ω"], D["X_validation"]);
# metrics(ŷ, D["y_validation"])
# y_training = Matrix(D["y_training"])[1, :]
# X_training = hcat(ones(size(D["X_training"], 2)), Matrix(D["X_training"])')
# b_hat = X_training \ y_training
# y_hat::CuArray{Float32, 2} = CuArray{typeof(b_hat[1]), 2}(hcat(hcat(ones(size(D["X_validation"], 2)), Matrix(D["X_validation"])') * b_hat)')
# metrics(y_hat, D["y_validation"])
function optim(
    Xy::Dict{String, CuArray{T, 2}};
    ν::Float64 = 0.10,
    opt_n_hidden_layers::Vector{Int64} = collect(1:3),
    opt_n_nodes_per_hidden_layer::Vector{Int64} = [size(Xy["X"], 1)-i for i in [0, 1]],
    opt_dropout_per_hidden_layer::Vector{Float64} = [0.0],
    opt_F_∂F::Vector{Dict{Symbol, Function}} = [Dict(:F => relu, :∂F => relu_derivative), Dict(:F => leakyrelu, :∂F => leakyrelu_derivative)],
    opt_C_∂C::Vector{Dict{Symbol, Function}} = [Dict(:C => MSE, :∂C => MSE_derivative)],
    opt_n_epochs::Vector{Int64} = [1_000, 10_000],
    opt_frac_patient_epochs::Vector{Float64} = [0.25, 0.50],
    opt_optimisers::Vector{String} = ["Adam"],
    seed::Int64 = 42,
)::Dict{String,Union{Network{T},Vector{T},Dict{String,T}}} where {T<:AbstractFloat}
    # Xy::Dict{String, CuArray{Float32, 2}} = simulate()
    # ν::Float64 = 0.10
    # opt_n_hidden_layers::Vector{Int64} = collect(1:3)
    # opt_n_nodes_per_hidden_layer::Vector{Int64} = [size(Xy["X"], 1)-i for i in [0, 5]]
    # opt_dropout_per_hidden_layer::Vector{Float64} = [0.0]
    # opt_F_∂F::Vector{Dict{Symbol, Function}} = [Dict(:F => relu, :∂F => relu_derivative), Dict(:F => leakyrelu, :∂F => leakyrelu_derivative)]
    # opt_C_∂C::Vector{Dict{Symbol, Function}} = [Dict(:C => MSE, :∂C => MSE_derivative)]
    # opt_n_epochs::Vector{Int64} = [1_000, 5_000, 10_000]
    # opt_frac_patient_epochs::Vector{Float64} = [0.25, 0.50]
    # opt_optimisers::Vector{String} = ["Adam"]
    # seed::Int64 = 42
    #
    # Sort by increasing complexity so we can jump when accuracy decreases (excluding activation and cost functions)
    opt_n_hidden_layers = sort(opt_n_hidden_layers)
    opt_n_nodes_per_hidden_layer = sort(opt_n_nodes_per_hidden_layer)
    opt_dropout_per_hidden_layer = sort(opt_dropout_per_hidden_layer)
    # opt_F_∂F = sort(opt_F_∂F)
    # opt_C_∂C = sort(opt_C_∂C)
    opt_n_epochs = sort(opt_n_epochs)
    opt_frac_patient_epochs = sort(opt_frac_patient_epochs)
    opt_optimisers = sort(opt_optimisers)
    # Prepare parameter space
    n_hidden_layers::Vector{Int64} = []
    n_nodes_per_hidden_layer::Vector{Int64} = []
    dropout_per_hidden_layer::Vector{Float64} = []
    F::Vector{Function} = []
    ∂F::Vector{Function} = []
    C::Vector{Function} = []
    ∂C::Vector{Function} = []
    n_epochs::Vector{Int64} = []
    n_patient_epochs::Vector{Int64} = []
    optimisers::Vector{String} = []
    for i in eachindex(opt_n_hidden_layers)
        for j in eachindex(opt_n_nodes_per_hidden_layer)
            for k in eachindex(opt_dropout_per_hidden_layer)
                for l in eachindex(opt_F_∂F)
                    for m in eachindex(opt_C_∂C)
                        for n in eachindex(opt_n_epochs)
                            for o in eachindex(opt_frac_patient_epochs)
                                for p in eachindex(opt_optimisers)
                                    push!(n_hidden_layers, opt_n_hidden_layers[i])
                                    push!(n_nodes_per_hidden_layer, opt_n_nodes_per_hidden_layer[j])
                                    push!(dropout_per_hidden_layer, opt_dropout_per_hidden_layer[k])
                                    push!(F, opt_F_∂F[l][:F])
                                    push!(∂F, opt_F_∂F[l][:∂F])
                                    push!(C, opt_C_∂C[m][:C])
                                    push!(∂C, opt_C_∂C[m][:∂C])
                                    push!(n_epochs, opt_n_epochs[n])
                                    push!(n_patient_epochs, Int64(ceil(opt_frac_patient_epochs[o] * opt_n_epochs[n])))
                                    push!(optimisers, opt_optimisers[p])
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    # Optimise
    P = length(n_hidden_layers)
    # P = 10
    mse::Vector{Float64} = repeat([NaN], P)
    println("Optimising along $P sets of parameters via grid search:")
    Threads.@threads for i in 1:P
        # i = 1
        dl = train(
            Xy,
            ν=ν,
            n_hidden_layers=n_hidden_layers[i],
            n_hidden_nodes=repeat([n_nodes_per_hidden_layer[i]], n_hidden_layers[i]),
            dropout_rates=repeat([dropout_per_hidden_layer[i]], n_hidden_layers[i]),
            F=F[i],
            ∂F=∂F[i],
            C=C[i],
            ∂C=∂C[i],
            n_epochs=n_epochs[i],
            n_patient_epochs=n_patient_epochs[i],
            optimiser=optimisers[i],
            seed=seed,
            verbose=false,
        )
        mse[i] = Float64(dl["metrics_validation"]["mse"])
        p = Int(round(100 * sum(.!isnan.(mse)) / P))
        print("\r$(repeat("█", p)) | $p% ")
    end
    println("")
    # Fit using the best set of parameters
    println("Fitting using the best set of parameters given the input parameter space:")
    idx = findall(.!isnan.(mse) .&& (mse .== minimum(mse[.!isnan.(mse)])))[1]
    dl_opt = train(
        Xy,
        ν=0.0,
        n_hidden_layers=n_hidden_layers[idx],
        n_hidden_nodes=repeat([n_nodes_per_hidden_layer[idx]], n_hidden_layers[idx]),
        dropout_rates=repeat([dropout_per_hidden_layer[idx]], n_hidden_layers[idx]),
        F=F[idx],
        ∂F=∂F[idx],
        C=C[idx],
        ∂C=∂C[idx],
        n_epochs=n_epochs[idx],
        n_patient_epochs=n_patient_epochs[idx],
        optimiser=optimisers[idx],
        seed=seed,
        verbose=true,
    )
    println("Total number of epochs ran: $(length(dl_opt["loss_training"]))")
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
    dl_opt
end


# ### Misc: LMM stuff
# using Distributions, LinearAlgebra, MultivariateStats, UnicodePlots, Random
# Random.seed!(42)
# n = 1_000
# m = 10
# p = 100
# h² = 0.50
# G = begin
#     g = rand(Normal(0.0, 1.0), p)
#     G = g  * g'
#     while !isposdef(G)
#         G[diagind(G)] .+= 0.01
#     end
#     G
# end
# UnicodePlots.heatmap(G)
# X = hcat(ones(n), rand(Bool, n, m-1))
# β = rand(Float64, m)
# Z = rand(Bool, n, p)
# u = rand(MvNormal(zeros(p), G))
# ϕ = (X * β) .+ (Z * u)
# σ² = var(ϕ) * ((1.0 - h²) / h²)
# R = begin
#     R = zeros(n, n)
#     R[diagind(R)] .= σ²
#     R
# end
# ϵ = rand(MvNormal(zeros(n), R))
# y = (X * β) .+ (Z * u) .+ ϵ
# y = (y .- mean(y)) ./ std(y)
# UnicodePlots.histogram(y, nbins=30, title="Phenotype distribution", xlabel="y", ylabel="Frequency")

# # X = (X .- mean(X, dims=1)) ./ std(X, dims=1); X[:,1] .= 1.0
# Z = (Z .- mean(Z, dims=1)) ./ std(Z, dims=1)

# b̂_ols = begin
#     A = hcat(X, Z)
#     inv(A' * A) * (A' * y)
# end

# b̂_lmm, û_lmm = begin
#     # cheating here by using the covariance matrices
#     # R = I(n)
#     # G = I(p)
#     Rinv = inv(R)
#     Ginv = inv(G)
#     A = vcat(
#         hcat(X'*Rinv*X, X'*Rinv*Z),
#         hcat(Z'*Rinv*X, (Z'*Rinv*Z) .+ Ginv),
#     )
#     B = vcat(
#         X'*Rinv*y,
#         Z'*Rinv*y
#     )
#     x̂ = inv(A) * B
#     (x̂[1:m], x̂[(m+1):(m+p)])
# end

# # Predict
# ŷ_ols = hcat(X, Z) * b̂_ols
# ŷ_lmm = (X * b̂_lmm) .+ (Z * û_lmm)
# UnicodePlots.scatterplot(y, ŷ_ols, title="OLS predictions (corr=$(round(100.0*cor(y, ŷ_ols)))%)", xlabel="y", ylabel="ŷ", xlims=(-5, 5), ylims=(-5, 5))
# UnicodePlots.scatterplot(y, ŷ_lmm, title="LMM predictions (corr=$(round(100.0*cor(y, ŷ_lmm)))%)", xlabel="y", ylabel="ŷ", xlims=(-5, 5), ylims=(-5, 5))
# UnicodePlots.scatterplot(b̂_ols, vcat(b̂_lmm, û_lmm), title="OLS vs LMM estimates (corr=$(round(100.0*cor(b̂_ols, vcat(b̂_lmm, û_lmm))))%)", xlabel="b̂_ols", ylabel="b̂_lmm & û_lmm", xlims=(-3, 3), ylims=(-3, 3))
