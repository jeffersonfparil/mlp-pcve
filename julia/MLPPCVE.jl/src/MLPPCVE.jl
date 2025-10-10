module MLPPCVE

using Random, LinearAlgebra, CUDA

# TODO: 
# Make this more portable, i.e. allowing use of CPU or any type of GPU available in the system
# Current attempt at portability using PackageCompiler.jl:
# - always compile CUDA with the graphics card you want to use the binary on
# - compile: `using PackageCompiler; create_app("MLPPCVE.jl/", "mlppcve-bin")`
# - do the following prior to running the binary: `export JULIA_CUDA_USE_COMPAT=false`
# - test run - no arguments show help docs: `./mlppcve-bin/bin/MLPPCVE`

# (0) Helper functions
include("helpers.jl")
# (1) Activation functions
include("activations.jl")
# (2) Cost functions and their derivatives
include("costs.jl")
# (3) Forward pass and backpropagation including the Network struct and initialisation
include("forwardbackward.jl")
# (4) Optimisers
include("optimisers.jl")
# (5) Training and validation
include("traintest.jl")
# (6) Input and output
include("io.jl")

export Network, init, mean, var, cov, std, sampleNormal, drawreplacenot, metrics, simulate
export Trial, squarest, levenshteindistance, isfuzzymatch, inittrial, simulatelineartrait!, simulatenonlineartrait!, simulatetrial
export writetrial, readtrial
export linear,
    linear_derivative,
    sigmoid,
    sigmoid_derivative,
    hyperbolictangent,
    hyperbolictangent_derivative,
    relu,
    relu_derivative,
    leakyrelu,
    leakyrelu_derivative
export MSE, MSE_derivative, MAE, MAE_derivative, HL, HL_derivative
export forwardpass!, backpropagation!
export gradientdescent!, Adam!, AdamMax!
export splitdata, predict, train, optim

# Testing
function args_parser()
    # ARGS is implicit?!
    if length(ARGS) == 0 || ARGS[1] == "--help" || ARGS[1] == "-h" || ARGS[1] == "help" || ARGS[1] == "h"
        println("Usage: ./mlppcve-bin/bin/MLPPCVE (1) path to the CSV file, (2) delimiter (character or string), (3) maximum number of hidden layers to test (integer), (4) number of batches (integer).")
        println("Example: ./mlppcve-bin/bin/MLPPCVE simulated_trial_data.csv , 1 2")
        exit(0)
    elseif ARGS[1] == "--example" || ARGS[1] == "example"
        println("Analysing a simulated trial with default parameters, i.e. max_layers=3 and n_batches=2.")
        fname = writetrial(simulatetrial(verbose=false))
        delimiter = ","
        max_layers = 3
        n_batches = 2
        (fname, delimiter, max_layers, n_batches)
    elseif length(ARGS) == 4
        fname = ARGS[1]
        delimiter = ARGS[2]
        max_layers = parse(Int64, ARGS[3])
        n_batches = parse(Int64, ARGS[4])
        (fname, delimiter, max_layers, n_batches)
    else
        println("Incorrect number of arguments provided. Provide: (1) path to the CSV file, (2) delimiter (character or string), (3) maximum number of hidden layers to test (integer), (4) number of batches (integer).")
        println("Example: ./mlppcve-bin/bin/MLPPCVE simulated_trial_data.csv , 3 2")
        exit(1)
    end
end

function julia_main()::Cint
    
    
    # TODO: a more ergonomic parser? Or just make it as simple no need for another external library?
    # TODO: output results to a file or fixed_explanatory_variables

    # Parse arguments
    fname, delimiter, max_layers, n_batches = args_parser()
    T::Type = Float32
    expected_labels_A::Vector{String} = ["years", "seasons", "harvests", "sites", "replications", "entries", "populations", "blocks", "rows", "cols"]
    opt_n_hidden_layers::Vector{Int64} = collect(0:max_layers)
    opt_n_nodes_per_hidden_layer::Vector{Int64} = [128, 256]
    opt_dropout_per_hidden_layer::Vector{Float64} = [0.0]
    opt_F_∂F::Vector{Dict{Symbol,Function}} = [Dict(:F => relu, :∂F => relu_derivative), Dict(:F => leakyrelu, :∂F => leakyrelu_derivative)]
    opt_C_∂C::Vector{Dict{Symbol,Function}} = [Dict(:C => MSE, :∂C => MSE_derivative)]
    opt_n_epochs::Vector{Int64} = [10_000]
    opt_n_burnin_epochs::Vector{Int64} = [100]
    opt_n_patient_epochs::Vector{Int64} = [5]
    opt_optimisers::Vector{String} = ["Adam"]
    n_threads::Int64 = 2
    seed::Int64 = 42
    verbose::Bool = true

    # Read the trial data
    trial = readtrial(fname=fname, delimiter=delimiter, expected_labels_A=expected_labels_A)
    # Remove fixed explanatory variables
    nonfixed_explanatory_variables = String[]
    fixed_explanatory_variables = String[]
    for j in 1:length(trial.labels_A)
        if length(unique(trial.A[:, j])) > 1
            push!(nonfixed_explanatory_variables, trial.labels_A[j])
        else
            push!(fixed_explanatory_variables, trial.labels_A[j])
        end
    end
    if verbose
        println("Fixed explanatory variables: $(fixed_explanatory_variables)")
        println("Non-fixed explanatory variables: $(nonfixed_explanatory_variables)")
    end
    idx = sort(vcat([findall(.!isnothing.(match.(Regex(x), trial.labels_X))) for x in nonfixed_explanatory_variables]...))
    Xy::Dict{String,CuArray{T,2}} = Dict(
        "X" => CuArray(T.(trial.X[:, idx]')), 
        "y" => CuArray(T.(trial.Y[:, 1]')),
    )
    labels_X_nonfixed = trial.labels_X[idx]
    for (j, trait) in enumerate(trial.labels_Y)
        # j = 2; trait = trial.labels_Y[j]
        if verbose
            println("Fitting trait ($j or $(length(trial.labels_Y))): $trait")
        end
        Xy["y"] = CuArray(T.(trial.Y[:, j]'))
        dl = train(Xy, n_hidden_layers=1, n_batches=n_batches, seed=seed, verbose=verbose)
        @show dl["metrics_training"]
        @show dl["metrics_training_ols"]
        @show dl["metrics_validation"]
        @show dl["metrics_validation_ols"]
        dl_optim = optim(
            Xy,
            n_batches=n_batches,
            opt_n_hidden_layers=opt_n_hidden_layers,
            opt_n_nodes_per_hidden_layer=[size(Xy["X"], 1)],
            opt_dropout_per_hidden_layer=opt_dropout_per_hidden_layer,
            opt_F_∂F=opt_F_∂F,
            opt_C_∂C=opt_C_∂C,
            opt_n_epochs=opt_n_epochs,
            opt_n_burnin_epochs=opt_n_burnin_epochs,
            opt_n_patient_epochs=opt_n_patient_epochs,
            opt_optimisers=opt_optimisers,
            n_threads=n_threads,
            seed=seed,
        )
        @show dl_optim["Full_fit"]["metrics_training"]
    end
    return 0
end

end
