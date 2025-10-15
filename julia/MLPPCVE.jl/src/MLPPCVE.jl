module MLPPCVE

using Random, LinearAlgebra, CUDA, JLD2

# TODO: 
# Make this more portable, i.e. allowing use of CPU or any type of GPU available in the system
# Current attempt at portability using PackageCompiler.jl and Julia version 1.11 (version 1.12 fails):
# - always compile CUDA with the graphics card you want to use the binary on
# - compile: `julia +1.11 -E 'using Pkg; Pkg.resolve(); using PackageCompiler; create_app("MLPPCVE.jl/", "mlppcve-bin")'`
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
        println("\t‣ Run example: ./mlppcve-bin/bin/MLPPCVE --example")
        println("\t‣ Run on some data: ./mlppcve-bin/bin/MLPPCVE some_trial_data.csv , 1 2")
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


        # TODO: fix issue with delimiter and input file parsing
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
    
    # Parse arguments
    fname, delimiter, max_layers, n_batches = args_parser()
    # fname, delimiter, max_layers, n_batches = (writetrial(simulatetrial(verbose=false)), ",", 3, 2)
    # # fname = "/group/pasture/forages/Ryegrass/STR/NUE_WUE_merged_2022_2025/phenotypes/STR-NUE_WUE-Hamilton_Tatura-2023_2025.tsv"; delimiter = "\t"
    T::Type = Float32
    expected_labels_A::Vector{String} = ["years", "seasons", "harvests", "sites", "replications", "entries", "populations", "blocks", "rows", "cols"]
    opt_n_hidden_layers::Vector{Int64} = collect(0:max_layers)
    opt_n_nodes_per_hidden_layer::Vector{Int64} = [128, 256]
    opt_dropout_per_hidden_layer::Vector{Float64} = [0.0]
    opt_F_∂F::Vector{Dict{Symbol,Function}} = [Dict(:F => relu, :∂F => relu_derivative), Dict(:F => leakyrelu, :∂F => leakyrelu_derivative)]
    opt_C_∂C::Vector{Dict{Symbol,Function}} = [Dict(:C => MSE, :∂C => MSE_derivative)]
    opt_n_epochs::Vector{Int64} = [10_000]
    opt_n_burnin_epochs::Vector{Int64} = [10, 100]
    opt_n_patient_epochs::Vector{Int64} = [5]
    opt_optimisers::Vector{String} = ["Adam"]
    n_threads::Int64 = 2
    seed::Int64 = 42
    output_prefix::String = "mlppcve"
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
    idx_vars = sort(vcat([findall(.!isnothing.(match.(Regex(x), trial.labels_X))) for x in nonfixed_explanatory_variables]...)) # remove fixed variables
    for (j, trait) in enumerate(trial.labels_Y)
        # j = 1; trait = trial.labels_Y[j]
        if verbose
            println("Fitting trait ($j or $(length(trial.labels_Y))): $trait")
        end
        idx_obs = findall(.!isnan.(trial.Y[:, j])) # remove observations with missing trait values
        Xy::Dict{String,CuArray{T,2}} = Dict(
            "X" => CuArray(T.(trial.X[idx_obs, idx_vars]')), 
            "y" => CuArray(T.(trial.Y[idx_obs, j]')),
        )
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
        # @show dl_optim["Full_fit"]["metrics_training"]
        # @show dl_optim["Full_fit"]["metrics_training_ols"]
        # Extract uncontextualised effects
        ϕ_nocontext::Vector{T} = []
        ϕ_nocontext_labels::Vector{String} = []
        x_new = CuArray(zeros(T, size(Xy["X"], 1), 1))
        for explanatory_variable in nonfixed_explanatory_variables
            # explanatory_variable = nonfixed_explanatory_variables[3]
            idx = findall(.!isnothing.(match.(Regex("^$explanatory_variable"), trial.labels_X)))
            ϕ_tmp = []
            for i in idx
                # i = idx[1]
                x_new .= 0.0
                x_new[i, :] .= 1.0
                push!(ϕ_tmp, Matrix(predict(dl_optim["Full_fit"]["Ω"], x_new))[1, 1])
                push!(ϕ_nocontext_labels, trial.labels_X[i])
            end
            # Standardised effects within each explanatory variable
            ϕ_tmp = (ϕ_tmp .- mean(hcat(ϕ_tmp))) ./ (std(hcat(ϕ_tmp)) .+ eps(Float64)) # add ϵ to avoid NaNsS
            ϕ_nocontext = vcat(ϕ_nocontext, ϕ_tmp[:, 1])
        end
        # Extract contextualised effects
        requested_contextualised_effects = ["seasons-entries", "sites-entries", "seasons-sites-entries"]
        Φ_context = Dict()
        for contextualised_effect in requested_contextualised_effects
            # contextualised_effect = requested_contextualised_effects[3]
            factors = String.(split(contextualised_effect, '-'))
            idx_factors = [findall(.!isnothing.(match.(Regex("^$factor"), trial.labels_X))) for factor in factors]
            idx_combinations = collect(Iterators.product(idx_factors...))
            x_new = CuArray(zeros(T, size(Xy["X"], 1), 1))
            ϕ_tmp = []
            ϕ_tmp_labels = []
            for idx_combination in idx_combinations
                # idx_combination = idx_combinations[1]
                x_new .= 0.0
                for i in collect(idx_combination)
                    x_new[i, :] .= 1.0
                end
                push!(ϕ_tmp, Matrix(predict(dl_optim["Full_fit"]["Ω"], x_new))[1, 1])
                push!(ϕ_tmp_labels, join(trial.labels_X[collect(idx_combination)], '-'))
            end
            # Standardised effects within each combination of explanatory variables
            ϕ_tmp = (ϕ_tmp .- mean(hcat(ϕ_tmp))) ./ std(hcat(ϕ_tmp))
            Φ_context[contextualised_effect] = Dict(
                "ϕ_context" => ϕ_tmp[:, 1],
                "ϕ_context_labels" => ϕ_tmp_labels,
            )
        end
        # Outputs
        output = Dict(
            "dl_optim" => dl_optim,
            "ϕ_nocontext" => ϕ_nocontext,
            "ϕ_nocontext_labels" => ϕ_nocontext_labels,
            "Φ_context" => Φ_context,
        )
        # Save outputs
        fnames_outputs = vcat(
            "$output_prefix-$trait-output.jld2",
            "$(output_prefix)-$(trait)-nocontext_standardised_effects.tsv",
            ["$(output_prefix)-$(trait)-$(replace(k, "-" => "_"))_standardised_effects.tsv" for k in keys(Φ_context)]
        )
        # Save all output into a JLD2 (HDF5-compatible format)
        JLD2.save(fnames_outputs[1], output)
        # Save context-less standardised effects of each explanatory variable level
        open(fnames_outputs[2], "w") do file
            write(file, "#explanatory_variables\tlevels\teffects\n")
            write(file, join(string.(replace.(ϕ_nocontext_labels, "|" => "\t"), "\t", ϕ_nocontext), "\n"))
        end
        # Save contextual standardised effects - with one file for each explanatory variable combination requested
        for (i, (k, v)) in enumerate(Φ_context)
            # k = String.(keys(Φ_context))[1]; v = Φ_context[k]
            open(fnames_outputs[i+2], "w") do file
                explanatory_variables = String.(split(k, "-"))
                for var in explanatory_variables
                    v["ϕ_context_labels"] = replace.(v["ϕ_context_labels"], "$var|" => "")
                end
                write(file, "#")
                write(file, join(vcat(explanatory_variables, "effects\n"), "\t"))
                write(file, join(string.(replace.(v["ϕ_context_labels"], "-" => "\t"), "\t", v["ϕ_context"]), "\n"))
            end
        end
        println("##################################################")
        println(join(vcat("$(trait) | output files:", fnames_outputs), "\n\t‣ "))
    end
    return 0
end

end
