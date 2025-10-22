function main(args::Dict)::Vector{String}
    # args = Dict(
    #     "fname" => writetrial(simulatetrial(verbose=false)),
    #     "delimiter" => ",",
    #     "n-batches" => 2,
    #     "expected-labels-A" => ["years","seasons","harvests","sites","replications","entries","populations","blocks","rows","cols",],
    #     "requested-contextualised-effects" => ["years-entries", "seasons-entries", "sites-entries", "years-seasons-sites-entries"],
    #     "traits" => [""],
    #     "opt-n-hidden-layers" => [0, 1, 2, 3],
    #     "opt-n-nodes-per-hidden-layer" => [128, 256],
    #     "opt-dropout-per-hidden-layer" => [0.0],
    #     "opt-n-epochs" => [1_000],
    #     "opt-n-burnin-epochs" => [10, 100],
    #     "opt-n-patient-epochs" => [5],
    #     "opt-optimisers" => ["Adam"],
    #     "opt-activation-functions" => [Dict(:F => relu, :∂F => relu_derivative), Dict(:F => leakyrelu, :∂F => leakyrelu_derivative)],
    #     "opt-cost-functions" => [Dict(:C => MSE, :∂C => MSE_derivative)],
    #     "learn-rate" => 0.001,
    #     "decay-rate-1" => 0.9,
    #     "decay-rate-2" => 0.999,
    #     "epsilon" => 1e-8,
    #     "n-threads" => 2,
    #     "seed" => 42,
    #     "output-prefix" => "mlppcve",
    #     "verbose" => true,
    # )
    # # args["fname"] = "/home/jp3h/Documents/mlp-pcve/julia/test_empirical.tsv"; args["delimiter"] = "\t";
    # Constant type which we may allow to be user-specified in the future
    T::Type = Float32
    fname = args["fname"]
    delimiter = args["delimiter"]
    n_batches = args["n-batches"]
    expected_labels_A = args["expected-labels-A"]
    requested_contextualised_effects = args["requested-contextualised-effects"]
    traits = args["traits"]
    opt_n_hidden_layers = args["opt-n-hidden-layers"]
    opt_n_nodes_per_hidden_layer = args["opt-n-nodes-per-hidden-layer"]
    opt_dropout_per_hidden_layer = args["opt-dropout-per-hidden-layer"]
    opt_n_epochs = args["opt-n-epochs"]
    opt_n_burnin_epochs = args["opt-n-burnin-epochs"]
    opt_n_patient_epochs = args["opt-n-patient-epochs"]
    opt_optimisers = args["opt-optimisers"]
    opt_F_∂F = args["opt-activation-functions"]
    opt_C_∂C = args["opt-cost-functions"]
    η = args["learn-rate"]
    β₁ = args["decay-rate-1"]
    β₂ = args["decay-rate-2"]
    ϵ = args["epsilon"]
    n_threads = args["n-threads"]
    seed = args["seed"]
    output_prefix = args["output-prefix"]
    verbose = args["verbose"]
    # Check for existing output files 
    fnames_output = readdir()
    idx = findall(.!isnothing.(match.(Regex("^$(output_prefix)-"), fnames_output)) .&& .!isnothing.(match.(Regex(".jld2\$"), fnames_output)))
    if length(idx) > 0
        error("Please re/move the existing output file/s with the prefix: \"$(output_prefix)\".")
    end
    # Read the trial data
    trial = readtrial(fname=fname, delimiter=delimiter, expected_labels_A=expected_labels_A)
    # Validate trait/s input
    traits = if (length(traits) == 1) && (traits[1] == "")
        trial.labels_Y
    else
        for trait in traits
            if trait ∉ trial.labels_Y
                error("The supplied trait: \"$trait\" does not exist in the input file: $fname. Valid traits are: \n\t‣ $(join(trial.labels_Y, "\nt\t‣ "))")
            end
        end
        traits
    end
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
    fnames_output = []
    idx_vars = sort(vcat([findall(.!isnothing.(match.(Regex(x), trial.labels_X))) for x in nonfixed_explanatory_variables]...)) # remove fixed variables
    if length(idx_vars) < 2
        error("There is less than 2 non-fixed explanatory variables:\n\t‣ $(join(trial.labels_X[idx_vars], "\n\t‣ "))")
    end
    if length(idx_vars) < length(trial.labels_X)
        idx_tmp = findall([!(x ∈ idx_vars) for x in collect(1:length(trial.labels_X))])
        @warn("Dropped the following explanatory variable level/s:\n\t‣ $(join(trial.labels_X[idx_tmp], "\n\t‣ "))")
    end
    labels_X = trial.labels_X[idx_vars]
    for (j, trait) in enumerate(traits)
       # j = 1; trait = trial.labels_Y[j]
        if verbose
            println("Fitting trait ($j or $(length(trial.labels_Y))): $trait")
        end
        y, μ_y, σ_y, idx_obs = begin
            y =  trial.Y[:, j]
            idx_obs = findall(.!isnan.(y)) # remove observations with missing trait values
            y = y[idx_obs]
            μ_y = mean(hcat(y))[1,1]
            σ_y = std(hcat(y))[1,1]
            y = (y .- μ_y) ./ (σ_y .+ eps(T))
            (y, μ_y, σ_y, idx_obs)
        end
        if length(y) < 10
            @warn("Skipped trait \"$trait\" because it does not have enough non-missing observations.")
            continue
        end
        if σ_y < 1e-12
            @warn("Skipped trait \"$trait\" because it does not have variation.")
            continue
        end
        Xy::Dict{String,CuArray{T,2}} = Dict(
            "X" => CuArray(T.(trial.X[idx_obs, idx_vars]')), 
            "y" => CuArray(T.(y')),
        )
        dl_optim = optim(
            Xy,
            n_batches=n_batches,
            opt_n_hidden_layers=opt_n_hidden_layers,
            # opt_n_nodes_per_hidden_layer=[size(Xy["X"], 1)],
            opt_n_nodes_per_hidden_layer=opt_n_nodes_per_hidden_layer,
            opt_dropout_per_hidden_layer=opt_dropout_per_hidden_layer,
            opt_F_∂F=opt_F_∂F,
            opt_C_∂C=opt_C_∂C,
            opt_n_epochs=opt_n_epochs,
            opt_n_burnin_epochs=opt_n_burnin_epochs,
            opt_n_patient_epochs=opt_n_patient_epochs,
            opt_optimisers=opt_optimisers,
            η=η,
            β₁=β₁,
            β₂=β₂,
            ϵ=ϵ,
            n_threads=n_threads,
            seed=seed,
        )
        @show dl_optim["Full_fit"]["metrics_training"]
        @show dl_optim["Full_fit"]["metrics_training_ols"]
        # Extract uncontextualised effects
        ϕ_nocontext::Vector{T} = []
        ϕ_nocontext_labels::Vector{String} = []
        x_new = CuArray(zeros(T, size(Xy["X"], 1), 1))
        for explanatory_variable in nonfixed_explanatory_variables
            # explanatory_variable = nonfixed_explanatory_variables[end]
            # @show explanatory_variable
            idx = findall(.!isnothing.(match.(Regex("^$explanatory_variable"), labels_X)))
            ϕ_tmp = []
            for i in idx
                # i = idx[1]
                x_new .= 0.0
                x_new[i, :] .= 1.0
                push!(ϕ_tmp, Matrix(predict(dl_optim["Full_fit"]["Ω"], x_new))[1, 1])
                push!(ϕ_nocontext_labels, labels_X[i])
            end
            # Standardised effects within each explanatory variable
            ϕ_tmp = (ϕ_tmp .- mean(hcat(ϕ_tmp))) ./ (std(hcat(ϕ_tmp)) .+ eps(Float64)) # add ϵ to avoid NaNsS
            ϕ_nocontext = vcat(ϕ_nocontext, ϕ_tmp[:, 1])
        end
        # Extract contextualised effects
        Φ_context = Dict()
        for contextualised_effect in requested_contextualised_effects
            # contextualised_effect = requested_contextualised_effects[1]
            factors = String.(split(contextualised_effect, '-'))
            idx_factors = [findall(.!isnothing.(match.(Regex("^$factor"), labels_X))) for factor in factors]
            idx_combinations = collect(Iterators.product(idx_factors...))
            x_new = CuArray(zeros(T, size(Xy["X"], 1), 1))
            ϕ_tmp = T[]
            ϕ_tmp_labels = String[]
            for idx_combination in idx_combinations
                # idx_combination = idx_combinations[1]
                x_new .= 0.0
                for i in collect(idx_combination)
                    x_new[i, :] .= 1.0
                end
                push!(ϕ_tmp, Matrix(predict(dl_optim["Full_fit"]["Ω"], x_new))[1, 1])
                # push!(ϕ_tmp_labels, join(labels_X[collect(idx_combination)], '-'))
                push!(ϕ_tmp_labels, join(labels_X[collect(idx_combination)], '\t'))
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
            "trait" => trait,
            "μ_y" => μ_y,
            "σ_y" => σ_y,
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
                # write(file, join(string.(replace.(v["ϕ_context_labels"], "-" => "\t"), "\t", v["ϕ_context"]), "\n"))
                write(file, join(string.(v["ϕ_context_labels"], "\t", v["ϕ_context"]), "\n"))
            end
        end
        println("##################################################")
        println(join(vcat("$(trait) | output files:", fnames_outputs), "\n\t‣ "))
        fnames_output = vcat(fnames_output, fnames_outputs)
    end
    return fnames_output
end
