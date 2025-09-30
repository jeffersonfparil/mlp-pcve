# Find the factors of N which results in the closest positive integer of the square-root
function squarest(N::Int64)
    selections = collect(1:Int(ceil(sqrt(N))))
    n_rows = selections[[N % x for x in selections].==0.0][end]
    n_cols = Int(N / n_rows)
    (n_rows, n_cols)
end

# Simulate explanatory variables
# D = simulatevars()
function simulatevars(;
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
    explanatory_variables = Dict(
        :years => ["2023_2024", "2024_2025", "2025_2026"],
        :seasons => ["Autumn", "Winter", "EarlySpring", "LateSpring", "Summer"],
        :harvests => ["1", "2"],
        :sites => ["Control", "HighN", "Drought", "Waterlogging", "HighTemp"],
        :entries =>
            ["entry_$(string(i, pad=length(string(n_entries))))" for i = 1:n_entries],
        :populations_cor_entries => [
            [
                "population_$(string(i, pad=length(string(n_populations))))" for
                i = 1:n_populations
            ][Int(ceil(rand() * n_populations))] for _ = 1:n_entries
        ],
        :populations => [
            "population_$(string(i, pad=length(string(n_populations))))" for
            i = 1:n_populations
        ],
        :replications => [
            "replication_$(string(i, pad=length(string(n_replications))))" for
            i = 1:n_replications
        ],
        :blocks => ["block_$(string(i, pad=length(string(n_blocks))))" for i = 1:n_blocks],
        :rows => ["block_$(string(i, pad=length(string(n_rows))))" for i = 1:n_rows],
        :cols => ["block_$(string(i, pad=length(string(n_cols))))" for i = 1:n_cols],
    ),
)::Dict{String}
    @assert n_rows * n_cols == n_plots
    labels_X::Vector{String} = []
    for var in [
        :years,
        :seasons,
        :harvests,
        :sites,
        :replications,
        :entries,
        :populations,
        :blocks,
        :rows,
        :cols,
    ]
        for x in explanatory_variables[var]
            push!(labels_X, string(var, "|", x))
        end
    end
    n =
        length(explanatory_variables[:years]) *
        length(explanatory_variables[:seasons]) *
        length(explanatory_variables[:harvests]) *
        length(explanatory_variables[:sites]) *
        length(explanatory_variables[:entries]) *
        length(explanatory_variables[:replications])
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
                        for idx_entries in
                            findall(.!isnothing.(match.(Regex("^entries"), labels_X)))
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
                                    explanatory_variables[:populations] .==
                                    explanatory_variables[:populations_cor_entries][idx_entries-idx_replications_last],
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
                        length(explanatory_variables[:replications]) *
                        length(explanatory_variables[:entries]) /
                        length(explanatory_variables[:blocks]),
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
                        length(explanatory_variables[:replications]) *
                        length(explanatory_variables[:entries]) /
                        length(explanatory_variables[:rows]),
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
                        length(explanatory_variables[:replications]) *
                        length(explanatory_variables[:entries]) /
                        length(explanatory_variables[:cols]),
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
    labels_A = [
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
    k = length(labels_A)
    A::Matrix{String} = reshape(repeat([""], n * k), n, k)
    for k in Symbol.(labels_A)
        # k = labels[1]
        v = explanatory_variables[k]
        println("k=$k; v=$v")
        idx_A = findfirst(labels_A .== string(k))
        idx_X = findall(.!isnothing.(match.(Regex(string(k)), labels_X)))
        A[:, idx_A] = [v[findfirst(X[i, idx_X] .== 1.0)] for i = 1:n]
    end
    # Output
    Dict("X" => X, "labels_X" => labels_X, "A" => A, "labels_A" => labels_A)
end

# Linear model
# y_linear = simulatelineartrait(X, labels_X=labels_X)
function simulatelineartrait(
    X;
    labels_X::Vector{String},
    linear_effects = Dict(
        "years" => (μ = 2.0, σ = 1.0),
        "seasons" => (μ = 2.0, σ = 2.0),
        "harvests" => (μ = 2.0, σ = 1.0),
        "sites" => (μ = 20.0, σ = 5.0),
        "replications" => (μ = 0.0, σ = 0.1),
        "entries" => (μ = 20.0, σ = 10.0),
        "populations" => (μ = 0.0, σ = 5.0),
        "blocks" => (μ = 0.0, σ = 0.1),
        "rows" => (μ = 0.0, σ = 0.1),
        "cols" => (μ = 0.0, σ = 0.1),
    ),
    seed::Int64 = 42,
)::Vector{Float64}
    n, p = size(X)
    β::Vector{Float64} = zeros(Float64, p)
    for (k, v) in linear_effects
        idx = findall(.!isnothing.(match.(Regex("^$(k)"), labels_X)))
        β[idx] = sampleNormal(
            length(idx),
            μ = linear_effects[k].μ,
            σ = linear_effects[k].σ,
            seed = seed,
        )
    end
    y = X * β
    y = (y .- mean(hcat(y))) ./ std(hcat(y))
    y[:, 1]
end

# Non-linear model
# y_nonlinear = simulatenonlineartrait(X)
function simulatenonlineartrait(
    X;
    n_hidden_layers::Int64 = 5,
    F::Function = relu,
    dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
    seed::Int64 = 123,
)::Vector{Float64}
    Ω = init(
        CuArray(Matrix(X')),
        n_hidden_layers = n_hidden_layers,
        F = relu,
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
    y[:, 1]
end

function simulatedata(; 
    n_entries::Int64 = 120,
    n_populations::Int64 = 3,
    n_replications::Int64 = 3,
    explanatory_variables::Union{Dict{Symbol, Vector{String}}, Nothing} = nothing,
    trait_specs::Dict{String} = Dict(
        "trait_1" => Dict(
            "linear" => true,
            "linear_effects" => Dict(
                "years" => (μ = 2.0, σ = 1.0),
                "seasons" => (μ = 2.0, σ = 2.0),
                "harvests" => (μ = 2.0, σ = 1.0),
                "sites" => (μ = 20.0, σ = 5.0),
                "replications" => (μ = 0.0, σ = 0.1),
                "entries" => (μ = 20.0, σ = 10.0),
                "populations" => (μ = 0.0, σ = 5.0),
                "blocks" => (μ = 0.0, σ = 0.1),
                "rows" => (μ = 0.0, σ = 0.1),
                "cols" => (μ = 0.0, σ = 0.1),
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
            "n_hidden_layers" => 5,
            "F" => leakyrelu,
            "dropout_rates" => repeat([0.0], 5),
        )
    ),
)::Dict{String}
    D = if isnothing(explanatory_variables)
        simulatevars(
            n_entries=n_entries,
            n_populations=n_populations,
            n_replications=n_replications,
        )
    else
        simulatevars(
            n_entries=n_entries,
            n_populations=n_populations,
            n_replications=n_replications,
            explanatory_variables=explanatory_variables,
        )
    end
    Y::Matrix{Float64} = fill(NaN, size(D["X"], 1), length(trait_specs))
    labels_Y = collect(keys(sort(trait_specs)))
    for (trait, specs) in sort(trait_specs)
        # trait = string.(keys(sort(trait_specs)))[1]; specs = trait_specs[trait]
        println("Simulating $trait ...")
        y = if specs["linear"]
            simulatelineartrait(
                D["X"],
                labels_X = D["labels_X"],
                linear_effects = specs["linear_effects"],
            )
        else
            simulatenonlineartrait(
                D["X"],
                n_hidden_layers = specs["n_hidden_layers"],
                F = specs["F"],
                dropout_rates = specs["dropout_rates"],
            )
        end
        idx = findfirst(labels_Y .== trait)
        Y[:, idx] = y
    end
    Dict("A" => D["A"], "Y" => Y, "labels_A" => D["labels_A"], "labels_Y" => labels_Y)
end

function write(;
    labels_A::Vector{String}, 
    A::Matrix{String}, 
    y::Vector{Float64}, 
    fname::String
)::String
    open(fname, "w") do io
        println(io, join(["years", "seasons", "harvests", "sites", "replications", "entries", "populations", "blocks", "rows", "cols", "y"], ','))
        n = size(A, 1)
        @inbounds for i = 1:n
            println(io, join(vcat(A[i, :], string(y[i])), ','))
        end
    end
end




function read()
    # Reconstitute X from A
    S = simulatedata()
    A = S["A"]
    y = S["y"]


    n = size(A, 1)
    labels_A = [
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
    levels_A = Dict()
    for (j, k) in enumerate(labels_A)
        levels_A[k] = sort(unique(A[:, j]))
    end
    labels_X::Vector{String} = []
    for k in labels_A
        for x in levels_A[k]
            push!(labels_X, string(k, "|", x))
        end
    end
    p = length(labels_X)
    X = zeros(Float64, n, p)
    @inbounds for (j_A, k) in enumerate(labels_A)
        # j_A = 1; k = labels_A[j_A]
        idx_X = findall(.!isnothing.(match.(Regex("^$k"), labels_X)))
        @inbounds for i = 1:n
            # i = 1
            j_X = idx_X[findfirst(levels_A[k] .== A[i, j_A])]
            X[i, j_X] = 1.0
        end
    end
    Dict("X" => X, "labels" => labels_X, "levels" => levels_A)
end

