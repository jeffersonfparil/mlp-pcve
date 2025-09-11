function read()
    n_entries::Int64 = 120; n_populations = 3; n_replications = 3; 
    n_plots = n_entries * n_replications
    function squarest(N::Int64)
        selections = collect(1:Int(ceil(sqrt(N))))
        n_rows = selections[[N % x for x in selections] .== 0.0][end]
        n_cols = Int(N / n_rows)
        (n_rows, n_cols)
    end
    n_blocks, n_plots_per_block = squarest(n_plots)
    n_block_rows, n_block_cols = squarest(n_blocks)
    n_rows_per_block, n_cols_per_block = squarest(n_plots_per_block)
    n_rows = n_block_rows * n_rows_per_block
    n_cols = n_block_cols * n_cols_per_block
    @assert n_rows*n_cols == n_plots
    years::Vector{String} = ["2023_2024", "2024_2025", "2025_2026"]
    seasons::Vector{String} = ["Autumn", "Winter", "EarlySpring", "LateSpring", "Summer"]
    harvests::Vector{String} = ["1", "2"]
    sites::Vector{String} = ["Control", "HighN", "Drought", "Waterlogging", "HighTemp"]
    entries::Vector{String} = ["entry_$(string(i, pad=length(string(n_entries))))" for i in 1:n_entries]
    populations::Vector{String} = [["population_$(string(i, pad=length(string(n_populations))))" for i in 1:n_populations][Int(ceil(rand()*n_populations))] for _ in 1:n]
    replications::Vector{String} = ["replication_$(string(i, pad=length(string(n_replications))))" for i in 1:n_replications]
    blocks::Vector{String} = ["block_$(string(i, pad=length(string(n_blocks))))" for i in 1:n_blocks]
    rows::Vector{String} = ["block_$(string(i, pad=length(string(n_rows))))" for i in 1:n_rows]
    cols::Vector{String} = ["block_$(string(i, pad=length(string(n_cols))))" for i in 1:n_cols]

end

function write()
end