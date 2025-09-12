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
    explanatory_variables = Dict(
        :years => ["2023_2024", "2024_2025", "2025_2026"],
        :seasons => ["Autumn", "Winter", "EarlySpring", "LateSpring", "Summer"],
        :harvests => ["1", "2"],
        :sites => ["Control", "HighN", "Drought", "Waterlogging", "HighTemp"],
        :entries => ["entry_$(string(i, pad=length(string(n_entries))))" for i in 1:n_entries],
        :populations => [["population_$(string(i, pad=length(string(n_populations))))" for i in 1:n_populations][Int(ceil(rand()*n_populations))] for _ in 1:n_entries],
        :replications => ["replication_$(string(i, pad=length(string(n_replications))))" for i in 1:n_replications],
        :blocks => ["block_$(string(i, pad=length(string(n_blocks))))" for i in 1:n_blocks],
        :rows => ["block_$(string(i, pad=length(string(n_rows))))" for i in 1:n_rows],
        :cols => ["block_$(string(i, pad=length(string(n_cols))))" for i in 1:n_cols],
    )

    m_years = length(explanatory_variables[:years])
    m_seasons = m_years + length(explanatory_variables[:seasons])
    m_harvests = m_seasons + length(explanatory_variables[:harvests])
    m_sites = m_harvests + length(explanatory_variables[:sites])
    m_replications = m_sites + length(explanatory_variables[:replications])
    m_entries = m_replications + length(explanatory_variables[:entries])
    # sorted unique populations as this has a one-to-one correspondence with the entries vector
    unique_populations = sort(unique(explanatory_variables[:populations]))
    m_populations = m_entries + length(unique_populations) 
    m_blocks = m_populations + length(explanatory_variables[:blocks])
    rep_blocks = Int64(length(explanatory_variables[:replications])*length(explanatory_variables[:entries]) / length(explanatory_variables[:blocks]))
    m_rows = m_blocks + length(explanatory_variables[:rows])
    rep_rows = Int64(length(explanatory_variables[:replications])*length(explanatory_variables[:entries]) / length(explanatory_variables[:rows]))
    m_cols = m_rows + length(explanatory_variables[:cols])
    rep_cols = Int64(length(explanatory_variables[:replications])*length(explanatory_variables[:entries]) / length(explanatory_variables[:cols]))

    n = length(explanatory_variables[:years]) * length(explanatory_variables[:seasons]) * length(explanatory_variables[:harvests]) * length(explanatory_variables[:sites]) * length(explanatory_variables[:entries]) * length(explanatory_variables[:replications])
    p = m_cols
    X = zeros(Float64, n, p)


    idx_row = 1
    for idx_years in eachindex(explanatory_variables[:years])
        for idx_seasons in m_years .+ eachindex(explanatory_variables[:seasons])
            for idx_harvests in m_seasons .+ eachindex(explanatory_variables[:harvests])
                for idx_sites in m_harvests .+ eachindex(explanatory_variables[:sites])


                    # Replications, entries, populations and the higher order grouping variables above
                    idx_row_sub = idx_row
                    for idx_replications in m_sites .+ .+ eachindex(explanatory_variables[:replications])
                        for idx_entries in m_replications .+ eachindex(explanatory_variables[:entries])
                            idx_populations = findall(unique_populations .== explanatory_variables[:populations][idx_entries - m_replications])[1]
                            idx_cols = [idx_years, idx_seasons, idx_harvests, idx_sites, idx_replications, idx_entries, idx_populations]
                            X[idx_row, idx_cols] .= 1.0
                            idx_row += 1
                        end
                    end

                    # Blocks
                    idx_row = idx_row_sub
                    for idx_cols in repeat(
                        m_populations .+ eachindex(explanatory_variables[:blocks]), 
                        inner=rep_blocks
                    )
                        X[idx_row, idx_cols] = 1.0
                        idx_row += 1
                    end

                    # Rows
                    idx_row = idx_row_sub
                    for idx_cols in repeat(
                        m_blocks .+ eachindex(explanatory_variables[:rows]), 
                        inner=rep_rows
                    )
                        X[idx_row, idx_cols] = 1.0
                        idx_row += 1
                    end

                    # Cols
                    idx_row = idx_row_sub
                    for idx_cols in repeat(
                        m_rows .+ eachindex(explanatory_variables[:cols]), 
                        inner=rep_cols
                    )
                        X[idx_row, idx_cols] = 1.0
                        idx_row += 1
                    end


                end
            end
        end
    end

    β = randn(p)
    ϵ = randn(n)
    y = X * β .+ ϵ

    
end

function write()
end