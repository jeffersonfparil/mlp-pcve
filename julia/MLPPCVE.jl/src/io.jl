"""
    writetrial(trial::Trial; fname::String="simulated_trial_data.csv", delimiter::Union{Char,String}=",")::String

Write trial data to a CSV file.

# Arguments
- `trial::Trial`: Trial object containing data to be written
- `fname::String="simulated_trial_data.csv"`: Output filename for the CSV file
- `delimiter::Union{Char,String}=","`: Delimiter character or string for CSV format

# Returns
- `String`: The filename where data was written

# Description
Writes trial data to a CSV file with the following format:
- First row contains headers combining trial.labels_A and trial.labels_Y
- Subsequent rows contain data from trial.A and trial.Y matrices
- Data fields are separated by the specified delimiter

# Example
```jldoctest; setup = :(using MLPPCVE)
julia> trial = simulatetrial(verbose=false);

julia> fname = writetrial(trial)
"simulated_trial_data.csv"
```
"""
function writetrial(
    trial::Trial;
    fname::String = "simulated_trial_data.csv",
    delimiter::String = ",",
)::String
    open(fname, "w") do io
        println(io, join(vcat(trial.labels_A, trial.labels_Y), delimiter))
        n = size(trial.A, 1)
        @inbounds for i = 1:n
            # i = 1
            println(
                io, 
                string(
                    join(trial.A[i, :], delimiter), 
                    delimiter, 
                    join(trial.Y[i, :], delimiter)
                )
            )
        end
    end
    fname
end

"""
    readtrial(; fname::String, delimiter::String=",", 
    missing_strings::Vector{String}=["", "NA", "NaN", "missing", "NAN", "na", "nan", "missing", "Missing", "MISSING"],
    expected_labels_A::Vector{String}=["years", "seasons", "harvests", "sites", "replications", 
    "entries", "populations", "blocks", "rows", "cols"])::Trial

Read trial data from a CSV-like file and construct a `Trial` object.

# Arguments
- `fname::String`: Path to the input file
- `delimiter::String=","`: Character or string used to separate values in the file
- `missing_strings::Vector{String}`: Strings to be interpreted as missing values and set to NaN
- `expected_labels_A::Vector{String}`: Expected factor names to match against column headers

# Returns
- `Trial`: A Trial object containing:
  - `X`: Design matrix (n × p) with dummy variables
  - `Y`: Response matrix (n × t) with observations (missing values as NaN)
  - `A`: Factor matrix (n × f) with factor levels as strings
  - `labels_X`: Vector of column names for X (length p)
  - `labels_Y`: Vector of column names for Y (length t)
  - `labels_A`: Vector of factor names (length f)

# Notes
- Missing values in response variables (Y) are stored as NaN instead of Julia's `missing` type
- The function recognises various string representations of missing values (e.g., "NA", "missing", "")
  and converts them to NaN in the Y matrix

# Format
The input file should be a delimited text file with:
- A header row containing factor names and response variable names
- Data rows containing factor levels and response values
- Factor names should match (fuzzy) with: years, seasons, harvests, sites, replications, 
  entries, populations, blocks, rows, cols

# Throws
- Error if file doesn't exist
- Error if file has less than 2 columns
- Error if any row has different number of columns than header
- Error if response columns cannot be parsed as Float64

# Example
```jldoctest; setup = :(using MLPPCVE)
julia> trial = simulatetrial(verbose=false);

julia> fname = writetrial(trial);

julia> trial_read = readtrial(fname=fname);

julia> trial.X[:, sortperm(trial.labels_X)] == trial_read.X[:, sortperm(trial_read.labels_X)]
true

julia> trial.Y[:, sortperm(trial.labels_Y)] == trial_read.Y[:, sortperm(trial_read.labels_Y)]
true

julia> trial.A[:, sortperm(trial.labels_A)] == trial_read.A[:, sortperm(trial_read.labels_A)]
true

julia> sort(trial.labels_X) == sort(trial_read.labels_X)
true

julia> sort(trial.labels_Y) == sort(trial_read.labels_Y)
true

julia> sort(trial.labels_A) == sort(trial_read.labels_A)
true
```
"""
function readtrial(;
    fname::String,
    delimiter::String = ",",
    missing_strings::Vector{String} = ["", "NA", "NaN", "missing", "NAN", "na", "nan", "missing", "Missing", "MISSING"],
    expected_labels_A::Vector{String} = [
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
    ],
)::Trial
    # fname = writetrial(simulatetrial())
    # delimiter = ","
    # expected_labels_A = ["years", "seasons", "harvests", "sites", "replications", "entries", "populations", "blocks", "rows", "cols"]
    isfile(fname) || error("File $fname does not exist.")
    # Extract header information, i.e. labels_A and labels_Y
    io = open(fname, "r")
    header = String.(split(readline(io), delimiter))
    if length(header) < 2
        error("File \"$fname\" has less than 2 columns. Delimiter: \"$delimiter\" may not be correct.")
    end
    close(io)
    labels_A = String[]
    labels_Y = String[]
    for h in header
        # h = header[1]
        if sum([isfuzzymatch(h, a) for a in expected_labels_A]) > 0
            push!(labels_A, h)
        else
            push!(labels_Y, h)
        end
    end
    # Extract A    
    n = countlines(fname) - 1
    A = fill("", n, length(labels_A))
    Y = fill(NaN64, n, length(labels_Y))
    idx_A = indexin(labels_A, header)
    idx_Y = indexin(labels_Y, header)
    open(fname, "r") do io
        readline(io) # skip header
        @inbounds for i = 1:n
            # i = 1
            ay = split(readline(io), delimiter)
            if length(ay) != length(header)
                error("Row $i in file \"$fname\" does not have the same number of columns as the header. Delimiter: \"$delimiter\" may not be correct.")
            end
            y = try
                [x ∈ missing_strings ? NaN : parse.(Float64, x) for x in ay[idx_Y]]
            catch
                error("Column/s $(join(labels_Y, ", ")) in file \"$fname\" cannot be parsed as Float64.")
            end
            A[i, :] = String.(ay[idx_A])
            Y[i, :] = y
        end
    end
    # Extract levels of A to get labels_X
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
    # Extract X
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
    Trial(
        X,
        Y,
        A,
        labels_X,
        labels_Y,
        labels_A,
    )
end

